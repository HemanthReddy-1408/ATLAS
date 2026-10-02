"""Retrieval quality gates, the agent loop, answer verification, the Groq client, and evaluation machinery."""

from __future__ import annotations

import json
import os

import httpx
import pytest

from atlas.config import Settings
from atlas.domain import Evidence
from atlas.evalset import EVAL_SET
from atlas.evaluate import (
    attribute_failure,
    eval_answers,
    eval_retrieval,
    key_point_recall,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    summarize_answers,
)
from atlas.generate import ClaimVerifier, normalize_citations, normalize_text
from atlas.llm import FakeLLM, GroqLLM, LLMError, parse_json
from atlas.retrieve import Filters


# -------------------------------------------------------- metric maths
def test_metric_definitions():
    ranked, gold = ["x", "a", "y", "b"], {"a", "b", "c"}
    assert recall_at_k(["d1", "d2", "d1"], {"d1", "d2", "d3", "d4"}, 3) == 0.5
    assert precision_at_k(ranked, gold, 4) == 0.5
    assert mrr(ranked, gold) == 0.5 and mrr(["x"], gold) == 0.0
    # ideal ranking => 1.0; relevant docs pushed down => <1
    assert ndcg_at_k(["a", "b", "c"], gold, 3) == pytest.approx(1.0)
    assert 0 < ndcg_at_k(ranked, gold, 4) < 1


def test_key_point_matching_supports_alternatives():
    assert key_point_recall("Meta shipped it on April 18, 2024", ["April 18, 2024", "Apache"]) == 0.5
    assert key_point_recall("uses AWS", ["Amazon|AWS"]) == 1.0


# ---------------------------------------------- retrieval quality gates
def test_hybrid_retrieval_meets_quality_gates_and_beats_weakest_single_retriever(corpus):
    scores = {s.mode: s for s in eval_retrieval(corpus)}
    assert scores["hybrid"].recall10 >= 0.9 and scores["hybrid"].mrr >= 0.85 and scores["hybrid"].ndcg10 >= 0.8
    assert scores["rrf"].recall5 >= max(scores["bm25"].recall5, scores["dense"].recall5) - 0.05
    assert scores["hybrid"].mrr > scores["graph"].mrr  # graph alone is a recall aid, not a ranker


def test_exact_technical_terms_favour_bm25_and_filters_are_honoured(corpus):
    top = corpus.search("Llama 4 Scout 17B 16 experts", "bm25", 3)
    assert "Llama 4" in corpus.idx.chunks[top[0]].title or "herd" in corpus.idx.chunks[top[0]].title
    old = corpus.search("language model release", "bm25", 10, Filters(date_to="2023-12-31"))
    assert old and all(corpus.idx.chunks[c].published_at <= "2023-12-31" for c in old)
    off = corpus.search("open weights", "dense", 10, Filters(source_types=frozenset({"news"})))
    assert off and all(corpus.idx.chunks[c].source_type == "news" for c in off)


def test_graph_retrieval_finds_multi_hop_bridge(corpus):
    ids = corpus.search("Which companies partner with NVIDIA and also build AI accelerators?", "graph", 12)
    titles = " ".join(corpus.idx.chunks[c].title for c in ids)
    assert "Maia" in titles and "Trainium2" in titles and "TPU" in titles


# ------------------------------------------------------------ the agent
def test_agent_answers_with_citations_and_full_trace(corpus):
    run = corpus.ask("When did Meta release Llama 3?", use_llm=False)
    assert "April 18, 2024" in run.answer and "[E1]" in run.answer
    assert run.evidence[0].url == "https://meta-ai.example/blog/llama-3" and run.evidence[0].source == "Meta AI"
    names = [s.name for s in run.trace.spans]
    assert names[:3] == ["understand", "route", "plan"] and {"retrieve", "context", "generate", "verify"} <= set(names)
    assert run.verification.faithfulness == 1.0


def test_temporal_question_is_decomposed_into_windows_and_ordered_chronologically(corpus):
    run = corpus.ask("How has Meta's Llama family evolved since 2023?", use_llm=False)
    assert any(s.sub.kind == "window" for s in run.steps)
    dates = [e.published_at for e in run.evidence]
    assert dates == sorted(dates) and dates[0] < "2024"


def test_agent_refines_when_evidence_is_insufficient(corpus):
    run = corpus.ask("How has the open-source LLM landscape changed in 2025, which companies are driving it?", use_llm=False)
    acts = [a for s in run.steps for a in s.actions]
    assert "relax_filters" in acts or run.iterations >= 1  # a hard 2025 filter starves some windows; the agent relaxes it
    assert run.evidence


def test_agent_abstains_when_the_knowledge_base_has_no_answer(corpus):
    run = corpus.ask("What is the parameter count of GPT-5?", use_llm=False)
    assert run.answer_mode == "none" and "could not find" in run.answer.lower()
    assert not run.verification.claims  # nothing asserted => nothing to hallucinate


def test_tools_cover_the_documented_toolbox(corpus):
    t = corpus.agent.tools
    names = {s["function"]["name"] for s in t.specs()}
    assert names >= {"search_bm25", "search_dense", "search_graph", "search_hybrid", "rewrite_query", "decompose_query", "get_entity",
                     "get_entity_history", "get_related_entities", "retrieve_document", "verify_claim", "get_latest_information"}
    assert t.call("get_entity", name="MoE")["name"] == "Mixture of Experts"
    hist = t.call("get_entity_history", name="Llama")["timeline"]
    assert any("Llama 4" in h["fact"] for h in hist)
    rel = t.call("get_related_entities", name="NVIDIA", relation="PARTNERS_WITH")["related"]
    assert {"Google", "Microsoft"} <= {r["entity"] for r in rel}
    latest = t.call("get_latest_information", query="DeepSeek model", k=2)["hits"]
    assert latest[0]["date"] >= latest[1]["date"]
    doc = t.call("retrieve_document", document_id="https://meta-ai.example/models/llama")
    assert len(doc["versions"]) == 2 and "Llama 4" in doc["text"]
    assert t.call("verify_claim", claim="Mistral 7B was released under the Apache 2.0 license")["verdict"] in ("SUPPORTED", "PARTIAL")
    assert t.call("verify_claim", claim="Mistral 7B has 500 billion parameters")["verdict"] == "UNSUPPORTED"
    assert "error" in t.call("nonexistent")


# ---------------------------------------------------- claim verification
def _ev(i, text, title="T", source="S"):
    return Evidence(f"E{i}", f"c{i}", "d", source, "official", "u", title, "sec", text, "2024-01-01", 1, 0, 0, 0, [])


EV = [_ev(1, "Meta released Llama 3 on April 18, 2024. Llama 3 comes in 8B and 70B parameter sizes."),
      _ev(2, "Mixtral is a sparse Mixture of Experts model released under the Apache 2.0 license.")]


@pytest.mark.parametrize("claim,verdict", [
    ("Meta introduced Llama 3 in April 2024 [E1].", "SUPPORTED"),                      # paraphrase (introduced≈released)
    ("Llama 3 was released on May 2, 2024 [E1].", "UNSUPPORTED"),                      # wrong date
    ("Llama 3 has 405B parameters [E1].", "UNSUPPORTED"),                              # invented number
    ("Mixtral uses MoE under an Apache 2.0 license [E2].", "SUPPORTED"),               # acronym ↔ expansion
    ("Mistral AI acquired Hugging Face last year [E2].", "UNSUPPORTED"),               # unrelated assertion
])
def test_claim_verdicts(claim, verdict):
    rep = ClaimVerifier().verify(claim, EV)
    assert rep.claims[0].verdict == verdict


def test_citation_accuracy_penalises_misattributed_citations():
    rep = ClaimVerifier().verify("Llama 3 comes in 8B and 70B sizes [E2]. Mixtral is released under Apache 2.0 [E2].", EV)
    assert [c.cited_ok for c in rep.claims] == [False, True] and rep.citation_accuracy == 0.5


def test_repair_removes_only_unsupported_lines():
    v = ClaimVerifier()
    ans = "- Meta released Llama 3 on April 18, 2024 [E1].\n- Llama 3 has 405B parameters [E1].\n- Mixtral uses Apache 2.0 [E2]."
    rep = v.verify(ans, EV)
    fixed = v.repair(ans, rep)
    assert "405B" not in fixed and "April 18" in fixed and "Apache" in fixed


def test_llm_output_normalisation():
    assert normalize_citations("done【E2】 and (E3) and [E1, E4]") == "done[E2] and [E3] and [E1][E4]"
    assert normalize_text("Mixture‑of‑Experts 8×7 B") == "Mixture-of-Experts 8x7B"


# ------------------------------------------------------ LLM integration
def groq(handler, **kw) -> GroqLLM:
    s = Settings(llm_provider="groq", groq_api_key="test-key", model="openai/gpt-oss-120b", backoff_base_s=0.0, **kw)
    return GroqLLM(s, transport=httpx.MockTransport(handler))


def ok(content: str, **extra) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}, **extra)


def test_groq_client_request_shape_cache_and_think_stripping():
    seen = []

    def h(req: httpx.Request):
        seen.append((req.headers["authorization"], json.loads(req.content)))
        return ok("<think>hidden</think>hello")

    llm = groq(h)
    assert llm.complete("sys", "hi", json_mode=True) == "hello"
    assert llm.complete("sys", "hi", json_mode=True) == "hello"  # second call served from cache
    auth, body = seen[0]
    assert auth == "Bearer test-key" and body["model"] == "openai/gpt-oss-120b" and body["response_format"] == {"type": "json_object"}
    assert body["reasoning_effort"] == "low" and len(seen) == 1 and llm.stats.cache_hits == 1


def test_groq_client_retries_rate_limits_then_raises(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = {"n": 0}

    def flaky(req):
        calls["n"] += 1
        return httpx.Response(429, headers={"retry-after": "0"}, text="slow down") if calls["n"] < 3 else ok("finally")

    assert groq(flaky).complete("s", "p") == "finally" and calls["n"] == 3
    with pytest.raises(LLMError):
        groq(lambda r: httpx.Response(500, text="boom")).complete("s", "p")
    with pytest.raises(LLMError):
        groq(lambda r: httpx.Response(401, text="bad key")).complete("s", "p")


def test_parse_json_tolerates_fences_and_prose():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! {"a": [1, 2]} hope that helps') == {"a": [1, 2]}
    with pytest.raises(LLMError):
        parse_json("no json here")


def _answer_llm(text_by_call):
    it = iter(text_by_call)
    return FakeLLM(fn=lambda s, p: next(it))


def test_llm_answer_is_verified_regenerated_and_repaired(corpus):
    bad = ("- Meta released Llama 3 on April 18, 2024【E1】.\n- Llama 3 was trained on 900 trillion tokens【E1】.")
    good = "- Meta released Llama 3 on April 18, 2024 [E1]."
    llm = _answer_llm([bad, good])
    corpus.generator.llm = llm
    try:
        run = corpus.agent.run("When did Meta release Llama 3?", use_llm=True)
    finally:
        corpus.generator.llm = None
    assert run.answer_mode == "llm" and run.regenerated and "900" not in run.answer
    assert "unsupported" in llm.calls[1][1]  # the second prompt carried the verifier's feedback
    assert run.verification.faithfulness == 1.0


def test_llm_failure_falls_back_to_extractive(corpus):
    corpus.generator.llm = FakeLLM(responses=[])  # raises LLMError on first call
    try:
        run = corpus.agent.run("Who developed Mixtral?", use_llm=True)
    finally:
        corpus.generator.llm = None
    assert run.answer_mode == "extractive" and "Mistral AI" in run.answer


def test_llm_judge_can_demote_but_not_forge_numbers(corpus):
    v = ClaimVerifier()
    rep = v.verify("Llama 3 has a flagship open model announced by Meta [E1].", EV)
    claim = rep.claims[0]
    llm = FakeLLM(fn=lambda s, p: json.dumps({"verdicts": [{"i": 1, "verdict": "supported"}]}))
    assert v.judge(rep, EV, llm).claims[0].verdict == "SUPPORTED" or claim.missing_numbers
    rep2 = v.verify("Llama 3 was released on May 2, 2024 [E1].", EV)
    assert v.judge(rep2, EV, llm).claims[0].verdict == "UNSUPPORTED"  # numbers are a hard guard the judge cannot override


@pytest.mark.live
@pytest.mark.skipif(not (os.environ.get("ATLAS_GROQ_API_KEY") or os.path.exists(".env")), reason="needs Groq credentials")
def test_live_groq_end_to_end(corpus):
    s = Settings.from_env()
    if s.llm_provider != "groq":
        pytest.skip("no Groq key configured")
    from atlas.llm import make_llm

    corpus.generator.llm = make_llm(s)
    try:
        run = corpus.agent.run("Which companies partner with NVIDIA and also build AI accelerators?", use_llm=True)
    finally:
        corpus.generator.llm = None
    assert run.answer_mode == "llm" and run.verification.hallucination_rate <= 0.2 and "Microsoft" in run.answer


# --------------------------------------------------- evaluation harness
def test_answer_evaluation_gates_and_failure_attribution(corpus):
    scores = eval_answers(corpus, use_llm=False)
    summary = summarize_answers(scores)
    assert summary["faithfulness"] >= 0.95 and summary["hallucination_rate"] <= 0.05
    assert summary["abstention_accuracy"] == 1.0 and summary["completeness"] >= 0.75
    assert set(summary["failures"]) <= {"OK", "RETRIEVAL", "RANKING", "CONTEXT", "GENERATION"}
    assert len(scores) == len(EVAL_SET)


def test_failure_attribution_localises_the_failing_stage(corpus):
    item = next(i for i in EVAL_SET if i.id == "f1")
    run = corpus.ask(item.question, use_llm=False)
    assert attribute_failure(corpus, item, run, 1.0, 1.0)[0] == "OK"
    assert attribute_failure(corpus, item, run, 0.0, 1.0)[0] == "GENERATION"  # evidence was there, answer missed it
    run.context.evidence = [e for e in run.context.evidence if "llama-3" not in e.url]
    assert attribute_failure(corpus, item, run, 0.0, 1.0)[0] == "CONTEXT"  # retrieved + ranked, then lost in context building
    for st in run.steps:
        st.result.candidates = []
    assert attribute_failure(corpus, item, run, 0.0, 1.0)[0] == "RANKING"
    for st in run.steps:
        st.result.trace.lists = {}
    assert attribute_failure(corpus, item, run, 0.0, 1.0)[0] == "RETRIEVAL"


def test_patient_context_extends_and_restores_the_rate_limit_wait():
    from atlas.llm import patient

    llm = groq(lambda r: ok("x"))
    before = llm.max_wait_s
    with patient(llm, 500):
        assert llm.max_wait_s == 500
    assert llm.max_wait_s == before


def test_empty_reasoning_only_completions_are_retried_with_a_bigger_budget():
    budgets = []

    def h(req):
        body = json.loads(req.content)
        budgets.append(body["max_completion_tokens"])
        return ok("" if len(budgets) == 1 else "answer")

    assert groq(h).complete("s", "p", max_tokens=100) == "answer" and budgets[1] == 2 * budgets[0]


def test_generator_fallback_reason_is_visible_in_the_trace(corpus):
    corpus.generator.llm = FakeLLM(responses=[])
    try:
        run = corpus.agent.run("Who developed Mixtral?", use_llm=True)
    finally:
        corpus.generator.llm = None
    gen = next(s for s in run.trace.spans if s.name == "generate")
    assert gen.attrs["mode"] == "extractive" and "no scripted response" in gen.attrs["llm_fallback"]
