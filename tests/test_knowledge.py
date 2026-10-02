"""Entity resolution, relation extraction, indexes, fusion, and query understanding."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from atlas.clock import FrozenClock
from atlas.domain import EntityType, QueryType
from atlas.indexes import BM25Index, HashingEmbedder, tokenize
from atlas.process import alias_key, split_sentences
from atlas.query import decompose, expand, parse_time, route
from atlas.retrieve import rrf


# ---------------------------------------------------------- resolution
@pytest.mark.parametrize("variants,canonical", [
    (["OpenAI", "OpenAI Inc.", "OpenAI, Inc.", "OpenAI's", "openai"], "openai"),
    (["GPT-4", "GPT 4", "GPT4", "gpt-4"], "gpt-4"),
    (["Llama 3.1", "Llama-3.1", "LLaMA 3.1"], "llama-3-1"),
    (["Mixture-of-Experts", "MoE", "mixture of experts"], "mixture-of-experts"),
    (["NVIDIA", "Nvidia Corporation"], "nvidia"),
])
def test_surface_variants_resolve_to_one_entity(fresh, variants, canonical):
    assert {fresh.resolver.resolve(v) for v in variants} == {canonical}


def test_versions_never_collapse(fresh):
    r = fresh.resolver
    ids = {r.resolve(x) for x in ["GPT-4", "GPT-4o", "GPT-5", "Llama 3", "Llama 3.1", "Llama 4"]}
    assert len(ids) == 6
    assert alias_key("Llama 3.1") != alias_key("Llama 31")


def test_unknown_model_versions_are_discovered_as_provisional_variants(fresh):
    ms = fresh.resolver.extract("Meta also shipped Llama 3.2 for edge devices.")
    names = {fresh.repo.entity(m.entity_id).canonical_name for m in ms}
    assert {"Meta", "Llama 3.2"} <= names
    e = fresh.repo.entity("llama-3-2")
    assert e.provisional and e.type == EntityType.MODEL


def test_common_words_are_not_entities_unless_capitalised(fresh):
    r = fresh.resolver
    assert not r.extract("a meta discussion about the math of rag", discover=False)
    assert [m.surface for m in r.extract("Meta uses RAG", discover=False)] == ["Meta", "RAG"]
    assert [m.surface for m in r.extract("meta rag", discover=False, lenient=True)] == ["meta", "rag"]  # queries are often lower-case


# -------------------------------------------------------------- relations
def rels(atlas, sentence):
    ms = atlas.resolver.extract(sentence)
    out = atlas.extractor.extract_sentence(sentence, ms, None)
    return {(x.src_id, x.rel, x.dst_id) for x in out}


@pytest.mark.parametrize("sentence,expected", [
    ("Meta released Llama 3 on April 18, 2024.", {("meta", "RELEASED", "llama-3")}),
    ("Llama 3 was released by Meta.", {("meta", "RELEASED", "llama-3")}),  # passive flips direction
    ("Meta released Llama 3 and Llama 3.1.", {("meta", "RELEASED", "llama-3"), ("meta", "RELEASED", "llama-3-1")}),  # conjunction
    ("Google acquired DeepMind.", set()),  # DeepMind is an alias of Google DeepMind (same company type) but not same entity? see below
    ("OpenAI partners with Microsoft.", {("openai", "PARTNERS_WITH", "microsoft")}),
    ("Mistral 7B outperforms Llama 2 on all benchmarks.", {("mistral-7b", "OUTPERFORMS", "llama-2")}),
    ("Mixtral uses a Mixture of Experts architecture.", {("mixtral", "USES", "mixture-of-experts")}),
    ("Maia 100 is an AI accelerator that Microsoft designed.", {("maia-100", "IS_A", "ai-accelerator")}),
    ("The paper Attention Is All You Need introduced the Transformer.", {("attention-is-all-you-need", "INTRODUCED", "transformer")}),
])
def test_relation_patterns(fresh, sentence, expected):
    got = rels(fresh, sentence)
    if sentence.startswith("Google acquired"):
        assert ("google", "ACQUIRED", "google-deepmind") in got
    else:
        assert expected <= got


def test_type_constraints_block_nonsense(fresh):
    assert not rels(fresh, "GPT-4 released Meta.")  # a model cannot release a company
    got = rels(fresh, "Meta released a statement criticising OpenAI's GPT-4.")
    assert not any(src == "meta" for src, _, _ in got)  # "released a statement ..." must not make Meta the releaser of GPT-4


def test_sentence_splitter_guards_abbreviations_and_decimals():
    s = [x[2] for x in split_sentences("OpenAI, Inc. released GPT-4 in 2023. Llama 3.1 has 405B parameters. It scores 90.0% on MMLU.")]
    assert len(s) == 3 and s[0].startswith("OpenAI, Inc. released")


# ------------------------------------------------------------- indexes
def test_tokenizer_keeps_technical_compounds():
    t = tokenize("Llama-3.1 405B and GPT-4o use 8x7B experts")
    assert {"llama-3.1", "llama31", "405b", "gpt-4o", "gpt4o", "8x7b"} <= set(t)


def test_bm25_ranks_exact_terms_and_supports_removal():
    from atlas.domain import Chunk

    def mk(i, text, title="t"):
        return Chunk(i, "d", text, "s", 0, 5, title=title)

    ix = BM25Index()
    for c in [mk("a", "Llama 4 Scout has 17B active parameters"), mk("b", "Mistral 7B is a small model"), mk("c", "general language models overview")]:
        ix.add(c)
    assert ix.search("Llama 4 17B", 3)[0][0] == "a"
    assert ix.search("Mistral 7B", 3)[0][0] == "b"
    ix.remove("a")
    assert all(c != "a" for c, _ in ix.search("Llama 4 17B", 3)) and len(ix) == 2
    assert ix.search("zzz", 3) == []


def test_hashing_embedder_is_deterministic_normalised_and_concept_aware():
    e = HashingEmbedder(256)
    v = e.embed(["Meta unveiled Llama 3", "Meta introduced Llama 3", "Quarterly revenue of a bakery"])
    assert abs((v[0] ** 2).sum() - 1) < 1e-5 and (e.embed(["Meta unveiled Llama 3"])[0] == v[0]).all()
    assert v[0] @ v[1] > v[0] @ v[2] + 0.3  # synonyms (unveiled/introduced) land together


def test_rrf_rewards_agreement_and_respects_weights():
    fused = rrf({"bm25": ["a", "b", "c"], "dense": ["b", "a", "d"]}, {"bm25": 1, "dense": 1}, k=60)
    assert fused["a"] == pytest.approx(fused["b"]) and fused["a"] > fused["c"] > 0 and fused["d"] > 0
    heavy = rrf({"bm25": ["a", "b"], "dense": ["b", "a"]}, {"bm25": 0.1, "dense": 2.0}, k=60)
    assert heavy["b"] > heavy["a"]


# -------------------------------------------------------- graph (corpus)
def test_graph_traversal_history_and_paths(corpus):
    g = corpus.idx.graph
    partners = {g.nodes[o].canonical_name for r, o in g.edges("nvidia", rel="PARTNERS_WITH")}
    assert {"Google", "Microsoft", "Meta", "Amazon Web Services"} <= partners  # symmetric edges seen from either side
    timeline = [r.valid_from for r in g.history("llama") if r.valid_from]
    assert timeline == sorted(timeline)
    path = g.path("openai", "nvidia", 3)
    assert path is not None and 1 <= len(path) <= 3
    assert g.path("openai", "attention-is-all-you-need", 1) is None


# ------------------------------------------------------ query intelligence
TODAY = date(2026, 6, 1)


@pytest.mark.parametrize("q,start,end,recency", [
    ("changes since 2023", date(2023, 1, 1), None, False),
    ("what happened in 2024", date(2024, 1, 1), date(2024, 12, 31), False),
    ("between 2022 and 2024", date(2022, 1, 1), date(2024, 12, 31), False),
    ("before 2024", None, date(2023, 12, 31), False),
    ("after 2023", date(2024, 1, 1), None, False),
    ("over the last 2 years", TODAY - timedelta(days=730), TODAY, False),
    ("latest NVIDIA announcement", None, None, True),
])
def test_time_constraints(q, start, end, recency):
    t = parse_time(q, TODAY)
    assert (t.start, t.end, t.recency) == (start, end, recency)


@pytest.mark.parametrize("q,qtype", [
    ("When did Meta release Llama 3?", QueryType.FACTUAL),
    ("What is LoRA?", QueryType.FACTUAL),
    ("Who developed Mixtral?", QueryType.RELATIONAL),
    ("Compare Llama 3.1 and Mistral 7B", QueryType.COMPARATIVE),
    ("How has Meta's Llama family evolved since 2023?", QueryType.TEMPORAL),
    ("Which companies partner with NVIDIA and also build AI accelerators?", QueryType.MULTI_HOP),
    ("Why do Mixture of Experts models need less compute?", QueryType.ANALYTICAL),
    ("Give me an overview of the AI accelerator landscape", QueryType.EXPLORATORY),
])
def test_query_classification(corpus, q, qtype):
    assert corpus.understanding.analyze(q).qtype == qtype


def test_the_flagship_question_is_understood_and_routed(corpus):
    q = "How has the open-source LLM landscape changed since 2023, which companies are driving it, and what architectural trends emerged?"
    a = corpus.understanding.analyze(q)
    assert a.qtype == QueryType.TEMPORAL and a.complex and len(a.clauses) == 3
    assert {"RELATIONAL", "ANALYTICAL"} <= {s.value for s in a.secondary} and a.time.start == date(2023, 1, 1)
    cfg = route(a)
    assert cfg.decompose and cfg.filters.date_from == "2023-01-01" and cfg.weights["graph"] > 0
    subs = decompose(a, corpus.resolver)
    assert [s.kind for s in subs].count("clause") == 3 and any(s.kind == "window" for s in subs)
    assert "(" in subs[1].text  # pronoun "it" resolved against the topic
    wins = [s for s in subs if s.kind == "window"]
    assert wins[0].date_from == "2023-01-01" and wins[-1].date_to.endswith("-12-31")


def test_comparative_decomposes_per_entity_and_expansion_adds_synonyms(corpus):
    a = corpus.understanding.analyze("Compare Gemma and Gemini")
    subs = decompose(a, corpus.resolver)
    assert [s.label for s in subs] == ["Gemma", "Gemini"] and all(s.entity_ids for s in subs)
    assert "open-weight" in expand("open-source LLM landscape")
    assert FrozenClock().now().year == 2026  # injectable clock
