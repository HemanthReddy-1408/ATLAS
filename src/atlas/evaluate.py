"""Evaluation: retrieval metrics, answer quality, and failure attribution (retrieval / ranking / context / generation)."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .agent import AgentRun
from .engine import Atlas
from .evalset import EVAL_SET, EvalItem
from .fixtures import doc_key_for_url


# ------------------------------------------------------------- relevance
def doc_key(atlas: Atlas, chunk_id: str) -> str | None:
    c = atlas.idx.chunks.get(chunk_id)
    return doc_key_for_url(c.url) if c else None


def relevant_chunks(atlas: Atlas, item: EvalItem) -> set[str]:
    """Chunk-level gold: chunk belongs to a relevant doc and (if phrases are given) contains one of them."""
    out = set()
    for cid, c in atlas.idx.chunks.items():
        if doc_key_for_url(c.url) in item.relevant_docs:
            if not item.relevant_phrases or any(p.lower() in c.text.lower() for p in item.relevant_phrases):
                out.add(cid)
    return out


# --------------------------------------------------------------- metrics
def recall_at_k(ranked_docs: list[str | None], gold_docs: set[str], k: int) -> float:
    return len(set(ranked_docs[:k]) & gold_docs) / len(gold_docs) if gold_docs else 0.0


def precision_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    top = ranked[:k]
    return sum(c in gold for c in top) / k if top else 0.0


def mrr(ranked: list[str], gold: set[str]) -> float:
    for i, c in enumerate(ranked, 1):
        if c in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    dcg = sum(1 / math.log2(i + 1) for i, c in enumerate(ranked[:k], 1) if c in gold)
    idcg = sum(1 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / idcg if idcg else 0.0


MODES = ["bm25", "dense", "graph", "rrf", "hybrid"]


@dataclass
class RetrievalScores:
    mode: str
    recall5: float
    recall10: float
    precision5: float
    mrr: float
    ndcg10: float
    n: int

    def row(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def eval_retrieval(atlas: Atlas, items: list[EvalItem] | None = None, modes: list[str] | None = None) -> list[RetrievalScores]:
    items = [i for i in (items or EVAL_SET) if not i.abstain]
    out = []
    for mode in modes or MODES:
        agg = {"r5": [], "r10": [], "p5": [], "mrr": [], "ndcg": []}
        for it in items:
            gold_c, gold_d = relevant_chunks(atlas, it), set(it.relevant_docs)
            ranked = atlas.search(it.question, mode, k=10)
            docs = [doc_key(atlas, c) for c in ranked]
            agg["r5"].append(recall_at_k(docs, gold_d, 5))
            agg["r10"].append(recall_at_k(docs, gold_d, 10))
            agg["p5"].append(precision_at_k(ranked, gold_c, 5))
            agg["mrr"].append(mrr(ranked, gold_c))
            agg["ndcg"].append(ndcg_at_k(ranked, gold_c, 10))
        m = lambda xs: sum(xs) / len(xs) if xs else 0.0  # noqa: E731
        out.append(RetrievalScores(mode, m(agg["r5"]), m(agg["r10"]), m(agg["p5"]), m(agg["mrr"]), m(agg["ndcg"]), len(items)))
    return out


# ---------------------------------------------------------------- answers
@dataclass
class AnswerScore:
    id: str
    question: str
    type: str
    answer: str
    key_point_recall: float
    faithfulness: float
    citation_accuracy: float | None
    hallucination_rate: float
    abstained: bool
    abstain_correct: bool | None
    failure: str
    stage_coverage: dict
    latency_ms: float
    mode: str

    def row(self) -> dict:
        d = asdict(self)
        d.pop("answer"), d.pop("stage_coverage")
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()}


def key_point_recall(answer: str, key_points: list[str]) -> float:
    if not key_points:
        return 1.0
    low = answer.lower()
    return sum(any(alt.lower() in low for alt in kp.split("|")) for kp in key_points) / len(key_points)


def attribute_failure(atlas: Atlas, item: EvalItem, run: AgentRun, kp: float, faith: float) -> tuple[str, dict]:
    """Which stage lost the evidence? Compare gold-doc coverage after each stage of the pipeline."""
    gold = set(item.relevant_docs)
    if not gold:
        return "OK", {}

    def cov(chunk_ids: set[str]) -> float:
        got = {doc_key(atlas, c) for c in chunk_ids if relevant_chunks_cache(atlas, item, c)}
        return len(got & gold) / len(gold)

    pool: set[str] = set()
    ranked: set[str] = set()
    for st in run.steps:
        if st.result:
            for ids in st.result.trace.lists.values():
                pool.update(ids)
            ranked.update(c.chunk_id for c in st.result.candidates)
    context = {e.chunk_id for e in run.evidence}
    cp, cr, cc = cov(pool), cov(ranked), cov(context)
    stages = {"pool": round(cp, 2), "reranked": round(cr, 2), "context": round(cc, 2)}
    if kp >= 0.99 and faith >= 0.8:
        return "OK", stages
    deficits = {"RETRIEVAL": 1 - cp, "RANKING": cp - cr, "CONTEXT": cr - cc}
    stage, d = max(deficits.items(), key=lambda kv: kv[1])
    if d > 0.001:
        return stage, stages
    return "GENERATION", stages


_rel_cache: dict[tuple[int, str], set[str]] = {}


def relevant_chunks_cache(atlas: Atlas, item: EvalItem, chunk_id: str) -> bool:
    key = (id(atlas), item.id)
    if key not in _rel_cache:
        _rel_cache[key] = relevant_chunks(atlas, item)
    return chunk_id in _rel_cache[key]


def eval_answers(atlas: Atlas, items: list[EvalItem] | None = None, use_llm: bool = False, progress=None) -> list[AnswerScore]:
    _rel_cache.clear()
    out = []
    for n, it in enumerate(items or EVAL_SET, 1):
        t0 = time.perf_counter()
        run = atlas.ask(it.question, use_llm=use_llm)
        ms = (time.perf_counter() - t0) * 1e3
        abstained = run.answer_mode == "none"
        if it.abstain:
            out.append(AnswerScore(it.id, it.question, it.type, run.answer, 0.0, 1.0, None, 0.0, abstained, abstained,
                                   "OK" if abstained else "GENERATION", {}, ms, run.answer_mode))
        else:
            kp = key_point_recall(run.answer, it.key_points)
            v = run.verification
            failure, stages = attribute_failure(atlas, it, run, kp, v.faithfulness)
            out.append(AnswerScore(it.id, it.question, it.type, run.answer, kp, v.faithfulness, v.citation_accuracy,
                                   v.hallucination_rate, abstained, None, failure, stages, ms, run.answer_mode))
        if progress:
            progress(n, len(items or EVAL_SET), it.question)
    return out


def summarize_answers(scores: list[AnswerScore]) -> dict:
    ans = [s for s in scores if s.abstain_correct is None]
    ab = [s for s in scores if s.abstain_correct is not None]
    m = lambda xs: round(sum(xs) / len(xs), 3) if xs else None  # noqa: E731
    fails: dict[str, int] = {}
    for s in scores:
        fails[s.failure] = fails.get(s.failure, 0) + 1
    by_type: dict[str, list[float]] = {}
    for s in ans:
        by_type.setdefault(s.type, []).append(s.key_point_recall)
    return {
        "completeness": m([s.key_point_recall for s in ans]),
        "faithfulness": m([s.faithfulness for s in ans]),
        "citation_accuracy": m([s.citation_accuracy for s in ans if s.citation_accuracy is not None]),
        "hallucination_rate": m([s.hallucination_rate for s in ans]),
        "abstention_accuracy": m([float(bool(s.abstain_correct)) for s in ab]),
        "p50_latency_ms": round(sorted(s.latency_ms for s in scores)[len(scores) // 2], 1) if scores else None,
        "failures": fails,
        "completeness_by_type": {k: m(v) for k, v in by_type.items()},
    }


def save_report(path: str | Path, retrieval: list[RetrievalScores], answers: list[AnswerScore]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({
        "retrieval": [r.row() for r in retrieval],
        "answers_summary": summarize_answers(answers),
        "answers": [{**s.row(), "answer": s.answer, "stage_coverage": s.stage_coverage} for s in answers],
    }, indent=2))


def format_table(rows: list[dict]) -> str:
    if not rows:
        return ""
    cols = list(rows[0])
    w = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    line = "  ".join(c.ljust(w[c]) for c in cols)
    return "\n".join([line, "  ".join("-" * w[c] for c in cols), *("  ".join(str(row[c]).ljust(w[c]) for c in cols) for row in rows)])

