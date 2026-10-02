"""Retrieval: metadata filters, BM25 / dense / graph retrievers, RRF fusion, reranking, hybrid orchestration."""

from __future__ import annotations

import itertools
import math
import re
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field

from .config import Settings
from .domain import Chunk, Relation
from .indexes import IndexManager, tokenize
from .ontology import REL_LABEL


# --------------------------------------------------------------- filters
@dataclass(frozen=True)
class Filters:
    source_types: frozenset[str] | None = None
    date_from: str | None = None  # ISO date, inclusive, compared with the chunk's published_at
    date_to: str | None = None
    entity_any: frozenset[str] | None = None
    document_ids: frozenset[str] | None = None
    kinds: frozenset[str] | None = None  # None => ordinary page chunks only; summaries are opt-in

    @property
    def empty(self) -> bool:
        return not any((self.source_types, self.date_from, self.date_to, self.entity_any, self.document_ids, self.kinds))

    def relaxed(self) -> Filters:
        """Drop the content constraints (dates, sources) but keep the kind scope."""
        return Filters(kinds=self.kinds)

    def matches(self, c: Chunk) -> bool:
        if c.kind not in (self.kinds or {"chunk"}):
            return False
        if self.source_types and str(c.source_type) not in self.source_types:
            return False
        if self.document_ids and c.document_id not in self.document_ids:
            return False
        if self.entity_any and not (set(c.entity_ids) & self.entity_any):
            return False
        if self.date_from or self.date_to:
            if not c.published_at:
                return False
            if self.date_from and c.published_at < self.date_from:
                return False
            if self.date_to and c.published_at > self.date_to:
                return False
        return True

    def describe(self) -> str:
        parts = []
        if self.source_types:
            parts.append("source∈" + "|".join(sorted(self.source_types)))
        if self.date_from or self.date_to:
            parts.append(f"date {self.date_from or '…'}→{self.date_to or '…'}")
        if self.entity_any:
            parts.append(f"entity∈{len(self.entity_any)}")
        return ", ".join(parts) or "none"


@dataclass
class RetrievalConfig:
    weights: dict[str, float] = field(default_factory=lambda: {"bm25": 1.0, "dense": 1.0, "graph": 0.0})
    hyde: bool = False
    decompose: bool = False
    multi_query: bool = True
    rerank: bool = True
    filters: Filters = field(default_factory=Filters)
    freshness_weight: float = 0.0
    diversity: float = 0.25  # 0 = pure relevance, 1 = pure diversity
    graph_hops: int = 2
    graph_active_only: bool = False
    include_summaries: bool = False
    describe: str = ""


@dataclass
class Variant:
    text: str
    weight: float = 1.0
    kind: str = "original"  # original | rewrite | expansion | hyde | subquery


@dataclass
class ScoredChunk:
    chunk_id: str
    fused: float = 0.0
    rerank: float = 0.0
    final: float = 0.0
    ranks: dict[str, int] = field(default_factory=dict)   # "bm25:original" -> rank
    scores: dict[str, float] = field(default_factory=dict)
    via: list[str] = field(default_factory=list)
    steps: set[str] = field(default_factory=set)

    @property
    def methods(self) -> list[str]:
        return sorted({k.split(":")[0] for k in self.ranks})


@dataclass
class RetrievalTrace:
    lists: dict[str, list[str]] = field(default_factory=dict)  # "bm25:original" -> ordered chunk ids
    fused: list[tuple[str, float]] = field(default_factory=list)
    reranked: list[tuple[str, float]] = field(default_factory=list)
    graph_facts: list[Relation] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    filters: str = "none"
    cache_hit: bool = False


@dataclass
class RetrievalResult:
    candidates: list[ScoredChunk]
    trace: RetrievalTrace


# -------------------------------------------------------------- reranker
RERANK_WEIGHTS = {"cov": 0.30, "bigram": 0.12, "entity": 0.18, "title": 0.08, "prox": 0.10, "year": 0.05,
                  "dense": 0.12, "bm25": 0.05}


class FeatureReranker:
    """Query-document *interaction* scorer (the role of a cross-encoder): term coverage weighted by IDF, phrase
    matches, entity coverage, heading hits, proximity and year agreement, with the retrievers' scores as a weak prior."""

    name = "feature"

    def __init__(self, idx: IndexManager) -> None:
        self.idx = idx

    def score(self, query: str, entity_ids: set[str], chunks: list[Chunk], priors: dict[str, dict[str, float]]) -> dict[str, float]:
        q_terms = [t for t in dict.fromkeys(tokenize(query))]
        weights = {t: self.idx.bm25.idf(t) or 0.5 for t in q_terms}
        total_w = sum(weights.values()) or 1.0
        q_seq = tokenize(query)
        q_bigrams = {(a, b) for a, b in itertools.pairwise(q_seq)}
        years = set(re.findall(r"\b(?:19|20)\d{2}\b", query))
        out: dict[str, float] = {}
        for c in chunks:
            toks = tokenize(c.text)
            tset = set(toks)
            cov = sum(w for t, w in weights.items() if t in tset) / total_w
            seq_bi = set(itertools.pairwise(toks))
            bigram = len(q_bigrams & seq_bi) / len(q_bigrams) if q_bigrams else 0.0
            ent = len(entity_ids & set(c.entity_ids)) / len(entity_ids) if entity_ids else 0.0
            head = set(tokenize(f"{c.title} {c.section_title}"))
            title = sum(w for t, w in weights.items() if t in head) / total_w
            pos = [i for i, t in enumerate(toks) if t in weights]
            prox = 0.0
            if pos:
                distinct = {toks[i] for i in pos}
                need = max(1, int(0.6 * len(distinct)))
                best = len(toks)
                seen: dict[str, int] = defaultdict(int)
                lo = 0
                for hi, p in enumerate(pos):
                    seen[toks[p]] += 1
                    while len([k for k, v in seen.items() if v > 0]) >= need:
                        best = min(best, pos[hi] - pos[lo] + 1)
                        seen[toks[pos[lo]]] -= 1
                        lo += 1
                prox = min(1.0, need / max(best, 1))
            year = 1.0 if years and any(y in c.text or y == (c.published_at or "")[:4] for y in years) else 0.0
            pr = priors.get(c.chunk_id, {})
            feats = {"cov": cov, "bigram": bigram, "entity": ent, "title": title, "prox": prox, "year": year,
                     "dense": max(0.0, pr.get("dense", 0.0)), "bm25": pr.get("bm25", 0.0)}
            out[c.chunk_id] = sum(RERANK_WEIGHTS[k] * v for k, v in feats.items())
        return out


class CrossEncoderReranker:
    """Optional neural reranker (pip install sentence-transformers): ATLAS_RERANKER=cross-encoder:<model>."""

    name = "cross-encoder"

    def __init__(self, model: str, idx: IndexManager) -> None:
        from sentence_transformers import CrossEncoder

        self.m, self.fallback = CrossEncoder(model), FeatureReranker(idx)

    def score(self, query: str, entity_ids: set[str], chunks: list[Chunk], priors: dict) -> dict[str, float]:
        raw = self.m.predict([(query, c.contextual_text) for c in chunks])
        base = self.fallback.score(query, entity_ids, chunks, priors)
        return {c.chunk_id: 0.7 / (1 + math.exp(-float(s))) + 0.3 * base[c.chunk_id] for c, s in zip(chunks, raw)}


def make_reranker(s: Settings, idx: IndexManager):
    if s.reranker.startswith("cross-encoder:"):
        return CrossEncoderReranker(s.reranker.split(":", 1)[1], idx)
    return FeatureReranker(idx)


# ---------------------------------------------------------------- fusion
def rrf(ranked_lists: dict[str, list[str]], weights: dict[str, float], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion: score(d) = Σ_lists w_l / (k + rank_l(d)), ranks 1-based."""
    fused: dict[str, float] = defaultdict(float)
    for name, ids in ranked_lists.items():
        w = weights.get(name, 1.0)
        for rank, cid in enumerate(ids, 1):
            fused[cid] += w / (k + rank)
    return dict(fused)


class _Cache:
    def __init__(self, size: int = 256, ttl: float = 600.0) -> None:
        self.size, self.ttl, self.d = size, ttl, OrderedDict()

    def get(self, key):
        v = self.d.get(key)
        if v and time.monotonic() - v[0] < self.ttl:
            self.d.move_to_end(key)
            return v[1]
        return None

    def put(self, key, val) -> None:
        self.d[key] = (time.monotonic(), val)
        if len(self.d) > self.size:
            self.d.popitem(last=False)


# ------------------------------------------------------------ retrievers
class Retrievers:
    def __init__(self, idx: IndexManager, settings: Settings) -> None:
        self.idx, self.s = idx, settings
        self._cache = _Cache()

    def _allow(self, f: Filters):
        chunks = self.idx.chunks
        return lambda cid: cid in chunks and f.matches(chunks[cid])

    def bm25(self, query: str, k: int, f: Filters) -> list[tuple[str, float]]:
        return self.idx.bm25.search(query, k, self._allow(f))

    def dense(self, query: str, k: int, f: Filters) -> list[tuple[str, float]]:
        key = ("dense", query, k, f, self.idx.version)
        hit = self._cache.get(key)
        if hit is None:
            hit = self.idx.vectors.search(query, k, self._allow(f))
            self._cache.put(key, hit)
        return hit

    def graph(self, entity_ids: list[str], rel_hints: set[str], k: int, f: Filters, hops: int = 2,
              active_only: bool = False) -> tuple[list[tuple[str, float]], dict[str, str], list[Relation]]:
        """Entity-linked retrieval. Returns ranked (chunk, score), chunk->explanation, and the supporting facts."""
        g = self.idx.graph
        seeds = [e for e in dict.fromkeys(entity_ids) if e in g.nodes]
        if not seeds:
            return [], {}, []
        reach: dict[str, dict[str, int]] = {s: {e: d for e, (d, _) in g.neighborhood([s], hops, active_only).items()} for s in seeds}
        # entities reachable from >=2 seeds are the multi-hop "bridge" nodes ("partners of X that also build Y")
        bridge = {e for e in g.nodes if sum(1 for s in seeds if e in reach[s]) >= 2} if len(seeds) > 1 else set()
        scores: dict[str, float] = defaultdict(float)
        why: dict[str, str] = {}
        facts: dict[str, tuple[float, Relation]] = {}
        for rel in g.rels.values():
            if active_only and not rel.active:
                continue
            ds = [min((reach[s].get(rel.src_id, 9), reach[s].get(rel.dst_id, 9)) ) for s in seeds]
            d = min(ds)
            if d > hops - 1:
                continue
            w = rel.confidence / (1 + d)
            if rel.rel in rel_hints:
                w *= 1.5
            if rel.src_id in bridge or rel.dst_id in bridge:
                w *= 1.3
            if rel.src_id in seeds or rel.dst_id in seeds:
                w *= 1.3
            if w < 0.12:
                continue
            facts[rel.relation_id] = (w, rel)
            if rel.chunk_id and rel.chunk_id in self.idx.chunks:
                if w > scores[rel.chunk_id]:
                    why[rel.chunk_id] = self.describe_fact(rel)
                scores[rel.chunk_id] = scores[rel.chunk_id] + w if rel.chunk_id in scores else w
        for cid, c in self.idx.chunks.items():
            hit = [s for s in seeds if s in c.entity_ids]
            if hit:
                scores[cid] += 0.3 * len(hit) + (0.4 if len(hit) > 1 else 0.0)
                why.setdefault(cid, "mentions " + ", ".join(g.nodes[h].canonical_name for h in hit))
        allow = self._allow(f)
        ranked = [(cid, s) for cid, s in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0])) if not allow or allow(cid)][:k]
        top_facts = [r for _, r in sorted(facts.values(), key=lambda x: -x[0])[:40]]
        return ranked, why, top_facts

    def describe_fact(self, r: Relation) -> str:
        n = self.idx.graph.nodes
        return f"{n[r.src_id].canonical_name} —{REL_LABEL.get(r.rel, r.rel)}→ {n[r.dst_id].canonical_name}"


# -------------------------------------------------------------- hybrid
class HybridRetriever:
    def __init__(self, idx: IndexManager, settings: Settings, reranker=None) -> None:
        self.idx, self.s = idx, settings
        self.r = Retrievers(idx, settings)
        self.reranker = reranker or make_reranker(settings, idx)

    def retrieve(self, query: str, entity_ids: list[str], rel_hints: set[str], variants: list[Variant],
                 cfg: RetrievalConfig, step_id: str = "main") -> RetrievalResult:
        t0 = time.perf_counter()
        trace = RetrievalTrace(filters=cfg.filters.describe())
        lists: dict[str, list[str]] = {}
        raw: dict[str, dict[str, float]] = defaultdict(dict)
        list_weights: dict[str, float] = {}
        k = self.s.candidates_per_retriever
        for v in variants:
            for method in ("bm25", "dense"):
                w = cfg.weights.get(method, 0.0)
                if w <= 0 or (method == "bm25" and v.kind == "hyde"):
                    continue
                hits = self.r.bm25(v.text, k, cfg.filters) if method == "bm25" else self.r.dense(v.text, k, cfg.filters)
                name = f"{method}:{v.kind}:{v.text[:40]}"
                lists[name] = [c for c, _ in hits]
                list_weights[name] = w * v.weight
                for c, s in hits:
                    raw[c][name] = s
        t1 = time.perf_counter()
        why: dict[str, str] = {}
        if cfg.weights.get("graph", 0) > 0 and entity_ids:
            hits, why, facts = self.r.graph(entity_ids, rel_hints, k, cfg.filters, cfg.graph_hops, cfg.graph_active_only)
            if hits:
                lists["graph:original"] = [c for c, _ in hits]
                list_weights["graph:original"] = cfg.weights["graph"]
                for c, s in hits:
                    raw[c]["graph:original"] = s
            trace.graph_facts = facts
        t2 = time.perf_counter()
        fused = rrf(lists, list_weights, self.s.rrf_k)
        order = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[: self.s.fused_k]
        trace.lists, trace.fused = lists, order
        cands: dict[str, ScoredChunk] = {}
        for cid, f in order:
            sc = ScoredChunk(cid, fused=f, steps={step_id})
            for name, ids in lists.items():
                if cid in ids:
                    sc.ranks[name] = ids.index(cid) + 1
                    sc.scores[name] = raw[cid][name]
            if cid in why and "graph:original" in sc.ranks:
                sc.via.append(why[cid])
            cands[cid] = sc
        chunks = [self.idx.chunks[c] for c in cands if c in self.idx.chunks]
        if cfg.rerank and chunks:
            best = {"dense": 0.0, "bm25": 0.0}
            for sc in cands.values():
                for name, s in sc.scores.items():
                    m = name.split(":")[0]
                    if m in best:
                        best[m] = max(best[m], s)
            priors = {cid: {m: (max((s for n, s in sc.scores.items() if n.startswith(m + ":")), default=0.0) / (best[m] or 1.0))
                            for m in best} for cid, sc in cands.items()}
            rr = self.reranker.score(query, set(entity_ids), chunks, priors)
            for cid, s in rr.items():
                cands[cid].rerank = s
            result = sorted(cands.values(), key=lambda s: (-s.rerank, -s.fused, s.chunk_id))[: self.s.rerank_k]
        else:
            result = sorted(cands.values(), key=lambda s: (-s.fused, s.chunk_id))[: self.s.rerank_k]
            ceiling = (sum(list_weights.values()) / (self.s.rrf_k + 1)) or 1.0  # best possible RRF score => scale to [0, 1]
            for sc in result:
                sc.rerank = min(1.0, sc.fused / ceiling)
        for sc in result:
            sc.final = sc.rerank
        trace.reranked = [(s.chunk_id, s.rerank) for s in result]
        t3 = time.perf_counter()
        trace.timings_ms = {"sparse+dense": (t1 - t0) * 1e3, "graph": (t2 - t1) * 1e3, "fuse+rerank": (t3 - t2) * 1e3}
        return RetrievalResult(result, trace)
