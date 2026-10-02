"""The research agent: a toolbox plus a plan → retrieve → assess → refine → answer → verify loop."""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .domain import Evidence, QueryType
from .generate import ContextResult, VerificationReport
from .observe import METRICS, Trace
from .ontology import REL_LABEL
from .query import (
    QueryAnalysis,
    SubQuestion,
    build_variants,
    decompose,
    expand,
    hyde_template,
    rewrite,
    route,
)
from .retrieve import Filters, RetrievalConfig, RetrievalResult, ScoredChunk, Variant

if TYPE_CHECKING:
    from .engine import Atlas


# ---------------------------------------------------------------- tools
@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict

    def json_schema(self) -> dict:
        """OpenAI/Groq-style function definition."""
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                  "parameters": {"type": "object", "properties": self.parameters,
                                                                 "required": [k for k, v in self.parameters.items() if v.get("required")]}}}


def _p(t: str, d: str, required: bool = False, **kw) -> dict:
    return {"type": t, "description": d, **({"required": True} if required else {}), **kw}


TOOL_SPECS = [
    ToolSpec("search_bm25", "Exact-term (BM25) search; best for model names, versions and numbers.",
             {"query": _p("string", "search text", True), "k": _p("integer", "results"), "date_from": _p("string", "ISO date"), "date_to": _p("string", "ISO date")}),
    ToolSpec("search_dense", "Semantic (vector) search; best for conceptual questions.",
             {"query": _p("string", "search text", True), "k": _p("integer", "results")}),
    ToolSpec("search_graph", "Knowledge-graph retrieval around named entities and relations.",
             {"entities": _p("array", "entity names", True, items={"type": "string"}), "relations": _p("array", "relation types", items={"type": "string"}), "k": _p("integer", "results")}),
    ToolSpec("search_hybrid", "Full pipeline: BM25 + dense + graph, RRF fusion, reranking.",
             {"query": _p("string", "search text", True), "k": _p("integer", "results"), "date_from": _p("string", "ISO date"), "date_to": _p("string", "ISO date")}),
    ToolSpec("rewrite_query", "Rewrite a question into retrieval-friendly queries (normalise entities, expand synonyms).",
             {"query": _p("string", "the question", True)}),
    ToolSpec("decompose_query", "Split a complex question into sub-questions.", {"query": _p("string", "the question", True)}),
    ToolSpec("get_entity", "Look up an entity: type, description, aliases, degree in the graph.", {"name": _p("string", "entity name or alias", True)}),
    ToolSpec("get_entity_history", "Chronological timeline of facts about an entity (including retired facts).", {"name": _p("string", "entity", True)}),
    ToolSpec("get_related_entities", "Entities connected to an entity, optionally by relation type.",
             {"name": _p("string", "entity", True), "relation": _p("string", "e.g. RELEASED, PARTNERS_WITH"), "direction": _p("string", "in|out|both")}),
    ToolSpec("retrieve_document", "Fetch a document's current text and version history.", {"document_id": _p("string", "document id or URL", True)}),
    ToolSpec("verify_claim", "Check a claim against retrieved evidence (supported / partial / unsupported).", {"claim": _p("string", "the claim", True)}),
    ToolSpec("get_latest_information", "Freshness-prioritised search: newest relevant material first.", {"query": _p("string", "topic", True), "k": _p("integer", "results")}),
]


class Tools:
    def __init__(self, atlas: Atlas) -> None:
        self.a = atlas

    def specs(self) -> list[dict]:
        return [t.json_schema() for t in TOOL_SPECS]

    def call(self, tool: str, /, **args: Any) -> dict:
        fn = getattr(self, tool, None)
        if fn is None or tool not in {t.name for t in TOOL_SPECS}:
            return {"error": f"unknown tool {tool}"}
        METRICS.inc(f"tool.{tool}")
        t0 = time.perf_counter()
        try:
            out = fn(**args)
        except TypeError as e:
            out = {"error": f"bad arguments: {e}"}
        out.setdefault("ms", round((time.perf_counter() - t0) * 1e3, 1))
        return out

    # -- helpers
    def _hits(self, ranked: list[tuple[str, float]], k: int) -> list[dict]:
        ch = self.a.idx.chunks
        return [{"chunk_id": c, "score": round(s, 4), "title": ch[c].title, "section": ch[c].section_title,
                 "source": ch[c].source_name, "date": ch[c].published_at, "snippet": ch[c].text[:240]}
                for c, s in ranked[:k] if c in ch]

    @staticmethod
    def _filters(date_from: str | None = None, date_to: str | None = None) -> Filters:
        return Filters(date_from=date_from, date_to=date_to)

    def _ids(self, names: list[str]) -> list[str]:
        out = []
        for n in names:
            i = self.a.resolver.lookup(n)
            if i:
                out.append(i)
        return out

    # -- tools
    def search_bm25(self, query: str, k: int = 10, date_from: str | None = None, date_to: str | None = None) -> dict:
        return {"hits": self._hits(self.a.retrievers.r.bm25(query, k, self._filters(date_from, date_to)), k)}

    def search_dense(self, query: str, k: int = 10) -> dict:
        return {"hits": self._hits(self.a.retrievers.r.dense(query, k, Filters()), k)}

    def search_graph(self, entities: list[str], relations: list[str] | None = None, k: int = 10) -> dict:
        ids = self._ids(entities)
        ranked, why, facts = self.a.retrievers.r.graph(ids, set(relations or []), k, Filters())
        hits = self._hits(ranked, k)
        for h in hits:
            h["why"] = why.get(h["chunk_id"], "")
        return {"entities_resolved": ids, "hits": hits, "facts": [self.a.retrievers.r.describe_fact(r) for r in facts[:15]]}

    def search_hybrid(self, query: str, k: int = 10, date_from: str | None = None, date_to: str | None = None) -> dict:
        an = self.a.understanding.analyze(query)
        cfg = dataclasses.replace(route(an), filters=self._filters(date_from, date_to))
        res = self.a.retrievers.retrieve(query, an.entity_ids, an.rel_hints, build_variants(an, query, cfg, self.a.llm, {}), cfg)
        return {"hits": self._hits([(s.chunk_id, s.rerank) for s in res.candidates], k), "route": cfg.describe}

    def rewrite_query(self, query: str) -> dict:
        an = self.a.understanding.analyze(query)
        return {"rewritten": rewrite(an), "expansions": expand(query), "entities": an.entity_names, "type": an.qtype.value}

    def decompose_query(self, query: str) -> dict:
        an = self.a.understanding.analyze(query)
        return {"sub_questions": [s.text for s in decompose(an, self.a.resolver, None)]}

    def get_entity(self, name: str) -> dict:
        eid = self.a.resolver.lookup(name)
        e = self.a.repo.entity(eid) if eid else None
        if not e:
            return {"error": f"unknown entity {name!r}"}
        g = self.a.idx.graph
        return {"entity_id": e.entity_id, "name": e.canonical_name, "type": e.type, "description": e.description,
                "aliases": list(e.aliases), "provisional": e.provisional, "degree": len(g.edges(e.entity_id))}

    def get_entity_history(self, name: str) -> dict:
        eid = self.a.resolver.lookup(name)
        if not eid:
            return {"error": f"unknown entity {name!r}"}
        n = self.a.idx.graph.nodes
        rows = [{"fact": f"{n[r.src_id].canonical_name} {REL_LABEL.get(r.rel, r.rel)} {n[r.dst_id].canonical_name}",
                 "valid_from": r.valid_from, "valid_until": r.valid_until, "active": r.active,
                 "confidence": r.confidence, "evidence": r.sentence} for r in self.a.idx.graph.history(eid)
                if r.src_id in n and r.dst_id in n]
        return {"entity": n[eid].canonical_name, "timeline": rows}

    def get_related_entities(self, name: str, relation: str | None = None, direction: str = "both") -> dict:
        eid = self.a.resolver.lookup(name)
        if not eid:
            return {"error": f"unknown entity {name!r}"}
        g = self.a.idx.graph
        rows = [{"relation": r.rel, "entity": g.nodes[o].canonical_name, "type": g.nodes[o].type, "valid_from": r.valid_from,
                 "active": r.active, "direction": "out" if r.src_id == eid else "in"}
                for r, o in g.edges(eid, direction, relation) if o in g.nodes]
        return {"entity": g.nodes[eid].canonical_name, "related": rows}

    def retrieve_document(self, document_id: str) -> dict:
        repo = self.a.repo
        d = repo.document(document_id) or repo.document_by_url(document_id)
        if not d:
            return {"error": "document not found"}
        cur = repo.current_version(d["document_id"])
        chunks = repo.doc_chunks(d["document_id"])
        return {"document_id": d["document_id"], "url": d["url"], "title": d["title"], "status": d["status"],
                "version": d["current_version"], "published_at": cur["published_at"] if cur else None,
                "versions": [{"version": v["version"], "created_at": v["created_at"], "summary": v["change_summary"]} for v in repo.versions(d["document_id"])],
                "text": "\n\n".join(c.text for c in chunks)[:6000]}

    def verify_claim(self, claim: str) -> dict:
        res = self.search_hybrid(claim, 6)
        ev = [Evidence(f"E{i}", h["chunk_id"], "", h["source"], "", "", h["title"], h["section"],
                       self.a.idx.chunks[h["chunk_id"]].text, h["date"], 1, 0, 0, 0, []) for i, h in enumerate(res["hits"], 1)]
        rep = self.a.verifier.verify(claim, ev)
        if not rep.claims:
            return {"verdict": "NOT_A_CLAIM"}
        c = rep.claims[0]
        return {"verdict": c.verdict, "coverage": c.coverage, "supporting": [ev[int(e[1:]) - 1].title for e in c.supporting if e[1:].isdigit()],
                "missing_numbers": c.missing_numbers, "missing_entities": c.missing_entities}

    def get_latest_information(self, query: str, k: int = 5) -> dict:
        an = self.a.understanding.analyze(query)
        cfg = dataclasses.replace(route(an), freshness_weight=0.6)
        res = self.a.retrievers.retrieve(query, an.entity_ids, an.rel_hints, build_variants(an, query, cfg, None, {}), cfg)
        hits = self._hits([(s.chunk_id, s.rerank) for s in res.candidates[:12]], 12)
        hits.sort(key=lambda h: h["date"] or "", reverse=True)
        return {"hits": hits[:k]}


# ----------------------------------------------------------------- state
@dataclass
class StepState:
    sub: SubQuestion
    cfg: RetrievalConfig
    variants: list[Variant] = field(default_factory=list)
    entity_ids: list[str] = field(default_factory=list)
    result: RetrievalResult | None = None
    sufficient: bool = False
    reason: str = ""
    attempts: int = 0
    actions: list[str] = field(default_factory=list)


@dataclass
class AgentRun:
    question: str
    analysis: QueryAnalysis
    route: str
    steps: list[StepState]
    context: ContextResult
    answer: str
    raw_answer: str
    answer_mode: str
    verification: VerificationReport
    regenerated: bool
    trace: Trace
    iterations: int
    llm_stats: dict = field(default_factory=dict)

    @property
    def evidence(self) -> list[Evidence]:
        return self.context.evidence


# ----------------------------------------------------------------- agent
class ResearchAgent:
    STRONG = 0.30  # reranker score at which a chunk counts as relevant evidence

    def __init__(self, atlas: Atlas) -> None:
        self.a = atlas
        self.tools = Tools(atlas)

    # ---- assessment
    def _assess(self, st: StepState, qtype: QueryType) -> None:
        res = st.result
        if res is None or not res.candidates:
            st.sufficient, st.reason = False, "no candidates retrieved"
            return
        strong = [c for c in res.candidates if c.rerank >= self.STRONG]
        need = 2 if qtype in (QueryType.EXPLORATORY, QueryType.MULTI_HOP, QueryType.COMPARATIVE, QueryType.TEMPORAL) and st.sub.kind != "window" else 1
        if st.sub.kind == "window":
            need = 1
        top_ents: set[str] = set()
        for c in res.candidates[:5]:
            top_ents |= set(self.a.idx.chunks[c.chunk_id].entity_ids)
        missing = [e for e in st.sub.entity_ids if e not in top_ents]
        if len(strong) < need:
            st.sufficient, st.reason = False, f"only {len(strong)} strong candidate(s) (<{need}); best={res.candidates[0].rerank:.2f}"
        elif missing and st.sub.kind == "entity":
            names = [self.a.idx.graph.nodes[m].canonical_name for m in missing if m in self.a.idx.graph.nodes]
            st.sufficient, st.reason = False, "no evidence mentioning " + ", ".join(names)
        else:
            st.sufficient, st.reason = True, f"{len(strong)} strong candidate(s); best={res.candidates[0].rerank:.2f}"

    # ---- refinement ladder
    def _refine(self, st: StepState, an: QueryAnalysis, trace: Trace) -> bool:
        ladder = ["expand", "relax_filters", "graph_bridge", "entity_lookup", "simplify"]
        while st.attempts < len(ladder):
            action = ladder[st.attempts]
            st.attempts += 1
            if action == "expand" and not any(v.kind in ("expansion", "hyde") for v in st.variants):
                ex = expand(st.sub.text)
                st.variants.append(Variant(st.sub.text + " " + " ".join(ex or an.keywords[:4]), 0.6, "expansion"))
                st.variants.append(Variant(hyde_template(an), 0.5, "hyde"))
                st.cfg = dataclasses.replace(st.cfg, hyde=True)
            elif action == "relax_filters" and not st.cfg.filters.empty:
                st.cfg = dataclasses.replace(st.cfg, filters=Filters())
            elif action == "graph_bridge" and st.entity_ids:
                g = self.a.idx.graph
                extra = {o for e in st.entity_ids for _, o in g.edges(e)[:6]}
                st.entity_ids = list(dict.fromkeys([*st.entity_ids, *sorted(extra)]))[:10]
                st.cfg = dataclasses.replace(st.cfg, weights={**st.cfg.weights, "graph": max(1.2, st.cfg.weights.get("graph", 0))})
            elif action == "entity_lookup" and st.sub.entity_ids:
                names = [self.a.idx.graph.nodes[e].canonical_name for e in st.sub.entity_ids if e in self.a.idx.graph.nodes]
                if names:
                    st.variants.append(Variant(" ".join(names) + " " + " ".join(an.keywords[:3]), 0.8, "rewrite"))
            elif action == "simplify":
                kws = sorted(set(an.keywords), key=lambda w: -self.a.idx.bm25.idf(w.lower()))[:4]
                st.variants.append(Variant(" ".join(kws), 0.7, "rewrite"))
            else:
                continue
            st.actions.append(action)
            trace.event("refine", step=st.sub.step_id, action=action, because=st.reason)
            return True
        return False

    def _should_abstain(self, an: QueryAnalysis, ctx: ContextResult) -> bool:
        """Refuse rather than guess: nothing retrieved, or none of the entities the user asked about appear in any evidence."""
        if not ctx.evidence:
            return True
        if an.entity_ids:
            seen: set[str] = set()
            for e in ctx.evidence:
                c = self.a.idx.chunks.get(e.chunk_id)
                if c:
                    seen |= set(c.entity_ids)
            if not seen & set(an.entity_ids):
                return True
        return max(e.rerank_score for e in ctx.evidence) < 0.12

    # ---- main loop
    def run(self, question: str, *, use_llm: bool = True, max_iterations: int | None = None, llm_judge: bool = False) -> AgentRun:
        a = self.a
        trace = Trace()
        llm = a.llm if use_llm else None
        max_it = max_iterations or a.settings.max_agent_iterations
        METRICS.inc("queries")
        with trace.span("understand") as sp:
            an = a.understanding.analyze(question)
            sp.attrs.update(an.summary())
        cfg = route(an, llm is not None)
        trace.event("route", config=cfg.describe, weights=str(cfg.weights), filters=cfg.filters.describe())
        with trace.span("plan"):
            subs = decompose(an, a.resolver, llm) if cfg.decompose else [SubQuestion("s1", question, "main", an.entity_ids)]
            hyde_cache: dict[str, str] = {}
            steps: list[StepState] = []
            for s in subs:
                scfg = cfg
                if s.kind == "window":
                    scfg = dataclasses.replace(cfg, filters=dataclasses.replace(cfg.filters, date_from=s.date_from, date_to=s.date_to))
                steps.append(StepState(s, scfg, build_variants(an, s.text, scfg, llm, hyde_cache),
                                       list(dict.fromkeys([*s.entity_ids, *(an.entity_ids if not s.entity_ids else [])]))))
        iterations = 0
        for it in range(max_it):
            todo = [s for s in steps if not s.sufficient]
            if not todo:
                break
            iterations = it + 1
            for st in todo:
                with trace.span("retrieve", step=st.sub.step_id, iteration=it + 1, query=st.sub.text) as sp:
                    st.result = a.retrievers.retrieve(st.sub.text, st.entity_ids, an.rel_hints, st.variants, st.cfg, st.sub.step_id)
                    self._assess(st, an.qtype)
                    sp.attrs.update(sufficient=st.sufficient, reason=st.reason, candidates=len(st.result.candidates))
                if not st.sufficient and it < max_it - 1:
                    self._refine(st, an, trace)
        # merge the per-step candidate lists
        merged: dict[str, ScoredChunk] = {}
        facts = []
        for st in steps:
            if not st.result:
                continue
            facts.extend(st.result.trace.graph_facts)
            for c in st.result.candidates:
                m = merged.get(c.chunk_id)
                if m is None:
                    merged[c.chunk_id] = dataclasses.replace(c, steps=set(c.steps), ranks=dict(c.ranks), scores=dict(c.scores), via=list(c.via))
                else:
                    m.steps |= c.steps
                    m.ranks.update(c.ranks)
                    m.scores.update(c.scores)
                    m.rerank = max(m.rerank, c.rerank)
                    m.fused = max(m.fused, c.fused)
                    m.via = list(dict.fromkeys([*m.via, *c.via]))
        seen_f: set[str] = set()
        facts = [f for f in facts if not (f.relation_id in seen_f or seen_f.add(f.relation_id))]
        n_real = len([s for s in subs if s.kind != "window"])
        with trace.span("context") as sp:
            ctx = a.context.build(list(merged.values()), an, cfg, question, facts, max_evidence=(5 if an.qtype in (QueryType.FACTUAL, QueryType.RELATIONAL) and not an.complex and "identify_entities" not in an.operations else min(12, a.settings.final_evidence + max(0, n_real - 1))))
            sp.attrs.update(ctx.stats)
        abstain = self._should_abstain(an, ctx)
        with trace.span("generate") as sp:
            gctx = dataclasses.replace(ctx, evidence=[], facts=[]) if abstain else ctx
            raw, mode = a.generator.generate(question, an, subs, gctx, use_llm=use_llm)
            sp.attrs.update(mode=mode)
        with trace.span("verify") as sp:
            rep = a.verifier.verify(raw, gctx.evidence)
            if llm_judge and llm and mode == "llm":
                rep = a.verifier.judge(rep, gctx.evidence, llm)
            regenerated = False
            if rep.unsupported and mode == "llm":
                fb = "\n".join(f"- unsupported: {c.text}" for c in rep.unsupported[:5])
                raw2, mode2 = a.generator.generate(question, an, subs, gctx, feedback=fb, use_llm=use_llm)
                if mode2 == "llm":
                    rep2 = a.verifier.verify(raw2, gctx.evidence)
                    if len(rep2.unsupported) <= len(rep.unsupported):
                        raw, rep, regenerated = raw2, rep2, True
            final = a.verifier.repair(raw, rep) if rep.unsupported else raw
            if final != raw:
                rep = a.verifier.verify(final, gctx.evidence)
            sp.attrs.update(faithfulness=round(rep.faithfulness, 3), unsupported=len(rep.unsupported), regenerated=regenerated)
        METRICS.observe("latency_ms.query", trace.total_ms)
        return AgentRun(question, an, cfg.describe, steps, ctx, final, raw, mode, rep, regenerated, trace, iterations,
                        llm.stats.as_dict() if llm else {})
