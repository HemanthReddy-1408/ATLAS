"""Atlas — Streamlit UI.   Run:  streamlit run app.py   (or: python -m atlas ui)"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd
import streamlit as st

from atlas.config import Settings
from atlas.engine import Atlas
from atlas.evalset import EVAL_SET
from atlas.evaluate import (
    eval_answers,
    eval_retrieval,
    save_report,
    summarize_answers,
)
from atlas.observe import METRICS
from atlas.sources import LIVE_SOURCES

st.set_page_config(page_title="Atlas · AI & Tech Intelligence", page_icon="🧭", layout="wide")

EXAMPLES = [
    "When did Meta release Llama 3?",
    "Which companies partner with NVIDIA and also build AI accelerators?",
    "How has the open-source LLM landscape changed since 2023, which companies are driving it, and what architectural trends emerged?",
    "Compare Llama 3.1 and Mistral 7B",
    "What are the trade-offs of Mixture of Experts models?",
    "What is the parameter count of GPT-5?",
]
TYPE_COLOR = {"COMPANY": "#4C78A8", "MODEL": "#F58518", "PRODUCT": "#54A24B", "TECHNOLOGY": "#B279A2", "HARDWARE": "#E45756",
              "PAPER": "#9D755D", "FRAMEWORK": "#72B7B2", "DATASET": "#BAB0AC", "BENCHMARK": "#EECA3B", "ORGANIZATION": "#4C78A8", "PERSON": "#FF9DA6"}


@st.cache_resource(show_spinner="Opening knowledge base…")
def get_atlas(db_path: str) -> Atlas:
    return Atlas(Settings.from_env().with_(db_path=db_path))


def cite_md(text: str) -> str:
    return re.sub(r"\[(E\d+)\]", r":blue[**[\1]**]", text)


def verdict_md(v: str) -> str:
    return {"SUPPORTED": ":green[✔ supported]", "PARTIAL": ":orange[◐ partial]", "UNSUPPORTED": ":red[✘ unsupported]"}[v]


# ------------------------------------------------------------------ sidebar
db_path = os.environ.get("ATLAS_DB_PATH", "atlas.db")
atlas = get_atlas(db_path)
stats = atlas.stats()

with st.sidebar:
    st.title("🧭 Atlas")
    st.caption("AI & technology intelligence — a living knowledge graph with an agentic RAG layer.")
    c1, c2 = st.columns(2)
    c1.metric("Documents", stats["documents"])
    c2.metric("Chunks", stats["chunks"])
    c1.metric("Entities", stats["entities"])
    c2.metric("Facts", stats["relations"])
    st.divider()
    st.markdown(f"**Database** `{db_path}`  \n**Embedder** `{stats['embedder']}`  \n**LLM** `{stats['llm']}`")
    if atlas.llm is None:
        st.warning("No Groq key found: answers use the deterministic extractive mode.", icon="⚠️")
    if stats["documents"] == 0:
        st.info("Empty knowledge base. Open the **Ingest** tab and load the demo corpus.", icon="👉")

tab_ask, tab_search, tab_graph, tab_corpus, tab_ingest, tab_eval = st.tabs(
    ["💬 Ask", "🔎 Search lab", "🕸️ Knowledge graph", "📚 Corpus", "⬇️ Ingest", "📊 Evaluation"])

# ---------------------------------------------------------------------- ask
with tab_ask:
    st.subheader("Ask a research question")
    if "q" not in st.session_state:
        st.session_state.q = EXAMPLES[1]
    cols = st.columns(3)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 3].button(ex if len(ex) < 60 else ex[:57] + "…", key=f"ex{i}", use_container_width=True):
            st.session_state.q = ex
            st.session_state.go = True
    q = st.text_area("Question", key="q", height=80, label_visibility="collapsed")
    o1, o2, o3 = st.columns([1, 1, 3])
    use_llm = o1.toggle("Use Groq for planning & writing", value=atlas.llm is not None, disabled=atlas.llm is None)
    judge = o2.toggle("LLM claim judge", value=False, disabled=atlas.llm is None, help="Extra Groq call to adjudicate borderline claims.")
    ask = o3.button("Ask Atlas", type="primary", disabled=not q.strip() or stats["documents"] == 0)
    if ask or st.session_state.pop("go", False):
        if stats["documents"] == 0:
            st.error("Load the demo corpus first (Ingest tab).")
        else:
            with st.spinner("Planning, retrieving, reasoning, verifying…"):
                st.session_state.run = atlas.ask(q, use_llm=use_llm, llm_judge=judge)

    run = st.session_state.get("run")
    if run:
        v = run.verification
        with st.container(border=True):
            st.markdown(cite_md(run.answer))
            st.caption(f"answer mode: **{run.answer_mode}** · {run.trace.total_ms:.0f} ms · {run.iterations} retrieval round(s)"
                       + (" · regenerated after verification" if run.regenerated else ""))
        m = st.columns(5)
        m[0].metric("Faithfulness", f"{v.faithfulness:.0%}", help="Share of claims fully supported by the evidence.")
        m[1].metric("Citation accuracy", "—" if v.citation_accuracy is None else f"{v.citation_accuracy:.0%}",
                    help="Of claims that carry a citation, how many are supported by the cited evidence.")
        m[2].metric("Unsupported claims", len(v.unsupported))
        m[3].metric("Evidence units", len(run.evidence))
        m[4].metric("Query type", run.analysis.qtype.value.title())

        t_ev, t_claims, t_plan, t_trace, t_ret = st.tabs(["Evidence", "Claim check", "Understanding & plan", "Agent trace", "Retrieval details"])
        with t_ev:
            if not run.evidence:
                st.info("No evidence was retrieved above the relevance floor.")
            for e in run.evidence:
                with st.expander(f"[{e.evidence_id}] {e.title} — {e.source} · {e.source_type}" + (f" · {e.published_at}" if e.published_at else "")):
                    st.write(e.text)
                    st.markdown(f"[{e.url}]({e.url})  \nsection **{e.section}** · v{e.version} · retrieved by `{'`, `'.join(e.retrieval_methods)}`")
                    st.progress(min(1.0, e.final_score), text=f"final {e.final_score:.2f} · rerank {e.rerank_score:.2f} · RRF {e.relevance_score:.4f}")
        with t_claims:
            if not v.claims:
                st.info("The answer makes no checkable claims (e.g. abstention).")
            else:
                st.dataframe(pd.DataFrame([{"verdict": c.verdict, "claim": c.text, "cited": ", ".join(c.cited), "evidence": ", ".join(c.supporting),
                                            "coverage": round(c.coverage, 2), "missing numbers/dates": ", ".join(c.missing_numbers),
                                            "missing names": ", ".join(c.missing_entities)} for c in v.claims]),
                             use_container_width=True, hide_index=True)
                st.caption("A claim is supported when an evidence passage covers its content words (synonym-aware) and every number, "
                           "date and proper name in it literally occurs there. Unsupported lines are removed or regenerated.")
        with t_plan:
            a = run.analysis
            c1, c2 = st.columns(2)
            c1.markdown("**Understanding**")
            c1.json({**a.summary(), "entity_types": a.entity_types})
            c2.markdown("**Routing decision**")
            c2.info(run.route)
            st.markdown("**Plan & execution**")
            st.dataframe(pd.DataFrame([{"step": s.sub.step_id, "kind": s.sub.kind, "sub-question": s.sub.text,
                                        "filters": s.cfg.filters.describe(), "sufficient": s.sufficient, "assessment": s.reason,
                                        "refinements": ", ".join(s.actions)} for s in run.steps]), use_container_width=True, hide_index=True)
        with t_trace:
            rows = pd.DataFrame(run.trace.as_rows())
            st.dataframe(rows, use_container_width=True, hide_index=True)
            timed = rows[rows["ms"] > 0].groupby("span")["ms"].sum()
            if not timed.empty:
                st.bar_chart(timed)
            if run.llm_stats:
                st.caption("Groq usage this query: " + ", ".join(f"{k}={v}" for k, v in run.llm_stats.items() if v))
        with t_ret:
            st.markdown(f"Context: {run.context.stats}")
            for s in run.steps:
                if not s.result:
                    continue
                with st.expander(f"Step {s.sub.step_id}: {s.sub.text}"):
                    tr = s.result.trace
                    st.caption(f"filters: {tr.filters} · timings (ms): { {k: round(v, 1) for k, v in tr.timings_ms.items()} }")
                    for name, ids in tr.lists.items():
                        st.markdown(f"`{name}` → " + " · ".join(atlas.idx.chunks[c].title[:28] for c in ids[:5] if c in atlas.idx.chunks))
            if run.context.dropped:
                st.markdown("**Dropped during context building**")
                st.dataframe(pd.DataFrame([{"chunk": atlas.idx.chunks[c].title if c in atlas.idx.chunks else c, "reason": r}
                                           for c, r in run.context.dropped]), use_container_width=True, hide_index=True)

# ------------------------------------------------------------------- search
with tab_search:
    st.subheader("Compare retrievers side by side")
    sq = st.text_input("Query", "Llama 4 Scout 17B active parameters")
    k = st.slider("Top-k", 3, 10, 5)
    if sq and stats["documents"]:
        modes = [("bm25", "BM25 (exact terms)"), ("dense", "Dense (vectors)"), ("graph", "Graph (entities)"),
                 ("rrf", "RRF fusion"), ("hybrid", "Fusion + rerank")]
        for col, (mode, label) in zip(st.columns(len(modes)), modes):
            with col:
                st.markdown(f"**{label}**")
                for i, cid in enumerate(atlas.search(sq, mode, k), 1):
                    c = atlas.idx.chunks[cid]
                    st.markdown(f"{i}. **{c.title[:40]}**  \n:gray[{c.section_title} · {c.source_name}]")
                    st.caption(c.text[:140] + "…")

# -------------------------------------------------------------------- graph
with tab_graph:
    st.subheader("Knowledge graph explorer")
    g = atlas.idx.graph
    names = sorted(((e.canonical_name, e.entity_id) for e in g.nodes.values()), key=lambda x: x[0].lower())
    if not names:
        st.info("No entities yet.")
    else:
        default = next((i for i, (n, _) in enumerate(names) if n == "NVIDIA"), 0)
        c1, c2, c3, c4 = st.columns([2, 1, 1, 2])
        pick = c1.selectbox("Entity", names, index=default, format_func=lambda x: x[0])
        hops = c2.slider("Hops", 1, 2, 1)
        retired = c3.checkbox("Include retired facts", value=True)
        rel_types = sorted({r.rel for r in g.rels.values()})
        chosen = c4.multiselect("Relations", rel_types, default=rel_types)
        eid = pick[1]
        ent = g.nodes[eid]
        st.markdown(f"### {ent.canonical_name}  `{ent.type}`")
        st.caption(ent.description + (f" · aliases: {', '.join(ent.aliases[:6])}" if ent.aliases else "") + (" · provisional (auto-discovered)" if ent.provisional else ""))
        hood = g.neighborhood([eid], hops, active_only=not retired)
        edges = {}
        for n in hood:
            for r, o in g.edges(n, rel=set(chosen) or None, active_only=not retired):
                if o in hood and r.relation_id not in edges:
                    edges[r.relation_id] = r
        edges_l = list(edges.values())[:70]
        dot = ["digraph G { rankdir=LR; bgcolor=\"transparent\"; node [shape=box, style=\"rounded,filled\", fontname=Helvetica, fontcolor=white, fontsize=11]; edge [fontname=Helvetica, fontsize=9];"]
        used = {eid}
        for r in edges_l:
            used |= {r.src_id, r.dst_id}
        for n in used:
            node = g.nodes[n]
            dot.append(f'"{n}" [label="{node.canonical_name}", fillcolor="{TYPE_COLOR.get(node.type, "#888")}"{", penwidth=3, color=black" if n == eid else ""}];')
        for r in edges_l:
            style = "solid" if r.active else "dashed"
            dot.append(f'"{r.src_id}" -> "{r.dst_id}" [label="{r.rel.lower().replace("_", " ")}", style={style}{", dir=both" if r.rel == "PARTNERS_WITH" else ""}];')
        dot.append("}")
        st.graphviz_chart("\n".join(dot), use_container_width=True)
        st.markdown("**Timeline** (dashed = fact no longer stated by its source)")
        tl = [{"since": r.valid_from or "", "until": r.valid_until or "", "active": r.active,
               "fact": f"{g.nodes[r.src_id].canonical_name} — {r.rel.lower().replace('_', ' ')} → {g.nodes[r.dst_id].canonical_name}",
               "confidence": r.confidence, "source sentence": r.sentence} for r in g.history(eid) if r.src_id in g.nodes and r.dst_id in g.nodes]
        st.dataframe(pd.DataFrame(tl), use_container_width=True, hide_index=True)

# ------------------------------------------------------------------- corpus
with tab_corpus:
    st.subheader("Corpus & versions")
    srcs = atlas.repo.sources()
    docs = atlas.repo.documents()
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("**Source registry**")
        st.dataframe(pd.DataFrame([{"source": s.name, "type": s.source_type, "priority": s.priority, "last crawled": s.last_crawled, "status": s.status}
                                   for s in srcs]), use_container_width=True, hide_index=True, height=360)
        st.markdown("**URL frontier**")
        st.bar_chart(pd.Series(atlas.repo.url_counts(), name="urls"))
    with c2:
        st.markdown("**Documents**")
        dd = pd.DataFrame([{"title": d["title"], "source": d["source_name"], "type": d["source_type"], "version": d["current_version"],
                            "status": d["status"], "url": d["url"]} for d in docs])
        st.dataframe(dd, use_container_width=True, hide_index=True, height=360)
    if docs:
        labels = {d["document_id"]: f"{d['title']}  (v{d['current_version']})" for d in docs}
        did = st.selectbox("Inspect a document", list(labels), format_func=labels.get)
        v1, v2 = st.columns(2)
        with v1:
            st.markdown("**Version history**")
            st.dataframe(pd.DataFrame([{"v": v["version"], "valid from": v["valid_from"], "valid until": v["valid_until"],
                                        "change": v["change_summary"]} for v in atlas.repo.versions(did)]), use_container_width=True, hide_index=True)
            st.markdown("**Chunks**")
            for c in atlas.repo.doc_chunks(did):
                with st.expander(f"{c.section_title} · {c.token_count} tokens · id {c.chunk_id[:8]}"):
                    st.write(c.text)
                    st.caption("entities: " + ", ".join(g.nodes[e].canonical_name for e in c.entity_ids if e in g.nodes))
        with v2:
            st.markdown("**Facts extracted from this document**")
            rr = atlas.repo.doc_relations(did)
            st.dataframe(pd.DataFrame([{"fact": f"{g.nodes[r.src_id].canonical_name} — {r.rel.lower().replace('_', ' ')} → {g.nodes[r.dst_id].canonical_name}",
                                        "since": r.valid_from, "active": r.active, "conf": r.confidence} for r in rr.values()
                                       if r.src_id in g.nodes and r.dst_id in g.nodes]), use_container_width=True, hide_index=True)

# ------------------------------------------------------------------- ingest
with tab_ingest:
    st.subheader("Knowledge update pipeline")
    st.caption("schedule → crawl (robots, retries, ETag) → extract → change-detect → version → chunk → entities/relations → embed → index")

    def show(rep) -> None:
        d = rep.as_dict()
        cols = st.columns(6)
        for col, key in zip(cols, ["fetched", "new_docs", "changed_docs", "unchanged_docs", "gone_docs", "failed"]):
            col.metric(key.replace("_", " "), d[key])
        cols = st.columns(6)
        for col, key in zip(cols, ["chunks_new", "chunks_reused", "chunks_removed", "embedded", "relations_new", "relations_retired"]):
            col.metric(key.replace("_", " "), d[key])
        if d["errors"]:
            st.error("\n".join(d["errors"]))

    b1, b2, b3 = st.columns(3)
    if b1.button("① Load demo corpus (crawl round 1)", use_container_width=True):
        with st.spinner("Crawling the offline fixture web…"):
            rep = atlas.load_fixtures(1)
        st.session_state.rep = rep
        st.rerun()
    if b2.button("② Apply later update (round 2)", use_container_width=True, help="New pages, edited pages and a removed page: watch only the changed chunks get re-embedded."):
        with st.spinner("Re-crawling…"):
            rep = atlas.load_fixtures(2)
        st.session_state.rep = rep
        st.rerun()
    if b3.button("Re-embed / rebuild indexes", use_container_width=True):
        atlas.idx.load()
        st.success("Indexes rebuilt from the database.")
    if "rep" in st.session_state:
        st.markdown("**Last run**")
        show(st.session_state.rep)

    st.divider()
    st.markdown("**Crawl the live web** (honours robots.txt; plain HTTP, no JavaScript rendering)")
    sel = st.multiselect("Sources", LIVE_SOURCES, default=LIVE_SOURCES[:1], format_func=lambda s: f"{s.name} ({s.source_type})")
    mp = st.number_input("Max pages", 5, 200, 20)
    if st.button("Crawl selected sources"):
        atlas.register_sources(sel)
        atlas.updater.s = atlas.settings.with_(per_host_delay_s=1.0)
        with st.spinner("Crawling… this can take a minute."):
            rep = atlas.ingest_sync(max_pages=int(mp))
        st.session_state.rep = rep
        st.rerun()

    st.divider()
    with st.expander("Danger zone"):
        if st.checkbox("I want to delete the whole knowledge base") and st.button("Reset database", type="secondary"):
            atlas.db.close()
            for suffix in ("", "-wal", "-shm"):
                Path(db_path + suffix).unlink(missing_ok=True)
            st.cache_resource.clear()
            st.session_state.clear()
            st.rerun()
    with st.expander("Process metrics"):
        st.json(METRICS.snapshot())

# --------------------------------------------------------------- evaluation
with tab_eval:
    st.subheader("Evaluation")
    st.caption(f"{len(EVAL_SET)} gold questions over the demo corpus (after both crawl rounds). Load rounds 1 and 2 first.")
    e1, e2 = st.columns(2)
    if e1.button("Run retrieval evaluation", use_container_width=True, disabled=stats["documents"] == 0):
        st.session_state.ret = eval_retrieval(atlas)
    if "ret" in st.session_state:
        df = pd.DataFrame([r.row() for r in st.session_state.ret]).set_index("mode")
        st.dataframe(df, use_container_width=True)
        st.bar_chart(df[["recall5", "recall10", "mrr", "ndcg10"]])
        st.caption("recall@k is document-level; MRR / nDCG / precision are chunk-level against phrase-constrained gold.")
    n_llm = e2.slider("Questions for the Groq run", 1, len(EVAL_SET), 4)
    c_a, c_b = st.columns(2)
    if c_a.button("Run answer evaluation (extractive, all questions)", use_container_width=True, disabled=stats["documents"] == 0):
        st.session_state.ans = eval_answers(atlas, use_llm=False)
    if c_b.button(f"Run answer evaluation with Groq ({n_llm} questions)", use_container_width=True, disabled=atlas.llm is None or stats["documents"] == 0):
        bar = st.progress(0.0, text="Evaluating…")
        st.session_state.ans = eval_answers(atlas, EVAL_SET[:n_llm], use_llm=True, progress=lambda n, t, q: bar.progress(n / t, text=q[:70]))
        bar.empty()
    if "ans" in st.session_state:
        sm = summarize_answers(st.session_state.ans)
        cols = st.columns(5)
        for col, key in zip(cols, ["completeness", "faithfulness", "citation_accuracy", "hallucination_rate", "abstention_accuracy"]):
            col.metric(key.replace("_", " ").title(), "—" if sm[key] is None else f"{sm[key]:.0%}")
        f = pd.Series(sm["failures"], name="questions")
        c1, c2 = st.columns(2)
        c1.markdown("**Failure attribution** (which stage lost the evidence)")
        c1.bar_chart(f)
        c2.markdown("**Completeness by question type**")
        c2.bar_chart(pd.Series(sm["completeness_by_type"]))
        st.dataframe(pd.DataFrame([s.row() for s in st.session_state.ans]), use_container_width=True, hide_index=True)
        if st.button("Save report to artifacts/"):
            path = Path("artifacts") / "ui_eval.json"
            save_report(path, st.session_state.get("ret", []), st.session_state.ans)
            st.success(f"saved {path}")
