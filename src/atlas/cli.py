"""Command-line entry point: `python -m atlas <command>` (or the `atlas` script)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .agent import AskOptions
from .config import PROJECT_ROOT, Settings
from .conversation import Conversation
from .engine import Atlas
from .evaluate import (
    eval_ablation,
    eval_answers,
    eval_judge,
    eval_retrieval,
    format_table,
    generate_questions,
    save_report,
    summarize_answers,
    summarize_judge,
)


def _atlas(args) -> Atlas:
    s = Settings.from_env()
    if getattr(args, "db", None):
        s = s.with_(db_path=args.db)
    return Atlas(s)


def cmd_demo(args) -> None:
    a = _atlas(args)
    for rnd in (1, 2) if args.updates else (1,):
        rep = a.load_fixtures(rnd)
        print(f"round {rnd}:", {k: v for k, v in rep.as_dict().items() if v and k != "errors"})
    print(json.dumps(a.stats(), indent=2))


def cmd_crawl(args) -> None:
    from .sources import LIVE_SOURCES

    a = _atlas(args)
    chosen = [s for s in LIVE_SOURCES if not args.source or s.source_id in args.source]
    a.register_sources(chosen)
    for s in a.repo.sources():
        if s.source_id not in {c.source_id for c in chosen}:
            a.repo.db.execute("UPDATE sources SET status='paused' WHERE source_id=?", (s.source_id,))
    a.settings = a.settings.with_(per_host_delay_s=1.0)
    a.updater.s = a.settings
    rep = a.ingest_sync(max_pages=args.max_pages)
    print(json.dumps(rep.as_dict(), indent=2))


def cmd_ask(args) -> None:
    a = _atlas(args)
    run = a.ask(args.question, mode=args.mode, options=AskOptions(use_llm=not args.no_llm, llm_judge=args.judge))
    print(run.answer, "\n")
    for m in run.transcript:
        if m.get("role") == "tool":
            print(f"  ⚙ {m['tool']}({m['args']})")
        elif "content" in m and m.get("role") not in ("assistant",):
            print(f"  [{m['role']}] {m['content'][:150]}")
    for c in run.conflicts:
        print(f"  ⚠ conflict: {c.fact} → {c.resolution['value']} ({c.resolution['source']}); {c.rationale}")
    print("— evidence —")
    for e in run.evidence:
        print(f"[{e.evidence_id}] {e.title} · {e.source} ({e.source_type}) {e.published_at or ''}\n     {e.url}")
    v = run.verification
    print(f"\nmode={run.answer_mode}  faithfulness={v.faithfulness:.2f}  citation_accuracy={v.citation_accuracy}  "
          f"unsupported={len(v.unsupported)}  iterations={run.iterations}  {run.trace.total_ms:.0f} ms")
    if args.debug:
        print(json.dumps(run.analysis.summary(), indent=2, default=str))
        print(format_table(run.trace.as_rows()))


def cmd_chat(args) -> None:
    a = _atlas(args)
    conv = Conversation()
    print("Atlas chat — blank line to quit.")
    while True:
        try:
            q = input("\nyou › ").strip()
        except EOFError:
            break
        if not q:
            break
        run = a.chat(conv, q, mode=args.mode, options=AskOptions(use_llm=not args.no_llm))
        if run.standalone_question:
            print(f"  (understood as: {run.standalone_question})")
        print(run.answer + ("\n  [served from semantic cache]" if run.cache_hit else ""))


def cmd_communities(args) -> None:
    a = _atlas(args)
    for c in a.graphrag.detect():
        print(f"L{c.level}#{c.index:<2} {len(c.members):>2} entities  {c.title}")
        if args.reports:
            print("   " + c.report.replace("\n", "\n   "))


def cmd_quarantine(args) -> None:
    for r in _atlas(args).repo.quarantined_chunks():
        print(f"risk {r['risk']:.2f}  {r['url']}\n   {r['risk_notes']}")


def cmd_ablate(args) -> None:
    a = _atlas(args)
    rows = eval_ablation(a, progress=lambda n, t, name: print(f"  [{n}/{t}] {name}", file=sys.stderr))
    print(format_table([r.row() for r in rows]))
    if args.out:
        Path(args.out).write_text(json.dumps([r.row() for r in rows], indent=2))


def cmd_synth(args) -> None:
    a = _atlas(args)
    if a.llm is None:
        sys.exit("synthetic question generation needs a Groq key")
    for it in generate_questions(a, args.n, a.llm):
        print(f"{it.id}: {it.question}\n     → {it.expected}  [{it.relevant_docs[0]}]")


def cmd_search(args) -> None:
    a = _atlas(args)
    for cid in a.search(args.query, args.mode, args.k):
        c = a.idx.chunks[cid]
        print(f"{c.title[:50]:50} | {c.section_title[:20]:20} | {c.text[:90]}")


def cmd_entity(args) -> None:
    a = _atlas(args)
    t = a.agent.tools
    print(json.dumps(t.call("get_entity", name=args.name), indent=2))
    for row in t.call("get_entity_history", name=args.name).get("timeline", []):
        print(f"  {row['valid_from'] or '—':10} {'✓' if row['active'] else '✗'} {row['fact']}")


def cmd_eval(args) -> None:
    a = _atlas(args)
    ret = eval_retrieval(a)
    print(format_table([r.row() for r in ret]))
    items = None
    if args.limit:
        from .evalset import EVAL_SET

        items = EVAL_SET[: args.limit]
    if args.judge:
        from .evalset import EVAL_SET as _E

        rows = eval_judge(a, (items or _E)[: args.limit or 6], progress=lambda n, t, q: print(f"  judge [{n}] {q[:60]}", file=sys.stderr))
        print(json.dumps(summarize_judge(rows), indent=2))
        return
    ans = eval_answers(a, items, use_llm=args.llm, progress=lambda n, t, q: print(f"  [{n}/{t}] {q[:70]}", file=sys.stderr))
    print(json.dumps(summarize_answers(ans), indent=2))
    out = Path(args.out or PROJECT_ROOT / "artifacts" / ("eval_llm.json" if args.llm else "eval.json"))
    save_report(out, ret, ans)
    print("report →", out)


def cmd_stats(args) -> None:
    print(json.dumps(_atlas(args).stats(), indent=2))


def cmd_ui(args) -> None:
    sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(PROJECT_ROOT / "app.py"), *args.extra]))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="atlas", description="Atlas — AI & technology intelligence engine")
    p.add_argument("--db", help="SQLite path (default: ATLAS_DB_PATH or atlas.db)")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("demo", help="load the offline fixture corpus")
    s.add_argument("--updates", action="store_true", help="also apply the second crawl round (edits, new pages, removals)")
    s.set_defaults(fn=cmd_demo)
    s = sub.add_parser("crawl", help="crawl the live source registry")
    s.add_argument("--source", action="append", help="source id (repeatable)")
    s.add_argument("--max-pages", type=int, default=40)
    s.set_defaults(fn=cmd_crawl)
    s = sub.add_parser("ask", help="ask the research agent")
    s.add_argument("question")
    s.add_argument("--no-llm", action="store_true", help="deterministic extractive answer, no Groq call")
    s.add_argument("--judge", action="store_true", help="LLM adjudication of borderline claims")
    s.add_argument("--mode", default="pipeline", choices=["pipeline", "autonomous", "multi-agent"],
                   help="pipeline = workflow agent; autonomous = LLM picks tools; multi-agent = planner/researchers/critic/writer")
    s.add_argument("--debug", action="store_true")
    s.set_defaults(fn=cmd_ask)
    s = sub.add_parser("search", help="raw retrieval")
    s.add_argument("query")
    s.add_argument("--mode", default="hybrid", choices=["bm25", "dense", "graph", "rrf", "hybrid"])
    s.add_argument("-k", type=int, default=8)
    s.set_defaults(fn=cmd_search)
    s = sub.add_parser("entity", help="entity card and timeline")
    s.add_argument("name")
    s.set_defaults(fn=cmd_entity)
    s = sub.add_parser("eval", help="retrieval + answer evaluation on the gold set")
    s.add_argument("--llm", action="store_true", help="evaluate Groq answers (slow: rate limited)")
    s.add_argument("--judge", action="store_true", help="RAGAS-style scoring by an LLM judge (Groq)")
    s.add_argument("--limit", type=int)
    s.add_argument("--out")
    s.set_defaults(fn=cmd_eval)
    s = sub.add_parser("chat", help="multi-turn conversation with memory and semantic cache")
    s.add_argument("--mode", default="pipeline", choices=["pipeline", "autonomous", "multi-agent"])
    s.add_argument("--no-llm", action="store_true")
    s.set_defaults(fn=cmd_chat)
    s = sub.add_parser("communities", help="GraphRAG communities")
    s.add_argument("--reports", action="store_true")
    s.set_defaults(fn=cmd_communities)
    sub.add_parser("quarantine", help="chunks withheld by the injection scanner").set_defaults(fn=cmd_quarantine)
    s = sub.add_parser("ablate", help="remove one pipeline component at a time and measure the damage")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_ablate)
    s = sub.add_parser("synth", help="generate synthetic gold questions with the LLM")
    s.add_argument("-n", type=int, default=4)
    s.set_defaults(fn=cmd_synth)
    sub.add_parser("stats").set_defaults(fn=cmd_stats)
    s = sub.add_parser("ui", help="launch the Streamlit app")
    s.add_argument("extra", nargs="*")
    s.set_defaults(fn=cmd_ui)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
