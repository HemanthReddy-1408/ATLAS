"""Multi-agent mode: a supervisor coordinating specialised roles over a shared blackboard.

    Planner     understands the question and splits it into self-contained sub-questions
    Researcher  one per sub-question; each independently analyses, routes (a relational sub-question gets a graph-first
                strategy, a definitional one BM25+dense, …), retrieves, grades (CRAG) and refines
    Critic      checks coverage and sends researchers back for follow-up rounds; detects *conflicting claims across
                sources* and resolves them by source authority and recency
    Writer      synthesises one cited answer; the shared claim verifier then audits it

Every message is kept, so the UI can show the agents' conversation."""

from __future__ import annotations

import dataclasses
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING

from .agent import AgentRun, AskOptions, Retrieved
from .domain import AUTHORITY, Relation
from .extract import MONTHS, extract_dates
from .observe import METRICS, Trace
from .ontology import REL_LABEL
from .query import SubQuestion, decompose, route
from .retrieve import ScoredChunk

if TYPE_CHECKING:
    from .engine import Atlas


@dataclass
class AgentMessage:
    role: str  # planner | researcher | critic | writer | supervisor
    content: str
    data: dict = field(default_factory=dict)


@dataclass
class Conflict:
    """Two or more sources give different dates for the same fact."""

    fact: str
    claims: list[dict]  # {value, source, source_type, url, chunk_id, authority, sentence}
    resolution: dict
    rationale: str
    entities: tuple[str, ...] = ()


def human_date(iso: str) -> str:
    try:
        d = date.fromisoformat(iso)
        return f"{['', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'][d.month]} {d.day}, {d.year}"
    except ValueError:
        return iso


def find_conflicts(atlas: Atlas, facts: list[Relation], chunk_ids: set[str] | None = None) -> list[Conflict]:
    """Group dated facts by (subject, relation, object). If different documents state different *sentence-level* dates,
    that is a conflict; the claim from the highest-authority (then most recent) source wins. Dates that merely come from
    a page's publication date are ignored: a late news article is not a contradiction."""
    g = atlas.idx.graph
    groups: dict[tuple[str, str, str], list[Relation]] = defaultdict(list)
    for r in facts:
        if r.src_id in g.nodes and r.dst_id in g.nodes and r.chunk_id in atlas.idx.chunks and extract_dates(r.sentence):
            groups[(r.src_id, r.rel, r.dst_id)].append(r)
    out = []
    for (s, rel, d), rs in groups.items():
        by_doc: dict[str, Relation] = {}
        for r in rs:
            by_doc.setdefault(r.document_id, r)
        claims = []
        for r in by_doc.values():
            day = next((x for x in extract_dates(r.sentence) if x.precision == "day"), None)
            if day is None:
                continue
            c = atlas.idx.chunks[r.chunk_id]
            claims.append({"value": day.iso, "source": c.source_name, "source_type": str(c.source_type), "url": c.url, "chunk_id": r.chunk_id,
                           "authority": AUTHORITY.get(str(c.source_type), 0.5), "sentence": r.sentence})
        if len({c["value"] for c in claims}) < 2:
            continue
        best = max(claims, key=lambda c: (c["authority"], c["value"]))
        names = f"{g.nodes[s].canonical_name} {REL_LABEL.get(rel, rel)} {g.nodes[d].canonical_name}"
        rationale = (f"{best['source']} ({best['source_type']}, authority {best['authority']:.1f}) outranks "
                     + ", ".join(f"{c['source']} ({c['source_type']}, {c['authority']:.1f})" for c in claims if c is not best))
        out.append(Conflict(names, sorted(claims, key=lambda c: -c["authority"]), best, rationale, (s, d)))
    return out


def conflict_notes(conflicts: list[Conflict]) -> str:
    lines = []
    for c in conflicts:
        vals = "; ".join(f"{x['source']} ({x['source_type']}) says {human_date(x['value'])}" for x in c.claims)
        lines.append(f"- CONFLICT on “{c.fact}”: {vals}. Prefer {human_date(c.resolution['value'])} ({c.rationale}); mention the discrepancy.")
    return "\n".join(lines)


class Supervisor:
    def __init__(self, atlas: Atlas) -> None:
        self.a = atlas

    def run(self, question: str, options: AskOptions | None = None, trace: Trace | None = None, max_rounds: int = 2) -> AgentRun:
        a, agent = self.a, self.a.agent
        opts = options or AskOptions()
        opts = dataclasses.replace(opts, use_llm=opts.use_llm and a.llm is not None)
        llm = a.llm if opts.use_llm else None
        trace = trace or Trace()
        board: list[AgentMessage] = []
        METRICS.inc("queries.multiagent")

        # ---------------------------------------------------------- planner
        with trace.span("planner") as sp:
            an = a.understanding.analyze(question)
            subs = decompose(an, a.resolver, llm)
            subs = [s for s in subs if s.kind != "window"] or subs
            if len(subs) == 1 and an.complex is False:
                subs = [SubQuestion("s1", question, "main", an.entity_ids)]
            plan_txt = "; ".join(f"{s.step_id}: {s.text}" for s in subs)
            board.append(AgentMessage("planner", f"{an.qtype.value} question → {len(subs)} sub-question(s): {plan_txt}", {"type": an.qtype.value}))
            sp.attrs.update(sub_questions=len(subs))

        # ------------------------------------------------------ researchers
        found: dict[str, Retrieved] = {}
        for s in subs:
            with trace.span("researcher", sub=s.step_id, question=s.text) as sp:
                rt = agent.research(s.text, opts, trace, llm)
                for st in rt.steps:
                    st.sub = dataclasses.replace(st.sub, step_id=f"{s.step_id}.{st.sub.step_id}")
                found[s.step_id] = rt
                strong = sum(1 for c in rt.candidates if c.rerank >= agent.STRONG)
                board.append(AgentMessage(f"researcher-{s.step_id}", f"{rt.analysis.qtype.value} → {rt.cfg.describe}. {len(rt.candidates)} candidates, "
                                          f"{strong} strong after grading; best score {max((c.rerank for c in rt.candidates), default=0):.2f}.",
                                          {"strong": strong, "sub": s.text}))
                sp.attrs.update(strong=strong, route=rt.cfg.describe[:80])

        # ----------------------------------------------------------- critic
        conflicts: list[Conflict] = []
        for rnd in range(1, max_rounds + 1):
            with trace.span("critic", round=rnd) as sp:
                weak = [s for s in subs if not any(st.sufficient for st in found[s.step_id].steps) or sum(1 for c in found[s.step_id].candidates if c.rerank >= agent.STRONG) == 0]
                all_facts = [f for rt in found.values() for f in rt.facts]
                relevant = set(an.entity_ids) | {e for s in subs for e in s.entity_ids}
                conflicts = [c for c in find_conflicts(a, all_facts) if set(c.entities) & relevant]  # only disputes about what was asked
                if conflicts:
                    board.append(AgentMessage("critic", f"{len(conflicts)} conflicting claim(s) across sources: " + "; ".join(
                        f"{c.fact} → resolved to {human_date(c.resolution['value'])} ({c.resolution['source']})" for c in conflicts)))
                if not weak or rnd == max_rounds or not opts.refine:
                    gaps = {t for s in weak for st in found[s.step_id].steps for t in st.sub.gap}
                    board.append(AgentMessage("critic", "coverage OK" if not weak else
                                              f"still weak: {', '.join(s.step_id for s in weak)}"
                                              + (f" — knowledge gap, never indexed: {', '.join(sorted(gaps))}" if gaps else "")))
                    sp.attrs.update(weak=len(weak), conflicts=len(conflicts))
                    break
                board.append(AgentMessage("critic", f"no strong evidence for {', '.join(s.step_id for s in weak)}: requesting follow-up research with broader strategy"))
                for s in weak:
                    broader = dataclasses.replace(opts, max_iterations=(opts.max_iterations or a.settings.max_agent_iterations) + 2, hyde=True, decompose=False)
                    rt2 = agent.research(s.text, broader, trace, llm)
                    for st in rt2.steps:
                        st.sub = dataclasses.replace(st.sub, step_id=f"{s.step_id}.r{rnd}")
                    found[s.step_id] = rt2
                    board.append(AgentMessage(f"researcher-{s.step_id}", f"follow-up round {rnd}: {len(rt2.candidates)} candidates, best "
                                              f"{max((c.rerank for c in rt2.candidates), default=0):.2f}"))
                sp.attrs.update(weak=len(weak), conflicts=len(conflicts), followup=True)

        # ------------------------------------------------------------ merge
        merged: dict[str, ScoredChunk] = {}
        for sid, rt in found.items():
            for c in rt.candidates:
                m = merged.get(c.chunk_id)
                if m is None:
                    merged[c.chunk_id] = dataclasses.replace(c, steps={sid}, ranks=dict(c.ranks), scores=dict(c.scores), via=list(c.via))
                else:
                    m.steps.add(sid)
                    m.rerank, m.fused = max(m.rerank, c.rerank), max(m.fused, c.fused)
        # conflicting sources must reach the writer even if they ranked low
        for c in conflicts:
            for cl in c.claims:
                if cl["chunk_id"] not in merged and cl["chunk_id"] in a.idx.chunks:
                    merged[cl["chunk_id"]] = ScoredChunk(cl["chunk_id"], rerank=0.35, final=0.35, steps={"critic"})
        seen: set[str] = set()
        facts = [f for rt in found.values() for f in rt.facts if not (f.relation_id in seen or seen.add(f.relation_id))]
        refined = {k: v for rt in found.values() for k, v in rt.refined.items()}
        top_cfg = dataclasses.replace(route(an), decompose=True)
        rt_all = Retrieved(an, top_cfg, subs, [st for rt in found.values() for st in rt.steps], list(merged.values()), facts,
                           max((rt.iterations for rt in found.values()), default=1), refined)

        # ----------------------------------------------------------- writer
        notes = conflict_notes(conflicts)
        with trace.span("writer"):
            run = agent.answer(question, rt_all, opts, trace, llm, notes=notes)
        if conflicts and run.answer_mode != "llm" and run.evidence:
            by_chunk = {e.chunk_id: e.evidence_id for e in run.evidence}
            lines = ["**Source conflicts**"]
            for c in conflicts:
                cites = "".join(f"[{by_chunk[x['chunk_id']]}]" for x in c.claims if x["chunk_id"] in by_chunk)
                vals = "; ".join(f"{x['source']} ({x['source_type']}) says {human_date(x['value'])}" for x in c.claims)
                lines.append(f"- {c.fact}: {vals}. Using {human_date(c.resolution['value'])} from {c.resolution['source']}, the higher-authority source. {cites}")
            run.answer = run.answer + "\n" + "\n".join(lines)
            run.verification = a.verifier.verify(run.answer, run.evidence)
        board.append(AgentMessage("writer", f"{run.answer_mode} answer, faithfulness {run.verification.faithfulness:.0%}, "
                                  f"{len(run.evidence)} evidence units"))
        covered = [s for s in subs if not any(st.sub.gap for st in found[s.step_id].steps) and (any(n.lower() in run.answer.lower() for n in [a.idx.graph.nodes[e].canonical_name for e in (s.entity_ids or []) if e in a.idx.graph.nodes])
                   or not s.entity_ids)]
        board.append(AgentMessage("critic", f"post-check: {len(covered)}/{len(subs)} sub-questions addressed by name in the answer; "
                                  f"{len(run.verification.unsupported)} unsupported claim(s) after verification"))
        run.mode, run.transcript, run.conflicts = "multi-agent", [dataclasses.asdict(m) for m in board], conflicts
        run.route = f"multi-agent: {len(subs)} researcher(s), {sum(1 for m in board if m.role == 'critic')} critic pass(es)"
        return run


__all__ = ["MONTHS", "Conflict", "Supervisor", "find_conflicts", "re"]
