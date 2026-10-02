"""GraphRAG-style global understanding: detect communities in the knowledge graph (Louvain modularity), write a
grounded report per community, and index those reports as *summary nodes* next to ordinary chunks.

Questions about a whole landscape ("how has the open-source ecosystem changed", "who are the key players") are poorly
served by top-k chunk retrieval; community reports answer them directly, and because each report is built from verbatim
source sentences its claims remain verifiable."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, field

from .domain import Chunk, Relation
from .indexes import IndexManager
from .llm import LLM, LLMError
from .ontology import REL_LABEL
from .safety import scan
from .store import Repository


# --------------------------------------------------------------- Louvain
def louvain(adj: dict[str, dict[str, float]], resolution: float = 1.0, max_passes: int = 25) -> dict[str, int]:
    """Phase 1 of the Louvain method (greedy modularity optimisation), deterministic node order."""
    nodes = sorted(adj)
    deg = {n: sum(adj[n].values()) for n in nodes}
    m2 = sum(deg.values()) or 1.0
    comm = {n: i for i, n in enumerate(nodes)}
    tot = {comm[n]: deg[n] for n in nodes}
    for _ in range(max_passes):
        moved = False
        for n in nodes:
            c0, k = comm[n], deg[n]
            tot[c0] -= k
            links: dict[int, float] = defaultdict(float)
            for o, w in adj[n].items():
                if o != n:
                    links[comm[o]] += w
            best, best_gain = c0, links.get(c0, 0.0) - resolution * tot[c0] * k / m2
            for c, w in sorted(links.items()):
                gain = w - resolution * tot[c] * k / m2
                if gain > best_gain + 1e-12:
                    best, best_gain = c, gain
            tot[best] = tot.get(best, 0.0) + k
            comm[n] = best
            moved |= best != c0
        if not moved:
            break
    order = {c: i for i, (c, _) in enumerate(sorted(((c, sum(1 for x in comm.values() if x == c)) for c in set(comm.values())),
                                                    key=lambda t: (-t[1], t[0])))}
    return {n: order[c] for n, c in comm.items()}


def aggregate(adj: dict[str, dict[str, float]], comm: dict[str, int]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for u, nb in adj.items():
        for v, w in nb.items():
            out[f"c{comm[u]}"][f"c{comm[v]}"] += w
    return {k: dict(v) for k, v in out.items()}


@dataclass
class Community:
    level: int
    index: int
    members: list[str]
    facts: list[Relation] = field(default_factory=list)
    title: str = ""
    report: str = ""
    narrative: str = ""


class GraphRAG:
    def __init__(self, repo: Repository, idx: IndexManager, clock) -> None:
        self.repo, self.idx, self.clock = repo, idx, clock
        self.communities: list[Community] = []
        self._narratives: dict[str, str] = {}

    # ---- detection
    def detect(self, min_conf: float = 0.5) -> list[Community]:
        g = self.idx.graph
        adj: dict[str, dict[str, float]] = defaultdict(dict)
        for r in g.rels.values():
            if r.confidence < min_conf or r.src_id == r.dst_id or r.src_id not in g.nodes or r.dst_id not in g.nodes:
                continue
            w = r.confidence * (0.5 if r.rel in ("IS_A", "VARIANT_OF") else 1.0) * (1.0 if r.active else 0.6)
            for a, b in ((r.src_id, r.dst_id), (r.dst_id, r.src_id)):
                adj[a][b] = adj[a].get(b, 0.0) + w
        out: list[Community] = []
        if not adj:
            self.communities = out
            return out
        c0 = louvain({k: dict(v) for k, v in adj.items()})
        groups0 = self._groups(c0)
        for i, members in enumerate(groups0):
            out.append(self._community(0, i, members))
        if len(groups0) > 3:  # level 1: communities of communities
            sup = louvain(aggregate(adj, c0))
            by_sup: dict[int, list[str]] = defaultdict(list)
            for c, s in sup.items():
                by_sup[s].extend(groups0[int(c[1:])])
            for i, members in enumerate(sorted(by_sup.values(), key=lambda m: (-len(m), sorted(m)[0]))):
                if len(members) > 3 and len(by_sup) < len(groups0):
                    out.append(self._community(1, i, sorted(members)))
        self.communities = out
        return out

    @staticmethod
    def _groups(comm: dict[str, int]) -> list[list[str]]:
        by: dict[int, list[str]] = defaultdict(list)
        for n, c in comm.items():
            by[c].append(n)
        return [sorted(by[c]) for c in sorted(by)]

    def _community(self, level: int, index: int, members: list[str]) -> Community:
        g = self.idx.graph
        mset = set(members)
        facts = [r for r in g.rels.values() if r.src_id in mset and r.dst_id in mset and r.confidence >= 0.5]
        facts.sort(key=lambda r: (-r.confidence * (1.0 if r.active else 0.7), r.valid_from or "", r.relation_id))
        deg = {m: sum(1 for r in facts if m in (r.src_id, r.dst_id)) for m in members}
        top = sorted(members, key=lambda m: (-deg[m], g.nodes[m].canonical_name))
        title = ", ".join(g.nodes[m].canonical_name for m in top[:3]) + (f" +{len(members) - 3}" if len(members) > 3 else "")
        c = Community(level, index, members, facts, title)
        c.report = self._report(c, top)
        return c

    def _report(self, c: Community, top: list[str]) -> str:
        g = self.idx.graph
        types: dict[str, int] = defaultdict(int)
        for m in c.members:
            types[g.nodes[m].type.lower()] += 1
        lines = [f"Community of {len(c.members)} related entities ({', '.join(f'{n} {t}' for t, n in sorted(types.items(), key=lambda kv: -kv[1]))}). "
                 f"Central entities: {', '.join(g.nodes[m].canonical_name for m in top[:6])}."]
        seen: set[str] = set()
        for r in c.facts:
            key = re.sub(r"\W+", " ", r.sentence.lower()).strip()
            if key in seen or not r.sentence:
                continue
            seen.add(key)
            when = f" ({r.valid_from[:7]})" if r.valid_from and r.valid_from[:4] in r.sentence else ""
            lines.append(f"- {r.sentence.strip()}{when}")
            if len(lines) >= 9:
                break
        text = "\n".join(lines)
        if c.narrative:
            text = c.narrative.strip() + "\n" + text
        return text

    # ---- indexing as summary nodes
    def rebuild(self, llm: LLM | None = None) -> dict:
        ts = self.clock.now_iso()
        comms = self.detect()
        if llm:
            self.narrate(comms, llm)
        new, refreshed, keep = [], [], set()
        for c in comms:
            if len(c.members) < 3 or len(c.facts) < 2 or scan(c.report).suspicious:
                continue
            dates = [r.valid_from for r in c.facts if r.valid_from]
            cid = hashlib.sha1(f"L{c.level}|{','.join(c.members)}|{c.report}".encode()).hexdigest()[:16]
            ch = Chunk(cid, f"community:L{c.level}-{c.index}", c.report, "Community report", c.index, len(c.report.split()),
                       title=f"Community: {c.title}", url=f"atlas://community/L{c.level}-{c.index}", source_id="graphrag",
                       source_name="GraphRAG communities", source_type="derived", published_at=max(dates) if dates else None,
                       version=1, created_at=ts, entity_ids=tuple(c.members), kind="community")
            keep.add(cid)
            is_new = self.repo.upsert_chunk(ch, ts)
            self.repo.set_chunk_entities(cid, set(c.members))
            (new if is_new else refreshed).append(ch)
        gone = self.repo.deactivate_kind("community", keep, ts)
        embedded = self.idx.apply(new, refreshed, gone)
        return {"communities": len(keep), "new": len(new), "removed": len(gone), "embedded": embedded}

    def narrate(self, comms: list[Community], llm: LLM, max_calls: int = 8) -> None:
        """Optional: one short Groq-written summary per community (cached by community content)."""
        for c in comms[:max_calls]:
            key = hashlib.sha1(c.report.encode()).hexdigest()
            if key not in self._narratives:
                try:
                    self._narratives[key] = llm.complete(
                        "You write 2-sentence summaries of clusters in a technology knowledge graph. Use only the facts given; "
                        "no new names, numbers or dates.", c.report, max_tokens=120, fast=True)
                except LLMError:
                    continue
            c.narrative = self._narratives[key]
            c.report = self._report(c, sorted(c.members, key=lambda m: -sum(1 for r in c.facts if m in (r.src_id, r.dst_id))))

    def describe(self, rel: Relation) -> str:
        n = self.idx.graph.nodes
        return f"{n[rel.src_id].canonical_name} {REL_LABEL.get(rel.rel, rel.rel)} {n[rel.dst_id].canonical_name}"
