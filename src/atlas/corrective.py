"""Corrective RAG (CRAG): grade every retrieved chunk, drop what is wrong, and refine what is only partly useful.

CORRECT   → use as is
AMBIGUOUS → adjudicated by the LLM when available; otherwise kept only if it names a query entity. Its text is
            *refined* (decompose into sentences, keep the relevant ones, recompose) before it reaches the prompt
INCORRECT → removed from the candidate set; if too little survives, the agent corrects course (rewrite / broaden)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from .indexes import IndexManager, tokenize
from .llm import LLM, LLMError
from .process import count_tokens, split_sentences
from .retrieve import ScoredChunk


class Grade(StrEnum):
    CORRECT = "correct"
    AMBIGUOUS = "ambiguous"
    INCORRECT = "incorrect"


@dataclass
class GradedChunk:
    chunk_id: str
    grade: Grade
    score: float
    reason: str
    refined: str | None = None


CORRECT_AT, AMBIGUOUS_AT = 0.30, 0.15


class CorrectiveGrader:
    def __init__(self, idx: IndexManager) -> None:
        self.idx = idx

    def grade(self, query: str, entity_ids: list[str], cands: list[ScoredChunk], llm: LLM | None = None) -> dict[str, GradedChunk]:
        out: dict[str, GradedChunk] = {}
        ents = set(entity_ids)
        for sc in cands:
            c = self.idx.chunks.get(sc.chunk_id)
            if c is None:
                continue
            hit = bool(ents & set(c.entity_ids))
            if sc.rerank >= CORRECT_AT or (sc.rerank >= 0.22 and hit):
                g, why = Grade.CORRECT, f"score {sc.rerank:.2f}" + (" + entity match" if hit else "")
            elif sc.rerank >= AMBIGUOUS_AT:
                g, why = Grade.AMBIGUOUS, f"score {sc.rerank:.2f}" + (" + entity match" if hit else ", no entity match")
            else:
                g, why = Grade.INCORRECT, f"score {sc.rerank:.2f} below floor"
            out[sc.chunk_id] = GradedChunk(sc.chunk_id, g, sc.rerank, why)
        amb = [x for x in out.values() if x.grade == Grade.AMBIGUOUS]
        if amb:
            self._adjudicate(query, amb, llm, ents)
        for x in out.values():
            if x.grade != Grade.INCORRECT:
                x.refined = self.refine(query, self.idx.chunks[x.chunk_id].text, ents)
        return out

    def _adjudicate(self, query: str, amb: list[GradedChunk], llm: LLM | None, ents: set[str]) -> None:
        if llm:
            batch = amb[:6]
            listing = "\n".join(f"{i}. {self.idx.chunks[x.chunk_id].title}: {self.idx.chunks[x.chunk_id].text[:280]}" for i, x in enumerate(batch, 1))
            try:
                d = llm.complete_json(
                    "You grade retrieved passages for a question. Return JSON {\"grades\": [{\"i\": 1, \"relevant\": true}]}. "
                    "relevant=true only if the passage contains information that helps answer the question.",
                    f"Question: {query}\n\nPassages:\n{listing}", max_tokens=200, fast=True)
                for row in (d.get("grades", []) if isinstance(d, dict) else d):
                    x = batch[int(row["i"]) - 1]
                    x.grade = Grade.CORRECT if row.get("relevant") else Grade.INCORRECT
                    x.reason += " → LLM: " + ("relevant" if row.get("relevant") else "irrelevant")
                return
            except (LLMError, KeyError, ValueError, IndexError, TypeError, AttributeError):
                pass
        for x in amb:  # deterministic adjudication: ambiguous passages survive only if they name a query entity
            c = self.idx.chunks[x.chunk_id]
            if ents & set(c.entity_ids):
                x.grade, x.reason = Grade.CORRECT, x.reason + " → kept (names a query entity)"
            else:
                x.grade, x.reason = Grade.INCORRECT, x.reason + " → dropped (no query entity)"

    def refine(self, query: str, text: str, ents: set[str]) -> str | None:
        """Knowledge refinement: keep the sentences that relate to the question. None = nothing to strip."""
        sents = [s for _, _, s in split_sentences(text)]
        if len(sents) < 4 or count_tokens(text) < 100:
            return None
        qt = {t for t in tokenize(query) if self.idx.bm25.idf(t) > 0.8}
        names = [self.idx.graph.nodes[e].canonical_name.lower() for e in ents if e in self.idx.graph.nodes]
        keep = []
        for i, s in enumerate(sents):
            low = s.lower()
            if i == 0 or set(tokenize(s)) & qt or any(n in low for n in names) or re.search(r"\b(19|20)\d{2}\b", s):
                keep.append(s)
        if len(keep) == len(sents) or not keep:
            return None
        return " ".join(keep)
