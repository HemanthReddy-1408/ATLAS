"""Conversational RAG: multi-turn memory with follow-up condensation, plus a semantic answer cache."""

from __future__ import annotations

import dataclasses
import re
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .agent import AgentRun, AskOptions
from .llm import LLMError

if TYPE_CHECKING:
    from .engine import Atlas

_PRONOUN = re.compile(r"\b(it|its|they|them|their|theirs|that|this|those|these|the same|former|latter|he|she)\b", re.I)
_CONTINUATION = re.compile(r"^\s*(and|also|what about|how about|why|how so|which one|then|so|but)\b", re.I)


@dataclass
class Turn:
    question: str
    standalone: str
    answer: str
    entity_ids: list[str]
    entity_names: list[str]
    mode: str
    evidence_urls: list[str]
    cache_hit: bool = False


@dataclass
class Conversation:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    turns: list[Turn] = field(default_factory=list)

    def history_messages(self, last: int = 2) -> list[dict]:
        out: list[dict] = []
        for t in self.turns[-last:]:
            out += [{"role": "user", "content": t.standalone}, {"role": "assistant", "content": t.answer[:900]}]
        return out


def condense(atlas: Atlas, conv: Conversation, text: str, use_llm: bool = True) -> tuple[str, str]:
    """Rewrite a follow-up ("what about its context window?") into a standalone question. Returns (question, how)."""
    if not conv.turns:
        return text, "first turn"
    an = atlas.understanding.analyze(text)
    last = conv.turns[-1]
    is_follow = bool(_PRONOUN.search(text) or _CONTINUATION.match(text) or (not an.entity_ids and len(text.split()) <= 9))
    if not is_follow or (an.entity_ids and not _PRONOUN.search(text) and not _CONTINUATION.match(text)):
        return text, "standalone"
    if use_llm and atlas.llm is not None:
        try:
            out = atlas.llm.complete(
                "Rewrite the user's follow-up as a single standalone question that needs no conversation context. Keep names exact. "
                "Return only the question.",
                f"Previous question: {last.standalone}\nPrevious answer (excerpt): {last.answer[:400]}\nFollow-up: {text}",
                max_tokens=80, fast=True).strip().strip('"')
            if 3 <= len(out.split()) <= 60:
                return out, "rewritten by LLM"
        except LLMError:
            pass
    names = last.entity_names[:2]
    if not names:
        return text, "no entity to carry over"
    if len(names) == 1 and _PRONOUN.search(text):
        poss = re.sub(r"\b(its|their)\b", f"{names[0]}'s", text, flags=re.I)
        return re.sub(r"\b(it|they|them|this|that)\b", names[0], poss, count=1, flags=re.I), f"carried over “{names[0]}”"
    return f"{text.rstrip('?')} (regarding {' and '.join(names)})?", f"carried over {', '.join(names)}"


class SemanticCache:
    """Answer cache keyed by question *meaning* (embedding cosine ≥ threshold), invalidated when the index changes."""

    def __init__(self, atlas: Atlas, threshold: float = 0.96, size: int = 64, ttl_s: float = 3600) -> None:
        self.a, self.threshold, self.size, self.ttl = atlas, threshold, size, ttl_s
        self.items: OrderedDict[int, tuple[np.ndarray, str, int, float, AgentRun]] = OrderedDict()
        self._n = 0
        self.hits = self.misses = 0

    def _key(self, q: str) -> np.ndarray:
        return self.a.idx.embedder.embed([q.lower().strip(" ?.!")])[0]

    def get(self, q: str, tag: str) -> AgentRun | None:
        v, now = self._key(q), time.monotonic()
        best, best_s = None, 0.0
        for k, (vec, t, ver, ts, _run) in list(self.items.items()):
            if t != tag or ver != self.a.idx.version or now - ts > self.ttl:
                continue
            s = float(v @ vec)
            if s > best_s:
                best, best_s = k, s
        if best is not None and best_s >= self.threshold:
            self.items.move_to_end(best)
            self.hits += 1
            return dataclasses.replace(self.items[best][4], cache_hit=True)
        self.misses += 1
        return None

    def put(self, q: str, tag: str, run: AgentRun) -> None:
        self._n += 1
        self.items[self._n] = (self._key(q), tag, self.a.idx.version, time.monotonic(), run)
        while len(self.items) > self.size:
            self.items.popitem(last=False)


def chat(atlas: Atlas, conv: Conversation, text: str, *, mode: str = "pipeline", options: AskOptions | None = None,
         trace=None, use_cache: bool = True) -> AgentRun:
    """One conversational turn: condense → (cache) → answer in the chosen mode → remember."""
    opts = options or AskOptions()
    use_llm = opts.use_llm and atlas.llm is not None
    standalone, how = condense(atlas, conv, text, use_llm)
    tag = f"{mode}|{use_llm}|{opts.label()}"
    run = atlas.cache.get(standalone, tag) if use_cache and mode != "autonomous" else None
    if run is None:
        history = conv.history_messages() if mode == "autonomous" else None
        run = atlas.ask(standalone, mode=mode, options=opts, trace=trace, history=history)
        if use_cache and mode != "autonomous":
            atlas.cache.put(standalone, tag, run)
    run.standalone_question = standalone if standalone != text else ""
    if how not in ("first turn", "standalone", "no entity to carry over"):
        run.trace.event("condense", how=how, standalone=standalone)
    names = run.analysis.entity_names or (conv.turns[-1].entity_names if conv.turns else [])
    ids = run.analysis.entity_ids or (conv.turns[-1].entity_ids if conv.turns else [])
    conv.turns.append(Turn(text, standalone, run.answer, ids, names, mode, [e.url for e in run.evidence], run.cache_hit))
    return run
