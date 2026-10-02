"""LLM access. Groq's OpenAI-compatible API, with rate limiting, caching and graceful failure.

Every LLM-assisted step in Atlas has a deterministic fallback, so an `LLMError` degrades quality, never availability.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import httpx

from .config import Settings


class LLMError(RuntimeError):
    pass


@dataclass
class LLMStats:
    calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    errors: int = 0
    waited_s: float = 0.0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class LLM:
    model = "none"
    stats: LLMStats

    def complete(self, system: str, prompt: str, *, json_mode: bool = False, max_tokens: int = 900,
                 temperature: float = 0.1, fast: bool = False) -> str:
        raise NotImplementedError

    def complete_json(self, system: str, prompt: str, **kw) -> dict | list:
        return parse_json(self.complete(system, prompt, json_mode=True, **kw))


def parse_json(text: str) -> dict | list:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except ValueError:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        a, b = text.find(open_c), text.rfind(close_c)
        if 0 <= a < b:
            try:
                return json.loads(text[a:b + 1])
            except ValueError:
                continue
    raise LLMError(f"model did not return valid JSON: {text[:120]!r}")


class _Limiter:
    """Sliding 60 s window over requests and (estimated) tokens."""

    def __init__(self, rpm: int, tpm: int) -> None:
        self.rpm, self.tpm = rpm, tpm
        self.events: deque[tuple[float, int]] = deque()
        self.lock = threading.Lock()

    def reserve(self, tokens: int, max_wait: float) -> float:
        waited = 0.0
        tokens = min(tokens, int(self.tpm * 0.95))
        while True:
            with self.lock:
                now = time.monotonic()
                while self.events and now - self.events[0][0] > 60:
                    self.events.popleft()
                used = sum(t for _, t in self.events)
                if len(self.events) < self.rpm and used + tokens <= self.tpm:
                    self.events.append((now, tokens))
                    return waited
                wait = max(0.5, 60 - (now - self.events[0][0]) + 0.1) if self.events else 1.0
            if waited + wait > max_wait:
                raise LLMError(f"rate limit: would wait {waited + wait:.0f}s (> {max_wait:.0f}s)")
            time.sleep(wait)
            waited += wait


class GroqLLM(LLM):
    def __init__(self, s: Settings, transport: httpx.BaseTransport | None = None) -> None:
        self.s, self.model, self.fast_model = s, s.model, s.fast_model
        self.client = httpx.Client(base_url=s.groq_base_url, timeout=s.llm_timeout_s, transport=transport,
                                   headers={"Authorization": f"Bearer {s.groq_api_key}"})
        self.stats = LLMStats()
        self._cache: dict[str, str] = {}
        self._limiter = _Limiter(int(os.environ.get("ATLAS_GROQ_RPM_LIMIT", 30)), int(os.environ.get("ATLAS_GROQ_TPM_LIMIT", 6000)))
        self.max_wait_s = 50.0

    def complete(self, system: str, prompt: str, *, json_mode: bool = False, max_tokens: int = 900,
                 temperature: float = 0.1, fast: bool = False) -> str:
        model = self.fast_model if fast else self.model
        key = hashlib.sha256(json.dumps([model, system, prompt, json_mode, max_tokens, temperature]).encode()).hexdigest()
        if key in self._cache:
            self.stats.cache_hits += 1
            return self._cache[key]
        body: dict = {
            "model": model, "temperature": temperature, "max_completion_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        }
        if "gpt-oss" in model:
            body["reasoning_effort"] = "low"
            body["max_completion_tokens"] = max_tokens + 400  # reasoning tokens count against the budget
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        est = (len(system) + len(prompt)) // 4 + body["max_completion_tokens"] // 2
        t0 = time.monotonic()
        last = "unknown"
        for attempt in range(3):
            self.stats.waited_s += self._limiter.reserve(est, self.max_wait_s)
            try:
                r = self.client.post("/chat/completions", json=body)
            except httpx.HTTPError as e:
                last = f"{type(e).__name__}: {e}"
                time.sleep(1.5 * (attempt + 1))
                continue
            if r.status_code == 200:
                data = r.json()
                text = data["choices"][0]["message"].get("content") or ""
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
                u = data.get("usage", {})
                self.stats.calls += 1
                self.stats.prompt_tokens += u.get("prompt_tokens", 0)
                self.stats.completion_tokens += u.get("completion_tokens", 0)
                self.stats.seconds += time.monotonic() - t0
                if not text:
                    raise LLMError("empty completion (token budget likely spent on reasoning)")
                self._cache[key] = text
                return text
            last = f"HTTP {r.status_code}: {r.text[:200]}"
            if r.status_code in (429, 500, 502, 503):
                ra = r.headers.get("retry-after")
                time.sleep(min(float(ra), 20.0) if ra and ra.replace(".", "").isdigit() else 2.0 * (attempt + 1))
                continue
            break
        self.stats.errors += 1
        raise LLMError(last)


@dataclass
class FakeLLM(LLM):
    """Scripted LLM for tests: responses are popped in order, or produced by `fn(system, prompt)`."""

    responses: list[str] = field(default_factory=list)
    fn: object = None
    calls: list[tuple[str, str]] = field(default_factory=list)
    model: str = "fake"
    stats: LLMStats = field(default_factory=LLMStats)

    def complete(self, system: str, prompt: str, **kw) -> str:
        self.calls.append((system, prompt))
        self.stats.calls += 1
        if self.fn:
            return self.fn(system, prompt)  # type: ignore[operator]
        if not self.responses:
            raise LLMError("no scripted response")
        return self.responses.pop(0)


def make_llm(s: Settings) -> LLM | None:
    if s.llm_provider == "groq" and s.groq_api_key:
        return GroqLLM(s)
    return None
