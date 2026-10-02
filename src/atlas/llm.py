"""LLM access. Groq's OpenAI-compatible API, with rate limiting, caching, function calling and graceful failure.

Every LLM-assisted step in Atlas has a deterministic fallback, so an `LLMError` degrades quality, never availability.
"""

from __future__ import annotations

import contextlib
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


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class ChatMessage:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)

    def as_message(self) -> dict:
        """The assistant turn in OpenAI wire format, to be appended to the running transcript."""
        m: dict = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            m["tool_calls"] = [{"id": t.id, "type": "function", "function": {"name": t.name, "arguments": json.dumps(t.arguments)}}
                               for t in self.tool_calls]
        return m


class LLM:
    model = "none"
    stats: LLMStats

    def complete(self, system: str, prompt: str, *, json_mode: bool = False, max_tokens: int = 900,
                 temperature: float = 0.1, fast: bool = False) -> str:
        raise NotImplementedError

    def chat(self, messages: list[dict], tools: list[dict] | None = None, *, max_tokens: int = 700,
             temperature: float = 0.1, fast: bool = False) -> ChatMessage:
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
        self.max_wait_s = float(os.environ.get("ATLAS_LLM_MAX_WAIT_S", 75))

    # -- transport shared by complete() and chat()
    def _post(self, body: dict, est_tokens: int) -> dict:
        t0 = time.monotonic()
        last = "unknown"
        for attempt in range(3):
            self.stats.waited_s += self._limiter.reserve(est_tokens, self.max_wait_s)
            try:
                r = self.client.post("/chat/completions", json=body)
            except httpx.HTTPError as e:
                last = f"{type(e).__name__}: {e}"
                time.sleep(1.5 * (attempt + 1))
                continue
            if r.status_code == 200:
                data = r.json()
                u = data.get("usage", {})
                self.stats.calls += 1
                self.stats.prompt_tokens += u.get("prompt_tokens", 0)
                self.stats.completion_tokens += u.get("completion_tokens", 0)
                self.stats.seconds += time.monotonic() - t0
                return data
            last = f"HTTP {r.status_code}: {r.text[:200]}"
            if r.status_code in (429, 500, 502, 503):
                ra = r.headers.get("retry-after")
                time.sleep(min(float(ra), 20.0) if ra and ra.replace(".", "").isdigit() else 2.0 * (attempt + 1))
                continue
            break
        self.stats.errors += 1
        raise LLMError(last)

    def _body(self, model: str, messages: list[dict], max_tokens: int, temperature: float) -> dict:
        body: dict = {"model": model, "temperature": temperature, "max_completion_tokens": max_tokens, "messages": messages}
        if "gpt-oss" in model:
            body["reasoning_effort"] = "low"
            body["max_completion_tokens"] = max_tokens + 400  # reasoning tokens count against the budget
        return body

    def complete(self, system: str, prompt: str, *, json_mode: bool = False, max_tokens: int = 900,
                 temperature: float = 0.1, fast: bool = False) -> str:
        model = self.fast_model if fast else self.model
        key = hashlib.sha256(json.dumps([model, system, prompt, json_mode, max_tokens, temperature]).encode()).hexdigest()
        if key in self._cache:
            self.stats.cache_hits += 1
            return self._cache[key]
        body = self._body(model, [{"role": "system", "content": system}, {"role": "user", "content": prompt}], max_tokens, temperature)
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        est = (len(system) + len(prompt)) // 4 + body["max_completion_tokens"] // 2
        text = ""
        for _attempt in range(2):  # reasoning models can spend the whole budget thinking: retry once with room to answer
            data = self._post(body, est)
            text = re.sub(r"<think>.*?</think>", "", data["choices"][0]["message"].get("content") or "", flags=re.S).strip()
            if text:
                break
            body["max_completion_tokens"] = int(body["max_completion_tokens"] * 2)
        if not text:
            raise LLMError("empty completion (token budget likely spent on reasoning)")
        self._cache[key] = text
        return text

    def chat(self, messages: list[dict], tools: list[dict] | None = None, *, max_tokens: int = 700,
             temperature: float = 0.1, fast: bool = False) -> ChatMessage:
        model = self.fast_model if fast else self.model
        body = self._body(model, messages, max_tokens, temperature)
        if tools:
            body["tools"], body["tool_choice"] = tools, "auto"
        est = len(json.dumps(messages)) // 4 + (len(json.dumps(tools)) // 4 if tools else 0) + body["max_completion_tokens"] // 3
        data = self._post(body, est)
        msg = data["choices"][0]["message"]
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except ValueError:
                args = {}
            calls.append(ToolCall(tc.get("id") or f"call_{len(calls)}", fn.get("name", ""), args if isinstance(args, dict) else {}))
        content = re.sub(r"<think>.*?</think>", "", msg.get("content") or "", flags=re.S).strip()
        if not content and not calls:
            raise LLMError("empty chat completion")
        return ChatMessage(content, calls)


@dataclass
class FakeLLM(LLM):
    """Scripted LLM for tests. `responses`/`fn` drive complete(); `chat_script` (ChatMessage items or callables) drives chat()."""

    responses: list[str] = field(default_factory=list)
    fn: object = None
    chat_script: list = field(default_factory=list)
    calls: list[tuple[str, str]] = field(default_factory=list)
    chats: list[list[dict]] = field(default_factory=list)
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

    def chat(self, messages: list[dict], tools: list[dict] | None = None, **kw) -> ChatMessage:
        self.chats.append([dict(m) for m in messages])
        self.stats.calls += 1
        if not self.chat_script:
            raise LLMError("no scripted chat response")
        item = self.chat_script.pop(0)
        return item(messages, tools) if callable(item) else item


@contextlib.contextmanager
def patient(llm: LLM | None, seconds: float = 240.0):
    """Batch jobs (judge, synthetic data, multi-step agents) may wait out a small tokens-per-minute tier instead of failing."""
    old = getattr(llm, "max_wait_s", None)
    if old is not None:
        llm.max_wait_s = max(old, seconds)  # type: ignore[union-attr]
    try:
        yield llm
    finally:
        if old is not None:
            llm.max_wait_s = old  # type: ignore[union-attr]


def make_llm(s: Settings) -> LLM | None:
    if s.llm_provider == "groq" and s.groq_api_key:
        return GroqLLM(s)
    return None
