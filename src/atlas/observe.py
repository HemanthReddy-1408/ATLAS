"""Observability: per-request traces (spans) and process-wide metrics."""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class Span:
    name: str
    start_ms: float
    duration_ms: float = 0.0
    attrs: dict = field(default_factory=dict)


class Trace:
    """Ordered record of what the system did for one request. Cheap enough to always keep."""

    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.spans: list[Span] = []

    @contextmanager
    def span(self, name: str, **attrs):
        sp = Span(name, (time.perf_counter() - self.t0) * 1e3, attrs=dict(attrs))
        self.spans.append(sp)
        try:
            yield sp
        finally:
            sp.duration_ms = (time.perf_counter() - self.t0) * 1e3 - sp.start_ms
            METRICS.observe(f"span_ms.{name}", sp.duration_ms)

    def event(self, name: str, **attrs) -> None:
        self.spans.append(Span(name, (time.perf_counter() - self.t0) * 1e3, 0.0, dict(attrs)))

    @property
    def total_ms(self) -> float:
        return (time.perf_counter() - self.t0) * 1e3

    def as_rows(self) -> list[dict]:
        return [{"span": s.name, "start_ms": round(s.start_ms, 1), "ms": round(s.duration_ms, 1),
                 **{k: (v if isinstance(v, (int, float, bool)) else str(v)[:160]) for k, v in s.attrs.items()}} for s in self.spans]


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.counters: dict[str, float] = defaultdict(float)
        self.hist: dict[str, list[float]] = defaultdict(list)

    def inc(self, name: str, n: float = 1) -> None:
        with self._lock:
            self.counters[name] += n

    def observe(self, name: str, v: float) -> None:
        with self._lock:
            h = self.hist[name]
            h.append(v)
            if len(h) > 2000:
                del h[:1000]

    def snapshot(self) -> dict:
        with self._lock:
            out = dict(self.counters)
            for k, h in self.hist.items():
                s = sorted(h)
                out[f"{k}.count"] = len(s)
                out[f"{k}.p50"] = round(s[len(s) // 2], 2)
                out[f"{k}.p95"] = round(s[min(len(s) - 1, int(len(s) * 0.95))], 2)
            return out

    def prometheus(self) -> str:
        return "\n".join(f"atlas_{k.replace('.', '_')} {v}" for k, v in sorted(self.snapshot().items())) + "\n"


METRICS = Metrics()
