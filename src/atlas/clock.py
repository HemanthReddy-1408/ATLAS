"""Injectable clock so crawl scheduling and temporal logic are testable."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta


def iso(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


class Clock:
    def now(self) -> datetime:
        return datetime.now(UTC)

    def now_iso(self) -> str:
        return iso(self.now())


class FrozenClock(Clock):
    def __init__(self, start: datetime | None = None) -> None:
        self._now = start or datetime(2026, 1, 1, tzinfo=UTC)

    def now(self) -> datetime:
        return self._now

    def advance(self, **kw: float) -> None:
        self._now += timedelta(**kw)
