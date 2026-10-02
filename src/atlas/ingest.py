"""Knowledge acquisition: URL normalisation, robots policy, the URL frontier, and the crawler."""

from __future__ import annotations

import asyncio
import random
import re
import time
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import httpx

from .clock import Clock
from .config import Settings
from .domain import Source, UrlRecord, UrlState
from .store import Repository

_TRACKING = re.compile(r"^(utm_|fbclid|gclid|mc_|ref$|ref_src|igshid|_hs)")


def normalize_url(url: str, base: str | None = None) -> str | None:
    """Canonical form used for dedup: lowercase host, no fragment/default port/tracking params,
    sorted query, no trailing slash (except root). Returns None for non-http(s) URLs."""
    url = urljoin(base, url.strip()) if base else url.strip()
    p = urlsplit(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        return None
    host = p.hostname.lower()
    port = p.port
    netloc = host if port in (None, 80, 443) else f"{host}:{port}"
    path = re.sub(r"/{2,}", "/", p.path or "/")
    if len(path) > 1:
        path = path.rstrip("/")
    q = sorted((k, v) for k, v in parse_qsl(p.query, keep_blank_values=True) if not _TRACKING.match(k.lower()))
    return urlunsplit((p.scheme, netloc, path, urlencode(q), ""))


# ---------------------------------------------------------------- robots
class RobotsPolicy:
    """robots.txt per RFC 9309: 4xx => allow all; 5xx/unreachable => disallow all (this round)."""

    def __init__(self, fetch, user_agent: str) -> None:  # fetch: async (url) -> (status, text)
        self._fetch, self._ua = fetch, user_agent
        self._cache: dict[str, tuple[RobotFileParser | None, bool]] = {}

    async def _load(self, url: str) -> tuple[RobotFileParser | None, bool]:
        p = urlsplit(url)
        origin = f"{p.scheme}://{p.netloc}"
        if origin not in self._cache:
            try:
                status, text = await self._fetch(f"{origin}/robots.txt")
            except Exception:
                self._cache[origin] = (None, False)
            else:
                if status == 200:
                    rp = RobotFileParser()
                    rp.parse(text.splitlines())
                    self._cache[origin] = (rp, True)
                elif 400 <= status < 500:
                    self._cache[origin] = (None, True)
                else:
                    self._cache[origin] = (None, False)
        return self._cache[origin]

    async def allowed(self, url: str) -> bool:
        rp, ok = await self._load(url)
        if rp is None:
            return ok
        return rp.can_fetch(self._ua, url)

    async def crawl_delay(self, url: str) -> float:
        rp, _ = await self._load(url)
        d = rp.crawl_delay(self._ua) if rp else None
        return float(d) if d else 0.0


# -------------------------------------------------------------- frontier
class Frontier:
    """SQLite-backed URL frontier: normalise -> dedupe -> domain/pattern validate -> prioritise -> schedule."""

    def __init__(self, repo: Repository, clock: Clock) -> None:
        self.repo, self.clock = repo, clock

    def accepts(self, url: str, src: Source, seed: bool = False) -> bool:
        host = urlsplit(url).hostname or ""
        doms = src.allowed_domains or [urlsplit(src.base_url).hostname or ""]
        if not any(host == d or host.endswith("." + d) for d in doms):
            return False
        if src.exclude_patterns and any(re.search(p, url) for p in src.exclude_patterns):
            return False
        if not seed and src.include_patterns and not any(re.search(p, url) for p in src.include_patterns):  # seeds are listing pages
            return False
        return not re.search(r"\.(png|jpe?g|gif|svg|webp|pdf|zip|gz|mp4|mp3|css|js|ico|woff2?)$", url, re.I)

    def add(self, url: str, src: Source, depth: int = 0, is_seed: bool = False, base: str | None = None) -> bool:
        n = normalize_url(url, base)
        if not n or not self.accepts(n, src, is_seed) or depth > src.max_depth:
            return False
        if self.repo.url(n):
            return False
        pr = src.priority + (3 if is_seed else 0) - depth
        self.repo.insert_url(UrlRecord(n, src.source_id, UrlState.QUEUED, pr, depth, is_seed), self.clock.now_iso())
        return True

    def schedule_due(self) -> int:
        """Re-queue seeds + previously crawled URLs whose source is past its crawl frequency."""
        now, n = self.clock.now(), 0
        from .clock import parse_iso

        for src in self.repo.sources(active_only=True):
            due = src.last_crawled is None or (now - parse_iso(src.last_crawled)).total_seconds() >= src.crawl_frequency_s
            if not due:
                continue
            for seed in src.seed_urls:
                nu = normalize_url(seed)
                if nu and self.accepts(nu, src, True):
                    if self.repo.url(nu) is None:
                        self.add(nu, src, 0, True)
                    else:
                        self.repo.set_url_state(nu, UrlState.QUEUED)
                    n += 1
            for r in self.repo.db.rows("SELECT url FROM urls WHERE source_id=? AND state IN (?,?,?)",
                                       (src.source_id, UrlState.CRAWLED, UrlState.UNCHANGED, UrlState.CHANGED)):
                self.repo.set_url_state(r["url"], UrlState.QUEUED)
                n += 1
        return n

    def next_batch(self, limit: int) -> list[UrlRecord]:
        return self.repo.urls_by_state(UrlState.QUEUED, limit)


# --------------------------------------------------------------- crawler
@dataclass(slots=True)
class FetchResult:
    url: str
    status: int
    html: str | None = None
    final_url: str | None = None
    etag: str | None = None
    last_modified: str | None = None
    error: str | None = None
    attempts: int = 1
    elapsed_s: float = 0.0

    @property
    def not_modified(self) -> bool:
        return self.status == 304


class Crawler:
    """Simple on purpose: HTTP + retry/backoff + conditional GET. No browser automation."""

    RETRY_STATUS = {429, 500, 502, 503, 504}

    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None, sleep=asyncio.sleep) -> None:
        self.s = settings
        self.client = httpx.AsyncClient(
            transport=transport, timeout=settings.http_timeout_s, follow_redirects=True,
            headers={"User-Agent": settings.user_agent, "Accept": "text/html,application/xhtml+xml"},
        )
        self._sleep = sleep
        self._host_lock: dict[str, asyncio.Lock] = {}
        self._host_last: dict[str, float] = {}
        self.robots = RobotsPolicy(self._robots_fetch, settings.user_agent)

    async def _robots_fetch(self, url: str) -> tuple[int, str]:
        r = await self.client.get(url)
        return r.status_code, r.text

    async def _polite(self, url: str) -> None:
        host = urlsplit(url).netloc
        lock = self._host_lock.setdefault(host, asyncio.Lock())
        async with lock:
            delay = max(self.s.per_host_delay_s, await self.robots.crawl_delay(url))
            wait = self._host_last.get(host, 0.0) + delay - time.monotonic()
            if wait > 0:
                await self._sleep(wait)
            self._host_last[host] = time.monotonic()

    async def fetch(self, url: str, etag: str | None = None, last_modified: str | None = None) -> FetchResult:
        headers = {}
        if etag:
            headers["If-None-Match"] = etag
        if last_modified:
            headers["If-Modified-Since"] = last_modified
        t0, last_err = time.monotonic(), None
        for attempt in range(1, self.s.max_retries + 2):
            await self._polite(url)
            try:
                r = await self.client.get(url, headers=headers)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last_err = f"{type(e).__name__}: {e}"
            else:
                if r.status_code in self.RETRY_STATUS:
                    last_err = f"HTTP {r.status_code}"
                    ra = r.headers.get("Retry-After")
                    if ra and ra.isdigit() and attempt <= self.s.max_retries:
                        await self._sleep(min(float(ra), 30.0))
                        continue
                else:
                    return self._result(url, r, attempt, time.monotonic() - t0)
            if attempt <= self.s.max_retries:
                await self._sleep(self.s.backoff_base_s * 2 ** (attempt - 1) + random.random() * 0.1)
        return FetchResult(url, 0, error=last_err or "unknown", attempts=self.s.max_retries + 1, elapsed_s=time.monotonic() - t0)

    def _result(self, url: str, r: httpx.Response, attempts: int, elapsed: float) -> FetchResult:
        base = FetchResult(url, r.status_code, final_url=str(r.url), attempts=attempts, elapsed_s=elapsed,
                           etag=r.headers.get("ETag"), last_modified=r.headers.get("Last-Modified"))
        if r.status_code == 304:
            return base
        if r.status_code >= 400:
            base.error = f"HTTP {r.status_code}"
            return base
        ctype = r.headers.get("content-type", "text/html").lower()
        if "html" not in ctype and "xml" not in ctype:
            base.error = f"unsupported content-type {ctype}"
            return base
        if len(r.content) > self.s.max_page_bytes:
            base.error = "page too large"
            return base
        base.html = r.text
        return base

    async def aclose(self) -> None:
        await self.client.aclose()
