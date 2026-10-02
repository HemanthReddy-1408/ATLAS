"""Knowledge update pipeline: schedule -> crawl -> change-detect -> version -> process -> index."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field

import httpx

from .clock import Clock
from .config import Settings
from .domain import AUTHORITY, Chunk, ParsedDocument, Source, UrlState
from .extract import content_hash, diff_documents, extract_dates, parse_html, raw_hash
from .indexes import IndexManager
from .ingest import Crawler, Frontier
from .process import (
    EntityResolver,
    RelationExtractor,
    build_relation,
    chunk_document,
    count_tokens,
    split_sentences,
)
from .safety import scan
from .store import Repository

log = logging.getLogger("atlas.pipeline")


@dataclass
class RunReport:
    fetched: int = 0
    new_docs: int = 0
    changed_docs: int = 0
    unchanged_docs: int = 0
    gone_docs: int = 0
    failed: int = 0
    skipped_robots: int = 0
    chunks_new: int = 0
    chunks_reused: int = 0
    chunks_removed: int = 0
    embedded: int = 0
    relations_new: int = 0
    relations_retired: int = 0
    quarantined: int = 0
    discovered: int = 0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["errors"] = self.errors[:10]
        return d


class Updater:
    def __init__(self, repo: Repository, resolver: EntityResolver, extractor: RelationExtractor,
                 indexes: IndexManager, settings: Settings, clock: Clock) -> None:
        self.repo, self.res, self.rx, self.idx, self.s, self.clock = repo, resolver, extractor, indexes, settings, clock
        self.frontier = Frontier(repo, clock)

    # ------------------------------------------------------------ one run
    async def run(self, transport: httpx.AsyncBaseTransport | None = None, max_pages: int = 500) -> RunReport:
        rep = RunReport()
        self.frontier.schedule_due()
        for r in self.repo.urls_by_state(UrlState.FAILED, 1000):  # retry failures on the next scheduled run
            self.repo.set_url_state(r.url, UrlState.QUEUED)
        crawler = Crawler(self.s, transport)
        lock, sem = asyncio.Lock(), asyncio.Semaphore(self.s.max_concurrency)
        touched: set[str] = set()
        try:
            while rep.fetched < max_pages:
                batch = self.frontier.next_batch(min(self.s.max_concurrency * 2, max_pages - rep.fetched))
                if not batch:
                    break
                for rec in batch:
                    self.repo.set_url_state(rec.url, UrlState.CRAWLING)
                rep.fetched += len(batch)

                async def one(rec, crawler=crawler, rep=rep, lock=lock, sem=sem) -> None:
                    async with sem:
                        try:
                            await self._handle(rec, crawler, rep, lock)
                        except Exception as e:  # one bad page must never kill the run
                            log.exception("page failed: %s", rec.url)
                            rep.failed += 1
                            rep.errors.append(f"{rec.url}: {type(e).__name__}: {e}")
                            self.repo.set_url_state(rec.url, UrlState.FAILED, last_error=str(e)[:300])

                await asyncio.gather(*(one(r) for r in batch))
                touched.update(r.source_id for r in batch)
        finally:
            await crawler.aclose()
        for sid in touched:
            self.repo.touch_source(sid, self.clock.now_iso())
        return rep

    async def _handle(self, rec, crawler: Crawler, rep: RunReport, lock: asyncio.Lock) -> None:
        src = self.repo.source(rec.source_id)
        assert src is not None
        if src.robots_policy == "respect" and not await crawler.robots.allowed(rec.url):
            self.repo.set_url_state(rec.url, UrlState.SKIPPED, last_error="disallowed by robots.txt")
            rep.skipped_robots += 1
            return
        res = await crawler.fetch(rec.url, rec.etag, rec.last_modified)
        now = self.clock.now_iso()
        if res.not_modified:
            self.repo.set_url_state(rec.url, UrlState.UNCHANGED, last_crawled=now)
            rep.unchanged_docs += 1
            return
        if res.status in (404, 410):
            async with lock:
                if self._tombstone(rec.url, now):
                    rep.gone_docs += 1
            self.repo.set_url_state(rec.url, UrlState.FAILED, last_error=res.error, last_crawled=now)
            return
        if res.error or res.html is None:
            self.repo.set_url_state(rec.url, UrlState.FAILED, last_error=res.error,
                                    fail_count=rec.fail_count + 1, last_crawled=now)
            rep.failed += 1
            return
        rh = raw_hash(res.html)
        if rec.content_hash == rh:  # byte-identical page: nothing to do
            self.repo.set_url_state(rec.url, UrlState.UNCHANGED, last_crawled=now, etag=res.etag, last_modified=res.last_modified)
            rep.unchanged_docs += 1
            return
        async with lock:
            state, links = await asyncio.to_thread(self._process_page, src, rec.url, res.html, rh, rep)
        self.repo.set_url_state(rec.url, state, last_crawled=now, etag=res.etag, last_modified=res.last_modified,
                                content_hash=rh, fail_count=0, last_error=None)
        for link in links:
            if self.frontier.add(link, src, rec.depth + 1):
                rep.discovered += 1

    # --------------------------------------------------------- processing
    def _process_page(self, src: Source, url: str, html: str, rh: str, rep: RunReport) -> tuple[str, list[str]]:
        now = self.clock.now_iso()
        parsed = parse_html(html, url)
        if count_tokens(parsed.text) < self.s.chunk_min_tokens:  # listing/stub page: follow its links, don't index it
            return UrlState.CRAWLED, parsed.links
        chash = content_hash(parsed)
        doc_id = hashlib.sha1(url.encode()).hexdigest()[:12]
        existing = self.repo.document_by_url(url)
        raw_id = self.repo.add_raw(url, src.source_id, now, 200, rh, html)
        old_parsed = None
        if existing:
            cur = self.repo.current_version(existing["document_id"])
            if cur and cur["content_hash"] == chash:
                self.repo.touch_document(doc_id, now)
                rep.unchanged_docs += 1
                return UrlState.UNCHANGED, parsed.links
            if cur:
                old_parsed = ParsedDocument.from_json(json.loads(cur["structure_json"]))
            version = existing["current_version"] + 1
            rep.changed_docs += 1
            state = UrlState.CHANGED
        else:
            self.repo.create_document(doc_id, url, src.source_id, parsed.title, now)
            version = 1
            rep.new_docs += 1
            state = UrlState.CHANGED
        diff = diff_documents(old_parsed, parsed)
        summary = "initial version" if old_parsed is None else diff.summary()
        valid_from = parsed.updated_at or parsed.published_at or now
        self.repo.add_version(doc_id, version, chash, raw_id, now, now if old_parsed else valid_from, summary,
                              parsed.title, parsed.published_at, parsed.updated_at, parsed.to_json())
        self._index_document(src, doc_id, parsed, version, now, rep)
        return state, parsed.links

    def _index_document(self, src: Source, doc_id: str, parsed, version: int, now: str, rep: RunReport) -> None:
        meta = dict(title=parsed.title, url=parsed.url, source_id=src.source_id, source_name=src.name,
                    source_type=str(src.source_type), published_at=parsed.published_at or parsed.updated_at,
                    version=version, created_at=now)
        chunks = chunk_document(parsed, doc_id, self.s, meta)
        new, refreshed, clean = [], [], []
        keep = {c.chunk_id for c in chunks}
        hidden = scan(parsed.hidden_text) if parsed.hidden_text else None
        with self.repo.db.tx():
            if hidden and hidden.suspicious:  # instructions hidden from readers but visible to a model: keep for audit only
                hc = Chunk(chunk_id=hashlib.sha1(f"{doc_id}|hidden|{parsed.hidden_text}".encode()).hexdigest()[:16],
                        document_id=doc_id, text=parsed.hidden_text, section_title="(hidden text)", ordinal=999,
                        token_count=count_tokens(parsed.hidden_text), **meta)
                hc.risk = max(hidden.score, 0.75)
                self.repo.upsert_chunk(hc, now, True, "hidden text: " + hidden.notes())
                keep.add(hc.chunk_id)
                rep.quarantined += 1
            for c in chunks:
                sc = scan(c.text)
                c.risk = sc.score
                if sc.quarantine:
                    self.repo.upsert_chunk(c, now, True, sc.notes())
                    self.repo.set_chunk_entities(c.chunk_id, set())
                    rep.quarantined += 1
                    log.warning("quarantined chunk %s of %s: %s", c.chunk_id, parsed.url, sc.notes())
                    continue
                is_new = self.repo.upsert_chunk(c, now)
                clean.append(c)
                ents: set[str] = set()
                for m in self.res.extract(c.text):
                    ents.add(m.entity_id)
                c.entity_ids = tuple(sorted(ents))
                self.repo.set_chunk_entities(c.chunk_id, ents)
                (new if is_new else refreshed).append(c)
            gone = self.repo.deactivate_chunks(doc_id, keep, now)
            self._reconcile_relations(doc_id, clean, parsed.published_at, version, now, rep, AUTHORITY.get(str(src.source_type), 0.5))
            summary = self._doc_summary(doc_id, parsed, clean, meta)
            if summary is not None:
                keep.add(summary.chunk_id)
                (new if self.repo.upsert_chunk(summary, now) else refreshed).append(summary)
                self.repo.set_chunk_entities(summary.chunk_id, set(summary.entity_ids))
        rep.chunks_new += len(new)
        rep.chunks_reused += len(refreshed)
        rep.chunks_removed += len(gone)
        rep.embedded += self.idx.apply(new, refreshed, gone)

    def _doc_summary(self, doc_id: str, parsed, clean: list[Chunk], meta: dict) -> Chunk | None:
        """Extractive document-level node (title + lead sentence of every *scanned, clean* section) for broad queries.
        Derived nodes are built from clean chunks only and scanned again: a quarantined sentence must never leak back in."""
        if not clean:
            return None
        parts, seen = [parsed.title + "."], set()
        for c in clean:
            if c.section_title in seen:
                continue
            seen.add(c.section_title)
            first = split_sentences(c.text)
            if first:
                parts.append(first[0][2].strip().lstrip("- "))
            if len(parts) > 6:
                break
        text = " ".join(parts)
        if count_tokens(text) < 12 or scan(text).suspicious:
            return None
        ents = sorted({e for c in clean for e in c.entity_ids})
        cid = hashlib.sha1(f"{doc_id}|summary|{text}".encode()).hexdigest()[:16]
        return Chunk(cid, doc_id, text, "Document summary", 1000, count_tokens(text), entity_ids=tuple(ents), kind="doc_summary", **meta)

    def _reconcile_relations(self, doc_id: str, chunks: list[Chunk], doc_date: str | None, version: int,
                             now: str, rep: RunReport, authority: float = 1.0) -> None:
        """Graph reconciliation: facts re-observed stay active; facts no longer stated get valid_until=now."""
        before = self.repo.doc_relations(doc_id)
        current: dict[str, object] = {}
        ranks: dict[str, tuple[bool, float]] = {}
        for c in chunks:
            for s_start, s_end, sent in split_sentences(c.text):
                ms = [m for m in self.res.extract(c.text[s_start:s_end])]
                if len(ms) < 2:
                    continue
                for x in self.rx.extract_sentence(sent, ms, doc_date):
                    x.confidence = round(x.confidence * (0.6 + 0.4 * authority), 3)  # weak sources => weaker facts
                    if x.confidence < 0.5:
                        continue
                    r = build_relation(x, doc_id, c.chunk_id, now, version)
                    rank = (bool(extract_dates(x.sentence)), x.confidence)  # same triple twice on a page: keep the best-dated sentence
                    if r.relation_id in current and ranks[r.relation_id] >= rank:
                        continue
                    ranks[r.relation_id] = rank
                    if r.relation_id in before:
                        r.first_version = before[r.relation_id].first_version
                        r.valid_from = before[r.relation_id].valid_from or r.valid_from
                        r.observed_at = before[r.relation_id].observed_at
                    current[r.relation_id] = r
        for rid, r in current.items():
            if rid not in before:
                rep.relations_new += 1
            self.repo.put_relation(r)  # type: ignore[arg-type]
        for rid, old in before.items():
            if rid not in current and old.active:
                self.repo.retire_relation(rid, now)
                rep.relations_retired += 1

    def _tombstone(self, url: str, now: str) -> bool:
        d = self.repo.document_by_url(url)
        if not d or d["status"] == "gone":
            return False
        did = d["document_id"]
        gone = self.repo.deactivate_chunks(did, set(), now)
        for rid, r in self.repo.doc_relations(did).items():
            if r.active:
                self.repo.retire_relation(rid, now)
        self.repo.db.execute("UPDATE documents SET status='gone', last_seen=? WHERE document_id=?", (now, did))
        self.idx.apply([], [], gone)
        return True
