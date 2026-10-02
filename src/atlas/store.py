"""SQLite system of record: sources, crawl state, raw docs, versions, chunks, entities, relations.

Indexes (BM25 / vectors / graph) are *derived* from this database and rebuilt from it.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np

from .domain import Chunk, Entity, Relation, Source, UrlRecord

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources(
  source_id TEXT PRIMARY KEY, name TEXT NOT NULL, base_url TEXT NOT NULL,
  source_type TEXT NOT NULL, crawl_frequency_s INTEGER NOT NULL DEFAULT 86400,
  priority INTEGER NOT NULL DEFAULT 5, robots_policy TEXT NOT NULL DEFAULT 'respect',
  allowed_domains TEXT NOT NULL DEFAULT '[]', seed_urls TEXT NOT NULL DEFAULT '[]',
  include_patterns TEXT NOT NULL DEFAULT '[]', exclude_patterns TEXT NOT NULL DEFAULT '[]',
  max_depth INTEGER NOT NULL DEFAULT 2, last_crawled TEXT, status TEXT NOT NULL DEFAULT 'active');
CREATE TABLE IF NOT EXISTS urls(
  url TEXT PRIMARY KEY, source_id TEXT NOT NULL, state TEXT NOT NULL, priority INTEGER NOT NULL DEFAULT 5,
  depth INTEGER NOT NULL DEFAULT 0, is_seed INTEGER NOT NULL DEFAULT 0, discovered_at TEXT,
  last_crawled TEXT, fail_count INTEGER NOT NULL DEFAULT 0, last_error TEXT,
  etag TEXT, last_modified TEXT, content_hash TEXT);
CREATE INDEX IF NOT EXISTS urls_state ON urls(state, priority DESC);
CREATE TABLE IF NOT EXISTS raw_documents(
  raw_id INTEGER PRIMARY KEY AUTOINCREMENT, url TEXT NOT NULL, source_id TEXT NOT NULL,
  crawl_timestamp TEXT NOT NULL, http_status INTEGER NOT NULL, content_hash TEXT NOT NULL, raw_html TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS documents(
  document_id TEXT PRIMARY KEY, url TEXT UNIQUE NOT NULL, source_id TEXT NOT NULL, title TEXT,
  current_version INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'active', first_seen TEXT, last_seen TEXT);
CREATE TABLE IF NOT EXISTS document_versions(
  version_id TEXT PRIMARY KEY, document_id TEXT NOT NULL, version INTEGER NOT NULL, content_hash TEXT NOT NULL,
  raw_id INTEGER, created_at TEXT NOT NULL, valid_from TEXT NOT NULL, valid_until TEXT, change_summary TEXT,
  title TEXT, published_at TEXT, updated_at TEXT, structure_json TEXT NOT NULL, UNIQUE(document_id, version));
CREATE TABLE IF NOT EXISTS chunks(
  chunk_id TEXT PRIMARY KEY, document_id TEXT NOT NULL, text TEXT NOT NULL, section_title TEXT, ordinal INTEGER,
  token_count INTEGER, title TEXT, url TEXT, source_id TEXT, source_name TEXT, source_type TEXT,
  published_at TEXT, first_version INTEGER, last_version INTEGER, created_at TEXT, valid_until TEXT,
  active INTEGER NOT NULL DEFAULT 1);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(document_id, active);
CREATE TABLE IF NOT EXISTS chunk_vectors(chunk_id TEXT PRIMARY KEY, model TEXT NOT NULL, dim INTEGER NOT NULL, vec BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS entities(
  entity_id TEXT PRIMARY KEY, canonical_name TEXT NOT NULL, type TEXT NOT NULL, description TEXT DEFAULT '',
  provisional INTEGER NOT NULL DEFAULT 0, created_at TEXT);
CREATE TABLE IF NOT EXISTS entity_aliases(
  alias_key TEXT NOT NULL, entity_id TEXT NOT NULL, alias_text TEXT NOT NULL, PRIMARY KEY(alias_key, entity_id));
CREATE TABLE IF NOT EXISTS chunk_entities(chunk_id TEXT NOT NULL, entity_id TEXT NOT NULL, PRIMARY KEY(chunk_id, entity_id));
CREATE TABLE IF NOT EXISTS relations(
  relation_id TEXT PRIMARY KEY, src_id TEXT NOT NULL, rel TEXT NOT NULL, dst_id TEXT NOT NULL,
  document_id TEXT NOT NULL, chunk_id TEXT, sentence TEXT, confidence REAL, observed_at TEXT,
  valid_from TEXT, valid_until TEXT, active INTEGER NOT NULL DEFAULT 1, first_version INTEGER, last_version INTEGER);
CREATE INDEX IF NOT EXISTS rel_src ON relations(src_id); CREATE INDEX IF NOT EXISTS rel_dst ON relations(dst_id);
CREATE INDEX IF NOT EXISTS rel_doc ON relations(document_id, active);
"""


class Database:
    def __init__(self, path: str = ":memory:") -> None:
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        if path != ":memory:":
            self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)

    def execute(self, sql: str, args: tuple | list = ()) -> sqlite3.Cursor:
        with self.lock:
            return self.conn.execute(sql, args)

    def rows(self, sql: str, args: tuple | list = ()) -> list[sqlite3.Row]:
        with self.lock:
            return self.conn.execute(sql, args).fetchall()

    def one(self, sql: str, args: tuple | list = ()) -> sqlite3.Row | None:
        with self.lock:
            return self.conn.execute(sql, args).fetchone()

    @contextmanager
    def tx(self) -> Iterator[None]:
        with self.lock:
            self.conn.execute("BEGIN")
            try:
                yield
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise
            else:
                self.conn.execute("COMMIT")

    def close(self) -> None:
        self.conn.close()


def _j(v: Any) -> str:
    return json.dumps(v)


class Repository:
    """Typed access to the tables. Thin on purpose: logic lives in the layers above."""

    def __init__(self, db: Database) -> None:
        self.db = db

    # ----------------------------------------------------------- sources
    def upsert_source(self, s: Source) -> None:
        self.db.execute(
            """INSERT INTO sources(source_id,name,base_url,source_type,crawl_frequency_s,priority,robots_policy,
               allowed_domains,seed_urls,include_patterns,exclude_patterns,max_depth,status)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(source_id) DO UPDATE SET name=excluded.name, base_url=excluded.base_url,
               source_type=excluded.source_type, crawl_frequency_s=excluded.crawl_frequency_s,
               priority=excluded.priority, robots_policy=excluded.robots_policy,
               allowed_domains=excluded.allowed_domains, seed_urls=excluded.seed_urls,
               include_patterns=excluded.include_patterns, exclude_patterns=excluded.exclude_patterns,
               max_depth=excluded.max_depth, status=excluded.status""",
            (s.source_id, s.name, s.base_url, str(s.source_type), s.crawl_frequency_s, s.priority,
             s.robots_policy, _j(s.allowed_domains), _j(s.seed_urls), _j(s.include_patterns),
             _j(s.exclude_patterns), s.max_depth, s.status),
        )

    @staticmethod
    def _source(r: sqlite3.Row) -> Source:
        return Source(
            r["source_id"], r["name"], r["base_url"], r["source_type"], json.loads(r["seed_urls"]),
            json.loads(r["allowed_domains"]), json.loads(r["include_patterns"]),
            json.loads(r["exclude_patterns"]), r["crawl_frequency_s"], r["priority"],
            r["robots_policy"], r["max_depth"], r["last_crawled"], r["status"],
        )

    def sources(self, active_only: bool = False) -> list[Source]:
        q = "SELECT * FROM sources" + (" WHERE status='active'" if active_only else "") + " ORDER BY priority DESC, name"
        return [self._source(r) for r in self.db.rows(q)]

    def source(self, source_id: str) -> Source | None:
        r = self.db.one("SELECT * FROM sources WHERE source_id=?", (source_id,))
        return self._source(r) if r else None

    def touch_source(self, source_id: str, ts: str) -> None:
        self.db.execute("UPDATE sources SET last_crawled=? WHERE source_id=?", (ts, source_id))

    # -------------------------------------------------------------- urls
    def url(self, url: str) -> UrlRecord | None:
        r = self.db.one("SELECT * FROM urls WHERE url=?", (url,))
        return self._url(r) if r else None

    @staticmethod
    def _url(r: sqlite3.Row) -> UrlRecord:
        return UrlRecord(r["url"], r["source_id"], r["state"], r["priority"], r["depth"],
                         bool(r["is_seed"]), r["etag"], r["last_modified"], r["content_hash"],
                         r["fail_count"], r["last_crawled"])

    def insert_url(self, u: UrlRecord, ts: str) -> None:
        self.db.execute(
            "INSERT OR IGNORE INTO urls(url,source_id,state,priority,depth,is_seed,discovered_at) VALUES(?,?,?,?,?,?,?)",
            (u.url, u.source_id, u.state, u.priority, u.depth, int(u.is_seed), ts),
        )

    def set_url_state(self, url: str, state: str, **fields: Any) -> None:
        cols = ["state=?"] + [f"{k}=?" for k in fields]
        self.db.execute(f"UPDATE urls SET {', '.join(cols)} WHERE url=?", (state, *fields.values(), url))

    def urls_by_state(self, state: str, limit: int = 100) -> list[UrlRecord]:
        rs = self.db.rows("SELECT * FROM urls WHERE state=? ORDER BY priority DESC, depth, discovered_at LIMIT ?", (state, limit))
        return [self._url(r) for r in rs]

    def url_counts(self) -> dict[str, int]:
        return {r["state"]: r["n"] for r in self.db.rows("SELECT state, COUNT(*) n FROM urls GROUP BY state")}

    # --------------------------------------------------------- raw / docs
    def add_raw(self, url: str, source_id: str, ts: str, status: int, chash: str, html: str) -> int:
        cur = self.db.execute(
            "INSERT INTO raw_documents(url,source_id,crawl_timestamp,http_status,content_hash,raw_html) VALUES(?,?,?,?,?,?)",
            (url, source_id, ts, status, chash, html),
        )
        return int(cur.lastrowid or 0)

    def document_by_url(self, url: str) -> sqlite3.Row | None:
        return self.db.one("SELECT * FROM documents WHERE url=?", (url,))

    def document(self, document_id: str) -> sqlite3.Row | None:
        return self.db.one("SELECT * FROM documents WHERE document_id=?", (document_id,))

    def documents(self) -> list[sqlite3.Row]:
        return self.db.rows("SELECT d.*, s.name source_name, s.source_type FROM documents d JOIN sources s USING(source_id) ORDER BY d.url")

    def current_version(self, document_id: str) -> sqlite3.Row | None:
        return self.db.one(
            "SELECT v.* FROM document_versions v JOIN documents d ON d.document_id=v.document_id AND d.current_version=v.version WHERE v.document_id=?",
            (document_id,),
        )

    def versions(self, document_id: str) -> list[sqlite3.Row]:
        return self.db.rows("SELECT * FROM document_versions WHERE document_id=? ORDER BY version", (document_id,))

    def create_document(self, document_id: str, url: str, source_id: str, title: str, ts: str) -> None:
        self.db.execute(
            "INSERT INTO documents(document_id,url,source_id,title,current_version,first_seen,last_seen) VALUES(?,?,?,?,0,?,?)",
            (document_id, url, source_id, title, ts, ts),
        )

    def add_version(self, document_id: str, version: int, chash: str, raw_id: int | None, ts: str,
                    valid_from: str, summary: str, title: str, published: str | None,
                    updated: str | None, structure: dict) -> str:
        vid = f"{document_id}@v{version}"
        with self.db.tx():
            self.db.execute(
                "UPDATE document_versions SET valid_until=? WHERE document_id=? AND valid_until IS NULL", (valid_from, document_id)
            )
            self.db.execute(
                """INSERT INTO document_versions(version_id,document_id,version,content_hash,raw_id,created_at,valid_from,
                   change_summary,title,published_at,updated_at,structure_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (vid, document_id, version, chash, raw_id, ts, valid_from, summary, title, published, updated, _j(structure)),
            )
            self.db.execute(
                "UPDATE documents SET current_version=?, title=?, last_seen=?, status='active' WHERE document_id=?",
                (version, title, ts, document_id),
            )
        return vid

    def touch_document(self, document_id: str, ts: str) -> None:
        self.db.execute("UPDATE documents SET last_seen=? WHERE document_id=?", (ts, document_id))

    # ------------------------------------------------------------ chunks
    @staticmethod
    def _chunk(r: sqlite3.Row, ents: tuple[str, ...] = ()) -> Chunk:
        return Chunk(r["chunk_id"], r["document_id"], r["text"], r["section_title"] or "", r["ordinal"] or 0,
                     r["token_count"] or 0, r["title"] or "", r["url"] or "", r["source_id"] or "",
                     r["source_name"] or "", r["source_type"] or "", r["published_at"], r["last_version"] or 1,
                     r["created_at"] or "", bool(r["active"]), ents)

    def doc_chunks(self, document_id: str, active_only: bool = True) -> list[Chunk]:
        q = "SELECT * FROM chunks WHERE document_id=?" + (" AND active=1" if active_only else "") + " ORDER BY ordinal"
        return [self._chunk(r, self.chunk_entities(r["chunk_id"])) for r in self.db.rows(q, (document_id,))]

    def all_active_chunks(self) -> list[Chunk]:
        ents: dict[str, list[str]] = {}
        for r in self.db.rows("SELECT ce.chunk_id, ce.entity_id FROM chunk_entities ce JOIN chunks c USING(chunk_id) WHERE c.active=1"):
            ents.setdefault(r["chunk_id"], []).append(r["entity_id"])
        return [self._chunk(r, tuple(ents.get(r["chunk_id"], ()))) for r in
                self.db.rows("SELECT * FROM chunks WHERE active=1 ORDER BY document_id, ordinal")]

    def chunk(self, chunk_id: str) -> Chunk | None:
        r = self.db.one("SELECT * FROM chunks WHERE chunk_id=?", (chunk_id,))
        return self._chunk(r, self.chunk_entities(chunk_id)) if r else None

    def chunk_entities(self, chunk_id: str) -> tuple[str, ...]:
        return tuple(r["entity_id"] for r in self.db.rows("SELECT entity_id FROM chunk_entities WHERE chunk_id=?", (chunk_id,)))

    def upsert_chunk(self, c: Chunk, ts: str) -> bool:
        """Insert or refresh a chunk. Returns True if the chunk is *new* (needs embedding)."""
        existing = self.db.one("SELECT chunk_id FROM chunks WHERE chunk_id=?", (c.chunk_id,))
        if existing:
            self.db.execute(
                """UPDATE chunks SET ordinal=?, section_title=?, title=?, url=?, source_name=?, source_type=?,
                   published_at=?, last_version=?, active=1, valid_until=NULL WHERE chunk_id=?""",
                (c.ordinal, c.section_title, c.title, c.url, c.source_name, c.source_type, c.published_at, c.version, c.chunk_id),
            )
            return False
        self.db.execute(
            """INSERT INTO chunks(chunk_id,document_id,text,section_title,ordinal,token_count,title,url,source_id,
               source_name,source_type,published_at,first_version,last_version,created_at,active)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)""",
            (c.chunk_id, c.document_id, c.text, c.section_title, c.ordinal, c.token_count, c.title, c.url,
             c.source_id, c.source_name, str(c.source_type), c.published_at, c.version, c.version, ts),
        )
        return True

    def deactivate_chunks(self, document_id: str, keep: set[str], ts: str) -> list[str]:
        gone = [r["chunk_id"] for r in self.db.rows("SELECT chunk_id FROM chunks WHERE document_id=? AND active=1", (document_id,))
                if r["chunk_id"] not in keep]
        for cid in gone:
            self.db.execute("UPDATE chunks SET active=0, valid_until=? WHERE chunk_id=?", (ts, cid))
        return gone

    def set_chunk_entities(self, chunk_id: str, entity_ids: set[str]) -> None:
        self.db.execute("DELETE FROM chunk_entities WHERE chunk_id=?", (chunk_id,))
        for e in entity_ids:
            self.db.execute("INSERT OR IGNORE INTO chunk_entities VALUES(?,?)", (chunk_id, e))

    # ----------------------------------------------------------- vectors
    def put_vector(self, chunk_id: str, model: str, vec: np.ndarray) -> None:
        v = vec.astype(np.float32)
        self.db.execute("INSERT OR REPLACE INTO chunk_vectors VALUES(?,?,?,?)", (chunk_id, model, int(v.shape[0]), v.tobytes()))

    def vectors(self, model: str) -> dict[str, np.ndarray]:
        rs = self.db.rows(
            "SELECT v.chunk_id, v.vec FROM chunk_vectors v JOIN chunks c USING(chunk_id) WHERE v.model=? AND c.active=1", (model,)
        )
        return {r["chunk_id"]: np.frombuffer(r["vec"], dtype=np.float32) for r in rs}

    # ---------------------------------------------------------- entities
    def entity(self, entity_id: str) -> Entity | None:
        r = self.db.one("SELECT * FROM entities WHERE entity_id=?", (entity_id,))
        if not r:
            return None
        al = tuple(a["alias_text"] for a in self.db.rows("SELECT alias_text FROM entity_aliases WHERE entity_id=?", (entity_id,)))
        return Entity(r["entity_id"], r["canonical_name"], r["type"], r["description"] or "", bool(r["provisional"]), al)

    def entities(self) -> list[Entity]:
        als: dict[str, list[str]] = {}
        for a in self.db.rows("SELECT entity_id, alias_text FROM entity_aliases"):
            als.setdefault(a["entity_id"], []).append(a["alias_text"])
        return [Entity(r["entity_id"], r["canonical_name"], r["type"], r["description"] or "", bool(r["provisional"]),
                       tuple(als.get(r["entity_id"], ()))) for r in self.db.rows("SELECT * FROM entities ORDER BY canonical_name")]

    def upsert_entity(self, e: Entity, ts: str) -> None:
        self.db.execute(
            "INSERT OR IGNORE INTO entities(entity_id,canonical_name,type,description,provisional,created_at) VALUES(?,?,?,?,?,?)",
            (e.entity_id, e.canonical_name, str(e.type), e.description, int(e.provisional), ts),
        )

    def add_alias(self, entity_id: str, key: str, text: str) -> None:
        self.db.execute("INSERT OR IGNORE INTO entity_aliases VALUES(?,?,?)", (key, entity_id, text))

    def alias_rows(self) -> list[sqlite3.Row]:
        return self.db.rows("SELECT a.alias_key, a.alias_text, a.entity_id, e.type FROM entity_aliases a JOIN entities e USING(entity_id)")

    # --------------------------------------------------------- relations
    @staticmethod
    def _rel(r: sqlite3.Row) -> Relation:
        return Relation(r["relation_id"], r["src_id"], r["rel"], r["dst_id"], r["document_id"], r["chunk_id"],
                        r["sentence"] or "", r["confidence"] or 0.0, r["observed_at"] or "", r["valid_from"],
                        r["valid_until"], bool(r["active"]), r["first_version"] or 1, r["last_version"] or 1)

    def relations(self, active_only: bool = False) -> list[Relation]:
        q = "SELECT * FROM relations" + (" WHERE active=1" if active_only else "")
        return [self._rel(r) for r in self.db.rows(q)]

    def doc_relations(self, document_id: str) -> dict[str, Relation]:
        return {r["relation_id"]: self._rel(r) for r in self.db.rows("SELECT * FROM relations WHERE document_id=?", (document_id,))}

    def put_relation(self, r: Relation) -> None:
        self.db.execute(
            """INSERT INTO relations(relation_id,src_id,rel,dst_id,document_id,chunk_id,sentence,confidence,observed_at,
               valid_from,valid_until,active,first_version,last_version) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(relation_id) DO UPDATE SET chunk_id=excluded.chunk_id, sentence=excluded.sentence,
               confidence=excluded.confidence, active=1, valid_until=NULL, last_version=excluded.last_version""",
            (r.relation_id, r.src_id, r.rel, r.dst_id, r.document_id, r.chunk_id, r.sentence, r.confidence,
             r.observed_at, r.valid_from, r.valid_until, int(r.active), r.first_version, r.last_version),
        )

    def retire_relation(self, relation_id: str, ts: str) -> None:
        self.db.execute("UPDATE relations SET active=0, valid_until=? WHERE relation_id=?", (ts, relation_id))

    # -------------------------------------------------------------- misc
    def stats(self) -> dict[str, int]:
        q = lambda t, w="": self.db.one(f"SELECT COUNT(*) n FROM {t} {w}")["n"]  # noqa: E731
        return {
            "sources": q("sources"), "urls": q("urls"), "documents": q("documents"),
            "versions": q("document_versions"), "chunks": q("chunks", "WHERE active=1"),
            "entities": q("entities"), "relations": q("relations", "WHERE active=1"),
            "relations_historic": q("relations", "WHERE active=0"),
        }
