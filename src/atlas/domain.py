"""Core vocabulary: enums and plain data records shared by every layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class SourceType(StrEnum):
    OFFICIAL = "official"
    DOCUMENTATION = "documentation"
    RESEARCH = "research"
    TECHNICAL_BLOG = "technical_blog"
    NEWS = "news"
    COMMUNITY = "community"


# Source authority is *metadata* consumed by context ranking, never a hard truth ranking.
AUTHORITY: dict[str, float] = {
    SourceType.OFFICIAL: 1.0,
    SourceType.DOCUMENTATION: 0.9,
    SourceType.RESEARCH: 0.85,
    SourceType.TECHNICAL_BLOG: 0.7,
    SourceType.NEWS: 0.5,
    SourceType.COMMUNITY: 0.3,
}


class UrlState(StrEnum):
    DISCOVERED = "DISCOVERED"
    QUEUED = "QUEUED"
    CRAWLING = "CRAWLING"
    CRAWLED = "CRAWLED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    CHANGED = "CHANGED"
    UNCHANGED = "UNCHANGED"


class EntityType(StrEnum):
    COMPANY = "COMPANY"
    PERSON = "PERSON"
    MODEL = "MODEL"
    PRODUCT = "PRODUCT"
    TECHNOLOGY = "TECHNOLOGY"
    FRAMEWORK = "FRAMEWORK"
    PAPER = "PAPER"
    DATASET = "DATASET"
    BENCHMARK = "BENCHMARK"
    ORGANIZATION = "ORGANIZATION"
    HARDWARE = "HARDWARE"  # extension of the base ontology: accelerators/chips


class QueryType(StrEnum):
    FACTUAL = "FACTUAL"
    COMPARATIVE = "COMPARATIVE"
    TEMPORAL = "TEMPORAL"
    RELATIONAL = "RELATIONAL"
    MULTI_HOP = "MULTI_HOP"
    EXPLORATORY = "EXPLORATORY"
    ANALYTICAL = "ANALYTICAL"


@dataclass(slots=True)
class Source:
    source_id: str
    name: str
    base_url: str
    source_type: str
    seed_urls: list[str] = field(default_factory=list)
    allowed_domains: list[str] = field(default_factory=list)
    include_patterns: list[str] = field(default_factory=list)
    exclude_patterns: list[str] = field(default_factory=list)
    crawl_frequency_s: int = 86400
    priority: int = 5
    robots_policy: str = "respect"
    max_depth: int = 2
    last_crawled: str | None = None
    status: str = "active"


@dataclass(slots=True)
class UrlRecord:
    url: str
    source_id: str
    state: str
    priority: int = 5
    depth: int = 0
    is_seed: bool = False
    etag: str | None = None
    last_modified: str | None = None
    content_hash: str | None = None
    fail_count: int = 0
    last_crawled: str | None = None


@dataclass(slots=True)
class Block:
    kind: str  # p | li | code | table | quote
    text: str


@dataclass(slots=True)
class Section:
    title: str
    level: int
    path: tuple[str, ...]
    blocks: list[Block] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(b.text for b in self.blocks)


@dataclass(slots=True)
class ParsedDocument:
    url: str
    title: str
    sections: list[Section]
    published_at: str | None = None
    updated_at: str | None = None
    canonical_url: str | None = None
    description: str = ""
    language: str = "en"
    links: list[str] = field(default_factory=list)
    hidden_text: str = ""  # text present in the DOM but invisible to readers (display:none, aria-hidden, …)

    @property
    def text(self) -> str:
        return "\n\n".join(s.text for s in self.sections if s.blocks)

    def to_json(self) -> dict[str, Any]:
        return {
            "url": self.url, "title": self.title, "published_at": self.published_at,
            "updated_at": self.updated_at, "canonical_url": self.canonical_url,
            "description": self.description, "language": self.language,
            "sections": [
                {"title": s.title, "level": s.level, "path": list(s.path),
                 "blocks": [[b.kind, b.text] for b in s.blocks]}
                for s in self.sections
            ],
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> ParsedDocument:
        return cls(
            url=d["url"], title=d["title"], published_at=d.get("published_at"),
            updated_at=d.get("updated_at"), canonical_url=d.get("canonical_url"),
            description=d.get("description", ""), language=d.get("language", "en"),
            sections=[
                Section(s["title"], s["level"], tuple(s["path"]),
                        [Block(k, t) for k, t in s["blocks"]])
                for s in d["sections"]
            ],
        )


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    section_title: str
    ordinal: int
    token_count: int
    title: str = ""
    url: str = ""
    source_id: str = ""
    source_name: str = ""
    source_type: str = SourceType.COMMUNITY
    published_at: str | None = None
    version: int = 1
    created_at: str = ""
    active: bool = True
    entity_ids: tuple[str, ...] = ()
    kind: str = "chunk"  # chunk | doc_summary | community
    risk: float = 0.0  # prompt-injection score from atlas.safety

    @property
    def contextual_text(self) -> str:
        """Text used for indexing: heading path travels with the chunk."""
        head = f"{self.title}. {self.section_title}." if self.section_title != self.title else f"{self.title}."
        return f"{head}\n{self.text}"


@dataclass(slots=True)
class Entity:
    entity_id: str
    canonical_name: str
    type: str
    description: str = ""
    provisional: bool = False
    aliases: tuple[str, ...] = ()


@dataclass(slots=True)
class Mention:
    entity_id: str
    surface: str
    start: int
    end: int
    type: str


@dataclass(slots=True)
class Relation:
    relation_id: str
    src_id: str
    rel: str
    dst_id: str
    document_id: str
    chunk_id: str | None
    sentence: str
    confidence: float
    observed_at: str
    valid_from: str | None = None
    valid_until: str | None = None
    active: bool = True
    first_version: int = 1
    last_version: int = 1


@dataclass(slots=True)
class Candidate:
    chunk_id: str
    score: float
    rank: int
    method: str
    via: str = ""  # the query variant / graph path that produced it


@dataclass(slots=True)
class Evidence:
    evidence_id: str
    chunk_id: str
    document_id: str
    source: str
    source_type: str
    url: str
    title: str
    section: str
    text: str
    published_at: str | None
    version: int
    relevance_score: float
    rerank_score: float
    final_score: float
    retrieval_methods: list[str]
    timestamp: str = ""
    kind: str = "chunk"  # chunk | graph_fact

    def cite(self) -> str:
        return f"[{self.evidence_id}]"
