"""The Atlas facade: wires storage, indexes, pipeline, retrieval and the agent together."""

from __future__ import annotations

import asyncio

import httpx

from .agent import AgentRun, ResearchAgent
from .clock import Clock
from .config import Settings
from .domain import Source
from .fixtures import FixtureSite, fixture_sources
from .generate import AnswerGenerator, ClaimVerifier, ContextBuilder
from .indexes import IndexManager, make_embedder
from .llm import LLM, make_llm
from .pipeline import RunReport, Updater
from .process import EntityResolver, RelationExtractor
from .query import QueryUnderstanding, route
from .retrieve import Filters, HybridRetriever, RetrievalConfig, Variant
from .store import Database, Repository


class Atlas:
    def __init__(self, settings: Settings | None = None, *, clock: Clock | None = None, llm: LLM | str | None = "auto",
                 db: Database | None = None, embedder=None) -> None:
        self.settings = settings or Settings.from_env()
        self.clock = clock or Clock()
        self.db = db or Database(self.settings.db_path)
        self.repo = Repository(self.db)
        self.resolver = EntityResolver(self.repo, self.clock)
        if not self.repo.entities():
            self.resolver.seed()
        self.extractor = RelationExtractor(self.resolver)
        self.idx = IndexManager(self.repo, self.settings, embedder or make_embedder(self.settings))
        self.idx.load()
        self.llm: LLM | None = make_llm(self.settings) if llm == "auto" else llm  # type: ignore[assignment]
        self.retrievers = HybridRetriever(self.idx, self.settings)
        self.understanding = QueryUnderstanding(self.resolver, self.clock)
        self.context = ContextBuilder(self.idx, self.settings, self.clock)
        self.generator = AnswerGenerator(self.idx, self.settings, self.llm)
        self.verifier = ClaimVerifier()
        self.updater = Updater(self.repo, self.resolver, self.extractor, self.idx, self.settings, self.clock)
        self.agent = ResearchAgent(self)

    # ------------------------------------------------------------ ingest
    def register_sources(self, sources: list[Source]) -> None:
        for s in sources:
            self.repo.upsert_source(s)

    async def ingest(self, transport: httpx.AsyncBaseTransport | None = None, max_pages: int = 500) -> RunReport:
        return await self.updater.run(transport, max_pages)

    def ingest_sync(self, transport: httpx.AsyncBaseTransport | None = None, max_pages: int = 500) -> RunReport:
        return asyncio.run(self.ingest(transport, max_pages))

    def load_fixtures(self, round: int = 1, site: FixtureSite | None = None) -> RunReport:
        """Crawl the offline fixture web (round 2 = later edits/removals, to exercise incremental updates)."""
        site = site or FixtureSite(round=round)
        self.register_sources(fixture_sources())
        # force every source due again so a second round actually re-crawls
        self.db.execute("UPDATE sources SET last_crawled=NULL")
        return self.ingest_sync(site.transport())

    # ---------------------------------------------------------- querying
    def ask(self, question: str, *, use_llm: bool = True, max_iterations: int | None = None, llm_judge: bool = False) -> AgentRun:
        return self.agent.run(question, use_llm=use_llm and self.llm is not None, max_iterations=max_iterations,
                              llm_judge=llm_judge)

    def search(self, question: str, mode: str = "hybrid", k: int = 10, filters: Filters | None = None) -> list[str]:
        """Single-shot retrieval returning ranked chunk ids. Modes: bm25 | dense | graph | rrf | hybrid."""
        an = self.understanding.analyze(question)
        f = filters or Filters()
        r = self.retrievers.r
        if mode == "bm25":
            return [c for c, _ in r.bm25(question, k, f)]
        if mode == "dense":
            return [c for c, _ in r.dense(question, k, f)]
        if mode == "graph":
            ranked, _, _ = r.graph(an.entity_ids, an.rel_hints, k, f)
            return [c for c, _ in ranked]
        cfg = RetrievalConfig(weights={"bm25": 1.0, "dense": 1.0, "graph": 1.0 if an.entity_ids else 0.0}, filters=f,
                              rerank=(mode == "hybrid"))
        res = self.retrievers.retrieve(question, an.entity_ids, an.rel_hints, [Variant(question)], cfg)
        return [s.chunk_id for s in res.candidates][:k]

    def stats(self) -> dict:
        return {**self.repo.stats(), "bm25_terms": len(self.idx.bm25.postings), "vectors": len(self.idx.vectors),
                "graph_edges": len(self.idx.graph.rels), "embedder": self.idx.embedder.name,
                "llm": self.settings.model if self.llm else "none (extractive mode)"}

    def routing_for(self, question: str) -> dict:
        an = self.understanding.analyze(question)
        cfg = route(an)
        return {"analysis": an.summary(), "route": cfg.describe, "weights": cfg.weights}

