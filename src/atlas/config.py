"""Runtime settings. Read from the environment (and a local .env), never hard-coded."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv is optional at runtime
        return
    load_dotenv(PROJECT_ROOT / ".env", override=False)


@dataclass(frozen=True)
class Settings:
    db_path: str = "atlas.db"
    user_agent: str = "AtlasBot/0.1 (+research crawler; respects robots.txt)"
    # --- crawling
    http_timeout_s: float = 20.0
    max_retries: int = 3
    backoff_base_s: float = 0.5
    max_concurrency: int = 8
    per_host_delay_s: float = 0.0
    max_page_bytes: int = 5_000_000
    # --- chunking
    chunk_target_tokens: int = 200
    chunk_max_tokens: int = 320
    chunk_min_tokens: int = 25
    # --- retrieval
    embedder: str = "hashing"  # "hashing" | "st:<sentence-transformers model>"
    embedding_dim: int = 512
    reranker: str = "feature"  # "feature" | "cross-encoder:<model>"
    rrf_k: int = 60
    candidates_per_retriever: int = 50
    fused_k: int = 50
    rerank_k: int = 15
    final_evidence: int = 9
    context_token_budget: int = 2500
    freshness_half_life_days: float = 365.0
    max_agent_iterations: int = 3
    # --- LLM (Groq, OpenAI-compatible)
    llm_provider: str = "none"  # "groq" | "none"
    groq_api_key: str = field(default="", repr=False)
    groq_base_url: str = "https://api.groq.com/openai/v1"
    model: str = "openai/gpt-oss-120b"
    fast_model: str = "openai/gpt-oss-120b"
    llm_timeout_s: float = 60.0

    @classmethod
    def from_env(cls) -> Settings:
        _load_dotenv()
        e = os.environ.get
        key = e("ATLAS_GROQ_API_KEY") or e("GROQ_API_KEY") or e("AEGIS_GROQ_API_KEY") or ""
        provider = e("ATLAS_LLM_PROVIDER", "groq" if key else "none")
        model = e("ATLAS_MODEL") or e("AEGIS_JUDGE_MODEL") or cls.model
        return cls(
            db_path=e("ATLAS_DB_PATH", cls.db_path),
            embedder=e("ATLAS_EMBEDDER", cls.embedder),
            reranker=e("ATLAS_RERANKER", cls.reranker),
            llm_provider=provider if key else "none",
            groq_api_key=key,
            groq_base_url=e("ATLAS_GROQ_BASE_URL") or e("AEGIS_GROQ_BASE_URL") or cls.groq_base_url,
            model=model,
            fast_model=e("ATLAS_FAST_MODEL", model),
        )

    def with_(self, **kw: object) -> Settings:
        return replace(self, **kw)  # type: ignore[arg-type]
