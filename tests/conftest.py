from __future__ import annotations

import pytest

from atlas.config import Settings
from atlas.engine import Atlas
from atlas.fixtures import FixtureSite


def offline_settings() -> Settings:
    return Settings(db_path=":memory:", llm_provider="none", max_retries=2, backoff_base_s=0.0)


@pytest.fixture(scope="session")
def corpus() -> Atlas:
    """Full fixture corpus after both crawl rounds. Read-only: tests must not mutate it."""
    a = Atlas(offline_settings(), llm=None)
    a.load_fixtures(1)
    a.load_fixtures(2)
    return a


@pytest.fixture()
def fresh() -> Atlas:
    return Atlas(offline_settings(), llm=None)


@pytest.fixture()
def site() -> FixtureSite:
    return FixtureSite()
