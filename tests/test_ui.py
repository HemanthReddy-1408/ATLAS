"""Drive the Streamlit app headlessly (streamlit.testing) against a seeded database."""

from __future__ import annotations

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from atlas.config import Settings
from atlas.engine import Atlas


@pytest.fixture()
def app(tmp_path, monkeypatch):
    db = str(tmp_path / "ui.db")
    a = Atlas(Settings(db_path=db, llm_provider="none", backoff_base_s=0.0), llm=None)
    a.load_fixtures(1)
    a.load_fixtures(2)
    a.db.close()
    monkeypatch.setenv("ATLAS_DB_PATH", db)
    monkeypatch.setenv("ATLAS_LLM_PROVIDER", "none")
    monkeypatch.delenv("ATLAS_GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("AEGIS_GROQ_API_KEY", raising=False)
    st.cache_resource.clear()
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=60)
    at.run()
    return at


def test_app_renders_every_tab_without_exceptions(app):
    assert not app.exception
    assert len(app.tabs) >= 8
    assert any("Documents" in m.label for m in app.sidebar.metric)


def test_asking_a_question_renders_answer_evidence_and_claims(app):
    app.chat_input[0].set_value("Which companies partner with NVIDIA and also build AI accelerators?").run()
    assert not app.exception
    md = " ".join(m.value for m in app.markdown)
    assert "Microsoft" in md and "[E" in md
    labels = {m.label for m in app.metric}
    assert {"Faithfulness", "Citation accuracy", "Evidence units"} <= labels


def test_evaluation_tab_runs_retrieval_eval(app):
    next(b for b in app.button if b.label == "Run retrieval evaluation").click()
    app.run()
    assert not app.exception
    assert any("hybrid" in str(df.value) for df in app.dataframe)
