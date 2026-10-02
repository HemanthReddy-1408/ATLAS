"""Acquisition → extraction → versioning → incremental indexing."""

from __future__ import annotations

import httpx
import pytest

from atlas.config import Settings
from atlas.domain import Source, UrlState
from atlas.engine import Atlas
from atlas.extract import content_hash, diff_documents, extract_dates, parse_date, parse_html
from atlas.ingest import Crawler, Frontier, normalize_url
from atlas.process import chunk_document

from .conftest import offline_settings


# ----------------------------------------------------------------- urls
def test_normalize_url_collapses_variants():
    base = "https://Example.COM:443/a//b/?utm_source=x&b=2&a=1#frag"
    assert normalize_url(base) == "https://example.com/a/b?a=1&b=2"
    assert normalize_url("/news/", "https://x.com/y") == "https://x.com/news"
    assert normalize_url("mailto:a@b.c") is None
    assert normalize_url("https://x.com") == "https://x.com/"


def test_frontier_validates_domain_dedupes_and_prioritises(fresh):
    src = Source("s", "S", "https://a.example/", "official", allowed_domains=["a.example"], exclude_patterns=[r"/private/"], priority=5)
    fresh.repo.upsert_source(src)
    f = Frontier(fresh.repo, fresh.clock)
    assert f.add("https://a.example/x", src, 1)
    assert not f.add("https://a.example/x/?utm_medium=y", src, 1)  # same URL after normalisation
    assert not f.add("https://evil.example/x", src, 1)  # other domain
    assert not f.add("https://a.example/private/y", src, 1)  # excluded
    assert not f.add("https://a.example/logo.png", src, 1)  # asset
    assert f.add("https://a.example/", src, 0, is_seed=True)
    assert next(u.url for u in f.next_batch(5)) == "https://a.example/"  # seeds first


# ---------------------------------------------------------------- dates
def test_extract_dates_precisions_and_iso_timestamps():
    ds = extract_dates("On March 14, 2023 OpenAI shipped; in Q3 2024 and 2025, also 2024-05-13T09:00:00Z.")
    assert [(d.iso, d.precision) for d in ds] == [("2023-03-14", "day"), ("2024-07-01", "quarter"), ("2025-01-01", "year"), ("2024-05-13", "day")]
    assert parse_date("2024-05-13T09:00:00Z") == "2024-05-13"
    assert extract_dates("GPT-4 has 175 billion in 2024B") == []  # bare years glued to units are not dates


# ----------------------------------------------------------- extraction
HTML = """<html><head><title>Big News | Site</title><meta property="article:published_time" content="2024-03-04T00:00:00Z">
<script>var x=1</script></head><body><nav><a href="/home">Home</a></nav><div class="cookie-banner">Accept cookies</div>
<article><h1>Big News</h1><p>Intro paragraph about <a href="/other?utm_x=1">other</a>.</p><h2>Details</h2>
<ul><li>First point</li><li><a href="/only-link">only a link</a></li></ul><pre>code here</pre>
<table><tr><td>A</td><td>B</td></tr></table></article><footer>© junk footer text</footer></body></html>"""


def test_parse_html_strips_boilerplate_and_keeps_structure():
    d = parse_html(HTML, "https://site.example/news")
    assert d.title == "Big News" and d.published_at == "2024-03-04"
    text = d.text
    assert "cookies" not in text and "junk footer" not in text and "var x" not in text and "only a link" not in text
    kinds = [(s.title, [b.kind for b in s.blocks]) for s in d.sections]
    assert kinds == [("Big News", ["p"]), ("Details", ["li", "code", "table"])]
    assert "https://site.example/other" in d.links and "https://site.example/only-link" in d.links


def test_content_hash_ignores_page_chrome_but_sees_content():
    a = parse_html(HTML, "https://site.example/n")
    b = parse_html(HTML.replace("Accept cookies", "Totally different banner").replace("var x=1", "var y=2"), "https://site.example/n")
    c = parse_html(HTML.replace("First point", "Changed point"), "https://site.example/n")
    assert content_hash(a) == content_hash(b) != content_hash(c)
    diff = diff_documents(a, c)
    assert diff.modified == ["Details"] and not diff.added and not diff.removed


def test_chunking_is_structure_aware_and_ids_are_stable():
    d = parse_html(HTML, "https://site.example/n")
    s = offline_settings()
    meta = dict(title=d.title, url=d.url, source_id="s", source_name="S", source_type="official", published_at=d.published_at, version=1, created_at="t")
    c1 = chunk_document(d, "doc1", s, meta)
    c2 = chunk_document(d, "doc1", s, meta)
    assert [c.chunk_id for c in c1] == [c.chunk_id for c in c2]
    assert {c.section_title for c in c1} == {"Big News", "Details"}  # never crosses a section
    d2 = parse_html(HTML.replace("First point", "Changed point"), "https://site.example/n")
    ids1, ids2 = {c.chunk_id for c in c1}, {c.chunk_id for c in chunk_document(d2, "doc1", s, meta)}
    assert ids1 & ids2 and ids1 != ids2  # intro chunk survives, edited section gets a new id


# -------------------------------------------------------------- crawler
async def test_crawler_retries_transient_errors_and_honours_etag(site):
    url = "https://openai.example/news/gpt-4"
    site.flaky[url] = 2
    c = Crawler(offline_settings(), site.transport(), sleep=lambda _: _noop())
    r = await c.fetch(url)
    assert r.status == 200 and r.attempts == 3 and r.html and r.etag
    again = await c.fetch(url, etag=r.etag)
    assert again.not_modified
    missing = await c.fetch("https://openai.example/nope")
    assert missing.status == 404 and missing.error
    await c.aclose()


async def _noop():
    return None


async def test_crawler_gives_up_after_retry_budget(site):
    url = "https://openai.example/news/gpt-4"
    site.flaky[url] = 99
    c = Crawler(offline_settings(), site.transport(), sleep=lambda _: _noop())
    r = await c.fetch(url)
    assert r.status == 0 and "503" in (r.error or "") and r.attempts == 3
    await c.aclose()


async def test_robots_policy_rules():
    def handler(req: httpx.Request) -> httpx.Response:
        host = req.url.host
        if req.url.path != "/robots.txt":
            return httpx.Response(200, text="<html></html>")
        return {"ok.example": httpx.Response(200, text="User-agent: *\nDisallow: /secret/\n"),
                "none.example": httpx.Response(404), "down.example": httpx.Response(503)}[host]

    c = Crawler(offline_settings(), httpx.MockTransport(handler))
    assert await c.robots.allowed("https://ok.example/public") and not await c.robots.allowed("https://ok.example/secret/x")
    assert await c.robots.allowed("https://none.example/anything")  # 4xx => allow
    assert not await c.robots.allowed("https://down.example/x")  # 5xx => disallow
    await c.aclose()


# ------------------------------------------------- end-to-end ingestion
def test_first_crawl_respects_robots_and_builds_all_layers(fresh, site):
    rep = fresh.load_fixtures(1, site)
    st = fresh.stats()
    assert rep.new_docs == 31 and rep.failed == 0 and rep.skipped_robots >= 17
    assert not any("/private/" in u or "/drafts/" in u for u in site.hits)  # disallowed paths are never requested
    assert st["chunks"] == st["vectors"] > 30 and st["relations"] > 40 and st["graph_edges"] == st["relations"]
    assert fresh.repo.url("https://nvidia.example/drafts/unreleased-roadmap").state == UrlState.SKIPPED


def test_second_crawl_is_incremental(fresh):
    r1 = fresh.load_fixtures(1)
    r2 = fresh.load_fixtures(2)
    assert r2.unchanged_docs >= 40  # byte-identical pages short-circuit via ETag / hash
    assert r2.changed_docs == 2 and r2.new_docs == 3 and r2.gone_docs == 1
    assert r2.embedded == r2.chunks_new < r1.embedded  # only new chunks are embedded
    assert r2.chunks_reused >= 1  # unchanged paragraphs of edited pages keep their chunk id
    hub = fresh.repo.document_by_url("https://meta-ai.example/models/llama")
    vs = fresh.repo.versions(hub["document_id"])
    assert [v["version"] for v in vs] == [1, 2] and vs[0]["valid_until"] == vs[1]["valid_from"]
    assert "added 1 section(s): Archive" in vs[1]["change_summary"]
    gone = fresh.repo.document_by_url("https://forum.example/t/best-open-model")
    assert gone["status"] == "gone" and not fresh.repo.doc_chunks(gone["document_id"])
    assert fresh.repo.stats()["relations_historic"] >= 1  # the forum thread's fact was retired, not erased


def _mini(html_body: str) -> httpx.MockTransport:
    page = f"<html><head><title>T</title></head><body><main><article><h1>T</h1>{html_body}</article></main></body></html>"
    return httpx.MockTransport(lambda r: httpx.Response(404) if r.url.path == "/robots.txt" else httpx.Response(200, text=page, headers={"content-type": "text/html"}))


def test_graph_reconciliation_retires_facts_no_longer_stated():
    a = Atlas(Settings(db_path=":memory:", llm_provider="none", backoff_base_s=0.0), llm=None)
    a.register_sources([Source("m", "Mini", "https://m.example/", "official", seed_urls=["https://m.example/"], allowed_domains=["m.example"], crawl_frequency_s=0)])
    p1 = "<p>OpenAI partners with Microsoft to run models on Azure. OpenAI released GPT-4 in March 2023. This sentence pads the page so it clears the stub threshold easily.</p>"
    a.ingest_sync(_mini(p1))
    live = {(r.src_id, r.rel, r.dst_id) for r in a.repo.relations(active_only=True)}
    assert ("openai", "PARTNERS_WITH", "microsoft") in live and ("openai", "RELEASED", "gpt-4") in live
    p2 = "<p>OpenAI released GPT-4 in March 2023. Anthropic released the Claude 3 family in March 2024. This sentence pads the page so it clears the stub threshold easily.</p>"
    rep = a.ingest_sync(_mini(p2))
    assert rep.changed_docs == 1 and rep.relations_retired == 1 and rep.relations_new == 1
    rels = {(r.src_id, r.rel, r.dst_id): r for r in a.repo.relations()}
    old = rels[("openai", "PARTNERS_WITH", "microsoft")]
    assert not old.active and old.valid_until  # closed interval, kept for history
    assert rels[("openai", "RELEASED", "gpt-4")].active and rels[("openai", "RELEASED", "gpt-4")].valid_from == "2023-03-01"
    assert ("anthropic", "RELEASED", "claude-3") in rels


def test_one_bad_page_does_not_kill_the_crawl(fresh, site):
    site.flaky["https://openai.example/news/gpt-4"] = 99
    rep = fresh.load_fixtures(1, site)
    assert rep.failed == 1 and rep.new_docs == 30
    assert fresh.repo.url("https://openai.example/news/gpt-4").state == UrlState.FAILED


@pytest.mark.parametrize("field", ["published_at", "title"])
def test_every_chunk_carries_provenance(corpus, field):
    for c in corpus.idx.chunks.values():
        assert c.url and c.source_name and c.source_type and getattr(c, field)
