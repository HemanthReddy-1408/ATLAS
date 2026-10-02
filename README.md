# Atlas — AI & Technology Intelligence Engine

Atlas continuously builds a **versioned knowledge base and temporal knowledge graph** of the AI/technology ecosystem and
answers research questions with an **agentic RAG** loop whose every claim is checked against evidence.

```
Web ─▶ crawl ─▶ extract ─▶ change-detect ─▶ version ─▶ chunk ─▶ entities · relations · dates ─▶ BM25 · vectors · graph
                                                                                                        │
question ─▶ understand ─▶ route ─▶ plan ─▶ retrieve (BM25+dense+graph) ─▶ RRF ─▶ rerank ─▶ assess ─▶ refine ─┐
                                   ▲                                                                       │
             answer ◀─ verify claims ◀─ generate (Groq) ◀─ context builder (dedupe·authority·freshness·MMR) ◀─┘
```

## Run it

```bash
cd ~/Desktop/Atlas
.venv/bin/python -m atlas demo --updates     # load the offline demo corpus (crawl round 1 + incremental round 2)
.venv/bin/python -m atlas ui                 # Streamlit app → http://localhost:8501
```

`atlas.db` is already seeded in this checkout. Groq is configured through `.env` (copied from the Aegis project:
`ATLAS_GROQ_API_KEY`, `ATLAS_MODEL=openai/gpt-oss-120b`); without a key everything still works in a deterministic extractive mode.

Other commands: `ask "…" [--no-llm] [--judge] [--debug]` · `search "…" --mode bm25|dense|graph|rrf|hybrid` · `entity NAME` ·
`eval [--llm --limit N]` · `crawl [--source ID]` (live web) · `stats`. Tests: `.venv/bin/python -m pytest -m "not live"`.

## The UI (Streamlit)

| Tab | What you see |
|---|---|
| **Ask** | Answer with `[E#]` citations · faithfulness / citation accuracy · evidence cards · per-claim verdicts · query understanding, routing decision and plan · agent trace with refinement steps · what each retriever returned and what context-building dropped |
| **Search lab** | BM25 / dense / graph / RRF / fusion+rerank side by side for any query |
| **Knowledge graph** | Entity card, neighbourhood diagram (dashed = fact no longer stated), per-entity timeline |
| **Corpus** | Source registry, URL frontier states, documents, version history with change summaries, chunks, extracted facts |
| **Ingest** | Load demo round 1 / apply round 2 and see exactly what was re-embedded; live crawl of the real source registry |
| **Evaluation** | Retrieval metrics per retriever, answer metrics, failure attribution by pipeline stage |

## Architecture (`src/atlas/`)

| Module | Responsibility |
|---|---|
| `ingest.py` | URL normalisation, `robots.txt` (RFC 9309), SQLite-backed frontier, async crawler (retry/backoff, `Retry-After`, ETag / `If-Modified-Since`) |
| `extract.py` | HTML → structured document (boilerplate removal, headings, lists, tables, code, JSON-LD / meta dates), date parsing, content hashing, section-level diffs |
| `process.py` | Sentence splitting, structure-aware chunking with content-addressed ids, entity resolution (aliases, fuzzy match with digit guard), entity extraction, relation extraction, temporal facts |
| `pipeline.py` | The update loop. Unchanged page → stop. Changed page → new version, diff, only *new* chunks embedded, graph reconciliation (facts no longer stated get `valid_until`) |
| `store.py` | SQLite system of record (sources, URLs, raw HTML, documents, versions, chunks, vectors, entities, relations) |
| `indexes.py` | BM25 (incremental, field-boosted), vector index, temporal knowledge graph, embedders |
| `retrieve.py` | Metadata filters, BM25 / dense / graph retrievers (graph does multi-hop "bridge" scoring), weighted RRF, reranker, hybrid orchestration |
| `query.py` | Entity linking, time constraints, 7-way classification, rewrite / expansion / HyDE / decomposition (time windows, per-entity, clause), router |
| `agent.py` | 12-tool toolbox with JSON schemas; plan → retrieve → assess → refine loop (expand → relax filters → graph bridge → entity lookup → simplify) |
| `generate.py` | Context builder, answer generation (Groq or extractive), claim extraction and verification, optional LLM judge, repair |
| `evaluate.py`, `evalset.py` | Recall@k, precision@k, MRR, nDCG, completeness, faithfulness, citation accuracy, hallucination rate, abstention, **failure attribution** |
| `llm.py` | Groq client (OpenAI-compatible): TPM/RPM limiter, retries, response cache, `<think>` stripping. Every LLM step has a non-LLM fallback |
| `fixtures.py` | Offline fixture web (34 pages on 17 `.example` hosts) served through `httpx.MockTransport`, incl. a second round of edits/new/removed pages |
| `observe.py` | Per-request span traces and process metrics |

### Design decisions worth knowing

- **Chunk ids are content-addressed**, so editing one section of a page leaves every other chunk's id, embedding and BM25 entry untouched.
- **Change detection hashes extracted content**, not raw HTML — ads, nonces and scripts never create a version.
- **Facts are temporal.** Each relation has `valid_from` (from the sentence's own date, else the page date) and `valid_until` when its source stops stating it. History is kept, not overwritten. Fact confidence is scaled by source authority.
- **Routing is deterministic and inspectable** (query type → retriever weights, filters, HyDE, decomposition) and shown in the UI. The agent overrides it only through logged refinement actions.
- **The verifier is strict about invented specifics**: a claim is supported only if its content words are covered (synonym-aware) *and* every number, date and proper name literally appears in the evidence. It caught a real model error during development (a news article's publication date used as the release date). The answer pipeline regenerates once with the verifier's feedback, then drops lines that are still unsupported.
- **The system refuses rather than guesses**: if no retrieved evidence mentions any entity the question names, it abstains.

## Measured results (demo corpus, 24 answerable + 2 unanswerable gold questions)

Retrieval — hashing embedder (default, offline) vs. `all-MiniLM-L6-v2` (`ATLAS_EMBEDDER=st:sentence-transformers/all-MiniLM-L6-v2`):

| mode | recall@10 | MRR | nDCG@10 | | recall@10 (MiniLM) | MRR (MiniLM) | nDCG@10 (MiniLM) |
|---|---|---|---|---|---|---|---|
| BM25 | 0.948 | 0.889 | 0.886 | | 0.948 | 0.889 | 0.886 |
| dense | 0.898 | 0.917 | 0.864 | | 0.912 | 0.938 | 0.864 |
| graph | 0.946 | 0.728 | 0.722 | | 0.946 | 0.728 | 0.722 |
| RRF | 0.954 | 0.833 | 0.839 | | 0.954 | 0.885 | 0.872 |
| **fusion + rerank** | **0.954** | **0.931** | **0.878** | | **0.958** | **0.938** | **0.900** |

Answers: extractive mode — completeness 0.81, faithfulness 1.00, hallucination rate 0.00, abstention accuracy 1.00. Groq
(`openai/gpt-oss-120b`) on the first four gold questions: 4/4 complete, faithfulness 1.00, ~0.7 s median. On the flagship
multi-part question Groq's answer scored faithfulness 0.71 / citation accuracy 0.94 / 0 unsupported claims — the remainder are
paraphrased synthesis sentences the lexical verifier marks "partial". Treat these as a regression harness on a small corpus,
not as a benchmark.

## Configuration

`.env` (see `config.py`): `ATLAS_GROQ_API_KEY`, `ATLAS_GROQ_BASE_URL`, `ATLAS_MODEL`, `ATLAS_FAST_MODEL`, `ATLAS_DB_PATH`,
`ATLAS_EMBEDDER`, `ATLAS_RERANKER` (`feature` or `cross-encoder:<model>`), `ATLAS_GROQ_RPM_LIMIT` / `ATLAS_GROQ_TPM_LIMIT`
(the client sleeps to stay under your tier's limits; default 30 RPM / 6000 TPM, so expect a wait between consecutive Groq answers).

## Known limits

- The default embedder is a deterministic hashing + synonym-concept baseline, not a neural model; MiniLM and a cross-encoder are supported but optional.
- Entity and relation extraction are pattern/gazetteer based (precise, limited recall). Unknown model versions (e.g. "Llama 3.2") are discovered as provisional entities; unknown companies are not.
- `LIVE_SOURCES` in `sources.py` are starting points; the crawler is plain HTTP (no JavaScript rendering) and I have not verified those sites' current layouts.
- "Fact retired" means *its source page stopped stating it*, not that the fact became false.
- The whole chunk table is held in memory; fine for tens of thousands of chunks, not millions.
- No Postgres / Qdrant / Neo4j / Redis: SQLite plus in-process indexes keep the project runnable with no services.
