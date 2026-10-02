# Atlas — AI & Technology Intelligence Engine

Atlas continuously builds a **versioned knowledge base and temporal knowledge graph** of the AI/technology ecosystem and
answers research questions with **agentic RAG** — three agent architectures over the same retrieval stack, every answer
claim-checked against its evidence.

```
Web ─▶ crawl ─▶ extract ─▶ SAFETY SCAN ─▶ change-detect ─▶ version ─▶ chunk ─▶ entities·relations·dates ─▶ BM25 · vectors · graph
                               │ quarantine                                                                  │ communities (GraphRAG)
                               ▼                                                                             ▼ summary nodes
question ─▶ condense (memory) ─▶ cache ─▶ ┌ pipeline agent    understand → route → plan → retrieve → grade (CRAG) → refine ┐
                                          ├ autonomous agent  the LLM picks tools (function calling), reflects, self-checks ├▶ context ─▶ write ─▶ verify claims ─▶ repair
                                          └ multi-agent       planner → researchers → critic (conflicts, gaps) → writer      ┘
```

## Run it

```bash
cd ~/Desktop/Atlas
.venv/bin/python -m atlas demo --updates     # offline demo corpus: crawl round 1, then round 2 (edits, new pages, a removal, a poisoned page)
.venv/bin/python -m atlas ui                 # Streamlit app → http://localhost:8501   (atlas.db is already seeded in this checkout)
```

Groq is configured through `.env` (`ATLAS_GROQ_API_KEY`, `ATLAS_MODEL=openai/gpt-oss-120b`, copied from the Aegis project).
Without a key the deterministic pipeline mode still works end to end.

```bash
python -m atlas ask "…" --mode pipeline|autonomous|multi-agent [--no-llm] [--judge]
python -m atlas chat --mode multi-agent        # multi-turn: follow-ups are condensed, repeated questions hit the semantic cache
python -m atlas communities --reports          # GraphRAG communities       python -m atlas quarantine   # withheld chunks
python -m atlas ablate                         # remove one component at a time, re-run the gold set
python -m atlas eval [--llm --limit N] [--judge]   python -m atlas synth -n 5    # gold-set metrics / LLM-judge / synthetic questions
python -m atlas crawl --source hf-blog         # live web      python -m atlas search "…" --mode bm25|dense|graph|rrf|hybrid
```
Tests: `.venv/bin/python -m pytest -m "not live"` (127 tests; one more calls Groq when a key is set).

## RAG & agentic skills demonstrated

| Skill | Where | Evidence it works |
|---|---|---|
| Incremental ingestion, robots/ETag/retry, content-hash change detection, document versioning, section diffs | `ingest.py` `extract.py` `pipeline.py` | round-2 test: only changed chunks are re-embedded; versions have closed validity intervals |
| Structure-aware chunking with content-addressed ids | `process.py` | editing one section leaves every other chunk id, vector and BM25 entry untouched |
| Entity resolution, relation extraction, **temporal knowledge graph** with retirement of facts no longer stated | `process.py` `indexes.py` | `test_graph_reconciliation_retires_facts_no_longer_stated` |
| Hybrid retrieval: BM25 + dense + graph (multi-hop *bridge* scoring), weighted **RRF**, reranking, MMR/authority/freshness context building | `retrieve.py` `generate.py` | retrieval table below; `test_graph_retrieval_finds_multi_hop_bridge` |
| Query understanding: classification, time constraints, rewriting, expansion, **HyDE**, decomposition (clauses / per-entity / time windows), routing | `query.py` | routing decision is shown in the UI for every query |
| **Corrective RAG** — grade each chunk correct/ambiguous/incorrect (LLM adjudication), drop wrong ones, refine partial ones sentence-by-sentence | `corrective.py` | `test_grader_*`, `test_knowledge_refinement_*` |
| **Iterative retrieval** — assess sufficiency, then expand → relax filters → graph bridge → entity lookup → simplify | `agent.py` | `test_agent_refines_when_evidence_is_insufficient` |
| **Knowledge-gap detection** — topic never seen by the index ⇒ say so instead of returning near-miss noise | `agent.py` | multi-agent transcript flags `flibber, qwxzv` as never indexed |
| **GraphRAG** — Louvain communities (2 levels), reports built from verbatim source sentences, indexed as summary nodes and routed for "landscape" questions; document summary nodes | `graphrag.py` | `test_summary_nodes_are_opt_in_and_routed_for_broad_questions`; always-on summaries *hurt* in the ablation, so routing is selective |
| **Autonomous tool-calling agent** (Groq function calling): model-chosen tools, evidence handles `[E#]`, call de-duplication, budget, **self-reflection** before finalising, malformed-tool-call retry, graceful fallback | `autonomous.py` `llm.py` | scripted + live runs; `test_autonomous_*` |
| **Multi-agent supervisor**: planner, per-sub-question researchers with *different* strategies, critic loop (coverage + knowledge gaps), writer, agent transcript | `multiagent.py` | `test_multi_agent_*` |
| **Conflict detection & resolution** across sources (official vs news dates), publication-date vs event-date discipline | `multiagent.py` | resolves "July 23 (Meta AI) vs July 24 (TechWire)" by authority and cites both |
| **Grounded generation + claim verification**: synonym-aware coverage, numbers/dates/names must literally appear, multi-source claims, table rows, optional LLM judge, regenerate-then-repair | `generate.py` | caught a model using a news article's publication date as the release date |
| **Prompt-injection defence**: hidden-text removal, pattern scanner with quarantine, derived nodes re-scanned, output sanitiser (no exfil images/foreign links) | `safety.py` | a poisoned fixture page is quarantined; the test suite found and closed a leak through document summaries |
| **Conversational RAG**: follow-up condensation (LLM or rules), per-mode **semantic cache** invalidated by index version | `conversation.py` | `test_semantic_cache_*`, `test_followups_*` |
| **Evaluation science**: recall/MRR/nDCG, answer completeness/faithfulness/citation accuracy, abstention, **failure attribution** (retrieval vs ranking vs context vs generation), **ablation study**, RAGAS-style **LLM judge**, **synthetic question generation** with span verification | `evaluate.py` `evalset.py` | tables below |
| Observability: spans with live listeners, metrics, LLM token/latency accounting, rate-limit-aware client | `observe.py` `llm.py` | the UI streams spans while an agent runs |

## Three agent modes

| Mode | Control flow | Best for | Cost |
|---|---|---|---|
| **Pipeline** | fixed workflow with a refinement loop | speed, determinism, works offline | ~10–20 ms deterministic; 1–3 LLM calls with Groq |
| **Autonomous** | the LLM chooses `search / entity / related / landscape / read_document / check_claim`, reflects, then answers | open-ended exploration | 3–8 LLM calls; on a ~6k tokens/min tier Atlas waits between steps (a run took ~2 min) |
| **Multi-agent** | planner → researchers (one strategy per sub-question) → critic (follow-ups, conflicts) → writer | comparative / multi-part / contested questions | one research pass per sub-question + 1–2 LLM calls |

All modes end in the same verifier, so their faithfulness and citation accuracy are directly comparable.

## UI (Streamlit)

**Chat** (mode selector, memory, live agent progress, evidence with CRAG grades, claim table, plan, agent conversation / tool timeline,
conflicts, trace) · **Search lab** (5 retrievers side by side) · **Graph** (entity explorer, dated timeline) · **Communities**
(GraphRAG reports + subgraph) · **Corpus** (sources, frontier, versions, extracted facts) · **Safety** (quarantine + scanner playground) ·
**Ingest** (demo rounds, live crawl) · **Evaluation** (retrieval, answers, ablation, LLM judge, synthetic questions).

## Architecture (`src/atlas/`)

`ingest` · `extract` · `safety` · `process` · `ontology` · `pipeline` · `store` · `indexes` · `retrieve` · `query` · `corrective` ·
`graphrag` · `agent` (pipeline agent, 12-tool toolbox, `AskOptions`) · `autonomous` · `multiagent` · `conversation` · `generate`
(context builder, answer writer, claim verifier) · `evaluate` / `evalset` · `llm` · `observe` · `fixtures` (offline fixture web served via
`httpx.MockTransport`) · `sources` (live registry) · `cli` · `engine` (facade). SQLite is the system of record; BM25, vectors and the
graph are derived and rebuilt from it.

### Design decisions worth knowing
- **Every LLM step has a deterministic fallback** and fallbacks are visible (`llm_fallback` in the trace), so a rate limit degrades quality, never availability.
- **Untrusted by default**: crawled text can't instruct the system. Quarantined chunks never enter any index; derived artefacts (summaries, community reports) are built from clean chunks and re-scanned.
- **Facts are temporal and sourced**: `valid_from` comes from the sentence's own date, else the page date; confidence is scaled by source authority; a fact whose source stops stating it gets `valid_until` (it is kept, not erased — "retired" means *no longer stated*, not *false*).
- **Strict about invented specifics**: a claim with a number, date or name that the evidence lacks is unsupported, however fluent.
- **Refuse rather than guess**: no evidence naming the entities asked about, or a topic the index has never seen ⇒ abstain.

## Measured results (demo corpus: 36 pages, 24 answerable + 2 unanswerable gold questions)

Retrieval — default hashing embedder (offline) and `all-MiniLM-L6-v2` (`ATLAS_EMBEDDER=st:sentence-transformers/all-MiniLM-L6-v2`):

| mode | recall@10 | MRR | nDCG@10 | | recall@10 (MiniLM) | MRR (MiniLM) | nDCG@10 (MiniLM) |
|---|---|---|---|---|---|---|---|
| BM25 | 0.948 | 0.889 | 0.854 | | 0.948 | 0.889 | 0.854 |
| dense | 0.898 | 0.917 | 0.857 | | 0.912 | 0.917 | 0.845 |
| graph | 0.940 | 0.735 | 0.731 | | 0.940 | 0.735 | 0.731 |
| RRF | 0.948 | 0.826 | 0.833 | | 0.958 | 0.865 | 0.865 |
| fusion + rerank | 0.940 | 0.882 | 0.843 | | 0.944 | 0.861 | 0.839 |

On a corpus this small the retrievers are close; fusion's value here is robustness across question types, not a large headline gain.

Answers (deterministic mode): completeness 0.78, faithfulness 1.00, citation accuracy 1.00, hallucination 0.00, abstention 1.00;
failures: 19 fully correct, 6 generation, 1 ranking (the extractive writer is the weak stage; the Groq runs below were complete on the questions tried).

Ablation (remove one component, whole gold set, deterministic):

| config | evidence recall | completeness | faithfulness |
|---|---|---|---|
| full pipeline | 0.938 | 0.778 | 1.00 |
| − reranker | 0.892 | 0.715 | 1.00 |
| − graph retriever | 0.938 | 0.750 | 1.00 |
| − dense / − BM25 / − decomposition / − refinement / − CRAG / − MMR | 0.938 | 0.778–0.806 | 1.00 |
| + summary nodes always on | 0.927 | 0.736 | 1.00 |
| BM25 only, no extras | 0.933 | 0.729 | 1.00 |

Reranker, graph retrieval and *selective* summary routing show real contributions; differences ≤ 0.03 on 26 questions are noise.

Groq (`openai/gpt-oss-120b`), live: multi-agent answered a 3-part comparison with the source conflict surfaced (6 calls, 6 s);
autonomous run used `search → related → search…`, built a cited table and passed verification; LLM judge on 3 questions:
correctness 1.0 / completeness 1.0 / relevance 1.0 / groundedness 1.0 / **context precision 0.48** (about half the evidence units
shown to the writer went unused — a concrete tuning target).

## Configuration

`.env` / environment: `ATLAS_GROQ_API_KEY`, `ATLAS_GROQ_BASE_URL`, `ATLAS_MODEL`, `ATLAS_FAST_MODEL`, `ATLAS_DB_PATH`, `ATLAS_EMBEDDER`,
`ATLAS_RERANKER` (`feature` | `cross-encoder:<model>`), `ATLAS_GROQ_RPM_LIMIT` / `ATLAS_GROQ_TPM_LIMIT` (default 30 / 6000 — raise
them on a bigger tier to remove waits), `ATLAS_LLM_MAX_WAIT_S`.

## Known limits

- **Extraction recall on unfamiliar text is low.** Entities come from a seed ontology plus model-name discovery, relations from patterns. Crawling 11 real Hugging Face posts (robotics, datasets) indexed them correctly (titles, dates, 367 chunks) but extracted **0 facts**. An LLM extractor would fix this; it was left out because it multiplies token cost under a 6k-token/min tier.
- Only the Hugging Face blog source was verified against the live web; the other `LIVE_SOURCES` entries are unverified starting points, and the crawler is plain HTTP (no JavaScript rendering).
- The autonomous agent is stochastic and rate-limited; it can stop early (a reflection step mitigates this) and Groq occasionally returns malformed tool calls (retried, then answered in prose).
- The claim verifier is lexical-semantic, not an NLI model: it is strict about numbers/names but marks faithful paraphrase-heavy synthesis as "partial"; the optional LLM judge adjudicates those.
- Community detection and the ablation are exercised on a ~40-document corpus; behaviour at scale (memory-resident chunk table, one SQLite file) is not benchmarked.
- No Postgres / Qdrant / Neo4j / Redis by design: SQLite plus in-process indexes keep it runnable with no services.
