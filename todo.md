# Postmortem RAG — Build Tracker

Work through these blocks in order. Test each before moving to the next.

---

## DONE

### Scaffolding
- [x] `pyproject.toml` — all dependencies declared
- [x] `docker-compose.yml` — ChromaDB service
- [x] `.env.example` — all env vars documented
- [x] `.gitignore`

### Week 1 — Data Pipeline

**Ingestion**
- [x] `data/ingestion/github_scraper.py` — clone danluu/post-mortems, parse markdown, fetch raw HTML → text
- [x] `data/ingestion/blog_scraper.py` — targeted fetches for known Cloudflare/GitHub/Stripe posts
- [x] `data/ingestion/structured_extractor.py` — LLM (gpt-4o-mini) → typed JSON schema per post-mortem
- [x] `data/ingestion/runbook_loader.py` — 5 synthetic runbooks (Kafka lag, Postgres, OOMKill, Redis, DNS)

**Indexing**
- [x] `indexing/chunking.py` — hierarchical chunker: parent (full doc) + children (per section)
- [x] `indexing/embedder.py` — batched OpenAI embeddings with rate-limit retry + cost tracking
- [x] `indexing/chroma_store.py` — ChromaDB collections (chunks + parents), metadata upsert, parent fetch
- [x] `indexing/bm25_index.py` — BM25Okapi index, custom tokenizer (preserves error codes), pickle persist
- [x] `indexing/pipeline.py` — idempotent end-to-end: raw → embed → upsert ChromaDB + build BM25

**Retrieval (partial)**
- [x] `retrieval/dense_retriever.py` — ChromaDB cosine ANN with metadata pre-filter
- [x] `retrieval/sparse_retriever.py` — BM25 search → fetch text+metadata from ChromaDB

---

## TODO

### Week 1 — Finish & Test
- [ ] **FIRST: set up `.env`** — copy `.env.example`, add:
  - `OPENAI_API_KEY` — needed for embeddings (text-embedding-3-small)
  - `GEMINI_API_KEY` — free key from aistudio.google.com, used for extraction
  - `ANTHROPIC_API_KEY` — for synthesis (Week 3)
- [ ] **Install deps** — `pip install -e ".[dev]"`
- [ ] **Run `runbook_loader.py`** — loads 5 runbooks to `data/processed/runbooks/`, no API calls needed
- [ ] **Run `github_scraper.py`** — scrape danluu/post-mortems (be patient, ~200 URLs, ~5 min)
- [ ] **Run `structured_extractor.py`** — LLM extraction over raw files (costs ~$0.05 total)
- [ ] **Run `indexing/pipeline.py`** — embeds + indexes everything into ChromaDB + BM25
- [ ] **Manual smoke test** — query ChromaDB and BM25 directly with 3–5 test queries, verify results look sensible
- [ ] Add `data/processed/schema.md` — document the JSON extraction schema (reference artifact)
- [ ] Add `__init__.py` files to each package directory

### Week 2 — Hybrid Retrieval + Reranker + Eval
- [ ] `retrieval/hybrid_retriever.py` — RRF fusion of dense + sparse (implement from scratch, ~30 lines)
- [ ] `retrieval/reranker.py` — Cohere Rerank API wrapper, graceful fallback to score-based ranking
- [ ] `evaluation/synthetic_qa_generator.py` — generate 200 QA pairs from post-mortems (5 per doc)
- [ ] `evaluation/ragas_evaluator.py` — run RAGAS metrics on a retrieval strategy
- [ ] `evaluation/ablation_study.py` — run all 4 strategies, write results to `evaluation/results/ablation_results.md`
- [ ] `tests/test_retrieval.py` — unit tests for each retriever
- [ ] Commit `evaluation/results/qa_pairs.json` and `ablation_results.md`

### Week 3 — LangGraph Agent + LangSmith
- [ ] `agents/state.py` — `AgentState` TypedDict (all 15 fields)
- [ ] `agents/query_analyzer.py` — Node 1: keyword extraction, failure category classification, strategy selection
- [ ] `agents/retriever_node.py` — Node 2: calls hybrid retriever + reranker, attaches scores
- [ ] `agents/self_reflection.py` — Node 3: gap detection prompt + conditional retry (max 2)
- [ ] `agents/synthesizer.py` — Node 4: grounded answer + citations as typed list
- [ ] `agents/graph.py` — LangGraph StateGraph assembly with conditional edge
- [ ] Add LangSmith tags: `incident_type`, `retrieval_strategy`, `retry_count` per run
- [ ] `tests/test_agents.py` — unit test each node with mocked state
- [ ] `tests/test_eval_gate.py` — CI RAGAS threshold gates (faithfulness ≥ 0.82, etc.)
- [ ] Run 20 real incident descriptions end-to-end, log failures

### Week 5 — Observability Stack
- [ ] Integrate OpenTelemetry tracing — instrument agent nodes (query_analyzer, retriever, reflection, synthesizer)
- [ ] Add Prometheus metrics — retrieval latency, reflection retry rate, answer length histogram
- [ ] Wire LangSmith with a real API key — per-run tags: `question_type`, `iterations`, `sufficient`
- [ ] Add structured JSON logging to each agent node (query, retrieved doc IDs, reflection verdict, latency)
- [ ] `ui/app.py` — expose agent trace sidebar (node timings, retry count, question_type)
- [ ] Grafana dashboard (optional) — if self-hosting, visualize Prometheus metrics
- [ ] Alerting rule — faithfulness drop below 0.80 triggers Slack/email

### Week 4 — UI + CI/CD + README + Deploy
- [ ] `ui/app.py` — Streamlit UI (incident input, root cause output, agent trace, confidence score)
- [ ] `.github/workflows/ci.yml` — lint → unit tests → integration test → RAGAS gate
- [ ] `README.md` — architecture diagram (Mermaid), ablation table, LangSmith screenshot, demo GIF, design decisions
- [ ] Run final RAGAS eval on full 200 QA pairs, update `evaluation/results/`
- [ ] Deploy to Hugging Face Spaces

---

## Next Step Right Now

```bash
# 1. Copy env file
cp .env.example .env
# edit .env and add your OPENAI_API_KEY

# 2. Install
pip install -e ".[dev]"

# 3. First smoke test (no API calls)
python -m data.ingestion.runbook_loader
# → should write 5 files to data/processed/runbooks/

# 4. Then scrape
python -m data.ingestion.github_scraper
```
