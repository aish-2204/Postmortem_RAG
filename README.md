# Postmortem RAG

A production-grade retrieval-augmented generation system over incident post-mortems, built to answer two kinds of questions:

1. **Incident match** — an automated monitoring agent sends current symptoms and gets back the closest past incident with its root cause and remediation steps
2. **Q&A** — a human asks a natural language question about any past incident or post-mortem

Built over ~200 real-world incident post-mortems (Cloudflare, GitHub, GitLab, Amazon, Facebook, Google, and more) sourced from [danluu/post-mortems](https://github.com/danluu/post-mortems).

---

## Architecture overview

```
Incident symptoms / Question
         │
         ▼
  query_analyzer           classify mode (Q&A vs incident match)
  + entity extraction      extract company, failure_category, services
         │
         ▼
  two-stage retriever
    Stage 1: semantic search on postmortem_parents → top matching incidents
    Stage 2: section chunks from matched incidents (section-filtered for Q&A)
         │
         ▼
  self_reflection          LLM judges if context is sufficient
    │        │
    │   insufficient + iter < 2
    │        └──► retry (drop section filter, all sections from Stage 1 docs)
    │
    ▼ sufficient
  synthesizer
    Q&A mode:       grounded prose answer with [DOC:id] citations
    incident mode:  structured report — matched incident, root cause,
                    remediation steps, confidence
         │
         ▼
    Streamlit UI  or  external agent API call
```

**Retrieval:** Hybrid RRF (dense + BM25) — chosen over dense-only, sparse-only, and reranked hybrid after a 4-strategy ablation study on 200 QA pairs.

**LLM:** Groq `llama-3.3-70b-versatile` (14,400 RPD free tier) for all agent nodes.

**Embeddings:** Google `gemini-embedding-001` (3072-dim) with sentence-transformers fallback for evaluation.

---

## Ablation results (200 QA pairs)

| Strategy      | Context Recall | Context Precision | Faithfulness | Answer Relevancy | CI Gate |
|---|---|---|---|---|---|
| dense         | 0.8900 ✓ | 0.3920 ✓ | 0.9350 ✓ | 0.7288 ✓ | PASS |
| sparse        | 0.8350 ✓ | 0.3180 ✗ | 0.6975 ✗ | 0.6510 ✓ | **FAIL** |
| **hybrid**    | **0.9150 ✓** | **0.3970 ✓** | **0.9450 ✓** | **0.7278 ✓** | **PASS** |
| hybrid+rerank | 0.9000 ✓ | 0.3820 ✓ | 0.5800 ✗ | 0.5987 ✗ | **FAIL** |

Hybrid RRF wins on every metric. Reranker (Cohere) hurt faithfulness by −0.37 due to training domain mismatch (see [architecture.md](architecture.md)).

---

## Setup

### Prerequisites

- Python 3.11+
- Docker (for ChromaDB)
- API keys: Gemini (free), Groq (free), Cohere (optional, unused after ablation)

### Install

```bash
pip install -e ".[dev]"
```

### Environment

Copy `.env.example` to `.env` and fill in:

```
GEMINI_API_KEY=...          # aistudio.google.com — free
GROQ_API_KEY=...            # console.groq.com — free 14,400 RPD
EMBEDDING_BACKEND=gemini    # or sentence_transformers (local, no quota)
```

### Start ChromaDB

```bash
docker compose up -d
```

### Index documents

```bash
python -m indexing.pipeline          # incremental — skips already-indexed docs
python -m indexing.pipeline --full   # force re-index everything
```

### Run the UI

```bash
streamlit run ui/app.py
```

---

## Usage

### Streamlit UI

Open `http://localhost:8501`. Two tabs:

- **Incident Match** — paste current symptoms, get structured report
- **Q&A** — ask a natural language question

### Python API

**Q&A:**
```python
from agents.graph import run
result = run("How did AWS remediate the S3 us-east-1 outage?")
print(result["answer"])
print(result["sources"])   # ["amazon_2017_02_28"]
```

**Incident match (for external agent integration):**
```python
from agents.graph import run_incident_match

result = run_incident_match(
    "Kafka consumer lag spike, OOMKill on K8s pods, 502 errors on API gateway"
)
print(result["answer"])
# MATCHED INCIDENT: ... (company, date)
# CONFIDENCE: high/medium/low
# ROOT CAUSE: ...
# REMEDIATION STEPS:
# - ...
```

---

## Project structure

```
indexing/
  chunking.py       hierarchical chunker (parent doc + section children)
  embedder.py       Gemini / sentence-transformers embedding with rate-limit handling
  chroma_store.py   ChromaDB upsert + parent fetch
  bm25_index.py     BM25Okapi index with custom tokenizer (preserves error codes)
  pipeline.py       end-to-end incremental indexing pipeline

retrieval/
  dense_retriever.py    ChromaDB cosine ANN with metadata filter
  sparse_retriever.py   BM25 search → text + metadata from ChromaDB
  hybrid_retriever.py   RRF fusion: dense + BM25, post-filter for section_type
  reranker.py           Cohere reranker (not used in production — see ablation)

agents/
  state.py              AgentState TypedDict — shared schema across all nodes
  query_analyzer.py     Node 1: mode detection, entity extraction, filter building
  retriever_node.py     Node 2: two-stage retrieval (doc-level → section-level)
  self_reflection.py    Node 3: LLM sufficiency judge, controls retry loop
  synthesizer.py        Node 4: mode-aware answer generation
  graph.py              LangGraph StateGraph, run() and run_incident_match() entry points

evaluation/
  synthetic_qa_generator.py   generate 200 QA pairs from source docs
  ragas_evaluator.py          RAGAS metrics with Groq fallback + checkpoint/resume
  ablation_study.py           runs all 4 strategies, writes ablation_results.md
  results/                    ablation_results.md, qa_pairs.json, per-strategy eval JSON

ui/
  app.py    Streamlit UI — two tabs, agent trace expander, source chips
```

---

## Running evaluation

```bash
# Generate QA pairs (requires Gemini)
python -m evaluation.synthetic_qa_generator

# Run full ablation (uses Groq as fallback if Gemini quota exhausted)
python -m evaluation.ablation_study
```

Evaluation checkpoints every 10 pairs — safe to interrupt and resume.
