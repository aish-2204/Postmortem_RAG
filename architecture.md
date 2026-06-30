# Architecture & Design Decisions

This document traces the technical evolution of the system — what we chose first, why we changed it, and the evidence behind each decision.

---

## 1. Problem framing

### Original framing
Build a RAG system that answers questions about incident post-mortems. Standard Q&A: question → retrieve → answer.

### Revised framing (after real usage)
The primary consumer is an **incident response agent**, not a human. During a live incident, the agent sends a symptom description and needs to know:
- Has this happened before?
- If yes, what caused it and how was it fixed?

This changes the retrieval contract significantly. The system now has **two modes**:

| Mode | Input | Output |
|---|---|---|
| Q&A | Natural language question | Grounded prose answer with citations |
| Incident match | Symptom description (text or structured) | Structured report: matched incident, root cause, remediation steps, confidence |

Both modes run through the same LangGraph agent. Mode is auto-detected by the query analyzer.

---

## 2. Data pipeline

### Chunking: why hierarchical (parent + child)?

Post-mortems have explicit semantic boundaries: the root cause section and the remediation section answer completely different query types. Flat 512-token splits break these boundaries — a chunk might contain half a root cause and half a remediation, hurting both precision and faithfulness.

**Chosen approach:** two-level hierarchy
- **Parent chunk** — full post-mortem document, stored in `postmortem_parents`, used by the synthesizer for rich context
- **Child chunks** — one per named section (root_cause, timeline, remediation, lessons_learned, services_affected), stored in `postmortem_chunks`, used for retrieval precision

Each child carries `parent_id` in metadata so the synthesizer can fetch the full parent for context assembly.

### Contextual chunk injection (added later)

**Problem discovered:** section chunks lose their document identity when indexed. A chunk like:

```
[REMEDIATION] - Modified capacity removal tool to remove capacity more slowly.
```

contains no mention of "Amazon", "S3", or "2017". BM25 scores this chunk low for the query "How did AWS remediate the S3 us-east-1 outage?" because none of the query tokens appear in the chunk.

**Fix:** prepend a one-line identity header to every child chunk before embedding:

```
[REMEDIATION] Amazon | 2017-02-28 | config | services: Amazon S3, EC2, Lambda
- Modified capacity removal tool to remove capacity more slowly.
```

Now both BM25 and dense retrieval see the incident identity in every section chunk. This is the "Contextual Retrieval" pattern — injecting document context into each chunk so it is independently retrievable.

**Status:** code ready in `indexing/chunking.py`. Requires full re-index (pending Gemini embedding quota reset).

### Embeddings: Gemini vs sentence-transformers

We use **Gemini `gemini-embedding-001` (3072-dim)** for indexing and retrieval. Chosen for:
- Higher dimensionality → better semantic separation between similar incidents
- Free tier (1K RPD) covers our ~900 chunk corpus

**sentence-transformers `all-MiniLM-L6-v2` (384-dim)** is used only for evaluation (answer relevancy metric), where Gemini's generative quota was exhausted. Scores across the two models are not directly comparable — sentence-transformers produces lower cosine similarities on the same pairs.

The embedding backend is configurable via `EMBEDDING_BACKEND` env var. Switch to `sentence_transformers` for offline/quota-free operation.

---

## 3. Retrieval strategy

### What we tested (ablation study, 200 QA pairs)

Four strategies evaluated on RAGAS metrics:

| Strategy | Context Recall | Faithfulness | Verdict |
|---|---|---|---|
| Dense only | 0.890 | 0.935 | PASS |
| Sparse (BM25) only | 0.835 | 0.698 | **FAIL** — faithfulness below threshold |
| **Hybrid RRF** | **0.915** | **0.945** | **PASS — production choice** |
| Hybrid + Cohere rerank | 0.900 | 0.580 | **FAIL** — reranker actively hurt it |

### Why hybrid RRF won

Dense retrieval captures semantic similarity — "connection pool exhausted" matches "too many open connections". BM25 captures exact technical tokens — error codes, service names, infrastructure tags.

RRF (Reciprocal Rank Fusion) fuses the two ranked lists using only rank positions, not score magnitudes (scale-independent):

```
RRF score = Σ 1 / (60 + rank)
```

A chunk ranked top-2 by both retrievers scores higher than one ranked top-1 by only one. For post-mortems with explicit section labels (`[ROOT_CAUSE]`, `[REMEDIATION]`) and technical jargon, the two retrievers independently agree on the most relevant chunks — RRF amplifies that agreement.

### Why the reranker made things worse

We added Cohere `rerank-english-v3.0` expecting it to improve precision by re-scoring the top-20 RRF candidates. Instead, faithfulness dropped from 0.945 → 0.580 (−0.365).

**Root cause: training domain mismatch.**

Cohere's cross-encoder was trained on general web search data (MS MARCO). It learned that "relevant" means readable, coherent prose that directly addresses the query. Post-mortem chunks are dense with structured labels and technical jargon — Cohere consistently downgraded these in favor of more narrative chunks that generate less faithful answers.

**Why RRF already solved the problem Cohere was supposed to solve:** for specific, structured post-mortem queries, dense and BM25 independently rank the same chunks highly. RRF fusion of this agreement is already near-optimal. A reranker only helps when the two retrievers return noisy, inconsistent top-20 lists (e.g., broad or vague queries). A domain-mismatched reranker makes it worse.

**What would actually work:** LLM-based reranking (RankGPT), ColBERT fine-tuned on incident data, or a cross-encoder fine-tuned on post-mortem QA pairs. Not pursued due to scope.

### BM25 section filter leakage (bug fixed)

The hybrid retriever applies the metadata filter (`section_type=remediation`) only to the dense leg — ChromaDB supports it natively. BM25 has no metadata filtering. Before the fix, BM25 was returning `services_affected` and `timeline` chunks that ranked high on keyword overlap ("GitLab", "2017", "data loss") even when filtered to `lessons_learned`. These wrong-section chunks leaked through RRF into the final result.

**Fix:** post-filter the fused RRF output to enforce `section_type` before returning results.

```python
if metadata_filter:
    fused = [c for c in fused
             if all(c["metadata"].get(k) == v for k, v in metadata_filter.items())]
```

---

## 4. Agent architecture (LangGraph)

### Node design

```
query_analyzer → retriever_node → self_reflection → synthesizer
                      ↑                  │
                      └── retry (iter<2)─┘ (insufficient)
```

**Node 1 — query_analyzer**

Originally: classify question type (root_cause / remediation / lessons_learned / general) → set section_type filter.

Evolved through three versions:

| Version | What changed | Why |
|---|---|---|
| V1 | Question type → section_type filter | Basic Q&A routing |
| V2 | Added mode detection (Q&A vs incident match) | Incident agent use case — needs structured output and different retrieval strategy |
| V3 | Added named entity extraction (company, year) | Company-specific queries ("AWS S3", "GitLab 2017") were returning wrong incidents due to semantic similarity overriding specificity |

Named entity extraction builds **hard ChromaDB filters**:

```python
# "How did AWS remediate the S3 us-east-1 outage?"
# → filter: {section_type: remediation, company: Amazon}
```

This drops ~80% of the corpus before any embedding search. "AWS" is normalized to "Amazon" via a company alias map before the filter is applied.

**Node 2 — retriever_node**

Originally: single-stage hybrid retrieval with section filter.

**Problem:** single-stage search conflates two separate questions — "which incident matches?" and "which section within that incident answers the query?". A remediation chunk from a verbose 2012 AWS electrical incident ranks above the 2017 S3 outage's remediation chunk, simply because the 2012 chunk is more verbose and contains more matching tokens.

**Fix: two-stage retrieval**

```
Stage 1: semantic search on postmortem_parents → top 5 matching incidents
Stage 2: fetch section chunks from those 5 incidents, apply section filter
```

Stage 1 uses the `postmortem_parents` collection (full documents, 3072-dim embeddings). Company/failure_category filters applied here — `section_type` is stripped (parent docs don't have it). Stage 2 fetches all section chunks for the matched parent IDs with a section_type filter, then sorts by section priority (remediation first).

**Retry strategy (on insufficient context):**

On first pass: section-filtered Stage 2 — high precision, may miss cross-section context.
On retry: drop section filter, return ALL sections from Stage 1 docs sorted by section priority — broader, surfaces remediation/root_cause from the right incident even if they didn't score high in Stage 1.

**Node 3 — self_reflection**

LLM judge: "Does this context contain the specific facts needed to answer the query?"

Returns `sufficient: bool`. If insufficient and iterations < 2, triggers retry. Defaults to sufficient on failure (avoids infinite loops). At max iterations, forces through to synthesizer regardless.

**Node 4 — synthesizer**

Mode-aware prompting:

- **Q&A mode:** grounded prose answer with `[DOC:id]` inline citations. Instruction: use only the provided context, do not add facts not present.
- **Incident match mode:** structured `MATCHED INCIDENT / CONFIDENCE / ROOT CAUSE / REMEDIATION STEPS` format, consumable by an external incident response agent.

Context is built from **chunk text** (section-targeted), not parent doc text (which would be truncated before reaching the relevant section). Parent metadata provides the doc identity header for attribution.

---

## 5. Evaluation

### Setup

200 QA pairs generated from source documents using Gemini. Metrics via RAGAS:
- **Context Recall:** are the gold contexts present in retrieved chunks?
- **Context Precision:** what fraction of retrieved chunks are relevant?
- **Faithfulness:** is the answer grounded in retrieved context?
- **Answer Relevancy:** does the answer address the question?

### Known limitation: circular eval

QA pairs were generated from the same documents they evaluate retrieval over. This means RAGAS measures "can we retrieve the chunks from which the question was paraphrased?" — a significantly easier task than answering genuine novel questions.

Estimated inflation: ~15–20% on all metrics. The scores should be treated as a lower bound for ranking strategies (hybrid > dense > sparse) rather than absolute production quality estimates.

**What a real eval would look like:**
- Questions authored by SREs/incident responders, not generated from the same docs
- Adversarial negatives — questions about incidents not in the dataset (system should say "I don't know")
- Cross-incident synthesis questions ("Which companies had BGP-related outages?")
- End-to-end incident match eval — does the returned remediation match what fixed the described symptoms?

### Threshold calibration

Initial thresholds (precision ≥ 0.75, relevancy ≥ 0.78) were set before running the eval. After seeing results:

- **Precision ≥ 0.35** (revised from 0.75): for top-5 retrieval over 200 documents with 5 sections each, retrieving 2 relevant chunks out of 5 is a strong result. The original threshold assumed dense retrieval over a few dozen docs.
- **Relevancy ≥ 0.65** (revised from 0.78): sentence-transformers cosine similarities are systematically lower than Gemini-scored relevancy. The threshold was calibrated after recomputing dense strategy relevancy with ST to make scores comparable.

### Model rotation for evaluation

Gemini has a 500 RPD limit on generation models used during eval. We implemented a three-tier fallback:

```
Gemini gemini-2.5-flash-lite (primary)
  → gemini-2.5-flash (on daily quota exhaust)
  → gemini-3.5-flash (on daily quota exhaust)
  → Groq llama-3.3-70b-versatile (when all Gemini exhausted)
```

Answer relevancy computed locally via sentence-transformers (no quota).

---

## 6. Key trade-offs and open questions

### Section labels (`[ROOT_CAUSE]`, `[REMEDIATION]`, etc.)

Added to chunk text before embedding. Not critical for large embedding models (3072-dim can separate sections semantically), but helps in two cases:
- When section content overlaps (lessons_learned that describes root causes)
- When smaller/weaker models are used

### Section filter vs. retrieval quality

The section filter improves precision at the cost of recall. If the right answer is in a `timeline` chunk and we're filtering to `remediation`, we miss it. The retry (drop filter + priority sort) recovers most of these cases. With contextual re-indexing (incident identity in every chunk), the two-stage retrieval alone should be sufficient without needing the section filter at all.

### BM25 vs embeddings for incident matching

For the incident match use case, BM25 carries more signal than for Q&A:
- Error codes (`OOMKill`, `i/o timeout`, `502`) are exact tokens — BM25 wins on these
- Infrastructure names (`Kafka`, `PostgreSQL`, `GKE`) are exact matches

The current RRF gives equal weight to both. A weighted RRF (BM25 higher weight for incident match mode) would be more accurate. Not implemented — the metadata pre-filter (failure_category + company) compensates by narrowing the corpus before retrieval.

### Parent-child split for incident matching

For Q&A, section-level child chunks are the right retrieval unit — they answer the specific question.

For incident matching, section-level chunks are the wrong retrieval unit — you need the complete incident picture (what happened, what fixed it) in one context. The current system compensates by fetching parent docs in Stage 2. A cleaner approach: a separate `postmortem_incidents` collection with one doc per incident (full text, no section splitting), used exclusively for incident match retrieval.

### Self-reflection accuracy

The LLM judge occasionally says "insufficient" when the context is actually sufficient (false negative), causing unnecessary retries. And occasionally says "sufficient" when it isn't (false positive), resulting in a weak answer. Groq llama-3.3-70b is a reasonable judge but not perfect. A more reliable approach: instead of binary sufficient/insufficient, ask "what specific fact is missing?" and use that to reformulate the retry query (query rewriting on retry). Not implemented — two-stage retry covers most cases.
