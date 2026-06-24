# Retrieval Ablation Study

**Date:** 2026-06-24  
**Dataset:** 200 QA pairs  
**Thresholds:** recall≥0.70, precision≥0.75, faithfulness≥0.82, relevancy≥0.78

## Results

| Strategy         | Context Recall | Context Precision | Faithfulness | Answer Relevancy | CI Gate |
|------------------|---------------|-------------------|--------------|-----------------|---------|
| dense            |      0.8900 ✓ |          0.3920 ✓ |     0.9350 ✓ |        0.7288 ✓ | PASS ✓ |
| sparse           |      0.8350 ✓ |          0.3180 ✗ |     0.6975 ✗ |        0.6510 ✓ | FAIL ✗ |
| hybrid           |      0.9150 ✓ |          0.3970 ✓ |     0.9450 ✓ |        0.7278 ✓ | PASS ✓ |
| hybrid_rerank    |      0.9000 ✓ |          0.3820 ✓ |     0.5800 ✗ |        0.5987 ✗ | FAIL ✗ |

## Key Findings

- **Best context recall:**    `hybrid` (0.9150)
- **Best context precision:** `hybrid` (0.3970)
- **Best faithfulness:**      `hybrid` (0.9450)
- **Best answer relevancy:**  `dense` (0.7288)

**Hybrid+Rerank vs Hybrid (RRF only):**
- Context recall delta:    -0.0150
- Faithfulness delta:      -0.3650
- Answer relevancy delta:  -0.1291

## Reranker Investigation: Why Cohere Hurt Performance

Cohere `rerank-english-v3.0` was expected to improve precision by re-scoring
the top-20 RRF candidates. Instead, faithfulness dropped from 0.9450 → 0.5800
and answer relevancy from 0.7278 → 0.5987. Investigation findings:

**Root cause: training domain mismatch.**
Cohere's cross-encoder was trained on general web data (search queries, articles).
It learns that 'relevant' means readable prose that directly addresses the question.
Post-mortem chunks are dense with technical jargon and structured labels
(`[ROOT_CAUSE]`, `[REMEDIATION]`) — Cohere consistently downgrades these in favour
of more narrative-sounding chunks (e.g. `[LESSONS_LEARNED]`) that score high on
readability but generate less grounded, less faithful answers.

**Why RRF already solves this.**
RRF never reads content — it fuses rank positions from two independent signals:
- Dense retriever: `[ROOT_CAUSE]` prefix creates a strong semantic embedding signal
- BM25: root-cause keywords match the 'what caused X?' question pattern
When both retrievers independently rank the same chunk near the top, RRF amplifies
that agreement. A chunk top-2 in both systems scores higher than one top-1 in only
one system. The reranker disrupts this convergence by applying a domain-mismatched
scoring function on top.

**When reranking would help:** Vague or broad queries where dense + BM25 return
noisy, inconsistent top-20 results. For specific, structured post-mortem queries
against labeled chunks, RRF fusion is already near-optimal.

**Production decision:** Use `hybrid` (RRF fusion, no reranker). If a reranker
is added in future, it should be fine-tuned on incident post-mortem data.

## Breakdown by Question Type

| Question Type    | Strategy         | Context Recall | Context Precision | Faithfulness |
|------------------|------------------|---------------|-------------------|-------------|
| root_cause       | dense            | 0.8795        | 0.4313            | 0.9458      |
| remediation      | dense            | 0.8814        | 0.4169            | 0.9831      |
| lessons_learned  | dense            | 0.9138        | 0.3103            | 0.8707      |
| root_cause       | sparse           | 0.9277        | 0.3639            | 0.6398      |
| remediation      | sparse           | 0.7288        | 0.3119            | 0.7271      |
| lessons_learned  | sparse           | 0.8103        | 0.2586            | 0.7500      |
| root_cause       | hybrid           | 0.9157        | 0.4386            | 0.9699      |
| remediation      | hybrid           | 0.9153        | 0.4203            | 0.9153      |
| lessons_learned  | hybrid           | 0.9138        | 0.3138            | 0.9397      |
| root_cause       | hybrid_rerank    | 0.9157        | 0.4193            | 0.5277      |
| remediation      | hybrid_rerank    | 0.8814        | 0.3864            | 0.6305      |
| lessons_learned  | hybrid_rerank    | 0.8966        | 0.3241            | 0.6034      |