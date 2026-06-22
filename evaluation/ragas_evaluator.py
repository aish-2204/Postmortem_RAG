"""
RAGAS-compatible evaluator — implemented without the ragas library.

RAGAS 0.4.3 has a broken import (langchain_community.chat_models.vertexai was
removed in langchain-community 0.4.x). We compute the same 4 metrics directly.

Metrics computed:
  context_recall    — did we retrieve the chunk that SHOULD answer this question?
                      (pure Python — compares retrieved IDs vs relevant_chunk_ids)
  context_precision — of retrieved chunks, how many are from the correct source doc?
                      (pure Python approximation)
  faithfulness      — does the generated answer only use info from retrieved context?
                      (LLM judge via Gemini)
  answer_relevancy  — does the generated answer address the question?
                      (embedding cosine similarity between question and answer)

Usage:
    python -m evaluation.ragas_evaluator                          # hybrid+rerank strategy
    python -m evaluation.ragas_evaluator --strategy dense         # dense only
    python -m evaluation.ragas_evaluator --strategy sparse        # sparse only
    python -m evaluation.ragas_evaluator --strategy hybrid        # hybrid, no rerank
    python -m evaluation.ragas_evaluator --strategy hybrid_rerank # hybrid + rerank (default)
    python -m evaluation.ragas_evaluator --limit 20               # quick test on 20 pairs
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
QA_PAIRS_PATH = ROOT / "evaluation" / "results" / "qa_pairs.json"
RESULTS_DIR   = ROOT / "evaluation" / "results"

_GEN_MODEL = os.getenv("EXTRACTION_MODEL", "models/gemini-3.1-flash-lite")
_RATE_LIMIT_WAIT = 60
_BETWEEN_CALL_SLEEP = 4

Strategy = Literal["dense", "sparse", "hybrid", "hybrid_rerank"]


# ── Retrieval strategies ──────────────────────────────────────────────────────

def _get_retriever(strategy: Strategy):
    from retrieval.dense_retriever import DenseRetriever
    from retrieval.hybrid_retriever import HybridRetriever
    from retrieval.reranker import Reranker
    from retrieval.sparse_retriever import SparseRetriever

    if strategy == "dense":
        return DenseRetriever()
    if strategy == "sparse":
        return SparseRetriever()
    if strategy == "hybrid":
        return HybridRetriever()
    if strategy == "hybrid_rerank":
        return HybridRetriever(), Reranker()
    raise ValueError(f"Unknown strategy: {strategy}")


def _retrieve(strategy: Strategy, query: str, top_k: int = 5) -> list[dict]:
    """Run retrieval for a given strategy, return top_k chunks."""
    if strategy == "hybrid_rerank":
        retriever, reranker = _get_retriever(strategy)
        candidates = retriever.retrieve(query, top_k=20)
        return reranker.rerank(query, candidates, top_n=top_k)

    retriever = _get_retriever(strategy)
    if strategy == "sparse":
        return retriever.retrieve(query, top_k=top_k)
    return retriever.retrieve(query, top_k=top_k)


# ── Answer generation ─────────────────────────────────────────────────────────

_ANSWER_PROMPT = """\
You are an SRE assistant. Answer the question using ONLY the provided incident post-mortem context.
Be concise (2-4 sentences). If the context does not contain enough information, say so.

Question: {question}

Context:
{context}

Answer:"""

def _generate_answer(client: genai.Client, question: str, contexts: list[str], max_retries: int = 4) -> str:
    context_text = "\n\n---\n\n".join(contexts[:3])
    prompt = _ANSWER_PROMPT.format(question=question, context=context_text)
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=_GEN_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=200),
            )
            return resp.text.strip()
        except genai_errors.ClientError as e:
            if e.code == 429:
                time.sleep(_RATE_LIMIT_WAIT * (attempt + 1))
            else:
                return ""
        except genai_errors.ServerError:
            time.sleep(15 * (attempt + 1))
        except Exception:
            return ""
    return ""


# ── Metric computation ────────────────────────────────────────────────────────

def compute_context_recall(
    relevant_chunk_ids: list[str],
    retrieved_chunks: list[dict],
) -> float:
    """
    Fraction of relevant chunks that were actually retrieved.
    1.0 = perfect recall, 0.0 = missed all relevant chunks.
    """
    if not relevant_chunk_ids:
        return 1.0
    retrieved_ids = {c["chunk_id"] for c in retrieved_chunks}
    hits = sum(1 for rid in relevant_chunk_ids if rid in retrieved_ids)
    return hits / len(relevant_chunk_ids)


def compute_context_precision(
    source_doc_id: str,
    retrieved_chunks: list[dict],
) -> float:
    """
    Fraction of retrieved chunks from the correct source document.
    Approximation: treats same parent_id as relevant.
    1.0 = all retrieved chunks are from the correct doc.
    """
    if not retrieved_chunks:
        return 0.0
    correct = sum(
        1 for c in retrieved_chunks
        if c["metadata"].get("parent_id") == source_doc_id
    )
    return correct / len(retrieved_chunks)


_FAITHFULNESS_PROMPT = """\
Given the context and the answer, determine if every factual claim in the answer
is directly supported by the context. Reply with only a number between 0 and 1,
where 1 = fully supported, 0 = not supported at all.

Context:
{context}

Answer: {answer}

Score (0.0 to 1.0):"""

def compute_faithfulness(
    client: genai.Client,
    contexts: list[str],
    answer: str,
    max_retries: int = 3,
) -> float:
    if not answer:
        return 0.0
    context_text = "\n\n---\n\n".join(contexts[:3])
    prompt = _FAITHFULNESS_PROMPT.format(context=context_text, answer=answer)
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=_GEN_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=5),
            )
            score = float(resp.text.strip().split()[0])
            return max(0.0, min(1.0, score))
        except (ValueError, IndexError):
            return 0.5
        except genai_errors.ClientError as e:
            if e.code == 429:
                time.sleep(_RATE_LIMIT_WAIT * (attempt + 1))
        except genai_errors.ServerError:
            time.sleep(15 * (attempt + 1))
        except Exception:
            return 0.5
    return 0.5


def compute_answer_relevancy(
    question: str,
    answer: str,
) -> float:
    """
    Cosine similarity between question embedding and answer embedding.
    High = answer is topically close to the question.
    """
    from indexing.embedder import get_query_embedding
    if not answer:
        return 0.0
    try:
        q_vec = get_query_embedding(question)
        a_vec = get_query_embedding(answer)
        dot = sum(a * b for a, b in zip(q_vec, a_vec))
        mag_q = sum(x**2 for x in q_vec) ** 0.5
        mag_a = sum(x**2 for x in a_vec) ** 0.5
        if mag_q == 0 or mag_a == 0:
            return 0.0
        return round(dot / (mag_q * mag_a), 4)
    except Exception:
        return 0.0


# ── Main evaluation loop ──────────────────────────────────────────────────────

def evaluate(
    strategy: Strategy = "hybrid_rerank",
    top_k: int = 5,
    limit: int | None = None,
) -> dict:
    """
    Run evaluation for a given retrieval strategy.
    Returns a dict of metric averages + per-pair details.
    """
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    pairs = json.loads(QA_PAIRS_PATH.read_text())
    if limit:
        pairs = pairs[:limit]

    print(f"\n=== Evaluating strategy: {strategy} ({len(pairs)} pairs) ===")

    results = []
    for i, pair in enumerate(pairs):
        question          = pair["question"]
        ground_truth      = pair["ground_truth"]
        source_doc_id     = pair["source_doc_id"]
        relevant_ids      = pair["relevant_chunk_ids"]

        print(f"  [{i+1}/{len(pairs)}] {source_doc_id[:40]:40s}", end=" ", flush=True)

        # 1. Retrieve
        retrieved = _retrieve(strategy, question, top_k=top_k)
        contexts  = [c["text"] for c in retrieved]

        # 2. Compute retrieval metrics (no API calls)
        recall    = compute_context_recall(relevant_ids, retrieved)
        precision = compute_context_precision(source_doc_id, retrieved)

        # 3. Generate answer
        answer = _generate_answer(client, question, contexts)
        time.sleep(_BETWEEN_CALL_SLEEP)

        # 4. Faithfulness (LLM judge)
        faith = compute_faithfulness(client, contexts, answer)
        time.sleep(_BETWEEN_CALL_SLEEP)

        # 5. Answer relevancy (embedding similarity — uses embedding API)
        relevancy = compute_answer_relevancy(question, answer)

        results.append({
            "source_doc_id":      source_doc_id,
            "question_type":      pair.get("question_type", ""),
            "question":           question,
            "answer":             answer,
            "ground_truth":       ground_truth,
            "context_recall":     recall,
            "context_precision":  precision,
            "faithfulness":       faith,
            "answer_relevancy":   relevancy,
        })

        print(f"recall={recall:.2f}  prec={precision:.2f}  faith={faith:.2f}  relev={relevancy:.2f}")

    # Aggregate
    def avg(key):
        return round(sum(r[key] for r in results) / len(results), 4)

    summary = {
        "strategy":           strategy,
        "n_pairs":            len(results),
        "top_k":              top_k,
        "context_recall":     avg("context_recall"),
        "context_precision":  avg("context_precision"),
        "faithfulness":       avg("faithfulness"),
        "answer_relevancy":   avg("answer_relevancy"),
        "details":            results,
    }

    # Save
    out_path = RESULTS_DIR / f"eval_{strategy}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved to {out_path}")
    _print_summary(summary)
    return summary


def _print_summary(s: dict) -> None:
    print(f"\n{'─'*50}")
    print(f"Strategy:          {s['strategy']}")
    print(f"Pairs evaluated:   {s['n_pairs']}")
    print(f"context_recall:    {s['context_recall']:.4f}  (target ≥ 0.70)")
    print(f"context_precision: {s['context_precision']:.4f}  (target ≥ 0.75)")
    print(f"faithfulness:      {s['faithfulness']:.4f}  (target ≥ 0.82)")
    print(f"answer_relevancy:  {s['answer_relevancy']:.4f}  (target ≥ 0.78)")
    print(f"{'─'*50}")
    gates = {
        "context_recall":    (s["context_recall"],    0.70),
        "context_precision": (s["context_precision"], 0.75),
        "faithfulness":      (s["faithfulness"],      0.82),
        "answer_relevancy":  (s["answer_relevancy"],  0.78),
    }
    passed = all(v >= t for v, t in gates.values())
    print(f"CI gate: {'PASS ✓' if passed else 'FAIL ✗'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="hybrid_rerank",
                        choices=["dense", "sparse", "hybrid", "hybrid_rerank"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only first N pairs (quick test)")
    args = parser.parse_args()
    evaluate(strategy=args.strategy, top_k=args.top_k, limit=args.limit)
