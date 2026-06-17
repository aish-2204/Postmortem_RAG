"""
Hybrid retrieval: RRF fusion of dense (ChromaDB) + sparse (BM25) results.

Why RRF instead of score averaging:
  Dense scores are cosine similarities (0–1).
  BM25 scores are unbounded floats (0–∞).
  Averaging them directly would let BM25 dominate on exact matches.
  RRF uses only rank position, not score magnitude — scale-independent.

RRF formula: score(chunk) = Σ  1 / (k + rank_in_list)
  k=60 is the standard constant that dampens the advantage of rank #1.
  A chunk ranked #1 by both retrievers gets the highest combined score.
"""

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()

_RRF_K = 60  # standard constant — lower k = top ranks dominate more


def rrf_fuse(
    dense_results: list[dict],
    sparse_results: list[dict],
    top_k: int = 20,
    k: int = _RRF_K,
) -> list[dict]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.
    Returns top_k results sorted by RRF score descending.
    """
    rrf_scores: dict[str, float] = {}
    chunk_data: dict[str, dict] = {}

    for rank, result in enumerate(dense_results):
        cid = result["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        chunk_data[cid] = result

    for rank, result in enumerate(sparse_results):
        cid = result["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        if cid not in chunk_data:
            chunk_data[cid] = result

    sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]

    return [
        {**chunk_data[cid], "score": round(rrf_scores[cid], 6), "retriever": "hybrid"}
        for cid in sorted_ids
    ]


class HybridRetriever:
    """
    Combines DenseRetriever + SparseRetriever via RRF.
    Optionally fetches parent documents for context-rich synthesis.
    """

    def __init__(self, dense=None, sparse=None, chroma_client=None):
        from retrieval.dense_retriever import DenseRetriever
        from retrieval.sparse_retriever import SparseRetriever

        self._dense = dense or DenseRetriever(client=chroma_client)
        self._sparse = sparse or SparseRetriever(chroma_client=chroma_client)
        self._chroma_client = chroma_client

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict]:
        """
        Returns top_k chunks fused by RRF.
        metadata_filter is applied only to the dense leg (BM25 has no filter support).
        """
        dense_results = self._dense.retrieve(query, top_k=top_k, metadata_filter=metadata_filter)
        sparse_results = self._sparse.retrieve(query, top_k=top_k)
        return rrf_fuse(dense_results, sparse_results, top_k=top_k)

    def retrieve_with_parents(
        self,
        query: str,
        top_k: int = 20,
        metadata_filter: dict[str, Any] | None = None,
    ) -> dict:
        """
        Returns fused child chunks + their full parent documents.

        Child chunks → precise match signal (what was retrieved and why).
        Parent docs  → rich context for the LLM synthesizer.

        Returns:
            {
                "chunks":  [top_k child chunk dicts with RRF scores],
                "parents": {parent_id: {"text": ..., "metadata": ...}},
            }
        """
        from indexing.chroma_store import fetch_parent

        chunks = self.retrieve(query, top_k=top_k, metadata_filter=metadata_filter)

        # Deduplicate parent IDs — multiple child chunks can share a parent
        parent_ids = list({c["metadata"]["parent_id"] for c in chunks})

        parents = {}
        for pid in parent_ids:
            doc = fetch_parent(pid, client=self._chroma_client)
            if doc:
                parents[pid] = doc

        return {"chunks": chunks, "parents": parents}
