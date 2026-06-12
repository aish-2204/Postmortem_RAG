"""
BM25 sparse retrieval.

Loads the persisted BM25 index and returns chunk IDs + scores.
To get full chunk text+metadata, we need to fetch from ChromaDB.
This is intentional: the BM25 index is the ranking oracle, ChromaDB
is the document store.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]


class SparseRetriever:
    def __init__(self, bm25_index=None, chroma_client=None):
        from indexing.bm25_index import load_bm25_index
        from indexing.chroma_store import get_chunks_collection, get_client

        self._index = bm25_index or load_bm25_index()
        self._col = get_chunks_collection(chroma_client or get_client())

    def retrieve(self, query: str, top_k: int = 20) -> list[dict]:
        """
        Returns top_k BM25 results: [{"chunk_id", "text", "metadata", "score"}]

        Scores are raw BM25 scores (TF-IDF based, unbounded) — do not compare
        directly against dense similarity scores. RRF handles normalization.
        """
        bm25_results = self._index.search(query, top_k=top_k)
        if not bm25_results:
            return []

        chunk_ids = [r["chunk_id"] for r in bm25_results]
        score_map = {r["chunk_id"]: r["score"] for r in bm25_results}

        # Fetch text + metadata from ChromaDB
        fetched = self._col.get(ids=chunk_ids, include=["documents", "metadatas"])

        output = []
        for cid, doc, meta in zip(
            fetched["ids"], fetched["documents"], fetched["metadatas"]
        ):
            output.append(
                {
                    "chunk_id": cid,
                    "text": doc,
                    "metadata": meta,
                    "score": score_map.get(cid, 0.0),
                    "retriever": "sparse",
                }
            )

        # Preserve BM25 rank order (get() doesn't guarantee order)
        id_to_idx = {cid: i for i, cid in enumerate(chunk_ids)}
        output.sort(key=lambda x: id_to_idx.get(x["chunk_id"], 999))
        return output
