"""
Dense retrieval via ChromaDB cosine similarity.

Supports optional metadata pre-filtering (e.g., failure_category="database")
to narrow the search space before ANN — this is O(1) in ChromaDB via where clauses.
"""

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()


class DenseRetriever:
    def __init__(self, client=None, openai_client=None):
        import chromadb
        from indexing.chroma_store import get_chunks_collection, get_client
        from indexing.embedder import get_query_embedding

        self._get_embedding = get_query_embedding
        self._col = get_chunks_collection(client or get_client())
        self._openai = openai_client

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict]:
        """
        Returns top_k results: [{"chunk_id", "text", "metadata", "score"}]
        score is cosine distance converted to similarity (1 - distance).
        """
        embedding = self._get_embedding(query, client=self._openai)

        where = _build_where(metadata_filter) if metadata_filter else None
        query_kwargs: dict[str, Any] = {
            "query_embeddings": [embedding],
            "n_results": min(top_k, self._col.count() or 1),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_kwargs["where"] = where

        results = self._col.query(**query_kwargs)

        output = []
        for chunk_id, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append(
                {
                    "chunk_id": chunk_id,
                    "text": doc,
                    "metadata": meta,
                    "score": float(1.0 - dist),  # cosine distance → similarity
                    "retriever": "dense",
                }
            )
        return output


def _build_where(filters: dict[str, Any]) -> dict:
    """
    Build ChromaDB where clause from a flat filter dict.
    Handles list values as $in operator.
    """
    if len(filters) == 1:
        key, val = next(iter(filters.items()))
        return {key: {"$in": val} if isinstance(val, list) else {"$eq": val}}

    conditions = []
    for key, val in filters.items():
        if isinstance(val, list):
            conditions.append({key: {"$in": val}})
        else:
            conditions.append({key: {"$eq": val}})
    return {"$and": conditions}
