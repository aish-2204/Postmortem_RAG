"""
Cohere cross-encoder reranker for precision stage of retrieval.

Architecture:
  Bi-encoder (dense retrieval): embeds query and doc independently → fast but approximate
  Cross-encoder (this):         processes query+doc together → slower but far more accurate

Used after HybridRetriever to rescore the top-20 RRF candidates down to top-5.
Gracefully degrades to RRF score ordering if COHERE_API_KEY is not set.
"""

import os

from dotenv import load_dotenv

load_dotenv()

_MODEL = "rerank-english-v3.0"
_DEFAULT_TOP_N = 5


class Reranker:
    def __init__(self):
        api_key = os.getenv("COHERE_API_KEY", "")
        self._enabled = bool(api_key and api_key != "...")
        self._client = None

        if self._enabled:
            import cohere
            self._client = cohere.Client(api_key=api_key)

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_n: int = _DEFAULT_TOP_N,
    ) -> list[dict]:
        """
        Rerank candidates using Cohere cross-encoder.
        Falls back to RRF score ordering if Cohere is unavailable.

        Args:
            query:      the original user query
            candidates: list of chunk dicts from HybridRetriever (must have "text" key)
            top_n:      number of results to return

        Returns:
            top_n chunk dicts with updated "score" (Cohere relevance 0–1) and
            "retriever" set to "reranked". Original RRF score preserved as "rrf_score".
        """
        if not candidates:
            return []

        top_n = min(top_n, len(candidates))

        if not self._enabled:
            return self._fallback(candidates, top_n)

        texts = [c["text"] for c in candidates]

        try:
            response = self._client.rerank(
                model=_MODEL,
                query=query,
                documents=texts,
                top_n=top_n,
                return_documents=False,  # we already have the text in candidates
            )

            reranked = []
            for r in response.results:
                candidate = candidates[r.index].copy()
                candidate["rrf_score"] = candidate["score"]      # preserve RRF score
                candidate["score"] = round(r.relevance_score, 6) # Cohere score replaces it
                candidate["retriever"] = "reranked"
                reranked.append(candidate)

            return reranked

        except Exception as e:
            print(f"  Reranker warning: Cohere call failed ({e}). Using RRF scores.")
            return self._fallback(candidates, top_n)

    def _fallback(self, candidates: list[dict], top_n: int) -> list[dict]:
        """Return top_n by existing RRF score — used when Cohere is unavailable."""
        sorted_candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
        return [
            {**c, "retriever": "rrf_fallback"}
            for c in sorted_candidates[:top_n]
        ]
