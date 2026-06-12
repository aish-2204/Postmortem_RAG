"""
BM25 sparse index over all post-mortem chunks.

Persisted to disk as a pickle so it survives restarts without re-fitting.
BM25 captures exact error code matches (OOMKilled, ECONNREFUSED, 504) that
embedding models are unreliable on — exact strings collapse to semantically
similar but incorrect embeddings.

Index only child chunks (not parents) — same granularity as dense retrieval.
"""

import os
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[1]
_INDEX_PATH = Path(os.getenv("BM25_INDEX_PATH", str(ROOT / "data" / "bm25_index.pkl")))


def _tokenize(text: str) -> list[str]:
    """
    Lightweight tokenizer that preserves:
    - Error codes: OOMKilled, ECONNREFUSED, 504
    - CamelCase split: KafkaConsumer → ["Kafka", "Consumer"]
    - Lowercase everything else
    """
    # Split CamelCase
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    # Split on non-alphanumeric but keep underscore-joined tokens (error codes)
    tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    # Filter very short tokens
    return [t for t in tokens if len(t) > 1]


class BM25Index:
    def __init__(self, corpus: list[str], chunk_ids: list[str]):
        self.chunk_ids = chunk_ids
        self._tokenized = [_tokenize(doc) for doc in corpus]
        self.bm25 = BM25Okapi(self._tokenized)

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """Returns top_k results as [{"chunk_id": ..., "score": ...}]."""
        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {"chunk_id": self.chunk_ids[i], "score": float(scores[i])}
            for i in top_indices
            if scores[i] > 0
        ]

    def save(self, path: Path = _INDEX_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"chunk_ids": self.chunk_ids, "bm25": self.bm25}, f)
        print(f"BM25 index saved: {path} ({len(self.chunk_ids)} docs)")

    @classmethod
    def load(cls, path: Path = _INDEX_PATH) -> "BM25Index":
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls.__new__(cls)
        obj.chunk_ids = data["chunk_ids"]
        obj.bm25 = data["bm25"]
        obj._tokenized = []
        return obj


def build_bm25_index(
    chunks: list,  # list[indexing.chunking.Chunk]
    save_path: Path = _INDEX_PATH,
) -> BM25Index:
    """Build BM25 index from child chunks only."""
    child_chunks = [c for c in chunks if not c.is_parent]
    corpus = [c.text for c in child_chunks]
    ids = [c.chunk_id for c in child_chunks]
    print(f"Building BM25 index over {len(child_chunks)} child chunks…")
    index = BM25Index(corpus, ids)
    index.save(save_path)
    return index


def load_bm25_index(path: Path = _INDEX_PATH) -> BM25Index:
    return BM25Index.load(path)
