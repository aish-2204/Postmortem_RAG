"""
Batch embedding with retry, rate-limit handling, and cost tracking.

Uses OpenAI text-embedding-3-small: 1536-dim, $0.02/1M tokens.
At ~300 chars/chunk avg: 200 docs × 6 chunks × 75 tokens ≈ 90k tokens ≈ $0.002 total.
"""

import os
import time
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

load_dotenv()

_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "100"))
_DIMS = 1536  # text-embedding-3-small dimensions


@retry(
    retry=retry_if_exception_type(RateLimitError),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=60),
)
def _embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=_MODEL, input=texts)
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]


def embed_texts(
    texts: list[str],
    client: OpenAI | None = None,
    show_progress: bool = True,
) -> list[list[float]]:
    """
    Embed a list of texts in batches. Returns embeddings in same order as input.
    Tracks token usage and prints cost estimate.
    """
    if client is None:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    all_embeddings: list[list[float]] = []
    total_batches = (len(texts) + _BATCH_SIZE - 1) // _BATCH_SIZE
    total_chars = sum(len(t) for t in texts)

    if show_progress:
        estimated_tokens = total_chars // 4
        estimated_cost = estimated_tokens / 1_000_000 * 0.02
        print(f"Embedding {len(texts)} texts (~{estimated_tokens:,} tokens, ~${estimated_cost:.4f})")

    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        batch_num = i // _BATCH_SIZE + 1

        if show_progress:
            print(f"  Batch {batch_num}/{total_batches} ({len(batch)} texts)…", end="\r")

        embeddings = _embed_batch(client, batch)
        all_embeddings.extend(embeddings)

        # Small delay to stay under TPM limits
        if batch_num < total_batches:
            time.sleep(0.1)

    if show_progress:
        print(f"  Embedding complete: {len(all_embeddings)} vectors ({_DIMS}d)")

    return all_embeddings


def get_query_embedding(text: str, client: OpenAI | None = None) -> list[float]:
    """Single-text embedding for query time — no batching overhead."""
    if client is None:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    result = _embed_batch(client, [text])
    return result[0]
