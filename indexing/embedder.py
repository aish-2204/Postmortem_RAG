"""
Batch embedding with rate-limit handling and cost tracking.

Uses Google gemini-embedding-001: 3072-dim, free tier.
Free tier limits: 100 RPM, 30K TPM, 1K RPD.

Rate limit math:
  batch_size=20, avg ~400 tokens/chunk → ~8K tokens per API call
  30K TPM ÷ 8K = 3.75 calls/minute → sleep 15s between calls to stay safe
  1200 chunks ÷ 20 = 60 calls total → ~15 min for a full index run
"""

import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

load_dotenv()

_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "20"))
_DIMS = 3072
_BETWEEN_BATCH_SLEEP = 15   # seconds between API calls to respect 30K TPM
_RATE_LIMIT_WAIT = 60        # seconds to wait on 429 before retry


def _get_client() -> genai.Client:
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def _embed_batch_with_retry(
    client: genai.Client,
    texts: list[str],
    task_type: str,
    max_retries: int = 5,
) -> list[list[float]]:
    """Call the Gemini embedding API with manual retry on rate limit (429)."""
    for attempt in range(max_retries):
        try:
            result = client.models.embed_content(
                model=_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            return [e.values for e in result.embeddings]
        except genai_errors.ClientError as e:
            if e.code == 429:
                wait = _RATE_LIMIT_WAIT * (attempt + 1)
                print(f"\n  Rate limited (429). Waiting {wait}s before retry {attempt + 1}/{max_retries}…")
                time.sleep(wait)
            else:
                raise
        except Exception:
            # Transient network error (e.g. RemoteProtocolError) — retry with backoff
            time.sleep(15 * (attempt + 1))
    raise RuntimeError(f"Embedding failed after {max_retries} retries.")


def embed_texts(
    texts: list[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
    show_progress: bool = True,
) -> list[list[float]]:
    """
    Embed a list of texts in batches. Returns embeddings in the same order as input.
    Sleeps between batches to stay within TPM limits.
    """
    client = _get_client()
    all_embeddings: list[list[float]] = []
    total_batches = (len(texts) + _BATCH_SIZE - 1) // _BATCH_SIZE

    if show_progress:
        print(f"  Embedding {len(texts)} chunks | {total_batches} API calls | ~{total_batches * _BETWEEN_BATCH_SLEEP}s")

    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        batch_num = i // _BATCH_SIZE + 1

        if show_progress:
            print(f"  [{batch_num}/{total_batches}] {len(batch)} texts…", end=" ", flush=True)

        embeddings = _embed_batch_with_retry(client, batch, task_type)
        all_embeddings.extend(embeddings)

        if show_progress:
            print("done")

        # Respect TPM limit — sleep between every batch except the last
        if batch_num < total_batches:
            time.sleep(_BETWEEN_BATCH_SLEEP)

    return all_embeddings


def get_query_embedding(text: str) -> list[float]:
    """Single-text embedding for query time — uses RETRIEVAL_QUERY task_type."""
    client = _get_client()
    result = _embed_batch_with_retry(client, [text], task_type="RETRIEVAL_QUERY")
    return result[0]
