"""
ChromaDB collection management.

Two collections:
  - postmortem_chunks: child chunks for dense retrieval (embedding per chunk)
  - postmortem_parents: parent docs for context fetching (no embedding needed at query time)

Metadata schema per chunk is documented in data/processed/schema.md.
"""

import json
import os
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]

_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(ROOT / "data" / "chroma_db"))
_CHROMA_HOST = os.getenv("CHROMA_HOST", "")
_CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))

CHUNKS_COLLECTION = "postmortem_chunks"
PARENTS_COLLECTION = "postmortem_parents"


def get_client() -> chromadb.ClientAPI:
    """Return a ChromaDB client — HTTP if CHROMA_HOST is set, else embedded persistent."""
    if _CHROMA_HOST:
        return chromadb.HttpClient(
            host=_CHROMA_HOST,
            port=_CHROMA_PORT,
            settings=Settings(anonymized_telemetry=False),
        )
    Path(_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )


def get_chunks_collection(client: chromadb.ClientAPI | None = None) -> chromadb.Collection:
    c = client or get_client()
    return c.get_or_create_collection(
        name=CHUNKS_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def get_parents_collection(client: chromadb.ClientAPI | None = None) -> chromadb.Collection:
    c = client or get_client()
    return c.get_or_create_collection(
        name=PARENTS_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def _sanitize_metadata(meta: dict) -> dict:
    """ChromaDB requires scalar metadata values; convert lists to JSON strings."""
    clean = {}
    for k, v in meta.items():
        if isinstance(v, list):
            clean[k] = json.dumps(v)
        elif v is None:
            clean[k] = ""
        else:
            clean[k] = v
    return clean


def upsert_chunks(
    chunks: list,  # list[indexing.chunking.Chunk]
    embeddings: list[list[float]],
    client: chromadb.ClientAPI | None = None,
    batch_size: int = 200,
) -> int:
    """
    Upsert child chunks into the chunks collection and parent docs into the parents collection.
    Returns number of chunks upserted.
    """
    c = client or get_client()
    chunks_col = get_chunks_collection(c)
    parents_col = get_parents_collection(c)

    child_chunks = [(chunk, emb) for chunk, emb in zip(chunks, embeddings) if not chunk.is_parent]
    parent_chunks = [(chunk, emb) for chunk, emb in zip(chunks, embeddings) if chunk.is_parent]

    # Upsert parents (no embedding — fetched by ID for context assembly)
    if parent_chunks:
        parent_ids = [c.chunk_id for c, _ in parent_chunks]
        parent_docs = [c.text for c, _ in parent_chunks]
        parent_metas = [_sanitize_metadata({**c.metadata, "parent_id": c.parent_id}) for c, _ in parent_chunks]
        parent_embeds = [e for _, e in parent_chunks]
        for i in range(0, len(parent_ids), batch_size):
            parents_col.upsert(
                ids=parent_ids[i : i + batch_size],
                embeddings=parent_embeds[i : i + batch_size],
                documents=parent_docs[i : i + batch_size],
                metadatas=parent_metas[i : i + batch_size],
            )

    # Upsert children
    for i in range(0, len(child_chunks), batch_size):
        batch = child_chunks[i : i + batch_size]
        ids = [c.chunk_id for c, _ in batch]
        embeds = [e for _, e in batch]
        docs = [c.text for c, _ in batch]
        metas = [_sanitize_metadata(c.metadata) for c, _ in batch]
        chunks_col.upsert(ids=ids, embeddings=embeds, documents=docs, metadatas=metas)

    print(
        f"Upserted {len(parent_chunks)} parents, {len(child_chunks)} child chunks to ChromaDB"
    )
    return len(child_chunks)


def fetch_parent(parent_id: str, client: chromadb.ClientAPI | None = None) -> dict | None:
    """Fetch a parent document by its doc ID for context assembly."""
    c = client or get_client()
    col = get_parents_collection(c)
    result = col.get(ids=[parent_id], include=["documents", "metadatas"])
    if not result["ids"]:
        return None
    return {
        "id": parent_id,
        "text": result["documents"][0],
        "metadata": result["metadatas"][0],
    }


def collection_stats(client: chromadb.ClientAPI | None = None) -> dict:
    c = client or get_client()
    chunks_col = get_chunks_collection(c)
    parents_col = get_parents_collection(c)
    return {
        "chunks": chunks_col.count(),
        "parents": parents_col.count(),
    }
