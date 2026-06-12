"""
End-to-end indexing pipeline: raw JSON → ChromaDB + BM25 index.

Idempotent: skips already-indexed documents by checking ChromaDB parent IDs.
Run this whenever new post-mortems are added to data/processed/.

Usage:
    python -m indexing.pipeline               # index everything
    python -m indexing.pipeline --full        # force re-index all
    python -m indexing.pipeline --stats       # print collection stats
"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]


def run(full_reindex: bool = False) -> None:
    # Import here to avoid circular deps in tests
    from indexing.bm25_index import build_bm25_index
    from indexing.chroma_store import collection_stats, get_client, upsert_chunks
    from indexing.chunking import chunk_all_documents
    from indexing.embedder import embed_texts

    processed_dir = ROOT / "data" / "processed"
    print(f"\n=== Postmortem RAG Indexing Pipeline ===")
    print(f"Source: {processed_dir}")

    # 1. Chunk all documents
    all_chunks = chunk_all_documents(processed_dir)
    if not all_chunks:
        print("No documents found in data/processed/. Run the ingestion pipeline first.")
        return

    # 2. Dedup: skip already-indexed parent IDs
    client = get_client()
    if not full_reindex:
        from indexing.chroma_store import get_parents_collection
        parents_col = get_parents_collection(client)
        existing_ids = set(parents_col.get()["ids"])
        if existing_ids:
            new_chunks = [c for c in all_chunks if c.parent_id not in existing_ids]
            print(f"Skipping {len(existing_ids)} already-indexed parents, {len(new_chunks)} new chunks to index")
            all_chunks = new_chunks

    if not all_chunks:
        print("All documents already indexed. Use --full to force re-index.")
        _print_stats(client)
        return

    # 3. Embed all chunks
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    texts = [c.text for c in all_chunks]
    embeddings = embed_texts(texts, client=openai_client)

    # 4. Upsert to ChromaDB
    upsert_chunks(all_chunks, embeddings, client=client)

    # 5. Build BM25 index (always rebuild to include new docs)
    from indexing.bm25_index import BM25_INDEX_PATH_DEFAULT, build_bm25_index
    build_bm25_index(all_chunks)

    # 6. Stats
    _print_stats(client)
    print("\nIndexing complete. Run `python -m indexing.pipeline --stats` to verify.")


def _print_stats(client=None) -> None:
    from indexing.chroma_store import collection_stats
    stats = collection_stats(client)
    print(f"\nChromaDB collections:")
    print(f"  postmortem_chunks (child):  {stats['chunks']}")
    print(f"  postmortem_parents (full):  {stats['parents']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the post-mortem indexing pipeline")
    parser.add_argument("--full", action="store_true", help="Force re-index all documents")
    parser.add_argument("--stats", action="store_true", help="Print collection stats and exit")
    args = parser.parse_args()

    if args.stats:
        from indexing.chroma_store import get_client
        _print_stats(get_client())
    else:
        run(full_reindex=args.full)
