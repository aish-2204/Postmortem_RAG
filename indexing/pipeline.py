"""
End-to-end indexing pipeline: raw JSON → ChromaDB + BM25 index.

Incremental by design:
  - Groups chunks by parent doc, embeds and upserts N docs at a time.
  - On any failure, already-upserted docs are saved in ChromaDB.
  - Re-running skips already-indexed docs automatically via parent ID dedup.

Usage:
    python -m indexing.pipeline               # index everything new
    python -m indexing.pipeline --full        # force re-index all
    python -m indexing.pipeline --stats       # print collection stats
    python -m indexing.pipeline --doc-batch 5 # smaller doc batches (more checkpoints)
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DOC_BATCH = 10  # docs to embed+upsert per checkpoint


def run(full_reindex: bool = False, doc_batch_size: int = _DEFAULT_DOC_BATCH) -> None:
    from indexing.bm25_index import build_bm25_index
    from indexing.chroma_store import get_client, get_parents_collection, upsert_chunks
    from indexing.chunking import chunk_all_documents
    from indexing.embedder import embed_texts

    processed_dir = ROOT / "data" / "processed"
    print(f"\n=== Postmortem RAG Indexing Pipeline ===")
    print(f"Source: {processed_dir}")

    # 1. Chunk all docs — fast, no API calls
    all_chunks = chunk_all_documents(processed_dir)
    if not all_chunks:
        print("No documents found in data/processed/.")
        return

    # 2. Dedup: find which parent IDs are not yet in ChromaDB
    client = get_client()
    chunks_to_index = all_chunks
    if not full_reindex:
        existing_ids = set(get_parents_collection(client).get()["ids"])
        if existing_ids:
            chunks_to_index = [c for c in all_chunks if c.parent_id not in existing_ids]
            skipped = len(all_chunks) - len(chunks_to_index)
            new_docs = sum(1 for c in chunks_to_index if c.is_parent)
            print(f"Already indexed: {len(existing_ids)} docs — skipping them")
            print(f"New docs to index: {new_docs}")

    if not chunks_to_index:
        print("All documents already indexed. Use --full to force re-index.")
        _print_stats(client)
        return

    # 3. Group new chunks by parent_id
    chunks_by_parent: dict[str, list] = {}
    for chunk in chunks_to_index:
        chunks_by_parent.setdefault(chunk.parent_id, []).append(chunk)

    parent_ids = list(chunks_by_parent.keys())
    total_docs = len(parent_ids)
    total_doc_batches = (total_docs + doc_batch_size - 1) // doc_batch_size
    print(f"\nIndexing {total_docs} docs in {total_doc_batches} batches of {doc_batch_size}")

    # 4. Embed + upsert per doc batch — progress is saved after every batch
    for i in range(0, total_docs, doc_batch_size):
        batch_parent_ids = parent_ids[i : i + doc_batch_size]
        batch_chunks = [c for pid in batch_parent_ids for c in chunks_by_parent[pid]]
        texts = [c.text for c in batch_chunks]

        doc_batch_num = i // doc_batch_size + 1
        print(f"\nDoc batch {doc_batch_num}/{total_doc_batches}"
              f" (docs {i+1}–{min(i+doc_batch_size, total_docs)}/{total_docs},"
              f" {len(texts)} chunks):")

        embeddings = embed_texts(texts, show_progress=True)
        upsert_chunks(batch_chunks, embeddings, client=client)
        print(f"  Saved to ChromaDB ✓")

    # 5. Build BM25 over full corpus (fast, no API calls — just re-tokenizes text)
    print("\nBuilding BM25 index from full corpus…")
    build_bm25_index(all_chunks)

    _print_stats(client)
    print("\nIndexing complete.")


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
    parser.add_argument("--doc-batch", type=int, default=_DEFAULT_DOC_BATCH,
                        help=f"Docs per embed+upsert checkpoint (default: {_DEFAULT_DOC_BATCH})")
    args = parser.parse_args()

    if args.stats:
        from indexing.chroma_store import get_client
        _print_stats(get_client())
    else:
        run(full_reindex=args.full, doc_batch_size=args.doc_batch)
