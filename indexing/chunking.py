"""
Hierarchical chunking strategy for post-mortems.

Structure:
  Level 0 (parent): full post-mortem document (~2000 tokens)
  Level 1 (child):  individual named sections (~400 tokens, 100 overlap)
      - root_cause, timeline, remediation, impact, lessons_learned

At retrieval time: search child chunks for precision, fetch parent for
context richness (parent_id stored on every child in ChromaDB metadata).

Why not flat 512-token splits: post-mortems have explicit semantic
boundaries — the root cause section and remediation section answer
completely different query types. Flat splits break these boundaries,
hurting both precision and faithfulness.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

_SECTION_KEYS: list[tuple[str, str]] = [
    ("root_cause_summary", "root_cause"),
    ("timeline", "timeline"),
    ("remediation_steps", "remediation"),
    ("lessons_learned", "lessons_learned"),
    ("services_affected", "services_affected"),
]

# Approximate token count per char (GPT tokenizer heuristic)
_CHARS_PER_TOKEN = 4
_CHILD_MAX_CHARS = 400 * _CHARS_PER_TOKEN   # ~1600 chars
_OVERLAP_CHARS = 100 * _CHARS_PER_TOKEN     # ~400 chars


@dataclass
class Chunk:
    chunk_id: str
    parent_id: str
    text: str
    metadata: dict[str, Any]
    is_parent: bool = False


def _section_text(doc: dict, key: str) -> str:
    """Convert a structured field to a human-readable string for embedding."""
    value = doc.get(key)
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        if not value:
            return ""
        if isinstance(value[0], dict):
            # timeline entries
            parts = []
            for item in value:
                offset = item.get("offset_minutes")
                evt = item.get("event", "")
                if offset is not None:
                    parts.append(f"T+{offset}m: {evt}")
                else:
                    parts.append(evt)
            return "\n".join(parts)
        return "\n".join(f"- {v}" for v in value)
    return str(value)


def _split_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """Slide a window over text with overlap for long sections."""
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def _doc_context_header(doc: dict) -> str:
    """
    One-line identity header prepended to every child chunk.
    Gives BM25 and dense retrieval the incident identity (company, date, services)
    so a section chunk can be found by company/date queries, not just by its content.

    e.g. "Amazon | 2017-02-28 | config | services: Amazon S3, EC2, Lambda"
    """
    company  = doc.get("company", "unknown")
    date     = doc.get("date", "")
    category = doc.get("failure_category", "")
    services = doc.get("services_affected", [])
    svc_str  = ", ".join(services[:3]) if isinstance(services, list) else str(services)
    parts    = [p for p in [company, date, category] if p and p != "unknown"]
    if svc_str:
        parts.append(f"services: {svc_str}")
    return " | ".join(parts)


def _base_metadata(doc: dict) -> dict[str, Any]:
    return {
        "company": doc.get("company", "unknown"),
        "date": doc.get("date") or "",
        "failure_category": doc.get("failure_category", "unknown"),
        "infrastructure_tags": doc.get("infrastructure_tags", []),
        "severity": doc.get("severity", "unknown"),
        "services_affected": doc.get("services_affected", []),
        "error_codes": doc.get("error_codes", []),
        "source_url": doc.get("source_url", ""),
        "source": doc.get("source", "unknown"),
    }


def chunk_document(doc: dict) -> list[Chunk]:
    """
    Returns one parent chunk + N child chunks for a structured post-mortem doc.
    The parent chunk embeds the full raw_text; children embed individual sections.
    """
    doc_id = doc["id"]
    base_meta = _base_metadata(doc)

    chunks: list[Chunk] = []

    # --- Parent chunk: full raw text ---
    parent_text = doc.get("raw_text", "")
    if not parent_text:
        # Reconstruct from structured fields
        parts = [f"Company: {doc.get('company', 'unknown')}"]
        for key, _ in _SECTION_KEYS:
            section = _section_text(doc, key)
            if section:
                parts.append(section)
        parent_text = "\n\n".join(parts)

    parent_chunk = Chunk(
        chunk_id=f"{doc_id}__parent",
        parent_id=doc_id,
        text=parent_text[:8000],  # cap at ~2k tokens
        metadata={**base_meta, "section_type": "full_document", "parent_id": doc_id},
        is_parent=True,
    )
    chunks.append(parent_chunk)

    # --- Child chunks: one per named section ---
    for field_key, section_name in _SECTION_KEYS:
        section_text = _section_text(doc, field_key)
        if not section_text.strip():
            continue

        # Prefix with section label + doc identity so every chunk is
        # retrievable by company/date queries, not just by content keywords
        ctx     = _doc_context_header(doc)
        labeled = f"[{section_name.upper()}] {ctx}\n{section_text}"
        splits = _split_text(labeled, _CHILD_MAX_CHARS, _OVERLAP_CHARS)

        for j, split in enumerate(splits):
            child_id = f"{doc_id}__{section_name}_{j}" if len(splits) > 1 else f"{doc_id}__{section_name}"
            chunks.append(
                Chunk(
                    chunk_id=child_id,
                    parent_id=doc_id,
                    text=split,
                    metadata={
                        **base_meta,
                        "section_type": section_name,
                        "parent_id": doc_id,
                        "chunk_index": j,
                    },
                    is_parent=False,
                )
            )

    return chunks


def chunk_all_documents(processed_dir: Path = PROCESSED_DIR) -> list[Chunk]:
    """Load all processed JSON docs and return all chunks."""
    all_chunks: list[Chunk] = []
    doc_paths = list(processed_dir.rglob("*.json"))
    print(f"Chunking {len(doc_paths)} documents…")

    for path in doc_paths:
        try:
            doc = json.loads(path.read_text())
            if "id" not in doc:
                continue
            chunks = chunk_document(doc)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"  ERROR chunking {path.name}: {e}")

    parent_count = sum(1 for c in all_chunks if c.is_parent)
    child_count = len(all_chunks) - parent_count
    print(f"Total chunks: {len(all_chunks)} ({parent_count} parents, {child_count} children)")
    return all_chunks
