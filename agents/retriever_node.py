"""
Node 2 — Two-Stage Retriever

Stage 1 — Doc selection (search postmortem_parents collection):
  Finds the most relevant incidents at document level.
  Company/year/failure_category filters applied here.
  Returns top N parent_ids.

Stage 2 — Section selection (search postmortem_chunks filtered to found parent_ids):
  Within the matched incidents, finds the most relevant sections.
  For Q&A: prefers the question_type section (remediation, root_cause, etc.)
  For incident_match: returns all sections from matched docs.

Why two stages:
  Single-stage search conflates "which incident matches?" with "which section answers?".
  A remediation chunk from the wrong incident can rank above the right incident's
  remediation chunk simply because its text is more verbose. Two stages separates
  these concerns: Stage 1 finds the right incident, Stage 2 finds the right section.

Retry (iterations > 0):
  Broaden Stage 2 — drop section filter, return all sections from Stage 1 docs.
"""

from agents.state import AgentState

_STAGE1_TOP_K = 5   # number of parent docs to find
_STAGE2_TOP_K = 10  # section chunks to return from those docs

_SECTION_PRIORITY = {
    "remediation":      0,
    "root_cause":       1,
    "lessons_learned":  2,
    "timeline":         3,
    "services_affected":4,
}


def retriever_node(state: AgentState) -> dict:
    import chromadb, os
    from dotenv import load_dotenv
    from indexing.chroma_store import get_chunks_collection, get_parents_collection, fetch_parent
    load_dotenv()

    query      = state["query"]
    iterations = state["iterations"]
    mode       = state.get("mode", "qa")
    meta_filter = state["metadata_filter"]

    client = chromadb.HttpClient(
        host=os.getenv("CHROMA_HOST", "localhost"),
        port=int(os.getenv("CHROMA_PORT", 8000)),
    )

    # ── Stage 1: find relevant parent docs ────────────────────────────────────
    parent_ids = _stage1_doc_selection(query, meta_filter, client)

    if not parent_ids:
        # No docs found with filter — fall back to unfiltered parent search
        print(f"  [retriever] stage1 found no docs with filter={meta_filter}, retrying unfiltered")
        parent_ids = _stage1_doc_selection(query, None, client)

    # ── Stage 2: get section chunks from matched docs ─────────────────────────
    # On retry: drop section constraint, get all sections from Stage 1 docs
    section_filter = None
    if iterations == 0 and mode == "qa":
        qt = state.get("question_type", "general")
        _SECTION_MAP = {"root_cause": "root_cause", "remediation": "remediation",
                        "lessons_learned": "lessons_learned"}
        section_filter = _SECTION_MAP.get(qt)
    elif iterations > 0:
        print(f"  [retriever] retry {iterations} — broadening to all sections from Stage 1 docs")

    chunks = _stage2_section_selection(parent_ids, section_filter, client)

    # Sort by section priority so most relevant section type surfaces first
    if section_filter is None:
        chunks.sort(key=lambda c: _SECTION_PRIORITY.get(c["metadata"].get("section_type", ""), 99))

    # Fetch parent docs for synthesizer context
    parents = {}
    for pid in parent_ids:
        doc = fetch_parent(pid, client=client)
        if doc:
            parents[pid] = doc

    print(f"  [retriever] stage1={len(parent_ids)} docs  stage2={len(chunks)} chunks"
          f"  (section={section_filter}  iter={iterations})")

    return {
        "retrieved_chunks": chunks[:_STAGE2_TOP_K],
        "parent_docs":      parents,
        "iterations":       iterations + 1,
    }


def _stage1_doc_selection(query: str, meta_filter: dict | None, client) -> list[str]:
    """
    Semantic search against postmortem_parents to find the top matching incidents.
    Applies company/failure_category filter if present (strips section_type —
    parent docs don't have that field).
    """
    from indexing.embedder import embed_texts

    parents_col = client.get_collection("postmortem_parents")
    query_emb   = embed_texts([query])[0]

    # Parent docs don't have section_type — strip it from the filter
    parent_filter = _strip_section_filter(meta_filter)

    kwargs: dict = {"query_embeddings": [query_emb], "n_results": min(_STAGE1_TOP_K, parents_col.count())}
    if parent_filter:
        kwargs["where"] = parent_filter

    try:
        results = parents_col.query(**kwargs)
        ids = results["ids"][0]  # ids[0] because single query
        return [i.replace("__parent", "") for i in ids]
    except Exception as e:
        print(f"  [retriever] stage1 error: {e}")
        return []


def _stage2_section_selection(parent_ids: list[str], section_filter: str | None, client) -> list[dict]:
    """
    Fetch section chunks from the matched parent docs.
    Filters by section_type if provided.
    """
    if not parent_ids:
        return []

    chunks_col = client.get_collection("postmortem_chunks")

    where: dict = {"parent_id": {"$in": parent_ids}}
    if section_filter:
        where = {"$and": [{"parent_id": {"$in": parent_ids}}, {"section_type": {"$eq": section_filter}}]}

    try:
        raw = chunks_col.get(where=where, include=["documents", "metadatas"])
        return [
            {"chunk_id": cid, "text": doc, "metadata": meta, "score": 0.0, "retriever": "two_stage"}
            for cid, doc, meta in zip(raw["ids"], raw["documents"], raw["metadatas"])
        ]
    except Exception as e:
        print(f"  [retriever] stage2 error: {e}")
        return []


def _strip_section_filter(meta_filter: dict | None) -> dict | None:
    """Remove section_type from filter — parent docs don't carry that field."""
    if not meta_filter:
        return None
    if "section_type" in meta_filter:
        rest = {k: v for k, v in meta_filter.items() if k != "section_type"}
        return rest or None
    if "$and" in meta_filter:
        clauses = [c for c in meta_filter["$and"] if "section_type" not in c]
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}
    return meta_filter
