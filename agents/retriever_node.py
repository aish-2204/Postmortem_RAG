"""
Node 2 — Retriever

Runs hybrid retrieval (RRF fusion — our ablation winner).
On the first attempt, applies the metadata_filter from query_analyzer
to prefer the target section. On retry (iterations > 0), drops the
filter to broaden the search.
"""

from agents.state import AgentState


def retriever_node(state: AgentState) -> dict:
    from retrieval.hybrid_retriever import HybridRetriever

    query      = state["query"]
    iterations = state["iterations"]

    # Drop section filter on retry — broaden to whole corpus
    metadata_filter = state["metadata_filter"] if iterations == 0 else None
    if iterations > 0:
        print(f"  [retriever] retry {iterations} — dropping section filter for broader search")

    retriever = HybridRetriever()
    result = retriever.retrieve_with_parents(
        query,
        top_k=5,
        metadata_filter=metadata_filter,
    )

    chunks  = result["chunks"]
    parents = result["parents"]

    print(f"  [retriever] retrieved {len(chunks)} chunks from {len(parents)} docs "
          f"(filter={metadata_filter})")

    return {
        "retrieved_chunks": chunks,
        "parent_docs":      parents,
        "iterations":       iterations + 1,
    }
