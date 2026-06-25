"""Shared state schema for the post-mortem RAG agent."""

from typing import TypedDict


class AgentState(TypedDict):
    # Input
    query: str

    # Set by query_analyzer
    question_type: str          # root_cause | remediation | lessons_learned | general
    metadata_filter: dict | None  # ChromaDB where clause, e.g. {"section": "root_cause"}

    # Set by retriever_node
    retrieved_chunks: list[dict]
    parent_docs: dict           # parent_id → {text, metadata}

    # Set by self_reflection
    sufficient: bool
    reflection: str             # LLM explanation of why context is/isn't sufficient

    # Set by synthesizer
    answer: str
    sources: list[str]          # list of source doc IDs cited in the answer

    # Loop control
    iterations: int             # how many retrieval attempts have been made
