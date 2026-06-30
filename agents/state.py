"""Shared state schema for the post-mortem RAG agent."""

from typing import TypedDict


class AgentState(TypedDict):
    # Input
    query: str

    # Mode — set by query_analyzer (auto-detected) or by the caller
    # "qa"             — human asking a natural language question
    # "incident_match" — external agent sending symptom description
    mode: str

    # Set by query_analyzer
    question_type: str          # qa mode:      root_cause | remediation | lessons_learned | general
    extracted_symptoms: dict    # incident mode: {failure_category, services_affected, infrastructure_tags, error_codes}
    metadata_filter: dict | None  # ChromaDB where clause — section_type (qa) or failure_category (incident)

    # Set by retriever_node
    retrieved_chunks: list[dict]
    parent_docs: dict           # parent_id → {text, metadata}

    # Set by self_reflection
    sufficient: bool
    reflection: str

    # Set by synthesizer
    answer: str                 # qa: prose answer  |  incident: structured report
    sources: list[str]

    # Loop control
    iterations: int
