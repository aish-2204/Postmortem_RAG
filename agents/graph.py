"""
LangGraph 4-node post-mortem RAG agent — dual-mode.

Q&A mode      (run):               human natural-language questions
Incident mode (run_incident_match): external agent sends current symptoms,
                                    gets back structured remediation report

Flow (same graph, both modes):
  START → query_analyzer → retriever_node → self_reflection
            ├─ sufficient=True OR iterations>=2  → synthesizer → END
            └─ sufficient=False                  → retriever_node (retry, no filter)
"""

from langgraph.graph import StateGraph, END

from agents.state           import AgentState
from agents.query_analyzer  import query_analyzer
from agents.retriever_node  import retriever_node
from agents.self_reflection import self_reflection
from agents.synthesizer     import synthesizer


def _route_after_reflection(state: AgentState) -> str:
    if state["sufficient"] or state["iterations"] >= 2:
        return "synthesizer"
    return "retriever_node"


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("query_analyzer",  query_analyzer)
    graph.add_node("retriever_node",  retriever_node)
    graph.add_node("self_reflection", self_reflection)
    graph.add_node("synthesizer",     synthesizer)

    graph.set_entry_point("query_analyzer")
    graph.add_edge("query_analyzer",  "retriever_node")
    graph.add_edge("retriever_node",  "self_reflection")
    graph.add_conditional_edges(
        "self_reflection",
        _route_after_reflection,
        {"retriever_node": "retriever_node", "synthesizer": "synthesizer"},
    )
    graph.add_edge("synthesizer", END)

    return graph


_compiled = None


def _get_compiled():
    global _compiled
    if _compiled is None:
        _compiled = build_graph().compile()
    return _compiled


def _base_state(query: str, mode: str) -> AgentState:
    return {
        "query":              query,
        "mode":               mode,
        "question_type":      "",
        "extracted_symptoms": {},
        "metadata_filter":    None,
        "retrieved_chunks":   [],
        "parent_docs":        {},
        "sufficient":         False,
        "reflection":         "",
        "answer":             "",
        "sources":            [],
        "iterations":         0,
    }


def run(query: str) -> dict:
    """Q&A mode — human natural language question. Mode auto-detected."""
    return _get_compiled().invoke(_base_state(query, mode=""))


def run_incident_match(symptoms: str) -> dict:
    """
    Incident match mode — called by an external incident response agent.

    symptoms: free-text description of current incident symptoms,
              e.g. "Kafka consumer lag spike, OOMKill on K8s pods, 502s on API"

    Returns AgentState with answer structured as:
      MATCHED INCIDENT / CONFIDENCE / ROOT CAUSE / REMEDIATION STEPS
    """
    return _get_compiled().invoke(_base_state(symptoms, mode="incident_match"))
