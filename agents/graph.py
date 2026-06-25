"""
LangGraph 4-node post-mortem RAG agent.

Flow:
  START
    └─► query_analyzer          # classify question type, set metadata_filter
          └─► retriever_node    # hybrid retrieval (section-filtered on first pass)
                └─► self_reflection
                      ├─ sufficient=False & iterations < 2  ──► retriever_node (retry, no filter)
                      └─ sufficient=True  OR  iterations >= 2  ──► synthesizer
                                                                         └─► END
"""

from langgraph.graph import StateGraph, END

from agents.state       import AgentState
from agents.query_analyzer  import query_analyzer
from agents.retriever_node  import retriever_node
from agents.self_reflection import self_reflection
from agents.synthesizer     import synthesizer


def _route_after_reflection(state: AgentState) -> str:
    """Conditional edge: loop back to retriever or proceed to synthesizer."""
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
        {
            "retriever_node": "retriever_node",
            "synthesizer":    "synthesizer",
        },
    )
    graph.add_edge("synthesizer", END)

    return graph


# Compiled graph — import this in your UI / tests
_compiled = None


def _get_compiled():
    global _compiled
    if _compiled is None:
        _compiled = build_graph().compile()
    return _compiled


def run(query: str) -> dict:
    """Run the agent for a single query. Returns the final AgentState dict."""
    initial_state: AgentState = {
        "query":            query,
        "question_type":    "",
        "metadata_filter":  None,
        "retrieved_chunks": [],
        "parent_docs":      {},
        "sufficient":       False,
        "reflection":       "",
        "answer":           "",
        "sources":          [],
        "iterations":       0,
    }
    app = _get_compiled()
    result = app.invoke(initial_state)
    return result
