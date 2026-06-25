"""
Post-Mortem RAG — Streamlit UI

Layout:
  Left sidebar  — app info, question type legend
  Main area     — query input → answer card + sources
  Expander      — agent trace (node timings, reflection verdict, retry info)
"""

import time

import streamlit as st

st.set_page_config(
    page_title="Post-Mortem RAG",
    page_icon="🔍",
    layout="wide",
)

# ── Styles ────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
.answer-card {
    background: #f0f4ff;
    border-left: 4px solid #4f6ef7;
    padding: 1rem 1.2rem;
    border-radius: 6px;
    font-size: 0.97rem;
    line-height: 1.7;
}
.source-chip {
    display: inline-block;
    background: #e8edf9;
    color: #2d3a6e;
    border-radius: 12px;
    padding: 2px 10px;
    margin: 2px 3px;
    font-size: 0.82rem;
    font-family: monospace;
}
.trace-row {
    display: flex;
    justify-content: space-between;
    padding: 4px 0;
    border-bottom: 1px solid #eee;
    font-size: 0.87rem;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Post-Mortem RAG")
    st.caption("LangGraph agent over ~200 incident post-mortems")
    st.markdown("---")
    st.markdown("""
**Question types detected automatically:**

| Type | Example |
|---|---|
| `root_cause` | *What caused the Cloudflare outage?* |
| `remediation` | *How did GitHub fix the 2018 outage?* |
| `lessons_learned` | *What did Stripe learn from their incident?* |
| `general` | *How long did the AWS S3 outage last?* |
""")
    st.markdown("---")
    st.markdown("""
**Retrieval strategy:** Hybrid RRF
**LLM:** Groq llama-3.3-70b-versatile
**Max retries:** 2
""")
    st.markdown("---")
    st.caption("Week 3 agent · ablation winner: hybrid RRF (faith=0.945)")


# ── Example queries ───────────────────────────────────────────────────────────

EXAMPLES = [
    "What caused the Cloudflare outage in 2019?",
    "How did GitHub remediate the October 2018 database incident?",
    "What lessons were learned from the AWS S3 us-east-1 outage?",
    "What were the contributing factors to the GitLab data loss incident?",
]

# ── Main ──────────────────────────────────────────────────────────────────────

st.header("Incident Post-Mortem Assistant")
st.markdown("Ask a question about any incident, outage, or post-mortem in the knowledge base.")

# Example buttons
st.markdown("**Try an example:**")
cols = st.columns(len(EXAMPLES))
for col, example in zip(cols, EXAMPLES):
    if col.button(example[:45] + "…", key=example, use_container_width=True):
        st.session_state["query_input"] = example

query = st.text_input(
    "Your question",
    key="query_input",
    placeholder="e.g. What caused the Cloudflare outage in 2019?",
)

run_btn = st.button("Ask", type="primary", disabled=not query)

# ── Run agent ─────────────────────────────────────────────────────────────────

if run_btn and query:
    with st.spinner("Running agent…"):
        from agents.graph import run as agent_run

        t0 = time.perf_counter()

        # Per-node timing via a thin wrapper — we instrument via state snapshots
        node_timings: list[dict] = []

        # We'll re-run the graph manually step-by-step to capture per-node timing
        import time as _time
        from agents.state import AgentState
        from agents.query_analyzer  import query_analyzer
        from agents.retriever_node  import retriever_node
        from agents.self_reflection import self_reflection
        from agents.synthesizer     import synthesizer

        state: dict = {
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

        def _run_node(name, fn, s):
            t = _time.perf_counter()
            out = fn(s)
            elapsed = round(_time.perf_counter() - t, 3)
            node_timings.append({"node": name, "ms": int(elapsed * 1000)})
            s.update(out)

        _run_node("query_analyzer",  query_analyzer,  state)
        _run_node("retriever_node",  retriever_node,  state)
        _run_node("self_reflection", self_reflection, state)

        retry_triggered = not state["sufficient"] and state["iterations"] < 2
        if retry_triggered:
            _run_node("retriever_node (retry)", retriever_node,  state)
            _run_node("self_reflection (retry)", self_reflection, state)

        _run_node("synthesizer", synthesizer, state)

        total_ms = int((time.perf_counter() - t0) * 1000)

    # ── Answer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    qtype_badge = {
        "root_cause":      "🔴 root_cause",
        "remediation":     "🟢 remediation",
        "lessons_learned": "🟡 lessons_learned",
        "general":         "🔵 general",
    }.get(state["question_type"], state["question_type"])

    col1, col2, col3 = st.columns([2, 1, 1])
    col1.markdown(f"**Question type:** {qtype_badge}")
    col2.metric("Total latency", f"{total_ms} ms")
    col3.metric("Retrieval iterations", state["iterations"])

    st.markdown("### Answer")
    st.markdown(
        f'<div class="answer-card">{state["answer"]}</div>',
        unsafe_allow_html=True,
    )

    # ── Sources ───────────────────────────────────────────────────────────────
    if state["sources"]:
        st.markdown("### Sources cited")
        chips = " ".join(
            f'<span class="source-chip">{s}</span>' for s in state["sources"]
        )
        st.markdown(chips, unsafe_allow_html=True)

    # ── Agent trace ───────────────────────────────────────────────────────────
    with st.expander("Agent trace", expanded=False):
        st.markdown(f"**Reflection verdict:** {state['reflection']}")
        st.markdown(f"**Section filter applied:** `{state['metadata_filter']}`")
        st.markdown(f"**Chunks retrieved (final pass):** {len(state['retrieved_chunks'])}")
        st.markdown("**Node timings:**")
        rows_html = ""
        for t in node_timings:
            rows_html += (
                f'<div class="trace-row">'
                f'<span>{t["node"]}</span>'
                f'<span><b>{t["ms"]} ms</b></span>'
                f'</div>'
            )
        st.markdown(rows_html, unsafe_allow_html=True)

        if state["retrieved_chunks"]:
            st.markdown("**Top retrieved chunks:**")
            for i, chunk in enumerate(state["retrieved_chunks"][:3], 1):
                meta = chunk.get("metadata", {})
                with st.expander(
                    f"Chunk {i} — {meta.get('doc_id', 'unknown')} [{meta.get('section', '?')}]",
                    expanded=False,
                ):
                    st.text(chunk["text"][:600])
