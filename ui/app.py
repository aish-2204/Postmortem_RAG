"""Post-Mortem RAG — Streamlit UI"""

import time

import streamlit as st

st.set_page_config(
    page_title="Post-Mortem RAG",
    page_icon="🔍",
    layout="centered",
)

# ── Hide Streamlit chrome (deploy button, hamburger menu, footer) ─────────────

st.markdown("""
<style>
#MainMenu        { visibility: hidden; }
header           { visibility: hidden; }
footer           { visibility: hidden; }

.answer-card {
    background: #f0f4ff;
    border-left: 4px solid #4f6ef7;
    padding: 1rem 1.2rem;
    border-radius: 6px;
    font-size: 0.97rem;
    line-height: 1.7;
    color: #1a1a2e !important;
}
.source-chip {
    display: inline-block;
    background: #dde4f7;
    color: #1e2d6b !important;
    border-radius: 12px;
    padding: 3px 12px;
    margin: 2px 3px;
    font-size: 0.82rem;
    font-family: monospace;
}
.trace-row {
    display: flex;
    justify-content: space-between;
    padding: 5px 0;
    border-bottom: 1px solid #dee2e6;
    font-size: 0.87rem;
    color: #1a1a2e !important;
}
</style>
""", unsafe_allow_html=True)

# ── Example queries ───────────────────────────────────────────────────────────

EXAMPLES = [
    "What caused the Cloudflare outage in 2020?",
    "How did AWS remediate the S3 us-east-1 outage?",
    "What lessons did GitLab learn from their 2017 data loss?",
    "What were the contributing factors to the GitLab data loss incident?",
]

# ── Header ────────────────────────────────────────────────────────────────────

st.title("Incident Post-Mortem Assistant")
st.caption("LangGraph agent · Hybrid RRF retrieval · Groq llama-3.3-70b")

mode_tab, qa_tab = st.tabs(["Incident Match", "Q&A"])

with mode_tab:
    st.markdown("Paste current incident symptoms. The agent finds the closest past incident and returns root cause + remediation steps.")
    INCIDENT_EXAMPLES = [
        "Kafka consumer lag spike, OOMKill on K8s pods, 502 errors on API gateway",
        "PostgreSQL connection pool exhausted, high CPU on database nodes, query timeouts",
        "BGP route withdrawal, DNS resolution failures, global traffic drop",
        "S3 PUT requests failing, high error rate on upstream services depending on object storage",
    ]
    st.markdown("**Examples:**")
    icols = st.columns(2)
    for i, ex in enumerate(INCIDENT_EXAMPLES):
        if icols[i % 2].button(ex[:55] + "…", key=f"inc_{i}", use_container_width=True):
            st.session_state["incident_input"] = ex
    incident_query = st.text_area(
        "Current symptoms",
        key="incident_input",
        placeholder="e.g. Kafka consumer lag spike, OOMKill on K8s pods, 502 errors on API",
        height=100,
        label_visibility="collapsed",
    )
    incident_btn = st.button("Find matching incident", type="primary",
                             disabled=not incident_query, use_container_width=True,
                             key="inc_btn")

with qa_tab:
    st.markdown("Ask a natural language question about any past incident or post-mortem.")
    QA_EXAMPLES = [
        "What caused the Cloudflare outage in 2020?",
        "How did AWS remediate the S3 us-east-1 outage?",
        "What lessons did GitLab learn from their 2017 data loss?",
        "What were the contributing factors to the GitLab data loss incident?",
    ]
    st.markdown("**Examples:**")
    qcols = st.columns(2)
    for i, ex in enumerate(QA_EXAMPLES):
        if qcols[i % 2].button(ex[:55] + "…", key=f"qa_{i}", use_container_width=True):
            st.session_state["qa_input"] = ex
    qa_query = st.text_input(
        "Your question",
        key="qa_input",
        placeholder="e.g. What caused the Cloudflare outage in 2020?",
        label_visibility="collapsed",
    )
    qa_btn = st.button("Ask", type="primary", disabled=not qa_query,
                       use_container_width=True, key="qa_btn")

query     = (incident_query if incident_btn else qa_query) or ""
run_btn   = incident_btn or qa_btn
use_mode  = "incident_match" if incident_btn else "qa"

# ── Run agent ─────────────────────────────────────────────────────────────────

if run_btn and query:
    with st.spinner("Running agent…"):
        import time as _time
        from agents.query_analyzer  import query_analyzer
        from agents.retriever_node  import retriever_node
        from agents.self_reflection import self_reflection
        from agents.synthesizer     import synthesizer
        from agents.graph           import _base_state

        state: dict = _base_state(query, mode=use_mode)

        node_timings: list[dict] = []

        def _run_node(name, fn, s):
            t = _time.perf_counter()
            s.update(fn(s))
            node_timings.append({"node": name, "ms": int((_time.perf_counter() - t) * 1000)})

        t0 = _time.perf_counter()
        _run_node("query_analyzer",  query_analyzer,  state)
        _run_node("retriever_node",  retriever_node,  state)
        _run_node("self_reflection", self_reflection, state)

        if not state["sufficient"] and state["iterations"] < 2:
            _run_node("retriever_node (retry)",  retriever_node,  state)
            _run_node("self_reflection (retry)", self_reflection, state)

        _run_node("synthesizer", synthesizer, state)
        total_ms = int((_time.perf_counter() - t0) * 1000)

    # ── Metadata row ──────────────────────────────────────────────────────────
    st.divider()

    QTYPE_LABEL = {
        "root_cause":      "🔴 root cause",
        "remediation":     "🟢 remediation",
        "lessons_learned": "🟡 lessons learned",
        "general":         "🔵 general",
    }
    c1, c2, c3 = st.columns(3)
    if state.get("mode") == "incident_match":
        symptoms = state.get("extracted_symptoms", {})
        c1.metric("Failure category", symptoms.get("failure_category", "—"))
    else:
        c1.metric("Question type", QTYPE_LABEL.get(state["question_type"], state["question_type"]))
    c2.metric("Latency", f"{total_ms} ms")
    c3.metric("Retrieval passes", state["iterations"])

    # ── Answer ────────────────────────────────────────────────────────────────
    if state.get("mode") == "incident_match":
        st.markdown("#### Incident Match Report")
        st.code(state["answer"], language=None)
    else:
        st.markdown("#### Answer")
        st.markdown(
            f'<div class="answer-card">{state["answer"]}</div>',
            unsafe_allow_html=True,
        )

    # ── Sources ───────────────────────────────────────────────────────────────
    if state["sources"]:
        st.markdown("#### Sources")
        chips = " ".join(
            f'<span class="source-chip">{s}</span>' for s in state["sources"]
        )
        st.markdown(chips, unsafe_allow_html=True)

    # ── Agent trace ───────────────────────────────────────────────────────────
    ROW_STYLE  = "display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #dee2e6;font-size:0.87rem;color:#1a1a2e;"
    BOLD_STYLE = "font-weight:600;color:#1a1a2e;"

    with st.expander("Agent trace", expanded=False):
        st.markdown(f"**Reflection:** {state['reflection']}")
        st.markdown(f"**Section filter:** `{state['metadata_filter']}`")

        st.markdown("**Node timings**")
        rows_html = "".join(
            f'<div style="{ROW_STYLE}"><span>{t["node"]}</span>'
            f'<span style="{BOLD_STYLE}">{t["ms"]} ms</span></div>'
            for t in node_timings
        )
        st.markdown(rows_html, unsafe_allow_html=True)

        if state["parent_docs"]:
            st.markdown("**Documents used in answer**")
            for pid, parent in list(state["parent_docs"].items())[:4]:
                meta     = parent.get("metadata", {})
                company  = meta.get("company") or ""
                date     = meta.get("date") or ""
                src_url  = meta.get("source_url", "")
                label    = f"{company} ({date})" if company and company != "unknown" else pid
                with st.expander(label, expanded=False):
                    if src_url:
                        st.caption(src_url)
                    st.text(parent["text"][:600])
