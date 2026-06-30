"""
Node 4 — Synthesizer

Mode-aware answer generation:

  qa mode:             grounded prose answer with [DOC:id] citations
  incident_match mode: structured report — matched incident, root cause,
                       remediation steps, confidence — for consumption by
                       an external incident response agent
"""

import os
import re

from dotenv import load_dotenv
from groq import Groq

from agents.state import AgentState

load_dotenv()

_MODEL = os.getenv("GROQ_EVAL_MODEL", "llama-3.3-70b-versatile")

# ── Prompts ───────────────────────────────────────────────────────────────────

_QA_PROMPT = """\
You are an expert at answering questions about software incident post-mortems.
Answer the question using ONLY the context below. Do not add facts not in the context.
Cite each source inline as [DOC:<id>].

Question: {query}

Context:
{context}

Answer:"""

_INCIDENT_PROMPT = """\
You are an incident response assistant. An automated monitoring agent has reported \
current symptoms. Your job is to find the best matching past incident in the context \
below and return a structured report.

Current symptoms: {query}
Extracted signals: failure_category={failure_category}, \
services={services_affected}, errors={error_codes}

Past incident context:
{context}

Respond in this exact format — no extra text:

MATCHED INCIDENT: <company> (<date>) [DOC:<id>]
CONFIDENCE: <high|medium|low>
SIMILARITY REASON: <one sentence — why this incident matches the current symptoms>
ROOT CAUSE: <concise description of what caused the past incident>
REMEDIATION STEPS:
- <step 1>
- <step 2>
- <step 3>
ADDITIONAL CONTEXT: <any other relevant detail from the post-mortem>"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_context(chunks: list[dict], parents: dict) -> str:
    """
    Build LLM context by grouping chunks under their parent doc header.

    We use chunk text (not parent doc text) because chunks are already
    section-targeted — the parent doc's full text buries relevant sections
    past any reasonable truncation limit.
    """
    # Group chunks by parent — preserves section ordering within each doc
    from collections import defaultdict
    by_parent: dict[str, list[dict]] = defaultdict(list)
    no_parent: list[dict] = []

    for chunk in chunks:
        pid = chunk.get("metadata", {}).get("parent_id", "")
        if pid:
            by_parent[pid].append(chunk)
        else:
            no_parent.append(chunk)

    parts: list[str] = []

    for pid, doc_chunks in by_parent.items():
        # Build header from parent metadata if available, else from chunk metadata
        if pid in parents:
            meta = parents[pid].get("metadata", {})
        else:
            meta = doc_chunks[0].get("metadata", {})

        company  = meta.get("company", "")
        date     = meta.get("date", "")
        category = meta.get("failure_category", "")
        services = meta.get("services_affected", "")
        header = (
            f"[DOC:{pid}] {company} ({date}) "
            f"| category={category} | services={services}"
        )

        # Concatenate all section chunks for this doc — each is already targeted
        chunk_texts = "\n\n".join(c["text"][:800] for c in doc_chunks)
        parts.append(f"{header}\n{chunk_texts}")

    for chunk in no_parent:
        cid = chunk.get("chunk_id", "unknown")
        parts.append(f"[DOC:{cid}]\n{chunk['text'][:600]}")

    return "\n\n---\n\n".join(parts)


# ── Node ──────────────────────────────────────────────────────────────────────

def synthesizer(state: AgentState) -> dict:
    query   = state["query"]
    mode    = state.get("mode", "qa")
    chunks  = state["retrieved_chunks"]
    parents = state["parent_docs"]

    if not chunks:
        return {
            "answer":  "No relevant incidents found in the knowledge base.",
            "sources": [],
        }

    context = _build_context(chunks, parents)

    if mode == "incident_match":
        symptoms = state.get("extracted_symptoms", {})
        prompt = _INCIDENT_PROMPT.format(
            query=query,
            failure_category=symptoms.get("failure_category", "unknown"),
            services_affected=symptoms.get("services_affected", []),
            error_codes=symptoms.get("error_codes", []),
            context=context,
        )
        max_tokens = 600
    else:
        prompt = _QA_PROMPT.format(query=query, context=context)
        max_tokens = 512

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    try:
        resp = client.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.1,
        )
        answer = resp.choices[0].message.content.strip()
    except Exception as exc:
        answer = f"Synthesis failed: {exc}"

    # Handle both [DOC:id] and [DOC:id1, DOC:id2] (comma-separated in one bracket)
    raw_ids = re.findall(r"\[DOC:([^\]]+)\]", answer)
    sources = list(dict.fromkeys(
        part.strip().removeprefix("DOC:")
        for raw in raw_ids
        for part in raw.split(",")
        if part.strip()
    ))
    print(f"  [synthesizer] mode={mode}  answer_len={len(answer)}  sources={sources}")

    return {"answer": answer, "sources": sources}
