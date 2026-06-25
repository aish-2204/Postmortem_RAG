"""
Node 4 — Synthesizer

Generates the final grounded answer from the retrieved parent documents.
Cites source doc IDs inline using [DOC:<id>] notation.
"""

import os

from dotenv import load_dotenv
from groq import Groq

from agents.state import AgentState

load_dotenv()

_MODEL = os.getenv("GROQ_EVAL_MODEL", "llama-3.3-70b-versatile")

_SYNTH_PROMPT = """\
You are an expert at answering questions about software incident post-mortems.
Answer the question using ONLY the context below. Do not add facts not present in the context.

For each claim, cite the source using [DOC:<id>] inline.
If the context does not contain enough information, say so clearly.

Question: {query}

Context:
{context}

Answer:"""


def synthesizer(state: AgentState) -> dict:
    query      = state["query"]
    chunks     = state["retrieved_chunks"]
    parents    = state["parent_docs"]

    # Build context from parent docs (full text) when available, else from chunks
    context_parts: list[str] = []
    seen_parents: set[str] = set()

    for chunk in chunks:
        parent_id = chunk.get("metadata", {}).get("parent_id") or chunk.get("id", "")
        if parent_id and parent_id in parents and parent_id not in seen_parents:
            seen_parents.add(parent_id)
            parent = parents[parent_id]
            context_parts.append(f"[DOC:{parent_id}]\n{parent['text'][:1200]}")
        elif not parent_id or parent_id not in parents:
            doc_id = chunk.get("id", "unknown")
            context_parts.append(f"[DOC:{doc_id}]\n{chunk['text'][:600]}")

    if not context_parts:
        return {
            "answer":  "I could not find relevant information in the post-mortem knowledge base.",
            "sources": [],
        }

    context = "\n\n---\n\n".join(context_parts)

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    try:
        resp = client.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": _SYNTH_PROMPT.format(
                query=query, context=context
            )}],
            max_tokens=512,
            temperature=0.1,
        )
        answer = resp.choices[0].message.content.strip()
    except Exception as exc:
        answer = f"Synthesis failed: {exc}"

    # Extract cited source IDs from the answer
    import re
    sources = list(dict.fromkeys(re.findall(r"\[DOC:([^\]]+)\]", answer)))

    print(f"  [synthesizer] answer length={len(answer)}  sources={sources}")

    return {"answer": answer, "sources": sources}
