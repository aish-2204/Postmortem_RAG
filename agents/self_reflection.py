"""
Node 3 — Self Reflection

Judges whether the retrieved context is sufficient to answer the query.
If not (and we haven't retried yet), signals the graph to loop back to
the retriever with a broader search.

Max 2 retrieval attempts total — after that, synthesizer runs regardless.
"""

import json
import os

from dotenv import load_dotenv
from groq import Groq

from agents.state import AgentState

load_dotenv()

_MODEL     = os.getenv("GROQ_EVAL_MODEL", "llama-3.3-70b-versatile")
_MAX_ITERS = 2

_REFLECT_PROMPT = """\
You are evaluating whether retrieved context is sufficient to answer a question.

Question: {query}

Retrieved context (top chunks):
{context}

Is this context sufficient to give a specific, accurate answer?
Consider: does it contain the actual facts needed, or only vague/unrelated content?

Respond with valid JSON only:
{{"sufficient": true/false, "reason": "<one sentence>"}}"""


def self_reflection(state: AgentState) -> dict:
    query  = state["query"]
    chunks = state["retrieved_chunks"]

    # Always pass through if we've hit the retry limit
    if state["iterations"] >= _MAX_ITERS:
        print("  [reflection] max iterations reached — proceeding to synthesizer")
        return {"sufficient": True, "reflection": "Max iterations reached, proceeding."}

    if not chunks:
        print("  [reflection] no chunks retrieved — retrying")
        return {"sufficient": False, "reflection": "No chunks retrieved."}

    context = "\n\n---\n\n".join(c["text"][:400] for c in chunks[:3])

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    try:
        resp = client.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": _REFLECT_PROMPT.format(
                query=query, context=context
            )}],
            max_tokens=80,
            temperature=0.0,
        )
        text = resp.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.split("```")[1].lstrip("json").strip()
        parsed     = json.loads(text)
        sufficient = bool(parsed.get("sufficient", True))
        reason     = parsed.get("reason", "")
    except Exception:
        sufficient = True
        reason     = "Reflection failed — defaulting to sufficient."

    status = "sufficient" if sufficient else "insufficient — will retry"
    print(f"  [reflection] {status}: {reason}")

    return {"sufficient": sufficient, "reflection": reason}
