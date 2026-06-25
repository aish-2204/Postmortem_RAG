"""
Node 1 — Query Analyzer

Classifies the incoming question into one of:
  root_cause      → what caused X?
  remediation     → how was X fixed / what steps were taken?
  lessons_learned → what was learned / what should change?
  general         → anything else (no section filter applied)

Outputs question_type and a ChromaDB metadata_filter so the retriever
can prefer the most relevant section of post-mortem docs.
"""

import json
import os

from dotenv import load_dotenv
from groq import Groq

from agents.state import AgentState

load_dotenv()

_MODEL = os.getenv("GROQ_EVAL_MODEL", "llama-3.3-70b-versatile")

_CLASSIFY_PROMPT = """\
You are analyzing a question asked to an incident post-mortem knowledge base.

Classify the question into exactly one type:
- root_cause:      asking what caused an incident, failure, or outage
- remediation:     asking how something was fixed, resolved, recovered, or mitigated
- lessons_learned: asking what was learned, what improvements were made, or what to prevent next time
- general:         any other question (e.g. timeline, impact, company history)

Question: {query}

Respond with valid JSON only, no explanation:
{{"question_type": "<type>", "reason": "<one sentence>"}}"""


def query_analyzer(state: AgentState) -> dict:
    query = state["query"]
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    try:
        resp = client.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": _CLASSIFY_PROMPT.format(query=query)}],
            max_tokens=80,
            temperature=0.0,
        )
        text = resp.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("```")[1].lstrip("json").strip()
        parsed = json.loads(text)
        question_type = parsed.get("question_type", "general")
    except Exception:
        question_type = "general"

    # Map question type → ChromaDB section filter
    _SECTION_MAP = {
        "root_cause":      "root_cause",
        "remediation":     "remediation",
        "lessons_learned": "lessons_learned",
    }
    section = _SECTION_MAP.get(question_type)
    metadata_filter = {"section": section} if section else None

    print(f"  [query_analyzer] type={question_type}  filter={metadata_filter}")

    return {
        "question_type":   question_type,
        "metadata_filter": metadata_filter,
        "iterations":      0,
    }
