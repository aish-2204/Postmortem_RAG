"""
Node 1 — Query Analyzer

Auto-detects mode and extracts retrieval signals in a single LLM call.

Q&A mode:
  - question_type  → section_type metadata filter
  - company_filter → narrows ChromaDB to a specific company (e.g. "Amazon")
  - year_filter    → narrows to a specific year if mentioned

Incident match mode:
  - failure_category, services_affected, infrastructure_tags, error_codes
  - Hard filter on failure_category + company if mentioned
"""

import json
import os

from dotenv import load_dotenv
from groq import Groq

from agents.state import AgentState

load_dotenv()

_MODEL = os.getenv("GROQ_EVAL_MODEL", "llama-3.3-70b-versatile")

# Common aliases → canonical name as stored in ChromaDB metadata
_COMPANY_NORM: dict[str, str] = {
    "aws": "Amazon", "amazon web services": "Amazon",
    "google cloud": "Google Cloud", "gcp": "Google Cloud", "google": "Google Cloud",
    "github": "GitHub", "gitlab": "GitLab",
    "cloudflare": "Cloudflare",
    "facebook": "Facebook", "meta": "Facebook",
    "stripe": "Stripe",
    "netflix": "Netflix",
    "slack": "Slack",
    "pagerduty": "PagerDuty",
    "launchdarkly": "LaunchDarkly",
}

_CLASSIFY_PROMPT = """\
You analyze input to a post-mortem incident knowledge base.

Determine whether the input is:
  A) A natural language QUESTION about a past incident
  B) An INCIDENT DESCRIPTION — symptoms or a request to match current issues against past incidents

Input: {query}

=== If (A) — Q&A mode: ===
  question_type: root_cause | remediation | lessons_learned | general
  company_filter: the specific company/service mentioned (use canonical name:
    AWS→Amazon, GCP→Google Cloud, Github→GitHub etc.) — null if not mentioned
  year_filter: 4-digit year if a specific year is mentioned — null if not

=== If (B) — Incident match mode: ===
  failure_category: compute | network | database | storage | deployment | dependency | config | unknown
  services_affected: list of service/technology names
  infrastructure_tags: list of infra components (K8s, Kafka, Redis, etc.)
  error_codes: list of specific error strings or named failure modes
  company_filter: specific company if mentioned — null if not

Respond with valid JSON only.

Q&A example:
{{"mode":"qa","question_type":"remediation","company_filter":"Amazon","year_filter":"2017","reason":"asking how AWS fixed S3 outage"}}

Incident match example:
{{"mode":"incident_match","failure_category":"compute","services_affected":["Kafka","K8s"],"infrastructure_tags":["K8s"],"error_codes":["OOMKill"],"company_filter":null,"reason":"symptom description"}}"""


_FAILURE_CATEGORIES = {
    "compute", "network", "database", "storage", "deployment", "dependency", "config", "unknown"
}


def _normalize_company(raw: str | None) -> str | None:
    if not raw:
        return None
    key = raw.strip().lower()
    return _COMPANY_NORM.get(key, raw.strip() if raw.strip() else None)


def query_analyzer(state: AgentState) -> dict:
    query       = state["query"]
    forced_mode = state.get("mode", "")

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    try:
        resp = client.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": _CLASSIFY_PROMPT.format(query=query)}],
            max_tokens=180,
            temperature=0.0,
        )
        text = resp.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.split("```")[1].lstrip("json").strip()
        parsed = json.loads(text)
    except Exception:
        parsed = {"mode": forced_mode or "qa", "question_type": "general"}

    mode            = forced_mode or parsed.get("mode", "qa")
    company_filter  = _normalize_company(parsed.get("company_filter"))
    year_filter     = str(parsed.get("year_filter", "") or "").strip() or None

    # ── Q&A mode ──────────────────────────────────────────────────────────────
    if mode == "qa":
        question_type = parsed.get("question_type", "general")
        _SECTION_MAP = {
            "root_cause":      "root_cause",
            "remediation":     "remediation",
            "lessons_learned": "lessons_learned",
        }
        section = _SECTION_MAP.get(question_type)

        # Build compound filter: section_type AND company (AND year if present)
        metadata_filter = _build_qa_filter(section, company_filter, year_filter)

        print(
            f"  [query_analyzer] mode=qa  type={question_type}  "
            f"company={company_filter}  year={year_filter}  filter={metadata_filter}"
        )
        return {
            "mode":               "qa",
            "question_type":      question_type,
            "extracted_symptoms": {},
            "metadata_filter":    metadata_filter,
            "iterations":         0,
        }

    # ── Incident match mode ───────────────────────────────────────────────────
    failure_category   = parsed.get("failure_category", "unknown")
    if failure_category not in _FAILURE_CATEGORIES:
        failure_category = "unknown"

    extracted_symptoms = {
        "failure_category":    failure_category,
        "services_affected":   parsed.get("services_affected", []),
        "infrastructure_tags": parsed.get("infrastructure_tags", []),
        "error_codes":         parsed.get("error_codes", []),
    }

    metadata_filter = _build_incident_filter(failure_category, company_filter)

    print(
        f"  [query_analyzer] mode=incident_match  "
        f"category={failure_category}  company={company_filter}  filter={metadata_filter}"
    )
    return {
        "mode":               "incident_match",
        "question_type":      "general",
        "extracted_symptoms": extracted_symptoms,
        "metadata_filter":    metadata_filter,
        "iterations":         0,
    }


def _build_qa_filter(section: str | None, company: str | None, year: str | None) -> dict | None:
    """
    Build ChromaDB $and filter from available signals.
    Year is intentionally excluded — company+section is specific enough,
    and ChromaDB range queries on string date fields are unreliable.
    """
    conditions = []
    if section:
        conditions.append({"section_type": {"$eq": section}})
    if company:
        conditions.append({"company": {"$eq": company}})

    if not conditions:
        return None
    if len(conditions) == 1:
        k, v = list(conditions[0].items())[0]
        return {k: v["$eq"]}
    return {"$and": conditions}


def _build_incident_filter(failure_category: str, company: str | None) -> dict | None:
    conditions = []
    if failure_category != "unknown":
        conditions.append({"failure_category": {"$eq": failure_category}})
    if company:
        conditions.append({"company": {"$eq": company}})
    if not conditions:
        return None
    if len(conditions) == 1:
        k, v = list(conditions[0].items())[0]
        return {k: v["$eq"]}
    return {"$and": conditions}
