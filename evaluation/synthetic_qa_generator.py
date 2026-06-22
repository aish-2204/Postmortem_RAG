"""
Synthetic QA pair generator for RAGAS evaluation.

Generates one question per post-mortem doc, cycling through 3 question types:
  root_cause   — "What caused the X incident?"
  remediation  — "How did [company] fix the X outage?"
  lessons      — "What lessons were learned from the X incident?"

Each QA pair has:
  question          — realistic SRE question (LLM-generated to sound natural)
  ground_truth      — correct answer (extracted directly from structured JSON fields)
  source_doc_id     — links back to the source doc
  question_type     — for stratified RAGAS analysis
  relevant_chunk_ids — which child chunks SHOULD be retrieved (for context recall)

Ground truth comes from structured fields (deterministic, not hallucinated).
LLM is only used to make the question sound natural, not to generate answers.

Usage:
    python -m evaluation.synthetic_qa_generator
    python -m evaluation.synthetic_qa_generator --limit 20   # quick test run
    python -m evaluation.synthetic_qa_generator --resume     # skip already-generated
"""

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_PATH = ROOT / "evaluation" / "results" / "qa_pairs.json"

_GEN_MODEL = os.getenv("EXTRACTION_MODEL", "models/gemini-3.1-flash-lite")
_BETWEEN_CALL_SLEEP = 5   # seconds between Gemini calls (free tier: 30 RPM)
_RATE_LIMIT_WAIT = 60

# Question types to cycle through — one per doc
_QUESTION_TYPES = ["root_cause", "remediation", "lessons_learned"]


# ── Prompts ──────────────────────────────────────────────────────────────────

_PROMPTS = {
    "root_cause": """\
You are a Site Reliability Engineer writing an incident investigation query.

Given this post-mortem summary:
  Company: {company}
  Date: {date}
  Root cause: {root_cause}
  Services affected: {services}

Write ONE concise question (1 sentence) that an SRE would ask when investigating a similar incident.
The question must be answerable using only the root cause summary above.
Ask about the technical cause, NOT about the company name or date.
Return ONLY the question. No explanation, no prefix, no quotes.""",

    "remediation": """\
You are a Site Reliability Engineer looking for incident response playbooks.

Given this post-mortem:
  Company: {company}
  Date: {date}
  Incident type: {failure_category}
  Remediation steps: {remediation}

Write ONE concise question (1 sentence) an SRE would ask when looking for remediation steps for a similar incident.
The question must be answerable using the remediation steps above.
Ask about the fix or recovery process, NOT about the company name or date.
Return ONLY the question. No explanation, no prefix, no quotes.""",

    "lessons_learned": """\
You are a Site Reliability Engineer building a reliability improvement program.

Given this post-mortem:
  Company: {company}
  Date: {date}
  Incident type: {failure_category}
  Lessons learned: {lessons}

Write ONE concise question (1 sentence) about what can be learned or improved from this type of incident.
The question must be answerable using the lessons learned above.
Ask about improvements or prevention, NOT about the company name or date.
Return ONLY the question. No explanation, no prefix, no quotes.""",
}


# ── Ground truth extraction ───────────────────────────────────────────────────

def _extract_ground_truth(doc: dict, question_type: str) -> str | None:
    """Extract ground truth answer directly from structured fields — no LLM."""
    if question_type == "root_cause":
        return doc.get("root_cause_summary") or None

    if question_type == "remediation":
        steps = doc.get("remediation_steps", [])
        if not steps:
            return None
        return " ".join(f"- {s}" for s in steps)

    if question_type == "lessons_learned":
        lessons = doc.get("lessons_learned", [])
        if not lessons:
            return None
        return " ".join(f"- {l}" for l in lessons)

    return None


def _relevant_chunk_ids(doc_id: str, question_type: str) -> list[str]:
    """Return the child chunk IDs that should surface for this question type."""
    type_to_section = {
        "root_cause":      "root_cause",
        "remediation":     "remediation",
        "lessons_learned": "lessons_learned",
    }
    section = type_to_section[question_type]
    return [f"{doc_id}__{section}"]


# ── LLM question generation ───────────────────────────────────────────────────

def _build_prompt(doc: dict, question_type: str) -> str:
    services = ", ".join(doc.get("services_affected", [])[:3]) or "unknown"
    remediation = "\n".join(f"- {s}" for s in doc.get("remediation_steps", [])[:5])
    lessons = "\n".join(f"- {l}" for l in doc.get("lessons_learned", [])[:5])

    return _PROMPTS[question_type].format(
        company=doc.get("company", "unknown"),
        date=doc.get("date", "unknown"),
        root_cause=doc.get("root_cause_summary", ""),
        services=services,
        failure_category=doc.get("failure_category", "unknown"),
        remediation=remediation or "Not available",
        lessons=lessons or "Not available",
    )


def _generate_question(
    client: genai.Client,
    doc: dict,
    question_type: str,
    max_retries: int = 5,
) -> str | None:
    prompt = _build_prompt(doc, question_type)
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=_GEN_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=80,
                ),
            )
            text = response.text.strip().strip('"').strip("'")
            if text and "?" in text:
                return text
            return text or None
        except genai_errors.ClientError as e:
            if e.code == 429:
                wait = _RATE_LIMIT_WAIT * (attempt + 1)
                print(f"\n  Rate limited (429). Waiting {wait}s…")
                time.sleep(wait)
            else:
                print(f"\n  Client error for {doc['id']}: {e}")
                return None
        except genai_errors.ServerError as e:
            # 503 transient overload — back off and retry
            wait = 15 * (attempt + 1)
            print(f"\n  Server overload (503). Waiting {wait}s before retry {attempt + 1}/{max_retries}…")
            time.sleep(wait)
        except Exception as e:
            wait = 10 * (attempt + 1)
            print(f"\n  Unexpected error ({type(e).__name__}). Waiting {wait}s…")
            time.sleep(wait)
    return None


# ── Main pipeline ─────────────────────────────────────────────────────────────

def load_existing(output_path: Path) -> dict[str, dict]:
    """Load already-generated pairs keyed by source_doc_id."""
    if not output_path.exists():
        return {}
    with open(output_path) as f:
        pairs = json.load(f)
    return {p["source_doc_id"]: p for p in pairs}


def save_pairs(pairs: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(pairs, f, indent=2)


def run(limit: int | None = None, resume: bool = True) -> None:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    # Load all processed docs
    doc_paths = sorted(PROCESSED_DIR.rglob("*.json"))
    docs = []
    for p in doc_paths:
        try:
            doc = json.loads(p.read_text())
            if "id" in doc and doc.get("root_cause_summary"):
                docs.append(doc)
        except Exception:
            continue

    print(f"Found {len(docs)} docs with root_cause_summary")

    # Resume: skip already-generated
    existing = load_existing(OUTPUT_PATH) if resume else {}
    if existing:
        print(f"Resuming — {len(existing)} pairs already generated, skipping them")

    if limit:
        docs = docs[:limit]

    pairs: list[dict] = list(existing.values())
    new_count = 0
    skipped = 0

    for i, doc in enumerate(docs):
        doc_id = doc["id"]

        # Cycle question types across docs
        question_type = _QUESTION_TYPES[i % len(_QUESTION_TYPES)]

        # Skip if already generated for this doc
        if resume and doc_id in existing:
            skipped += 1
            continue

        ground_truth = _extract_ground_truth(doc, question_type)
        if not ground_truth:
            # Try next question type if primary field is empty
            for qt in _QUESTION_TYPES:
                ground_truth = _extract_ground_truth(doc, qt)
                if ground_truth:
                    question_type = qt
                    break
        if not ground_truth:
            print(f"  [{i+1}/{len(docs)}] SKIP {doc_id} — no usable fields")
            continue

        print(f"  [{i+1}/{len(docs)}] {doc_id[:45]:45s} [{question_type}]", end=" ", flush=True)

        question = _generate_question(client, doc, question_type)
        if not question:
            print("FAILED")
            continue

        pair = {
            "question":           question,
            "ground_truth":       ground_truth,
            "source_doc_id":      doc_id,
            "company":            doc.get("company", "unknown"),
            "date":               doc.get("date", ""),
            "failure_category":   doc.get("failure_category", "unknown"),
            "question_type":      question_type,
            "relevant_chunk_ids": _relevant_chunk_ids(doc_id, question_type),
        }
        pairs.append(pair)
        existing[doc_id] = pair
        new_count += 1

        print(f"✓  Q: {question[:60]}…" if len(question) > 60 else f"✓  Q: {question}")

        # Save after every 10 new pairs — checkpoint in case of rate limit
        if new_count % 10 == 0:
            save_pairs(pairs, OUTPUT_PATH)
            print(f"  [checkpoint] {len(pairs)} pairs saved to {OUTPUT_PATH}")

        time.sleep(_BETWEEN_CALL_SLEEP)

    # Final save
    save_pairs(pairs, OUTPUT_PATH)
    print(f"\nDone. {new_count} new pairs generated, {skipped} skipped.")
    print(f"Total: {len(pairs)} QA pairs saved to {OUTPUT_PATH}")
    _print_stats(pairs)


def _print_stats(pairs: list[dict]) -> None:
    from collections import Counter
    by_type = Counter(p["question_type"] for p in pairs)
    by_cat  = Counter(p["failure_category"] for p in pairs)
    print("\nBreakdown by question type:")
    for qt, n in sorted(by_type.items()):
        print(f"  {qt:20s} {n}")
    print("\nBreakdown by failure category:")
    for cat, n in by_cat.most_common(8):
        print(f"  {cat:20s} {n}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N docs (for quick test runs)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Regenerate all pairs even if output file exists")
    args = parser.parse_args()
    run(limit=args.limit, resume=not args.no_resume)
