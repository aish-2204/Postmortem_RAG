"""
LLM-powered structured extraction: raw post-mortem text → typed JSON schema.

Supports two providers via EXTRACTION_PROVIDER env var:
  - "gemini" (default): Gemini 3.1 Flash Lite via Google AI Studio — free tier.
    Get key at aistudio.google.com.
  - "openai": gpt-4o-mini — ~$0.002/doc.

Both use the OpenAI Python SDK; Gemini is accessed via its OpenAI-compatible
endpoint (same SDK, different base_url + key).

Output: data/processed/<id>.json
"""

import json
import os
import re
import time
import traceback
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import openai
from openai import NotFoundError, OpenAI, RateLimitError

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

_PROVIDER = os.getenv("EXTRACTION_PROVIDER", "gemini").lower()
_DEFAULTS = {
    "gemini": "gemini-3.1-flash-lite",
    "openai": "gpt-4o-mini",
}
_MODEL = os.getenv("EXTRACTION_MODEL", _DEFAULTS.get(_PROVIDER, "gemini-2.5-flash"))
_MAX_INPUT_CHARS = 50_000  # ~12k tokens — Gemini Flash handles this; covers 45k-char status pages

EXTRACTION_PROMPT = """You are an expert SRE analyst. Extract structured information from this production incident post-mortem.

Return ONLY valid JSON matching this exact schema. Use null for unknown fields.

Schema:
{
  "id": "<company>_<YYYY>_<MM>_<DD>",
  "company": "string",
  "date": "YYYY-MM-DD or null",
  "duration_minutes": integer or null,
  "services_affected": ["string"],
  "infrastructure_tags": ["string — infra components: Kafka, PostgreSQL, Redis, BGP, DNS, K8s, etc."],
  "failure_category": "network|database|messaging|compute|storage|config|capacity|dependency|unknown",
  "root_cause_summary": "1–2 sentence factual summary",
  "timeline": [{"offset_minutes": int_or_null, "event": "string"}],
  "remediation_steps": ["string"],
  "error_codes": ["string — exact error codes: OOMKilled, 504, ECONNREFUSED, etc."],
  "severity": "P0|P1|P2|P3|unknown",
  "lessons_learned": ["string"],
  "raw_text": "<verbatim truncated input>"
}

Post-mortem text:
{text}"""


def _make_client() -> OpenAI:
    """Return an OpenAI-compatible client for the configured provider."""
    if _PROVIDER == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set. Get a free key at aistudio.google.com.")
        return OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    # openai
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=api_key)


def _build_processed_index(processed_dir: Path) -> dict[str, Path]:
    """
    Scan already-processed docs and return a map of source_url → file path.
    Used for dedup: if a URL was already extracted, skip the LLM call.
    This is O(n) at startup but cheap — files are small JSON, no embeddings.
    """
    index: dict[str, Path] = {}
    for path in processed_dir.rglob("*.json"):
        try:
            doc = json.loads(path.read_text())
            url = doc.get("source_url", "")
            if url:
                index[url] = path
        except Exception:
            pass
    return index


def _retry_delay_from_error(exc: RateLimitError) -> float:
    """
    Parse the server-provided retryDelay from a Gemini 429 response body.
    The body looks like: "Please retry after 58.83 seconds" or retryDelay field.
    Falls back to 65s if unparseable.
    """
    try:
        body = str(exc)
        match = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?([\d.]+)", body)
        if match:
            return float(match.group(1)) + 2  # add 2s buffer
        match = re.search(r"retry.*?(\d+\.?\d*)\s*s", body, re.IGNORECASE)
        if match:
            return float(match.group(1)) + 2
    except Exception:
        pass
    return 65.0


def _trim_to_budget(text: str, budget: int) -> str:
    """
    For docs longer than budget: take first 60% + last 40%, joined by a marker.
    This ensures the LLM sees both the incident description (top) and
    remediation/lessons_learned sections (bottom) which often appear at the end.

    Strategy: head+tail heuristic — works because post-mortems have predictable structure
    (root cause at top, remediation/lessons at bottom, verbose timeline body in middle).
    The middle section is dropped; total chars sent == budget.

    EVAL NOTE: If extraction shows poor remediation_steps or lessons_learned recall,
    switch to map-reduce: split into N chunks, extract from each, merge results in a
    second LLM call. Cost: ~3x API calls but full coverage regardless of doc structure.
    Tracked: evaluate after RAGAS eval run on full corpus.
    """
    if len(text) <= budget:
        return text
    head = int(budget * 0.6)
    tail = budget - head
    return text[:head] + "\n\n[...truncated...]\n\n" + text[-tail:]


def _make_id(company: str | None, date: str | None, url: str) -> str:
    co = re.sub(r"[^a-zA-Z0-9]", "_", (company or "unknown").lower())[:20]
    if date and re.match(r"\d{4}-\d{2}-\d{2}", date):
        return f"{co}_{date.replace('-', '_')}"
    # Fall back to URL hash fragment
    fragment = re.sub(r"[^a-zA-Z0-9]", "_", url.split("//")[-1])[:30]
    return f"{co}_{fragment}"


def _call_llm(client: OpenAI, text: str) -> dict[str, Any]:
    """
    Call the LLM with one retry on RateLimitError, sleeping for the server-provided
    retryDelay so we don't guess at timing. Raises immediately on NotFoundError.
    """
    trimmed = _trim_to_budget(text, _MAX_INPUT_CHARS)
    prompt = EXTRACTION_PROMPT.replace("{text}", trimmed, 1)

    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                parsed = parsed[0] if parsed else {}
            return parsed
        except NotFoundError:
            raise
        except RateLimitError as e:
            # 429 — wait exactly as long as the server asks
            if attempt == 1:
                raise
            delay = _retry_delay_from_error(e)
            print(f"    Rate limited (429) — waiting {delay:.0f}s…")
            time.sleep(delay)
        except openai.APIStatusError as e:
            # 503 — transient server overload, short wait is enough
            if e.status_code == 503 and attempt == 0:
                print(f"    Service unavailable (503) — waiting 15s…")
                time.sleep(15)
            else:
                raise


def extract_one(raw_path: Path, client: OpenAI) -> Path | None:
    """
    Extract structured data from a single raw file.
    Dedup (skip-if-already-processed) is handled by the caller via
    _build_processed_index — this function always runs the LLM.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw = json.loads(raw_path.read_text())
    text = raw.get("raw_text", "")

    if len(text) < 200:
        return None

    try:
        structured = _call_llm(client, text)
    except Exception as e:
        print(f"  LLM extraction failed for {raw_path.name}")
        print(f"    Type:    {type(e).__name__}")
        print(f"    Message: {e}")
        print(f"    Detail:  {traceback.format_exc().splitlines()[-1]}")
        return None

    structured["company"] = structured.get("company") or raw.get("company") or "unknown"
    structured.setdefault("failure_category", "unknown")
    structured.setdefault("services_affected", [])
    structured.setdefault("infrastructure_tags", [])
    structured.setdefault("error_codes", [])
    structured.setdefault("timeline", [])
    structured.setdefault("remediation_steps", [])
    structured.setdefault("lessons_learned", [])
    structured["raw_text"] = text  # full extracted text — chunker handles splitting at index time
    structured["source_url"] = raw.get("url", "")
    structured["source"] = raw.get("source", "unknown")

    doc_id = _make_id(
        structured["company"],
        structured.get("date"),
        raw.get("url", raw_path.stem),
    )
    structured["id"] = doc_id
    out_path = PROCESSED_DIR / f"{doc_id}.json"
    out_path.write_text(json.dumps(structured, ensure_ascii=False, indent=2))
    return out_path


def extract_all(
    raw_dir: Path = RAW_DIR,
    skip_existing: bool = True,
    max_docs: int | None = None,
) -> list[Path]:
    client = _make_client()
    raw_files = sorted(raw_dir.glob("*.json"))
    print(f"Extracting structured data from {len(raw_files)} raw files…")
    print(f"Provider: {_PROVIDER} / Model: {_MODEL}")

    # Build URL index once — O(n) scan of already-processed files
    processed_urls: dict[str, Path] = _build_processed_index(PROCESSED_DIR) if skip_existing else {}
    if processed_urls:
        print(f"Skipping {len(processed_urls)} already-processed URLs")

    results: list[Path] = []
    new_count = 0

    for i, raw_path in enumerate(raw_files):
        if max_docs is not None and new_count >= max_docs:
            print(f"Reached max_docs={max_docs}, stopping.")
            break

        raw = json.loads(raw_path.read_text())
        source_url = raw.get("url", "")
        text = raw.get("raw_text", "")

        if len(text) > 400_000:
            print(f"  SKIP (PDF/too large {len(text):,} chars): {raw_path.name}")
            continue

        if skip_existing and source_url in processed_urls:
            results.append(processed_urls[source_url])
            continue

        print(f"  [{new_count+1}/{max_docs or len(raw_files)}] {raw_path.name}")
        out = extract_one(raw_path, client)
        if out:
            results.append(out)
            new_count += 1
            if source_url:
                processed_urls[source_url] = out
        time.sleep(10)  # 10s between calls → ~6 RPM, matching observed gemini-2.5-flash free tier limit

    print(f"Extraction complete: {len(results)} total ({new_count} new) in {PROCESSED_DIR}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-docs", type=int, default=None, help="Stop after extracting N new docs")
    parser.add_argument("--no-skip", action="store_true", help="Re-process already-extracted docs")
    args = parser.parse_args()
    extract_all(max_docs=args.max_docs, skip_existing=not args.no_skip)
