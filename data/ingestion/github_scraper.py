"""
Scrapes danluu/post-mortems: parses the README index, fetches each linked
post-mortem page, and extracts raw text via unstructured.

Output: data/raw/<slug>.json  with fields: {url, company, raw_text, source}
"""

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse


import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from data.ingestion.content_extractor import extract_text

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
DANLUU_README = "https://raw.githubusercontent.com/danluu/post-mortems/master/README.md"

# Sites that block scrapers — skip gracefully
_BLOCKED_DOMAINS = {"twitter.com", "x.com", "facebook.com"}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; PostmortemRAG/1.0; "
        "+https://github.com/example/postmortem-rag)"
    )
}


def _slug(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.netloc + parsed.path
    return re.sub(r"[^a-zA-Z0-9_-]", "_", path)[:120]


def _extract_entries(readme: str) -> list[dict]:
    """
    Parse post-mortem entries from the danluu README.

    Entries are plain paragraphs, one per line, starting with a markdown link:
        [Allegro](https://allegro.tech/...). Description text...
        [Amazon](https://aws.amazon.com/...). Description text...

    Table-of-contents lines look like: ' - **[Config Errors](#config-errors)**'
    Those link to #anchors, not https URLs, so the https?:// regex excludes them.
    """
    entries = []
    for line in readme.splitlines():
        line = line.strip()
        if not line.startswith("["):
            continue
        matches = re.findall(r"\[([^\]]+)\]\((https?://[^\)]+)\)", line)
        for text, url in matches:
            domain = urlparse(url).netloc.lstrip("www.")
            company = text.split(" ")[0].strip(",").strip()
            entries.append({"company": company, "description": text, "url": url, "domain": domain})
    return entries


def _extract_original_url(url: str) -> str | None:
    """
    Pull the original URL out of a Wayback Machine wrapper URL.
      http://web.archive.org/web/20160720200842/https://stackstatus.net/post/...
      → https://stackstatus.net/post/...
    Returns None if the input is not an archive URL.
    """
    match = re.search(r"web\.archive\.org/web/\d+(?:if_)?/(https?://.+)", url)
    return match.group(1) if match else None


def _archive_url_with_flag(url: str) -> str:
    """Add if_ flag to suppress Wayback Machine toolbar injection."""
    return re.sub(r"(web\.archive\.org/web/\d+)(/)", r"\1if_\2", url)


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=8))
def _fetch_url(url: str, timeout: int = 20) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def _fetch_html(url: str) -> str:
    """
    Fetch HTML, with special handling for Wayback Machine URLs:
      1. Try the original embedded URL (live site, cleanest content)
      2. Fall back to archive URL with if_ flag (raw archived HTML, no toolbar)
    """
    original = _extract_original_url(url)
    if original:
        try:
            return _fetch_url(original, timeout=20)
        except Exception:
            return _fetch_url(_archive_url_with_flag(url), timeout=45)
    return _fetch_url(url, timeout=20)


def _fetch_and_save(entry: dict, skip_existing: bool) -> Path | None:
    """Fetch and save a single post-mortem entry. Designed to run in a thread."""
    url = entry["url"]
    domain = urlparse(url).netloc.lstrip("www.")

    if any(blocked in domain for blocked in _BLOCKED_DOMAINS):
        return None

    out_path = RAW_DIR / f"{_slug(url)}.json"
    if skip_existing and out_path.exists():
        return out_path

    try:
        html = _fetch_html(url)
        raw_text = extract_text(html, url=url)
        if len(raw_text) < 200:
            return None

        record = {
            "url": url,
            "company": entry["company"],
            "description": entry["description"],
            "raw_text": raw_text,
            "source": "danluu/post-mortems",
        }
        # write_text is atomic per-file — no lock needed across different paths
        out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
        return out_path

    except Exception as e:
        print(f"  ERROR {url}: {e}")
        return None


def scrape(
    max_entries: int | None = None,
    skip_existing: bool = True,
    max_workers: int = 10,
) -> list[Path]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching danluu/post-mortems README…")
    readme = _fetch_url(DANLUU_README, timeout=20)
    entries = _extract_entries(readme)
    print(f"Found {len(entries)} post-mortem links")

    if max_entries:
        entries = entries[:max_entries]

    # Filter already-done before submitting to the pool
    todo = [e for e in entries if not (skip_existing and (RAW_DIR / f"{_slug(e['url'])}.json").exists())]
    already_done = len(entries) - len(todo)
    if already_done:
        print(f"Skipping {already_done} already-scraped entries")
    print(f"Fetching {len(todo)} entries with {max_workers} workers…")

    saved: list[Path] = []
    # Thread-safe counter for progress printing
    _lock = threading.Lock()
    completed = [0]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_and_save, entry, skip_existing): entry for entry in todo}

        for future in as_completed(futures):
            with _lock:
                completed[0] += 1
                n = completed[0]
            result = future.result()
            entry = futures[future]
            status = "OK" if result else "SKIP"
            print(f"  [{n}/{len(todo)}] {status} {entry['company']:20s} {entry['url']}")
            if result:
                saved.append(result)

    # Add back the ones we skipped
    for e in entries:
        p = RAW_DIR / f"{_slug(e['url'])}.json"
        if p.exists() and p not in saved:
            saved.append(p)

    print(f"Done — {len(saved)} post-mortems in {RAW_DIR}")
    return saved


if __name__ == "__main__":
    scrape()
