"""
Supplemental scraper for engineering blog post-mortems from high-signal sources.

Each entry is a known, stable URL to a public post-mortem or status page report.
These are static — no crawling, just targeted fetches.
"""

import json
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from data.ingestion.content_extractor import extract_text

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; PostmortemRAG/1.0; "
        "+https://github.com/example/postmortem-rag)"
    )
}

# Curated high-quality post-mortems not in danluu's list
KNOWN_POST_MORTEMS: list[dict] = [
    {
        "company": "Cloudflare",
        "url": "https://blog.cloudflare.com/cloudflare-incident-on-june-21-2022/",
        "tags": ["BGP", "routing", "network"],
    },
    {
        "company": "Cloudflare",
        "url": "https://blog.cloudflare.com/cloudflare-outage-on-june-21-2022/",
        "tags": ["network", "BGP"],
    },
    {
        "company": "GitHub",
        "url": "https://github.blog/2018-10-30-oct21-post-incident-analysis/",
        "tags": ["database", "MySQL", "replication"],
    },
    {
        "company": "GitHub",
        "url": "https://github.blog/2012-12-26-github-availability-this-week/",
        "tags": ["database", "Percona", "replication"],
    },
    {
        "company": "Stripe",
        "url": "https://stripe.com/blog/payment-api-design",
        "tags": ["API", "idempotency"],
    },
    {
        "company": "PagerDuty",
        "url": "https://www.pagerduty.com/blog/outage-post-mortem-culture/",
        "tags": ["culture", "process"],
    },
    {
        "company": "Honeycomb",
        "url": "https://www.honeycomb.io/blog/incident-review-you-cant-have-reliability-without-costs",
        "tags": ["reliability", "cost"],
    },
]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _fetch(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=25)
    resp.raise_for_status()
    return resp.text



def scrape_blogs(skip_existing: bool = True) -> list[Path]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for entry in KNOWN_POST_MORTEMS:
        url = entry["url"]
        domain = urlparse(url).netloc.lstrip("www.")
        slug = domain.replace(".", "_") + "_" + url.split("/")[-2][:40]
        out_path = RAW_DIR / f"blog_{slug}.json"

        if skip_existing and out_path.exists():
            saved.append(out_path)
            continue

        try:
            print(f"  Fetching blog post-mortem: {url}")
            html = _fetch(url)
            text = extract_text(html, url=url)
            if len(text) < 300:
                print(f"    Too short, skipping")
                continue

            record = {
                "url": url,
                "company": entry["company"],
                "description": f"{entry['company']} post-mortem",
                "raw_text": text,
                "source": "blog_scraper",
                "tags": entry.get("tags", []),
            }
            out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
            saved.append(out_path)
            time.sleep(0.5)

        except Exception as e:
            print(f"    ERROR: {e}")

    print(f"Blog scrape complete: {len(saved)} posts")
    return saved


if __name__ == "__main__":
    scrape_blogs()
