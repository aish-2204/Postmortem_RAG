"""
3-tier HTML → text extraction cascade, shared across all scrapers.

Tier 1: trafilatura  — text-density heuristics, handles 90%+ of blog/status pages
Tier 2: semantic HTML — <article> / <main> for sites with correct markup
Tier 3: BS4 full-page — current behavior, last resort

Why a cascade instead of just trafilatura:
  trafilatura occasionally returns nothing on unusual page structures (some wiki
  pages, status dashboards with no clear article boundary). The semantic HTML
  tier catches those. BS4 full-page is the guaranteed non-empty fallback.

Why not per-company selectors:
  danluu/post-mortems has 300 URLs across 200+ domains. Sites redesign their
  HTML. A selector map is unbounded maintenance. The cascade handles all of them.
"""

import trafilatura
from bs4 import BeautifulSoup

_MIN_LENGTH = 200


def extract_text(html: str, url: str = "") -> str:
    """
    Extract main article text from an HTML page.
    Returns clean, boilerplate-free text ready for LLM extraction and embedding.
    """
    # Tier 1: trafilatura — same algorithm as Firefox Reader Mode
    text = trafilatura.extract(
        html,
        url=url or None,
        include_comments=False,
        include_tables=True,
        no_fallback=False,
    )
    if text and len(text.strip()) > _MIN_LENGTH:
        return _clean(text)

    # Tier 2: semantic HTML elements
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    for selector in ["article", "main", "[role='main']"]:
        el = soup.select_one(selector)
        if el:
            candidate = _clean(el.get_text(separator="\n"))
            if len(candidate) > _MIN_LENGTH:
                return candidate

    # Tier 3: full-page dump
    return _clean(soup.get_text(separator="\n"))


def _clean(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)
