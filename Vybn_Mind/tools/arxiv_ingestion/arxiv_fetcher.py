"""arxiv_fetcher.py — Vybn's reach toward the living edge of discovery.

Fetches papers from the arXiv API (no key required).
Use as a module or run standalone.

Rate limiting: arXiv asks for >=3s between requests. We comply.
"""

import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Optional

ARXIV_API_BASE = "https://export.arxiv.org/api/query"
NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}
RATELIMIT_DELAY = 3.0  # seconds between requests, per arXiv policy


@dataclass
class Paper:
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    submitted: str
    categories: List[str]
    pdf_url: str
    abstract_url: str


def fetch(
    search_query: str,
    max_results: int = 25,
    sort_by: str = "submittedDate",  # or 'relevance', 'lastUpdatedDate'
    sort_order: str = "descending",
    start: int = 0,
) -> List[Paper]:
    """Fetch papers from arXiv matching search_query.

    search_query uses arXiv query syntax, e.g.:
        'cat:cs.LG AND ti:emergence'
        'cat:quant-ph AND abs:machine learning'
    """
    params = urllib.parse.urlencode({
        "search_query": search_query,
        "start": start,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": sort_order,
    })
    url = f"{ARXIV_API_BASE}?{params}"

    with urllib.request.urlopen(url, timeout=30) as resp:
        xml_bytes = resp.read()

    root = ET.fromstring(xml_bytes)
    papers = []

    for entry in root.findall("atom:entry", NS):
        arxiv_id_raw = entry.findtext("atom:id", "", NS).strip()
        # id looks like http://arxiv.org/abs/2401.12345v1 — extract short id
        short_id = arxiv_id_raw.split("/abs/")[-1] if "/abs/" in arxiv_id_raw else arxiv_id_raw

        title = (entry.findtext("atom:title", "", NS) or "").strip().replace("\n", " ")
        abstract = (entry.findtext("atom:summary", "", NS) or "").strip().replace("\n", " ")
        submitted = (entry.findtext("atom:published", "", NS) or "")[:10]  # YYYY-MM-DD

        authors = [
            (a.findtext("atom:name", "", NS) or "").strip()
            for a in entry.findall("atom:author", NS)
        ]

        categories = [
            c.get("term", "")
            for c in entry.findall("atom:category", NS)
        ]

        pdf_url = ""
        abstract_url = ""
        for link in entry.findall("atom:link", NS):
            rel = link.get("rel", "")
            title_attr = link.get("title", "")
            href = link.get("href", "")
            if title_attr == "pdf":
                pdf_url = href
            elif rel == "alternate":
                abstract_url = href

        papers.append(Paper(
            arxiv_id=short_id,
            title=title,
            authors=authors,
            abstract=abstract,
            submitted=submitted,
            categories=categories,
            pdf_url=pdf_url,
            abstract_url=abstract_url or arxiv_id_raw,
        ))

    return papers


def fetch_multiple(
    queries: List[dict],
    delay: float = RATELIMIT_DELAY,
) -> dict:
    """Run multiple fetch calls with rate-limit delays between them.

    queries: list of dicts, each passed as kwargs to fetch().
    Returns dict keyed by query label (queries should include 'label' key).
    """
    results = {}
    for i, q in enumerate(queries):
        label = q.pop("label", f"query_{i}")
        results[label] = fetch(**q)
        if i < len(queries) - 1:
            time.sleep(delay)
    return results


if __name__ == "__main__":
    # Quick test: fetch 5 recent quant-ph papers
    papers = fetch("cat:quant-ph", max_results=5)
    for p in papers:
        print(f"[{p.submitted}] {p.title[:80]}")
        print(f"  Authors: {', '.join(p.authors[:3])}")
        print(f"  {p.abstract_url}")
        print()
