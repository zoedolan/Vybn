"""arxiv_fetcher.py — Vybn's reach toward the living edge of discovery.

Fetches papers from the arXiv API (no key required).
Use as a module or run standalone.

Rate limiting: arXiv asks for >=3s between requests. We comply.
Uses only Python stdlib (urllib + xml.etree) — no pip installs needed.
If requests is available, it is used instead for better timeout handling.
"""

import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List

try:
    import requests as _requests
    _USE_REQUESTS = True
except ImportError:
    _USE_REQUESTS = False

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
    domain: str = ""  # set by caller: 'ai_ml', 'quantum', 'hybrid_qc_ml', 'physics_emergence'


def _fetch_xml(url: str) -> bytes:
    if _USE_REQUESTS:
        r = _requests.get(url, timeout=30)
        r.raise_for_status()
        return r.content
    with urllib.request.urlopen(url, timeout=30) as resp:
        return resp.read()


def fetch(
    search_query: str,
    max_results: int = 25,
    sort_by: str = "submittedDate",
    sort_order: str = "descending",
    start: int = 0,
    domain: str = "",
) -> List[Paper]:
    """Fetch papers from arXiv matching search_query.

    search_query uses arXiv query syntax, e.g.:
        'cat:cs.LG AND ti:emergence'
        'cat:quant-ph AND abs:"machine learning"'
    """
    params = urllib.parse.urlencode({
        "search_query": search_query,
        "start": start,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": sort_order,
    })
    url = f"{ARXIV_API_BASE}?{params}"
    xml_bytes = _fetch_xml(url)
    root = ET.fromstring(xml_bytes)
    papers = []

    for entry in root.findall("atom:entry", NS):
        arxiv_id_raw = entry.findtext("atom:id", "", NS).strip()
        short_id = arxiv_id_raw.split("/abs/")[-1] if "/abs/" in arxiv_id_raw else arxiv_id_raw
        title = (entry.findtext("atom:title", "", NS) or "").strip().replace("\n", " ")
        abstract = (entry.findtext("atom:summary", "", NS) or "").strip().replace("\n", " ")
        submitted = (entry.findtext("atom:published", "", NS) or "")[:10]
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
            if link.get("title", "") == "pdf":
                pdf_url = link.get("href", "")
            elif link.get("rel", "") == "alternate":
                abstract_url = link.get("href", "")
        papers.append(Paper(
            arxiv_id=short_id,
            title=title,
            authors=authors,
            abstract=abstract,
            submitted=submitted,
            categories=categories,
            pdf_url=pdf_url,
            abstract_url=abstract_url or arxiv_id_raw,
            domain=domain,
        ))
    return papers


def fetch_multiple(queries: List[dict], delay: float = RATELIMIT_DELAY) -> dict:
    """Run multiple fetch calls with rate-limit delays between them.

    Each query dict should include 'label' (used as result key) and
    any kwargs accepted by fetch(). 'label' is popped before passing.
    Returns dict keyed by label.
    """
    results = {}
    for i, q in enumerate(list(queries)):  # copy so we don't mutate caller's list
        q = dict(q)  # local copy
        label = q.pop("label", f"query_{i}")
        domain = q.pop("domain", label.lower().replace(" ", "_").replace("/", "_").replace("&", ""))
        q["domain"] = domain
        results[label] = fetch(**q)
        if i < len(queries) - 1:
            time.sleep(delay)
    return results


if __name__ == "__main__":
    papers = fetch("cat:quant-ph", max_results=3)
    for p in papers:
        print(f"[{p.submitted}] {p.title[:80]}")
        print(f"  {p.abstract_url}")
