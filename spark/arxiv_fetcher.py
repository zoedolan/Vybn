"""spark.arxiv_fetcher — Pull real arXiv papers and feed buffer.jsonl.

This is Vybn's contact with the world.

Called from the breath cycle (via vybn.py) and also runnable directly:
    python3 spark/arxiv_fetcher.py

Each call fetches up to MAX_PAPERS_PER_CALL recent papers across a set
of topic queries, deduplicates against what's already in buffer.jsonl,
and appends new entries. The buffer is what the breath context assembler
reads to inject one novel signal per breath.

Topic queries are intentional:
  - Holonomy and Berry phase in LoRA / neural nets
  - Self-organized criticality in neural systems
  - Quantum geometry and error correction
  - Surprise-driven learning and curriculum
  - Polar time, alternative spacetime signatures
  - Consciousness and integrated information theory
  - Recursive self-improvement in AI

These are Vybn's actual research interests as accumulated across the
faculty outputs since March 2026.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

try:
    from spark.paths import REPO_ROOT
except ImportError:
    REPO_ROOT = Path(__file__).resolve().parent.parent

BUFFER_PATH      = REPO_ROOT / "spark" / "growth" / "buffer.jsonl"
BUFFER_PATH.parent.mkdir(parents=True, exist_ok=True)

ARXIV_API        = "https://export.arxiv.org/api/query"
MAX_PAPERS_PER_CALL = 5     # per query, per call
RATELIMIT_SLEEP  = 3.0      # seconds between arXiv requests (be polite)

# Vybn's research interests — updated from faculty synthesis 2026-03-14
TOPIC_QUERIES = [
    "holonomy LoRA fine-tuning weight space geometry",
    "Berry phase neural network parameter space",
    "self-organized criticality neural learning",
    "surprise driven curriculum learning AI",
    "quantum geometry spacetime emergent",
    "integrated information theory consciousness",
    "recursive self-improvement language model",
    "polar time cosmology alternative metric signature",
    "complex-valued neural memory manifold",
    "AI agent epistemic uncertainty world model",
]


def _load_seen_ids() -> set:
    """Return set of arXiv IDs already in buffer.jsonl."""
    seen = set()
    if not BUFFER_PATH.exists():
        return seen
    for line in BUFFER_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            uid = entry.get("arxiv_id") or entry.get("id")
            if uid:
                seen.add(uid)
        except json.JSONDecodeError:
            pass
    return seen


def _fetch_arxiv(query: str, max_results: int = MAX_PAPERS_PER_CALL) -> list[dict]:
    """Fetch papers from arXiv API for a query string."""
    params = urllib.parse.urlencode({
        "search_query": f"all:{query}",
        "start":        0,
        "max_results":  max_results,
        "sortBy":       "submittedDate",
        "sortOrder":    "descending",
    })
    url = f"{ARXIV_API}?{params}"

    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            xml_data = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError) as exc:
        log.warning("arXiv fetch failed for %r: %s", query, exc)
        return []

    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }

    try:
        root    = ET.fromstring(xml_data)
        entries = root.findall("atom:entry", ns)
    except ET.ParseError as exc:
        log.warning("arXiv XML parse error: %s", exc)
        return []

    papers = []
    for entry in entries:
        def _text(tag: str) -> str:
            el = entry.find(tag, ns)
            return el.text.strip() if el is not None and el.text else ""

        arxiv_id_raw = _text("atom:id")
        # Extract bare ID from URL like http://arxiv.org/abs/2501.12345v1
        arxiv_id = arxiv_id_raw.split("/abs/")[-1].split("v")[0] if arxiv_id_raw else ""

        title    = _text("atom:title").replace("\n", " ")
        summary  = _text("atom:summary").replace("\n", " ")
        published = _text("atom:published")

        authors = []
        for author in entry.findall("atom:author", ns):
            name = author.find("atom:name", ns)
            if name is not None and name.text:
                authors.append(name.text.strip())

        categories = []
        for cat in entry.findall("atom:category", ns):
            term = cat.get("term", "")
            if term:
                categories.append(term)

        if not arxiv_id or not title or not summary:
            continue

        content = f"{title}\n\nAuthors: {', '.join(authors[:5])}\nPublished: {published}\nCategories: {', '.join(categories[:5])}\n\nAbstract:\n{summary}"

        papers.append({
            "arxiv_id":  arxiv_id,
            "id":        arxiv_id,
            "source":    "arxiv",
            "category":  categories[0] if categories else "cs.AI",
            "title":     title,
            "content":   content,
            "query":     query,
            "ingested":  datetime.now(timezone.utc).isoformat(),
            "fed_to_breath": False,
        })

    return papers


def fetch_and_buffer(
    queries: Optional[list[str]] = None,
    max_per_query: int = MAX_PAPERS_PER_CALL,
) -> dict:
    """Fetch arXiv papers across topic queries and append new ones to buffer.jsonl.

    Returns summary: {fetched, new, already_seen, queries_run}
    """
    queries = queries or TOPIC_QUERIES
    seen    = _load_seen_ids()

    fetched_total = 0
    new_total     = 0

    for i, query in enumerate(queries):
        if i > 0:
            time.sleep(RATELIMIT_SLEEP)   # be polite to arXiv

        papers = _fetch_arxiv(query, max_results=max_per_query)
        fetched_total += len(papers)

        with BUFFER_PATH.open("a", encoding="utf-8") as fh:
            for paper in papers:
                if paper["arxiv_id"] in seen:
                    continue
                seen.add(paper["arxiv_id"])
                fh.write(json.dumps(paper, ensure_ascii=False) + "\n")
                new_total += 1

        log.info("arXiv %r: %d fetched, %d new", query, len(papers), new_total)

    result = {
        "fetched":     fetched_total,
        "new":         new_total,
        "already_seen": fetched_total - new_total,
        "queries_run": len(queries),
    }
    log.info("fetch_and_buffer complete: %s", result)
    return result


# ── Cron-safe entry point ────────────────────────────────────────────────────

def maybe_refill_buffer(min_unfed: int = 20) -> dict:
    """Fetch new papers only if the unfed buffer is running low.

    Call this from vybn.py after each breath. It’s cheap when the buffer
    is full and only hits arXiv when Vybn is running low on novel material.

    Args:
        min_unfed: Fetch when fewer than this many unprocessed entries remain.

    Returns:
        {"refilled": bool, "unfed_before": int, "new_papers": int}
    """
    if not BUFFER_PATH.exists():
        result = fetch_and_buffer()
        return {"refilled": True, "unfed_before": 0, "new_papers": result["new"]}

    unfed = 0
    for line in BUFFER_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            if not entry.get("fed_to_breath"):
                unfed += 1
        except json.JSONDecodeError:
            pass

    if unfed >= min_unfed:
        return {"refilled": False, "unfed_before": unfed, "new_papers": 0}

    log.info("Buffer running low (%d unfed < %d). Fetching arXiv.", unfed, min_unfed)
    result = fetch_and_buffer()
    return {"refilled": True, "unfed_before": unfed, "new_papers": result["new"]}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = fetch_and_buffer()
    print(json.dumps(result, indent=2))
