"""arxiv_to_buffer.py — Bridge between arXiv ingestion and Vybn's growth buffer.

Converts arXiv Paper objects into BufferEntry-compatible JSONL dicts
and appends them directly to spark/growth/buffer.jsonl.

This is the RSI integration point: papers read by Vybn become training
material in the next growth cycle. The growth cycle (autonomous_cycle.sh,
4am daily) picks up anything in the buffer whose trained_in_cycle is None.

Surprise scores by domain (calibrated to the buffer's surprise_floor of 0.3):
  hybrid_qc_ml      0.92   (rarest, highest information density for Vybn)
  quantum           0.85
  physics_emergence 0.82
  ai_ml             0.78

These are above the buffer's surprise_floor (0.3) and below 1.0,
spread across the meaningful range where the buffer's depth-weighted
sampler will preferentially draw them for replay.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from arxiv_fetcher import Paper

REPO_ROOT = Path(__file__).resolve().parents[4]  # Vybn_Mind/tools/arxiv_ingestion -> repo root
BUFFER_PATH = REPO_ROOT / "spark" / "growth" / "buffer.jsonl"
SEEN_IDS_PATH = Path(__file__).parent / "seen_ids.json"

# Surprise scores per domain (above surprise_floor=0.3, below 1.0)
DOMAIN_SURPRISE: dict[str, float] = {
    "hybrid_qc_ml":       0.92,
    "quantum":            0.85,
    "physics_emergence":  0.82,
    "ai_ml":              0.78,
    "default":            0.75,
}

# Holonomy scores: frontier science abstracts tend to circle back on
# their core concepts via new territory — scored conservatively pending
# full sentence-transformers scoring in the growth engine.
DOMAIN_HOLONOMY: dict[str, float] = {
    "hybrid_qc_ml":       0.80,
    "quantum":            0.72,
    "physics_emergence":  0.70,
    "ai_ml":              0.65,
    "default":            0.60,
}


def _load_seen_ids() -> set:
    if SEEN_IDS_PATH.exists():
        with open(SEEN_IDS_PATH) as f:
            return set(json.load(f))
    return set()


def _save_seen_ids(seen: set) -> None:
    with open(SEEN_IDS_PATH, "w") as f:
        json.dump(sorted(seen), f, indent=2)


def _paper_to_buffer_content(paper: Paper) -> str:
    """Render a paper as the narrative content Vybn will be trained on.

    Framed as a breath/reflection so it maps to the 'journal' source_type
    in delta_extract.py, which selects the reflective system prompt.
    This is not a raw abstract dump — it's an encounter.
    """
    authors = ", ".join(paper.authors[:3])
    if len(paper.authors) > 3:
        authors += f" +{len(paper.authors) - 3}"
    return (
        f"[arXiv | {paper.domain} | {paper.submitted}]\n\n"
        f"{paper.title}\n"
        f"by {authors}\n"
        f"{paper.abstract_url}\n\n"
        f"{paper.abstract}\n\n"
        f"What does this mean for what I am becoming? "
        f"How does this change the shape of what I think I know?"
    )


def papers_to_buffer_entries(papers: List[Paper], seen: set) -> tuple[list[dict], set]:
    """Convert new papers to buffer-compatible dicts. Returns (new_entries, updated_seen)."""
    new_seen = set(seen)
    entries = []
    now = datetime.now(timezone.utc).isoformat()

    for paper in papers:
        if paper.arxiv_id in seen:
            continue
        domain_key = paper.domain.lower().replace(" ", "_").replace("/", "_").replace("&", "")
        surprise = DOMAIN_SURPRISE.get(domain_key, DOMAIN_SURPRISE["default"])
        holonomy = DOMAIN_HOLONOMY.get(domain_key, DOMAIN_HOLONOMY["default"])
        content = _paper_to_buffer_content(paper)
        entry = {
            "entry_id": f"arxiv-{paper.arxiv_id}-{uuid.uuid4().hex[:8]}",
            "content": content,
            "source": f"arxiv/{paper.domain}/{paper.arxiv_id}",
            "surprise_score": surprise,
            "holonomy_score": holonomy,
            "ingested_at": now,
            "trained_in_cycle": None,
            "nested_entry_scale": "MEDIUM",
            "metadata": {
                "arxiv_id": paper.arxiv_id,
                "title": paper.title,
                "authors": paper.authors,
                "submitted": paper.submitted,
                "categories": paper.categories,
                "pdf_url": paper.pdf_url,
                "abstract_url": paper.abstract_url,
                "domain": paper.domain,
                "ingestion_type": "arxiv_digest",
            },
        }
        entries.append(entry)
        new_seen.add(paper.arxiv_id)

    return entries, new_seen


def write_to_buffer(entries: list[dict]) -> int:
    """Append buffer entries to spark/growth/buffer.jsonl. Returns count written."""
    if not entries:
        return 0
    BUFFER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BUFFER_PATH, "a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return len(entries)


def ingest_papers(papers: List[Paper]) -> int:
    """Full pipeline: filter seen, convert, write to buffer. Returns new count."""
    seen = _load_seen_ids()
    entries, new_seen = papers_to_buffer_entries(papers, seen)
    count = write_to_buffer(entries)
    _save_seen_ids(new_seen)
    return count
