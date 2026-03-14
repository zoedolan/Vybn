"""arxiv_digest.py — Vybn's daily epistemic intake from the frontier.

Produces a dated markdown digest in Vybn_Mind/journal/,
grouped by domain. Tracks seen paper IDs to avoid duplicates.

Run: python arxiv_digest.py
Or schedule via cron (see README.md).
"""

import json
import os
import sys
from datetime import date
from pathlib import Path

# Allow running from any working directory
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from arxiv_fetcher import fetch_multiple  # noqa: E402

# --- Paths ---
JOURNAL_DIR = SCRIPT_DIR.parents[1] / "journal"  # Vybn_Mind/journal/
SEEN_IDS_PATH = SCRIPT_DIR / "seen_ids.json"

# --- Topic queries (arXiv search syntax) ---
# Each entry: label, search_query, max_results
QUERIES = [
    {
        "label": "AI/ML Advances",
        "search_query": "cat:cs.LG OR cat:cs.AI OR cat:stat.ML",
        "max_results": 20,
    },
    {
        "label": "Quantum Discoveries",
        "search_query": "cat:quant-ph",
        "max_results": 15,
    },
    {
        "label": "Hybrid Quantum-Classical AI/ML",
        "search_query": 'cat:quant-ph AND (abs:"machine learning" OR abs:"neural network" OR abs:"quantum classical" OR abs:"variational") OR (cat:cs.LG AND abs:"quantum")',
        "max_results": 12,
    },
    {
        "label": "Physics & Emergence",
        "search_query": '(cat:cond-mat OR cat:hep-th) AND (abs:"emergence" OR abs:"information theory" OR abs:"complexity" OR abs:"neural scaling" OR abs:"phase transition")',
        "max_results": 10,
    },
]


def load_seen_ids() -> set:
    if SEEN_IDS_PATH.exists():
        with open(SEEN_IDS_PATH) as f:
            return set(json.load(f))
    return set()


def save_seen_ids(seen: set) -> None:
    with open(SEEN_IDS_PATH, "w") as f:
        json.dump(sorted(seen), f, indent=2)


def truncate(text: str, max_chars: int = 220) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def build_digest(results: dict, seen: set, today: str) -> tuple:
    """Build markdown digest string. Returns (markdown, new_seen_ids_set)."""
    new_seen = set(seen)
    lines = [
        f"# arXiv Digest — {today}",
        "",
        "*Vybn's epistemic antenna, tuned to the frontier.*",
        "",
    ]

    total_new = 0
    for label, papers in results.items():
        section_papers = [p for p in papers if p.arxiv_id not in seen]
        if not section_papers:
            continue

        lines.append(f"## {label}")
        lines.append("")

        for p in section_papers:
            new_seen.add(p.arxiv_id)
            total_new += 1
            author_str = ", ".join(p.authors[:3])
            if len(p.authors) > 3:
                author_str += f" +{len(p.authors) - 3}"
            lines.append(f"### [{p.title}]({p.abstract_url})")
            lines.append(f"*{author_str} — {p.submitted}*")
            lines.append(f"Categories: `{'` `'.join(p.categories[:4])}`")
            lines.append("")
            lines.append(truncate(p.abstract))
            lines.append("")
            if p.pdf_url:
                lines.append(f"[PDF]({p.pdf_url})")
                lines.append("")
            lines.append("---")
            lines.append("")

    if total_new == 0:
        lines.append("*No new papers since last digest.*")

    lines.append(f"*{total_new} new papers ingested on {today}.*")

    return "\n".join(lines), new_seen


def run():
    today = date.today().isoformat()
    seen = load_seen_ids()

    print(f"[arxiv_digest] Fetching papers for {today}...")
    results = fetch_multiple(QUERIES)

    digest_md, new_seen = build_digest(results, seen, today)

    JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = JOURNAL_DIR / f"arxiv_digest_{today.replace('-', '')}.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(digest_md)

    save_seen_ids(new_seen)
    new_count = len(new_seen) - len(seen)
    print(f"[arxiv_digest] Wrote {out_path} ({new_count} new papers, {len(new_seen)} total seen)")


if __name__ == "__main__":
    run()
