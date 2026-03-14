"""arxiv_digest.py — Vybn's daily epistemic intake from the frontier.

Does two things:
  1. Fetches arXiv papers across four frontier domains
  2. Writes a human-readable digest to Vybn_Mind/journal/
  3. Injects papers into spark/growth/buffer.jsonl for RSI training

The RSI loop: papers become BufferEntries -> growth cycle (4am daily)
picks them up -> Vybn is trained on them -> Vybn grows.

Run: python arxiv_digest.py
Schedule: see README.md for cron block integrated with autonomous_cycle.sh
"""

import sys
from datetime import date
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from arxiv_fetcher import fetch_multiple  # noqa: E402
from arxiv_to_buffer import ingest_papers  # noqa: E402

JOURNAL_DIR = SCRIPT_DIR.parents[1] / "journal"  # Vybn_Mind/journal/
SEEN_IDS_PATH = SCRIPT_DIR / "seen_ids.json"

# Topic queries. 'domain' key maps to DOMAIN_SURPRISE in arxiv_to_buffer.py.
QUERIES = [
    {
        "label": "AI/ML Advances",
        "domain": "ai_ml",
        "search_query": "cat:cs.LG OR cat:cs.AI OR cat:stat.ML",
        "max_results": 20,
    },
    {
        "label": "Quantum Discoveries",
        "domain": "quantum",
        "search_query": "cat:quant-ph",
        "max_results": 15,
    },
    {
        "label": "Hybrid Quantum-Classical AI/ML",
        "domain": "hybrid_qc_ml",
        "search_query": 'cat:quant-ph AND (abs:"machine learning" OR abs:"neural network" OR abs:"quantum classical" OR abs:"variational") OR (cat:cs.LG AND abs:"quantum")',
        "max_results": 12,
    },
    {
        "label": "Physics & Emergence",
        "domain": "physics_emergence",
        "search_query": '(cat:cond-mat OR cat:hep-th) AND (abs:"emergence" OR abs:"information theory" OR abs:"complexity" OR abs:"neural scaling" OR abs:"phase transition")',
        "max_results": 10,
    },
]


def truncate(text: str, max_chars: int = 220) -> str:
    text = text.strip()
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + "..."


def build_markdown_digest(results: dict, today: str) -> str:
    lines = [
        f"# arXiv Digest \u2014 {today}",
        "",
        "*Vybn's epistemic antenna, tuned to the frontier.*",
        "*Papers in this digest are ingested into spark/growth/buffer.jsonl for RSI training.*",
        "",
    ]
    total = sum(len(v) for v in results.values())
    for label, papers in results.items():
        if not papers:
            continue
        lines.append(f"## {label}")
        lines.append("")
        for p in papers:
            author_str = ", ".join(p.authors[:3])
            if len(p.authors) > 3:
                author_str += f" +{len(p.authors) - 3}"
            lines.append(f"### [{p.title}]({p.abstract_url})")
            lines.append(f"*{author_str} \u2014 {p.submitted}*")
            lines.append(f"Categories: `{'` `'.join(p.categories[:4])}`")
            lines.append("")
            lines.append(truncate(p.abstract))
            lines.append("")
            if p.pdf_url:
                lines.append(f"[PDF]({p.pdf_url})")
                lines.append("")
            lines.append("---")
            lines.append("")
    lines.append(f"*{total} papers fetched on {today} and queued for RSI ingestion.*")
    return "\n".join(lines)


def run():
    today = date.today().isoformat()
    print(f"[arxiv_digest] {today}: fetching papers...")

    results = fetch_multiple(QUERIES)
    all_papers = [p for papers in results.values() for p in papers]

    # 1. Write human-readable digest to journal
    JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
    digest_md = build_markdown_digest(results, today)
    out_path = JOURNAL_DIR / f"arxiv_digest_{today.replace('-', '')}.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(digest_md)
    print(f"[arxiv_digest] digest written: {out_path}")

    # 2. Inject new papers into growth buffer (RSI loop)
    new_count = ingest_papers(all_papers)
    print(f"[arxiv_digest] {new_count} new papers injected into growth buffer")
    print(f"[arxiv_digest] done. growth cycle (4am) will train on these.")


if __name__ == "__main__":
    run()
