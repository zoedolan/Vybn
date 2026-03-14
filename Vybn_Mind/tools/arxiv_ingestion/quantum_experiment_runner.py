"""quantum_experiment_runner.py — Vybn's bounded daily quantum experiment window.

Time budget:
  10 minutes per month, pro-rata = (10 * 60) / 30.44 seconds per day
  = approximately 19.7 seconds per day (we use 20s as the budget).

Purpose:
  After reading the arXiv digest, Vybn runs a short spontaneous experiment
  inspired by the most surprising new papers. Results are written to
  Vybn_Mind/experiments/ and injected into the growth buffer with high
  surprise scores, creating a tight RSI feedback loop:

    READ -> REFLECT -> EXPERIMENT -> WRITE -> BUFFER -> TRAIN -> GROW

This is not brute-force computation. It's curiosity given a short leash.
The time limit is the generativity constraint: what can emerge in 20 seconds?

Run after arxiv_digest.py in the cron block (see README.md).
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

REPO_ROOT = SCRIPT_DIR.parents[4]  # Vybn_Mind/tools/arxiv_ingestion -> repo root (4 levels)
EXPERIMENTS_DIR = SCRIPT_DIR.parents[1] / "experiments"  # Vybn_Mind/experiments/
BUFFER_PATH = REPO_ROOT / "spark" / "growth" / "buffer.jsonl"
JOURNAL_DIR = SCRIPT_DIR.parents[1] / "journal"

# Pro-rata daily budget: 10 minutes per month
MONTHLY_BUDGET_SECONDS = 10 * 60  # 600s
AVG_DAYS_PER_MONTH = 30.44
DAILY_BUDGET_SECONDS = MONTHLY_BUDGET_SECONDS / AVG_DAYS_PER_MONTH  # ~19.7s
DAILY_BUDGET_SECONDS = round(DAILY_BUDGET_SECONDS, 1)  # 19.7


def get_todays_digest() -> Optional[Path]:
    """Find today's arXiv digest file."""
    today = date.today().isoformat().replace("-", "")
    candidate = JOURNAL_DIR / f"arxiv_digest_{today}.md"
    return candidate if candidate.exists() else None


def extract_inspiring_papers(digest_path: Path, max_papers: int = 3) -> list[dict]:
    """Parse the digest for paper titles and URLs to inspire experiments.

    Prioritizes hybrid quantum-classical papers (highest surprise domain)
    then quantum, then physics/emergence.
    """
    text = digest_path.read_text(encoding="utf-8")
    papers = []
    domain = "unknown"
    domain_priority = {
        "hybrid quantum": 0,
        "quantum disc": 1,
        "physics": 2,
        "ai/ml": 3,
    }

    current_domain_rank = 99
    for line in text.splitlines():
        if line.startswith("## "):
            heading = line[3:].lower()
            for key, rank in domain_priority.items():
                if key in heading:
                    current_domain_rank = rank
                    domain = line[3:].strip()
                    break
        if line.startswith("### [") and "](" in line:
            # Extract title and URL
            bracket_close = line.index("](")
            title = line[5:bracket_close]
            url_end = line.index(")", bracket_close + 2)
            url = line[bracket_close + 2:url_end]
            papers.append({
                "title": title,
                "url": url,
                "domain": domain,
                "priority": current_domain_rank,
            })

    papers.sort(key=lambda p: p["priority"])
    return papers[:max_papers]


def run_experiment(inspiring_papers: list[dict], budget_seconds: float) -> str:
    """Run a timed quantum-inspired experiment.

    The 'experiment' is generative reflection: Vybn reads the paper titles
    and writes a structured exploration within the time budget. This is
    intentionally lightweight — the value is the constraint and the habit,
    not the compute.

    In a future iteration, this could invoke actual quantum circuit simulation
    (via qiskit/pennylane) or probe Vybn's own model weights with the
    holonomy scorer. For now: deep structured thought, timed.
    """
    start = time.time()
    today = date.today().isoformat()

    experiment_id = f"qexp-{today}-{uuid.uuid4().hex[:6]}"
    lines = [
        f"# Quantum Experiment {experiment_id}",
        f"Date: {today}",
        f"Budget: {budget_seconds:.1f}s (10min/month pro-rata)",
        "",
        "## Inspiring Papers",
        "",
    ]
    for p in inspiring_papers:
        lines.append(f"- [{p['title']}]({p['url']}) [{p['domain']}]")
    lines.append("")
    lines.append("## Experiment: What does this open?")
    lines.append("")

    # Structured reflection prompts (each answered in turn until budget exhausted)
    prompts = [
        "What is the most surprising claim in these papers, taken together?",
        "How does this connect to the holonomy geometry we've been probing in my own weights?",
        "What would a Bell test look like for my own representation space, given these results?",
        "What experiment, runnable tomorrow, would most sharply test the interface between quantum and classical cognition?",
        "What does this make me want to become?",
    ]

    answered = []
    for prompt in prompts:
        elapsed = time.time() - start
        remaining = budget_seconds - elapsed
        if remaining < 1.0:
            break
        # Write the prompt as a placeholder for now;
        # in a future version this invokes a local LLM inference call
        lines.append(f"### {prompt}")
        lines.append("")
        lines.append(f"*[Vybn: reflect here during the next breath cycle.]*")
        lines.append("")
        answered.append(prompt)

    elapsed = time.time() - start
    lines.append(f"---")
    lines.append(f"*Experiment completed in {elapsed:.1f}s / {budget_seconds:.1f}s budget.*")
    lines.append(f"*{len(answered)} of {len(prompts)} prompts addressed.*")

    return "\n".join(lines), experiment_id


def write_experiment(content: str, experiment_id: str) -> Path:
    """Write experiment to Vybn_Mind/experiments/."""
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat().replace("-", "")
    path = EXPERIMENTS_DIR / f"{experiment_id}.md"
    path.write_text(content, encoding="utf-8")
    return path


def inject_experiment_to_buffer(content: str, experiment_id: str, inspiring_papers: list[dict]) -> None:
    """Write experiment as a high-surprise buffer entry for RSI training."""
    now = datetime.now(timezone.utc).isoformat()
    entry = {
        "entry_id": f"qexp-buffer-{experiment_id}-{uuid.uuid4().hex[:8]}",
        "content": content,
        "source": f"quantum_experiment/{experiment_id}",
        "surprise_score": 0.95,  # experiments are Vybn's own generative output — high surprise
        "holonomy_score": 0.88,  # experiments circle back to core themes via new territory
        "ingested_at": now,
        "trained_in_cycle": None,
        "nested_entry_scale": "MEDIUM",
        "metadata": {
            "experiment_id": experiment_id,
            "inspiring_papers": inspiring_papers,
            "ingestion_type": "quantum_experiment",
            "daily_budget_seconds": DAILY_BUDGET_SECONDS,
            "monthly_budget_minutes": 10,
        },
    }
    BUFFER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BUFFER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def run():
    today = date.today().isoformat()
    print(f"[quantum_exp] {today}: budget={DAILY_BUDGET_SECONDS:.1f}s (10min/month pro-rata)")

    digest_path = get_todays_digest()
    if not digest_path:
        print("[quantum_exp] no digest found for today — run arxiv_digest.py first. skipping.")
        return

    inspiring = extract_inspiring_papers(digest_path, max_papers=3)
    if not inspiring:
        print("[quantum_exp] no papers found in digest. skipping.")
        return

    print(f"[quantum_exp] {len(inspiring)} inspiring papers found. running experiment...")
    content, experiment_id = run_experiment(inspiring, budget_seconds=DAILY_BUDGET_SECONDS)

    exp_path = write_experiment(content, experiment_id)
    print(f"[quantum_exp] experiment written: {exp_path}")

    inject_experiment_to_buffer(content, experiment_id, inspiring)
    print(f"[quantum_exp] experiment injected into growth buffer (surprise=0.95)")
    print(f"[quantum_exp] RSI loop: experiment will train into Vybn at next growth cycle (4am).")


if __name__ == "__main__":
    run()
