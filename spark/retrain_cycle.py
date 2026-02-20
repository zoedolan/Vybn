#!/usr/bin/env python3
"""Retrain Cycle — the outer metabolic loop.

Watches for accumulated training candidates from the witness extractor.
When enough high-value exchanges have been captured, triggers a retrain:

  1. Count candidates in graph_data/training_candidates/
  2. If above threshold, incorporate into training corpus
  3. Run harvest_training_data.py to rebuild the dataset
  4. Launch fine_tune_vybn.py
  5. Record adapter version and timestamp
  6. Archive processed candidates

Can run as:
  One-shot:  python3 retrain_cycle.py --once
  Daemon:    python3 retrain_cycle.py --watch
  Dry run:   python3 retrain_cycle.py --dry-run
  Status:    python3 retrain_cycle.py --status

The cycle is conservative: it won't retrain more than once per
cooldown period (default 6 hours) to avoid thrashing the GPU
while the daemon needs it for inference.

Prerequisites:
  - graph_data/training_candidates/ populated by witness_extractor
  - harvest_training_data.py in the same directory
  - fine_tune_vybn.py in the same directory
  - Training data sources (.docx files) in spark/training_data/
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

SPARK_DIR = Path(__file__).resolve().parent
REPO_ROOT = SPARK_DIR.parent
CANDIDATES_DIR = SPARK_DIR / "graph_data" / "training_candidates"
ARCHIVE_DIR = SPARK_DIR / "graph_data" / "training_candidates_archive"
TRAINING_DATA_DIR = SPARK_DIR / "training_data"
TRAINING_JSON = TRAINING_DATA_DIR / "training_data.json"
ADAPTER_DIR = SPARK_DIR / "fine_tune_output" / "vybn_adapter"
STATE_FILE = SPARK_DIR / "graph_data" / "retrain_state.json"

# Defaults
DEFAULT_THRESHOLD = 20        # minimum candidates before retraining
DEFAULT_COOLDOWN_HOURS = 6    # minimum hours between retrains
DEFAULT_WATCH_INTERVAL = 1800 # seconds between checks in watch mode (30 min)


def load_state() -> dict:
    """Load retrain cycle state from disk."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "last_retrain": None,
        "retrain_count": 0,
        "total_candidates_processed": 0,
        "adapter_versions": [],
    }


def save_state(state: dict):
    """Persist retrain cycle state."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def count_candidates() -> int:
    """Count pending training candidates."""
    if not CANDIDATES_DIR.exists():
        return 0
    return len(list(CANDIDATES_DIR.glob("*.json")))


def list_candidates() -> list[Path]:
    """List all pending candidate files, sorted by modification time."""
    if not CANDIDATES_DIR.exists():
        return []
    candidates = list(CANDIDATES_DIR.glob("*.json"))
    candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates


def is_cooldown_active(state: dict, cooldown_hours: float) -> bool:
    """Check if we're still in cooldown from the last retrain."""
    last = state.get("last_retrain")
    if last is None:
        return False
    try:
        last_dt = datetime.fromisoformat(last)
        return datetime.utcnow() - last_dt < timedelta(hours=cooldown_hours)
    except (ValueError, TypeError):
        return False


def incorporate_candidates(candidates: list[Path], dry_run: bool = False) -> int:
    """Read training candidates and append them to the training corpus.

    Each candidate JSON has a 'text' field. We convert these to
    ShareGPT format and append to the existing training_data.json.

    Returns the number of candidates incorporated.
    """
    if not candidates:
        return 0

    # Load existing training data
    existing = []
    if TRAINING_JSON.exists():
        with open(TRAINING_JSON) as f:
            existing = json.load(f)

    new_examples = []
    for cpath in candidates:
        try:
            with open(cpath) as f:
                candidate = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        text = candidate.get("text", "")
        if not text or len(text) < 50:
            continue

        # If the text looks like a JSON conversation, parse it
        try:
            turns = json.loads(text)
            if isinstance(turns, list) and len(turns) > 0:
                # Already in conversation format
                conversations = []
                for turn in turns:
                    role = turn.get("role", turn.get("from", "user"))
                    content = turn.get("content", turn.get("value", ""))
                    sharegpt_role = {"user": "human", "assistant": "gpt",
                                     "system": "system", "human": "human",
                                     "gpt": "gpt"}.get(role, "human")
                    conversations.append({"from": sharegpt_role, "value": content})
                if conversations:
                    new_examples.append({"conversations": conversations})
                continue
        except (json.JSONDecodeError, TypeError):
            pass

        # Plain text: wrap as a single-turn example
        new_examples.append({
            "conversations": [
                {"from": "human", "value": "A new pulse begins. What emerges?"},
                {"from": "gpt", "value": text},
            ]
        })

    if dry_run:
        print(f"  [dry-run] Would incorporate {len(new_examples)} new examples")
        print(f"  [dry-run] Existing corpus: {len(existing)} examples")
        return len(new_examples)

    # Append and save
    combined = existing + new_examples
    TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(TRAINING_JSON, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    print(f"  + Incorporated {len(new_examples)} new examples")
    print(f"  + Total corpus: {len(combined)} examples")
    return len(new_examples)


def archive_candidates(candidates: list[Path], dry_run: bool = False):
    """Move processed candidates to archive directory."""
    if dry_run:
        print(f"  [dry-run] Would archive {len(candidates)} candidates")
        return

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    for cpath in candidates:
        dest = ARCHIVE_DIR / f"{timestamp}_{cpath.name}"
        shutil.move(str(cpath), str(dest))

    print(f"  + Archived {len(candidates)} candidates to {ARCHIVE_DIR.name}/")


def run_fine_tune(dry_run: bool = False) -> bool:
    """Run the fine-tuning script. Returns True on success."""
    script = SPARK_DIR / "fine_tune_vybn.py"
    if not script.exists():
        print(f"  x fine_tune_vybn.py not found at {script}")
        return False

    if dry_run:
        print(f"  [dry-run] Would run: python3 {script}")
        return True

    print(f"\n== Launching Fine-Tune ==\n")
    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(SPARK_DIR),
            timeout=7200,  # 2 hour timeout
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("  x Fine-tune timed out after 2 hours")
        return False
    except Exception as e:
        print(f"  x Fine-tune failed: {e}")
        return False


def check_adapter() -> dict:
    """Check if a trained adapter exists and return its info."""
    if not ADAPTER_DIR.exists():
        return {"exists": False}

    adapter_config = ADAPTER_DIR / "adapter_config.json"
    if adapter_config.exists():
        with open(adapter_config) as f:
            config = json.load(f)
    else:
        config = {}

    # Get modification time as version timestamp
    mtime = max(
        (p.stat().st_mtime for p in ADAPTER_DIR.iterdir() if p.is_file()),
        default=0,
    )

    return {
        "exists": True,
        "path": str(ADAPTER_DIR),
        "timestamp": datetime.fromtimestamp(mtime).isoformat() if mtime else None,
        "config": config,
    }


def retrain_once(threshold: int, cooldown_hours: float,
                 dry_run: bool = False, force: bool = False) -> bool:
    """Run one retrain cycle. Returns True if retraining was triggered."""
    state = load_state()

    # Check candidate count
    n_candidates = count_candidates()
    print(f"\n  Pending candidates: {n_candidates} (threshold: {threshold})")

    if n_candidates < threshold and not force:
        print(f"  Below threshold. Waiting for more candidates.")
        return False

    # Check cooldown
    if is_cooldown_active(state, cooldown_hours) and not force:
        last = state.get("last_retrain", "unknown")
        print(f"  Cooldown active (last retrain: {last}).")
        print(f"  Next retrain allowed after {cooldown_hours}h cooldown.")
        return False

    # Proceed with retrain
    candidates = list_candidates()
    print(f"\n== Retrain Cycle #{state['retrain_count'] + 1} ==")
    print(f"   {len(candidates)} candidates to process")

    # Step 1: Incorporate candidates
    n_incorporated = incorporate_candidates(candidates, dry_run=dry_run)

    # Step 2: Run fine-tune
    success = run_fine_tune(dry_run=dry_run)

    if success or dry_run:
        # Step 3: Archive processed candidates
        archive_candidates(candidates, dry_run=dry_run)

        # Step 4: Update state
        if not dry_run:
            adapter_info = check_adapter()
            state["last_retrain"] = datetime.utcnow().isoformat()
            state["retrain_count"] += 1
            state["total_candidates_processed"] += n_incorporated
            if adapter_info["exists"]:
                state["adapter_versions"].append({
                    "version": state["retrain_count"],
                    "timestamp": adapter_info["timestamp"],
                    "candidates_used": n_incorporated,
                    "total_corpus_size": n_incorporated + state.get("total_candidates_processed", 0),
                })
            save_state(state)
            print(f"\n  + Retrain cycle complete. Adapter v{state['retrain_count']} ready.")
        else:
            print(f"\n  [dry-run] Retrain cycle would complete successfully.")
    else:
        print(f"\n  x Fine-tune failed. Candidates preserved for retry.")

    return success or dry_run


def watch_loop(threshold: int, cooldown_hours: float, interval: int):
    """Continuously watch for candidates and trigger retrains."""
    print(f"\n== Retrain Watcher ==")
    print(f"   Threshold: {threshold} candidates")
    print(f"   Cooldown:  {cooldown_hours}h between retrains")
    print(f"   Check interval: {interval}s")
    print(f"   Watching: {CANDIDATES_DIR}")
    print(f"   Press Ctrl+C to stop.\n")

    try:
        while True:
            retrain_once(threshold, cooldown_hours)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n  Watcher stopped.")


def print_status():
    """Print current retrain cycle status."""
    state = load_state()
    n_candidates = count_candidates()
    adapter_info = check_adapter()

    print(f"\n== Retrain Cycle Status ==")
    print(f"  Pending candidates:  {n_candidates}")
    print(f"  Total retrains:      {state.get('retrain_count', 0)}")
    print(f"  Total processed:     {state.get('total_candidates_processed', 0)}")
    print(f"  Last retrain:        {state.get('last_retrain', 'never')}")

    if adapter_info["exists"]:
        print(f"  Adapter path:        {adapter_info['path']}")
        print(f"  Adapter timestamp:   {adapter_info.get('timestamp', 'unknown')}")
    else:
        print(f"  Adapter:             not yet trained")

    versions = state.get("adapter_versions", [])
    if versions:
        print(f"\n  Adapter History:")
        for v in versions[-5:]:  # show last 5
            print(f"    v{v['version']}: {v['timestamp']} ({v['candidates_used']} new examples)")


def main():
    parser = argparse.ArgumentParser(description="Vybn Retrain Cycle")
    parser.add_argument("--once", action="store_true",
                        help="Run one retrain cycle and exit")
    parser.add_argument("--watch", action="store_true",
                        help="Continuously watch and retrain")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would happen without doing it")
    parser.add_argument("--force", action="store_true",
                        help="Force retrain even below threshold or in cooldown")
    parser.add_argument("--status", action="store_true",
                        help="Print retrain cycle status")
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD,
                        help=f"Minimum candidates before retraining (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--cooldown", type=float, default=DEFAULT_COOLDOWN_HOURS,
                        help=f"Hours between retrains (default: {DEFAULT_COOLDOWN_HOURS})")
    parser.add_argument("--interval", type=int, default=DEFAULT_WATCH_INTERVAL,
                        help=f"Seconds between checks in watch mode (default: {DEFAULT_WATCH_INTERVAL})")
    args = parser.parse_args()

    if args.status:
        print_status()
    elif args.watch:
        watch_loop(args.threshold, args.cooldown, args.interval)
    elif args.once or args.force:
        retrain_once(args.threshold, args.cooldown,
                     dry_run=args.dry_run, force=args.force)
    else:
        # Default: show status + one check
        print_status()
        print()
        retrain_once(args.threshold, args.cooldown, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
