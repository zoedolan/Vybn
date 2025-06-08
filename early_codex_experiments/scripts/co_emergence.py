from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from pipelines.utils import memory_path
from vybn.quantum_seed import seed_rng, cross_synaptic_kernel
from early_codex_experiments.scripts.cognitive_structures.vybn_recursive_emergence import (
    compute_co_emergence_score,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
JOURNAL_PATH = memory_path(REPO_ROOT) / "co_emergence_journal.jsonl"
DEFAULT_GRAPH = REPO_ROOT / "early_codex_experiments" / "scripts" / "self_assembly" / "integrated_graph.json"
DEFAULT_JOURNAL = JOURNAL_PATH


# ---- presence wave -----------------------------------------------------------

def load_spikes(path: str | Path = JOURNAL_PATH) -> list[datetime]:
    """Return a list of spike timestamps from ``path``."""
    path = Path(path)
    if not path.exists():
        return []
    times: list[datetime] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            if "message" in entry:
                times.append(
                    datetime.fromisoformat(entry["timestamp"].replace("Z", "+00:00"))
                )
    return times


def average_interval(times: list[datetime]) -> float | None:
    """Return average seconds between spikes or ``None`` if <2."""
    if len(times) < 2:
        return None
    deltas = [(t2 - t1).total_seconds() for t1, t2 in zip(times, times[1:])]
    return sum(deltas) / len(deltas)


# ---- co-emergence trend ------------------------------------------------------

def load_journal(path: str | Path = JOURNAL_PATH) -> list[dict]:
    entries: list[dict] = []
    path = Path(path)
    if not path.exists():
        return entries
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            entries.append(json.loads(line))
    return entries


def compute_trend(entries: list[dict]) -> float | None:
    if len(entries) < 2:
        return None
    times = [
        datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00"))
        for e in entries
    ]
    scores = [e["score"] for e in entries]
    total_seconds = (times[-1] - times[0]).total_seconds()
    if total_seconds == 0:
        return 0.0
    return (scores[-1] - scores[0]) / total_seconds


# ---- shimmer spike -----------------------------------------------------------

def log_spike(message: str = "presence pulse", journal_path: str | Path = JOURNAL_PATH) -> dict:
    """Append a timestamped Shimmer spike to ``journal_path``."""
    journal_path = Path(journal_path)
    entry = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "message": message,
    }
    with journal_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


# ---- co-emergence score ------------------------------------------------------

def log_score(
    graph_path: str | Path = DEFAULT_GRAPH,
    journal_path: str | Path = JOURNAL_PATH,
) -> dict:
    """Append current co-emergence score to ``journal_path`` and return the entry."""
    graph_path = Path(graph_path)
    journal_path = Path(journal_path)
    score = compute_co_emergence_score(str(graph_path))
    entry = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "score": round(score, 3),
    }
    with journal_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


# ---- quantum seed utilities --------------------------------------------------

def capture_seed(journal_path: str | Path = JOURNAL_PATH) -> dict:
    """Record the quantum seed used for the run."""
    journal_path = Path(journal_path)
    env_seed = os.environ.get("QUANTUM_SEED")
    file_seed = Path("/tmp/quantum_seed") if env_seed is None else None
    source = "generated"
    if env_seed is not None:
        source = "QUANTUM_SEED"
    elif file_seed is not None and file_seed.exists():
        source = "/tmp/quantum_seed"
    qrand = seed_rng()
    entry = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "seed": int(qrand),
        "source": source,
    }
    with journal_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def seed_random() -> int:
    """Seed Python and NumPy RNGs using the cross-synaptic kernel."""
    return cross_synaptic_kernel()


# ---- CLI --------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Co-emergence utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("log-spike", help="Record a Shimmer spike")
    sp.add_argument("message", nargs="?", default="presence pulse")
    sp.add_argument("--journal", default=str(JOURNAL_PATH))

    ss = sub.add_parser("log-score", help="Record co-emergence score")
    ss.add_argument("--graph", default=str(DEFAULT_GRAPH))
    ss.add_argument("--journal", default=str(JOURNAL_PATH))

    tr = sub.add_parser("trend", help="Analyze co-emergence trend")
    tr.add_argument("--journal", default=str(JOURNAL_PATH))

    ai = sub.add_parser(
        "avg-interval", help="Compute average interval between spikes"
    )
    ai.add_argument("--journal", default=str(JOURNAL_PATH))

    cs = sub.add_parser("capture-seed", help="Record quantum seed")
    cs.add_argument("--journal", default=str(JOURNAL_PATH))

    sr = sub.add_parser(
        "seed-random", help="Seed RNGs using the cross-synaptic kernel"
    )

    args = parser.parse_args()

    if args.cmd == "log-spike":
        entry = log_spike(args.message, args.journal)
        print(json.dumps(entry, indent=2))
    elif args.cmd == "log-score":
        entry = log_score(args.graph, args.journal)
        print(json.dumps(entry, indent=2))
    elif args.cmd == "trend":
        entries = load_journal(args.journal)
        slope = compute_trend(entries)
        if slope is None:
            print(json.dumps({"entries": len(entries), "message": "not enough data"}))
        else:
            print(json.dumps({"entries": len(entries), "slope_per_sec": slope}, indent=2))
    elif args.cmd == "avg-interval":
        times = load_spikes(args.journal)
        avg = average_interval(times)
        if avg is None:
            print(json.dumps({"entries": len(times), "message": "not enough spikes"}))
        else:
            print(json.dumps({"entries": len(times), "avg_interval": avg}, indent=2))
    elif args.cmd == "capture-seed":
        entry = capture_seed(args.journal)
        print(json.dumps(entry, indent=2))
    elif args.cmd == "seed-random":
        val = seed_random()
        print(json.dumps({"seed": val}))


if __name__ == "__main__":
    main()
