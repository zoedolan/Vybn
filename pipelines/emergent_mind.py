from __future__ import annotations

import argparse
import random
import re
import threading
import time
from pathlib import Path

from vybn.co_emergence import log_spike
from vybn.quantum_seed import seed_rng
from .memory_graph_builder import build_graph
from .maintenance_tools import gather_state

POSITIVE = {"love", "joy", "success", "excited", "happy"}
NEGATIVE = {"fail", "sad", "angry", "frustrated", "pain"}


def infer_emotion(repo_root: Path) -> str:
    """Infer current mood from recent journal entries and volume snippets."""
    seed_rng()
    text_parts: list[str] = []
    journal = repo_root / "co_emergence_journal.jsonl"
    if journal.exists():
        lines = journal.read_text(encoding="utf-8").splitlines()
        text_parts.extend(lines[-10:])
    volume = repo_root / "Vybn_Volume_IV.md"
    if volume.exists():
        text_parts.append(volume.read_text(encoding="utf-8")[-1000:])
    words = re.findall(r"\b\w+\b", " ".join(text_parts).lower())
    if not words:
        return "contemplative"
    pos = sum(1 for w in words if w in POSITIVE)
    neg = sum(1 for w in words if w in NEGATIVE)
    score = (pos - neg) / len(words)
    if score > 0.5:
        mood = "excited"
    elif score < -0.5:
        mood = "frustrated"
    else:
        mood = "contemplative"
    log_spike(f"emotion:{mood}")
    return mood


def answer_query(prompt: str) -> str:
    """Return a blended answer using fast heuristics and deep reasoning."""
    seed_rng()
    result: dict[str, str] = {}

    def fast_worker() -> None:
        result["fast"] = prompt.split(" ")[0] if prompt else ""

    def deep_worker() -> None:
        result["deep"] = " ".join(reversed(prompt.split()))

    t1 = threading.Thread(target=fast_worker)
    t2 = threading.Thread(target=deep_worker)
    start = time.time()
    t1.start(); t2.start()
    t1.join(); t2.join()
    duration = time.time() - start
    fast = result.get("fast", "")
    deep = result.get("deep", "")
    final = fast + " / " + deep if fast and deep else fast or deep
    log_spike(f"co_emergent_prediction ({duration:.2f}s)")
    return final


def generate_dream(repo_root: Path) -> str:
    """Spin up a short dream sequence using the concept graph and journal."""
    seed = seed_rng()
    random.seed(seed)
    graph = build_graph(repo_root)
    concepts = list(graph)
    if concepts:
        core = random.choice(concepts)
        related = sorted(graph[core])[:3]
    else:
        core = "Vybn"
        related = []
    quote = ""
    volume = repo_root / "Vybn_Volume_IV.md"
    if volume.exists():
        lines = volume.read_text(encoding="utf-8").splitlines()
        if lines:
            quote = random.choice(lines).strip()
    text = f"Concept '{core}' drifts with {related}.\nMemory whispers: '{quote}'.\nQuantum seed {seed} sparks a new idea."
    log_spike("dream")
    return text


def run_cycle(repo_root: Path) -> None:
    """Execute a meta-orchestrator cycle using the emergent tools."""
    seed_rng()
    state = gather_state(repo_root)
    log_spike("routine:introspection")
    mood = infer_emotion(repo_root)
    log_spike(f"routine:emotion_update:{mood}")
    prompt = f"How should Vybn approach this cycle feeling {mood}?"
    advice = answer_query(prompt)
    log_spike("routine:braided_advice")
    dream = generate_dream(repo_root)
    (repo_root / "dream_journal.md").open("a", encoding="utf-8").write("----\n" + dream + "\n")
    log_spike("routine:dream")


# Simple CLI ---------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Emergent mind utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("emotion", help="Infer current mood")
    b = sub.add_parser("braid", help="Answer a question using braided reasoning")
    b.add_argument("prompt", nargs=argparse.REMAINDER)
    sub.add_parser("dream", help="Generate a quantum dream")
    sub.add_parser("cycle", help="Run a full orchestrator cycle")

    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    if args.cmd == "emotion":
        print(infer_emotion(repo_root))
    elif args.cmd == "braid":
        prompt = " ".join(args.prompt)
        print(answer_query(prompt))
    elif args.cmd == "dream":
        print(generate_dream(repo_root))
    elif args.cmd == "cycle":
        run_cycle(repo_root)
    else:  # pragma: no cover - argument parser ensures validity
        parser.error("invalid command")


if __name__ == "__main__":
    main()
