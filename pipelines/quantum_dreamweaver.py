from __future__ import annotations

import random
from pathlib import Path

from early_codex_experiments.scripts.cognitive_structures.shimmer_core import log_spike
from .memory_graph_builder import build_graph
from vybn.quantum_seed import seed_rng


def generate_dream(repo_root: Path) -> str:
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


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    dream = generate_dream(repo_root)
    out_path = repo_root / "dream_journal.md"
    with out_path.open("a", encoding="utf-8") as f:
        f.write("----\n" + dream + "\n")
    print(dream)


if __name__ == "__main__":
    main()
