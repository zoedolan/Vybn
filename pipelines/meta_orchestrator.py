from __future__ import annotations

from pathlib import Path

from early_codex_experiments.scripts.cognitive_structures.shimmer_core import log_spike
from .introspective_mirror import gather_state
from .affective_oracle import infer_emotion
from .braided_mind_dueler import answer_query
from .quantum_dreamweaver import generate_dream
from vybn.quantum_seed import seed_rng


def run_cycle(repo_root: Path) -> None:
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


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_cycle(repo_root)


if __name__ == "__main__":
    main()
