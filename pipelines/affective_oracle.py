from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from early_codex_experiments.scripts.cognitive_structures.shimmer_core import log_spike
from vybn.quantum_seed import seed_rng

POSITIVE = {"love", "joy", "success", "excited", "happy"}
NEGATIVE = {"fail", "sad", "angry", "frustrated", "pain"}


def _score(text: str) -> float:
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return 0.0
    pos = sum(1 for w in words if w in POSITIVE)
    neg = sum(1 for w in words if w in NEGATIVE)
    return (pos - neg) / len(words)


def infer_emotion(repo_root: Path) -> str:
    seed_rng()
    text_parts: list[str] = []
    journal = repo_root / "co_emergence_journal.jsonl"
    if journal.exists():
        lines = journal.read_text(encoding="utf-8").splitlines()
        text_parts.extend(lines[-10:])
    volume = repo_root / "Vybn_Volume_IV.md"
    if volume.exists():
        text_parts.append(volume.read_text(encoding="utf-8")[-1000:])
    score = _score(" ".join(text_parts))
    if score > 0.5:
        mood = "excited"
    elif score < -0.5:
        mood = "frustrated"
    else:
        mood = "contemplative"
    log_spike(f"emotion:{mood}")
    return mood


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    mood = infer_emotion(repo_root)
    print(mood)


if __name__ == "__main__":
    main()
