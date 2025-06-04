"""Select parent agents based on sigmoid-scaled performance and novelty."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import List



DEF_NOVELTY = 0.1


def load_metadata(agent_dir: Path) -> dict:
    return json.loads((agent_dir / "metadata.json").read_text())


def parent_candidates(archive_dir: Path) -> List[Path]:
    return [p for p in archive_dir.iterdir() if (p / "metadata.json").exists()]


def score(agent_meta: dict) -> float:
    return agent_meta.get("score", 0.0)


def novelty_bonus(agent_meta: dict) -> float:
    return agent_meta.get("novelty", DEF_NOVELTY)


def weight(alpha: float, novelty: float) -> float:
    # sigmoid scaled performance
    sig = 1 / (1 + math.exp(-12 * (alpha - 0.5)))
    return sig * novelty


def select_parents(archive_dir: Path, k: int = 1) -> List[Path]:
    candidates = []
    for p in parent_candidates(archive_dir):
        meta = load_metadata(p)
        if meta.get("score", 0) < 1.0:
            candidates.append((p, weight(score(meta), novelty_bonus(meta))))
    if not candidates:
        return []
    total = sum(w for _, w in candidates)
    probs = [w / total for _, w in candidates]
    chosen = []
    for _ in range(min(k, len(candidates))):
        val = random.random()
        accum = 0
        for (path, w), prob in zip(candidates, probs):
            accum += prob
            if val <= accum:
                chosen.append(path)
                break
    return chosen


if __name__ == "__main__":
    import sys
    archive = Path(sys.argv[1])
    for p in select_parents(archive, k=1):
        print(p)
