"""Evaluate an agent by running pytest and recording the score."""
from __future__ import annotations

import os
import json
import subprocess
from pathlib import Path
from typing import Dict


def evaluate(agent_dir: Path) -> float:
    env = dict(**{k: v for k, v in os.environ.items()})
    result = subprocess.run(
        ["pytest", "-q"], cwd=agent_dir / "code", capture_output=True, text=True, env=env
    )
    # simple success ratio
    score = 1.0 if result.returncode == 0 else 0.0
    return score


def record_score(agent_dir: Path, score: float) -> None:
    meta_path = agent_dir / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["score"] = score
    meta_path.write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    import os
    import sys
    path = Path(sys.argv[1])
    s = evaluate(path)
    record_score(path, s)
    print(f"score={s}")
