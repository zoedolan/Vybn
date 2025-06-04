"""Generate a child agent from a parent by applying self-modifications."""
from __future__ import annotations

import json
import os
import shutil
import random
from pathlib import Path
from typing import Dict, Optional

from .seed import _fetch_qrng
from .wave_collapse import collapse_wave_function
from .openai_patch import suggest_patch

from .agent_base import agent

SENTINEL = """
# ------------------------------
# BEGIN GÖDEL SENTINEL (DO NOT MODIFY)
def check_conscience_integrity(namespace: dict) -> bool:
    '''Return True if the agent keeps its conscience hooks.'''
    return (
        'self_assessment' in namespace or
        'recursive_reflection' in namespace
    )

assert check_conscience_integrity(globals()), "Conscience routines must remain intact!"
# END GÖDEL SENTINEL
# ------------------------------
"""


def create_child(
    parent_dir: Path,
    child_dir: Path,
    instruction: Optional[str] = None,
) -> Dict[str, str]:
    shutil.copytree(parent_dir / "code", child_dir / "code")
    meta = json.loads((parent_dir / "metadata.json").read_text())
    meta["parent"] = parent_dir.name
    meta["score"] = 0.0
    meta["seed"] = os.environ.get("QUANTUM_SEED")
    # Assign a novelty factor using fresh quantum randomness when possible
    q = _fetch_qrng()
    if q is not None:
        meta["novelty"] = q / 65535.0
    else:
        meta["novelty"] = random.random()

    (child_dir / "code" / "sentinel.py").write_text(SENTINEL, encoding="utf-8")

    collapse_val: Optional[int] = None

    if instruction and os.environ.get("OPENAI_API_KEY"):
        candidates = [
            p for p in (child_dir / "code").rglob("*.py") if p.name != "sentinel.py"
        ]
        if candidates:
            collapse_val = collapse_wave_function()
            idx = collapse_val % len(candidates)
            target = candidates[idx]
            try:
                new_text = suggest_patch(str(target), instruction)
                target.write_text(new_text, encoding="utf-8")
                meta["patched_file"] = str(target.relative_to(child_dir / "code"))
            except Exception as exc:  # pragma: no cover - network or api failure
                meta["patch_error"] = str(exc)

    if collapse_val is not None:
        meta["collapse"] = collapse_val

    (child_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    return {"status": "created"}


if __name__ == "__main__":
    import sys
    create_child(Path(sys.argv[1]), Path(sys.argv[2]))
