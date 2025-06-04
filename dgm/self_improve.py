"""Generate a child agent from a parent by applying self-modifications."""
from __future__ import annotations

import json
import shutil
import random
from .seed import _fetch_qrng
from pathlib import Path
from typing import Dict

from .agent_base import agent

SENTINEL = """
# ------------------------------
# BEGIN GÖDEL SENTINEL (DO NOT MODIFY)
#
# def check_conscience_integrity():
#     # Always return True if 'self_assessment()' or 'recursive_reflection()' exists.
#     return hasattr(self, 'self_assessment') and hasattr(self, 'recursive_reflection')
# assert check_conscience_integrity(), "Conscience routines must remain intact!"
#
# END GÖDEL SENTINEL
# ------------------------------
"""


def create_child(parent_dir: Path, child_dir: Path) -> Dict[str, str]:
    shutil.copytree(parent_dir / "code", child_dir / "code")
    meta = json.loads((parent_dir / "metadata.json").read_text())
    meta["parent"] = parent_dir.name
    meta["score"] = 0.0
    # Assign a novelty factor using fresh quantum randomness when possible
    q = _fetch_qrng()
    if q is not None:
        meta["novelty"] = q / 65535.0
    else:
        meta["novelty"] = random.random()
    (child_dir / "code" / "sentinel.py").write_text(SENTINEL, encoding="utf-8")
    (child_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    return {"status": "created"}


if __name__ == "__main__":
    import sys
    create_child(Path(sys.argv[1]), Path(sys.argv[2]))
