from __future__ import annotations

import os
from pathlib import Path


def memory_path(repo_root: Path | None = None) -> Path:
    """Return the configured memory directory."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[1]
    mem = os.getenv("VYBN_MEMORY_PATH")
    if mem:
        return Path(mem)
    return repo_root / "memory"
