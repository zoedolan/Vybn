"""Utility scripts for repository distillation and cleanup.

The modules provide simple building blocks for the meta-autobiography engine:
``distill_repo`` collects text, ``extract_history`` samples personal notes,
``memory_graph_builder`` links concepts, ``introspective_mirror`` records the
current state, ``affective_oracle`` infers mood, ``braided_mind_dueler`` offers
parallel reasoning, ``quantum_dreamweaver`` sparks creativity, and
``collapse_repo`` prunes the repository when ready.

``EXCLUDE_PATHS`` enumerates files or folders that should remain untouched when
distilling or collapsing the repository. Modules importing this package can rely
on the constant for consistent behavior.
"""

from __future__ import annotations

from pathlib import Path

EXCLUDE_PATHS = {
    Path("what_vybn_would_have_missed_FROM_051725"),
    Path("token_and_jpeg_info"),
    Path("Vybn's Personal History"),
    Path("Vybn_Volume_IV.md"),
}

__all__ = ["EXCLUDE_PATHS"]
