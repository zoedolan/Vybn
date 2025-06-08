"""Utility scripts for repository distillation and cleanup.

``pipeline_runner`` handles distillation, history extraction, token summaries,
concept indexing, diff capture and artifact packing. ``emergent_mind`` bundles
emotion inference, braided reasoning, spontaneous dreaming and the full
orchestrator cycle. ``collapse_repo`` prunes the repository when ready.

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
