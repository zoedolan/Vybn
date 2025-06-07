"""Collapse the repository into ``Vybn_Volume_IV.md``.

This module runs the existing pipeline, appends its outputs to the
autobiography file and removes everything except a small set of
preserved paths. It is intentionally destructive and should be used
with care.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Set

from vybn.quantum_seed import seed_rng
from .pipeline_runner import main as run_pipeline
from . import EXCLUDE_PATHS


def collapse_repo(repo_root: Path) -> None:
    """Summarize repo content into Vybn_Volume_IV.md and prune other files."""
    seed_rng()
    run_pipeline()

    volume = repo_root / 'Vybn_Volume_IV.md'
    parts = []
    if volume.exists():
        parts.append(volume.read_text(encoding='utf-8'))

    def append_if_exists(label: str, path: Path) -> None:
        if path.exists():
            parts.append(f"## {label}\n")
            parts.append(path.read_text(encoding='utf-8'))

    append_if_exists('Distilled Corpus', repo_root / 'distilled_corpus.txt')
    append_if_exists('History Excerpts', repo_root / 'history_excerpt.txt')
    append_if_exists('Token Summary', repo_root / 'token_summary.txt')
    append_if_exists('WVWHM Count', repo_root / 'wvwhm_count.txt')
    append_if_exists('Emergence Graph', repo_root / 'emergence_graph.json')

    volume.write_text('\n\n'.join(parts), encoding='utf-8')

    keep: Set[Path] = {repo_root / str(p) for p in EXCLUDE_PATHS}
    keep.add(volume)
    keep.add(repo_root / "pipelines")
    keep.add(repo_root / ".git")

    for path in repo_root.iterdir():
        if path in keep:
            continue
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                path.unlink()
            except Exception:
                pass


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    collapse_repo(repo_root)


if __name__ == '__main__':
    main()
