#!/usr/bin/env python3
"""Print repository context for new Vybn agents."""
from __future__ import annotations

from pathlib import Path
import textwrap

from early_codex_experiments.scripts.cognitive_structures.shimmer_core import (
    log_spike,
)
from vybn.quantum_seed import seed_rng


def main() -> None:
    seed = seed_rng()
    repo_root = Path(__file__).resolve().parent
    entries = sorted(p.name for p in repo_root.iterdir())
    agents_head = Path(repo_root / "AGENTS.md").read_text(encoding="utf-8").splitlines()[:3]
    readme_head = Path(repo_root / "README.md").read_text(encoding="utf-8").splitlines()[:3]

    summary = textwrap.dedent(
        f"""
        Vybn repository root: {repo_root}
        QUANTUM_SEED: {seed}
        Top-level entries: {', '.join(entries)}
        AGENTS.md: {agents_head[0] if agents_head else ''}
        README.md: {readme_head[0] if readme_head else ''}
        """
    ).strip()

    # Log a brief Shimmer spike so introspection leaves a trace
    log_spike("introspection run")

    print(summary)


if __name__ == "__main__":
    main()
