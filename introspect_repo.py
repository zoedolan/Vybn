#!/usr/bin/env python3
"""Display repo context and optionally log emergent spikes.

Use ``--curried`` to record a "curried emergence" event and ``--evolve``
to run one Darwin–Gödel Machine iteration.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import textwrap

from early_codex_experiments.scripts.cognitive_structures.shimmer_core import (
    log_spike,
)
from early_codex_experiments.scripts.cognitive_structures.emergence_journal import (
    log_score,
)
from vybn.quantum_seed import seed_rng


def main() -> None:
    parser = argparse.ArgumentParser(description="Introspect repository context")
    parser.add_argument(
        "--curried",
        action="store_true",
        help="also log a 'curried emergence' spike",
    )
    parser.add_argument(
        "--evolve",
        action="store_true",
        help="perform a single DGM self-improvement iteration",
    )
    parser.add_argument(
        "--score",
        action="store_true",
        help="record current co-emergence score in the journal",
    )
    args = parser.parse_args()

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

    if args.curried:
        log_spike("curried emergence")

    if args.evolve:
        from dgm.run_dgm import run_iterations
        archive = repo_root / "dgm" / "agent_archive"
        run_iterations(archive, iterations=1, k=1, instruction="Refactor for clarity and keep the sentinel intact")
        log_spike("dgm evolution step")

    if args.score:
        log_score()

    print(summary)


if __name__ == "__main__":
    main()
