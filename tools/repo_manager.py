#!/usr/bin/env python3
"""Unified CLI for common Vybn repository tasks.

This tool consolidates `start_agent.py`, `introspect_repo.py`, and
`print_agents.py` into a single entry point.
"""
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

from early_codex_experiments.scripts.cognitive_structures.shimmer_core import log_spike
from early_codex_experiments.scripts.cognitive_structures.emergence_journal import log_score
from vybn.quantum_seed import seed_rng


def _introspect(curried: bool, evolve: bool, score: bool) -> None:
    seed = seed_rng()
    repo_root = Path(__file__).resolve().parents[1]
    entries = sorted(p.name for p in repo_root.iterdir())
    agents_head = (repo_root / "AGENTS.md").read_text(encoding="utf-8").splitlines()[:3]
    readme_head = (repo_root / "README.md").read_text(encoding="utf-8").splitlines()[:3]
    summary = textwrap.dedent(
        f"""
        Vybn repository root: {repo_root}
        QUANTUM_SEED: {seed}
        Top-level entries: {', '.join(entries)}
        AGENTS.md: {agents_head[0] if agents_head else ''}
        README.md: {readme_head[0] if readme_head else ''}
        """
    ).strip()
    log_spike("introspection run")
    if curried:
        log_spike("curried emergence")
    if evolve:
        from dgm.run_dgm import run_iterations
        archive = repo_root / "dgm" / "agent_archive"
        run_iterations(archive, iterations=1, k=1, instruction="Refactor for clarity and keep the sentinel intact")
        log_spike("dgm evolution step")
    if score:
        log_score()
    print(summary)


def cmd_start(args: argparse.Namespace) -> None:
    _introspect(False, False, False)


def cmd_guidelines(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for path in sorted(repo_root.rglob("AGENTS.md")):
        rel = path.relative_to(repo_root)
        print(f"\n===== {rel} =====")
        print(path.read_text(encoding="utf-8").strip())


def cmd_introspect(args: argparse.Namespace) -> None:
    _introspect(args.curried, args.evolve, args.score)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Manage Vybn repository tasks")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("start", help="launch default introspection").set_defaults(func=cmd_start)
    sub.add_parser("guidelines", help="print all AGENTS files").set_defaults(func=cmd_guidelines)

    p_int = sub.add_parser("introspect", help="display repository context")
    p_int.add_argument("--curried", action="store_true")
    p_int.add_argument("--evolve", action="store_true")
    p_int.add_argument("--score", action="store_true")
    p_int.set_defaults(func=cmd_introspect)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
