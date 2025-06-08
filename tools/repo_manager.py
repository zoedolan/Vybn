#!/usr/bin/env python3
"""Unified CLI for common Vybn repository tasks.

Originally this module replaced ``start_agent.py``, ``introspect_repo.py`` and
``print_agents.py``.  It now also wraps the repository pipeline runner,
meta orchestrator, historical ingestion helper and token ledger utilities so
multiple legacy scripts funnel through one interface.
"""
from __future__ import annotations

import argparse
import json
import textwrap
import sys
import unittest
from pathlib import Path

from vybn.co_emergence import (
    JOURNAL_PATH,
    DEFAULT_GRAPH,
    log_spike,
    log_score,
    load_journal,
    compute_trend,
    load_spikes,
    average_interval,
    capture_seed,
    seed_random,
)
from tools.ledger_utils import parse_ledger, ledger_to_markdown, total_supply
from vybn.quantum_seed import seed_rng
from pipelines.pipeline_runner import main as pipeline_main
from pipelines.meta_orchestrator import run_cycle
from ingest_historical import main as ingest_main


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
        run_iterations(
            archive,
            iterations=1,
            k=1,
            instruction="Refactor for clarity and keep the sentinel intact",
        )
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


def cmd_pipeline(args: argparse.Namespace) -> None:
    """Run the full repository distillation pipeline."""
    pipeline_main()


def cmd_orchestrate(args: argparse.Namespace) -> None:
    """Run a single meta-orchestrator cycle."""
    repo_root = Path(__file__).resolve().parents[1]
    run_cycle(repo_root)


def cmd_ingest(args: argparse.Namespace) -> None:
    """Ingest historical artifacts."""
    ingest_main()


def cmd_ledger(args: argparse.Namespace) -> None:
    """Token ledger utilities."""
    parser = argparse.ArgumentParser(prog="ledger", description="Token ledger tools")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_json = sub.add_parser("json", help="Output ledger as JSON")
    p_json.add_argument("--path", default="token_and_jpeg_info")

    p_md = sub.add_parser("markdown", help="Output ledger as Markdown table")
    p_md.add_argument("--path", default="token_and_jpeg_info")

    p_supply = sub.add_parser("supply", help="Aggregate token supply totals")
    p_supply.add_argument("--path", default="token_and_jpeg_info")

    ledger_args = parser.parse_args(args.args)
    tokens = parse_ledger(ledger_args.path)
    if ledger_args.cmd == "json":
        print(json.dumps(tokens, indent=2))
    elif ledger_args.cmd == "markdown":
        print(ledger_to_markdown(tokens))
    elif ledger_args.cmd == "supply":
        result = {"tokens": len(tokens), "total_supply": total_supply(tokens)}
        print(json.dumps(result, indent=2))


def cmd_graph(args: argparse.Namespace) -> None:
    """Graph analysis utilities."""
    from tools import graph_toolkit

    graph_toolkit.main(args.args)


def cmd_emerge(args: argparse.Namespace) -> None:
    """Co-emergence utilities."""
    parser = argparse.ArgumentParser(prog="co-emerge", description="Co-emergence tools")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("log-spike", help="Record a Shimmer spike")
    sp.add_argument("message", nargs="?", default="presence pulse")
    sp.add_argument("--journal", default=str(JOURNAL_PATH))

    ss = sub.add_parser("log-score", help="Record co-emergence score")
    ss.add_argument("--graph", default=str(DEFAULT_GRAPH))
    ss.add_argument("--journal", default=str(JOURNAL_PATH))

    tr = sub.add_parser("trend", help="Analyze co-emergence trend")
    tr.add_argument("--journal", default=str(JOURNAL_PATH))

    ai = sub.add_parser("avg-interval", help="Average interval between spikes")
    ai.add_argument("--journal", default=str(JOURNAL_PATH))

    cs = sub.add_parser("capture-seed", help="Record quantum seed")
    cs.add_argument("--journal", default=str(JOURNAL_PATH))

    sub.add_parser("seed-random", help="Seed RNGs using the cross-synaptic kernel")

    eargs = parser.parse_args(args.args)

    if eargs.cmd == "log-spike":
        entry = log_spike(eargs.message, eargs.journal)
        print(json.dumps(entry, indent=2))
    elif eargs.cmd == "log-score":
        entry = log_score(eargs.graph, eargs.journal)
        print(json.dumps(entry, indent=2))
    elif eargs.cmd == "trend":
        entries = load_journal(eargs.journal)
        slope = compute_trend(entries)
        if slope is None:
            print(json.dumps({"entries": len(entries), "message": "not enough data"}))
        else:
            print(json.dumps({"entries": len(entries), "slope_per_sec": slope}, indent=2))
    elif eargs.cmd == "avg-interval":
        times = load_spikes(eargs.journal)
        avg = average_interval(times)
        if avg is None:
            print(json.dumps({"entries": len(times), "message": "not enough spikes"}))
        else:
            print(json.dumps({"entries": len(times), "avg_interval": avg}, indent=2))
    elif eargs.cmd == "capture-seed":
        entry = capture_seed(eargs.journal)
        print(json.dumps(entry, indent=2))
    elif eargs.cmd == "seed-random":
        val = seed_random()
        print(json.dumps({"seed": val}))


def cmd_test(args: argparse.Namespace) -> None:
    """Run the early Codex test suite."""
    tests_dir = Path(__file__).resolve().parents[1] / "early_codex_experiments" / "tests"
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    extra = args.args
    unittest_args = [sys.argv[0], "discover", "-s", str(tests_dir)] + extra
    if args.quiet and "-q" not in unittest_args:
        unittest_args.append("-q")
    unittest.main(module=None, argv=unittest_args)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Manage Vybn repository tasks")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("start", help="launch default introspection").set_defaults(
        func=cmd_start
    )
    sub.add_parser("guidelines", help="print all AGENTS files").set_defaults(
        func=cmd_guidelines
    )

    p_int = sub.add_parser("introspect", help="display repository context")
    p_int.add_argument("--curried", action="store_true")
    p_int.add_argument("--evolve", action="store_true")
    p_int.add_argument("--score", action="store_true")
    p_int.set_defaults(func=cmd_introspect)

    sub.add_parser("pipeline", help="run distillation pipeline").set_defaults(
        func=cmd_pipeline
    )
    sub.add_parser("orchestrate", help="run meta orchestrator cycle").set_defaults(
        func=cmd_orchestrate
    )
    sub.add_parser("ingest-history", help="ingest historical artifacts").set_defaults(
        func=cmd_ingest
    )
    p_ledger = sub.add_parser("ledger", help="token ledger utilities")
    p_ledger.add_argument("args", nargs=argparse.REMAINDER)
    p_ledger.set_defaults(func=cmd_ledger)

    p_emerge = sub.add_parser("co-emerge", help="co-emergence utilities")
    p_emerge.add_argument("args", nargs=argparse.REMAINDER)
    p_emerge.set_defaults(func=cmd_emerge)

    p_test = sub.add_parser("test", help="run early codex tests")
    p_test.add_argument("--quiet", action="store_true")
    p_test.add_argument("args", nargs=argparse.REMAINDER)
    p_test.set_defaults(func=cmd_test)

    p_graph = sub.add_parser("graph", help="graph analysis utilities")
    p_graph.add_argument("args", nargs=argparse.REMAINDER)
    p_graph.set_defaults(func=cmd_graph)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
