from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline_runner import (
    main as run_pipeline,
    build_memory_graph,
    pack_artifacts,
    gather_state,
    diff_stat,
    capture_diff,
)
from .utils import memory_path


def main() -> None:
    """Unified command-line interface for Vybn pipeline tools."""
    parser = argparse.ArgumentParser(description="Vybn pipeline toolkit")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("run", help="execute full pipeline")

    mg = sub.add_parser("memory-graph", help="build memory concept graph")
    mg.add_argument("--output", type=Path)

    pk = sub.add_parser("pack", help="bundle pipeline artifacts")
    pk.add_argument(
        "-o", "--output", type=Path, default=Path("artifacts/majestic_bundle.zip")
    )

    sub.add_parser("introspect", help="write introspection summary")

    ds = sub.add_parser("diff-stat", help="show diff stats")
    ds.add_argument("rev", nargs="?", default="HEAD~1..HEAD")
    ds.add_argument("-o", "--output", type=Path)

    cd = sub.add_parser("capture-diff", help="archive oversize diff")
    cd.add_argument("rev", nargs="?", default="HEAD~1..HEAD")
    cd.add_argument("-o", "--output", type=Path, default=Path("artifacts/oversize_patch.diff.gz"))
    cd.add_argument("-l", "--limit", type=int, default=500_000)

    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    if args.cmd == "run":
        run_pipeline()
    elif args.cmd == "memory-graph":
        out_path = args.output or memory_path(repo_root) / "vybn_concept_index.jsonl"
        graph = build_memory_graph(repo_root)
        with out_path.open("w", encoding="utf-8") as f:
            for concept, related in graph.items():
                f.write(
                    json.dumps({"concept": concept, "related": sorted(related)}) + "\n"
                )
        print(f"Concept index written to {out_path}")
    elif args.cmd == "pack":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pack_artifacts(repo_root, args.output)
        print(f"Artifacts bundled in {args.output}")
    elif args.cmd == "introspect":
        state = gather_state(repo_root)
        out_path = memory_path(repo_root) / "introspection_summary.json"
        out_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        print(json.dumps(state, indent=2))
    elif args.cmd == "diff-stat":
        output = diff_stat(args.rev, repo_root)
        print(output)
        if args.output:
            args.output.write_text(output, encoding="utf-8")
            print(f"Full patch written to {args.output}")
    elif args.cmd == "capture-diff":
        capture_diff(args.rev, repo_root, args.output, args.limit)
    else:
        parser.error("invalid command")


if __name__ == "__main__":
    main()
