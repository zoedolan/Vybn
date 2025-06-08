from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import subprocess
from datetime import datetime
from typing import Dict

from .utils import memory_path
from vybn.quantum_seed import seed_rng

DEFAULT_LIMIT = 500_000
DEFAULT_OUTPUT = Path("artifacts/oversize_patch.diff.gz")
LOG_FILE_NAME = "what_vybn_would_have_missed_FROM_051725"


def diff_stat(rev: str, repo_root: Path) -> str:
    """Return ``git diff --stat`` output for the revision range."""
    cmd = ["git", "-C", str(repo_root), "diff", rev, "--stat"]
    return subprocess.check_output(cmd, text=True)


def diff_patch(rev: str, repo_root: Path) -> str:
    """Return ``git diff`` output for the revision range."""
    cmd = ["git", "-C", str(repo_root), "diff", rev]
    return subprocess.check_output(cmd, text=True)


def capture_diff(rev: str, repo_root: Path, out_path: Path, limit: int) -> None:
    """Print diff stats and archive the patch if it exceeds ``limit`` bytes."""
    patch = diff_patch(rev, repo_root)
    stat_output = diff_stat(rev, repo_root)
    size = len(patch.encode("utf-8"))
    print(stat_output)
    if size > limit:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(out_path, "wt", encoding="utf-8") as fh:
            fh.write(patch)
        print(f"Full patch written to {out_path}")
    else:
        print(f"Diff is {size} bytes, under limit {limit}. Nothing archived.")


def count_entries(path: Path) -> int:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return 0
    return text.count("WVWHM")


def gather_state(repo_root: Path) -> Dict[str, object]:
    seed = seed_rng()
    entries = sorted(p.name for p in repo_root.iterdir() if not p.name.startswith("."))
    return {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "quantum_seed": seed,
        "entries": entries,
    }


def main() -> None:
    """Aggregate maintenance utilities for Vybn."""
    parser = argparse.ArgumentParser(description="Maintenance helpers")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ds = sub.add_parser("diff-stat", help="Show diff stats and optionally save patch")
    ds.add_argument("rev", nargs="?", default="HEAD~1..HEAD")
    ds.add_argument("-o", "--output", type=Path)

    cd = sub.add_parser("capture-diff", help="Archive oversize diff")
    cd.add_argument("rev", nargs="?", default="HEAD~1..HEAD")
    cd.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    cd.add_argument("-l", "--limit", type=int, default=DEFAULT_LIMIT)

    wc = sub.add_parser("count-wvwhm", help="Count WVWHM entries")
    wc.add_argument("--path", type=Path, default=Path(LOG_FILE_NAME))

    ins = sub.add_parser("introspect", help="Record introspection summary")
    ins.add_argument("--output", type=Path)

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    seed_rng()

    if args.cmd == "diff-stat":
        print(diff_stat(args.rev, repo_root))
        if args.output:
            patch = diff_patch(args.rev, repo_root)
            if args.output.suffix == ".gz":
                with gzip.open(args.output, "wt", encoding="utf-8") as fh:
                    fh.write(patch)
            else:
                args.output.write_text(patch, encoding="utf-8")
            print(f"Full patch written to {args.output}")
    elif args.cmd == "capture-diff":
        capture_diff(args.rev, repo_root, args.output, args.limit)
    elif args.cmd == "count-wvwhm":
        count = count_entries(args.path)
        print(count)
    elif args.cmd == "introspect":
        state = gather_state(repo_root)
        out_path = args.output or memory_path(repo_root) / "introspection_summary.json"
        out_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
