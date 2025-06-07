#!/usr/bin/env python
"""Archive oversize git diffs for later inspection.

The script calculates the patch for a revision range and measures its
size. When that size surpasses the user-provided threshold, it writes the
patch to ``artifacts/oversize_patch.diff.gz`` (or a path of your
choosing) and prints ``git diff --stat``. The archive can be unpacked with
``gzip -d`` or viewed directly with ``zless``. This way the heavy diff is
saved only when necessary and remains accessible for posterity.
"""
from __future__ import annotations

import argparse
import gzip
from pathlib import Path

from .diff_stat import diff_patch, diff_stat
from vybn.quantum_seed import seed_rng

DEFAULT_LIMIT = 500_000
DEFAULT_OUTPUT = Path("artifacts/oversize_patch.diff.gz")


def capture(rev: str, repo_root: Path, out_path: Path, limit: int) -> None:
    """Print diff stats and archive the patch if it exceeds ``limit`` bytes."""
    seed_rng()
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive oversize git diffs")
    parser.add_argument(
        "rev",
        nargs="?",
        default="HEAD~1..HEAD",
        help="revision range (default: HEAD~1..HEAD)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="destination file",
    )
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="size threshold in bytes",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    capture(args.rev, repo_root, args.output, args.limit)


if __name__ == "__main__":
    main()
