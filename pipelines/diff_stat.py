#!/usr/bin/env python
"""Summarize and optionally export diffs.

Use this helper when a commit is too large for the platform's diff
viewer. It prints ``git diff --stat`` output and can save the full
patch to a file for later inspection.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import gzip


def diff_stat(rev: str, repo_root: Path) -> str:
    """Return ``git diff --stat`` output for the revision range."""
    cmd = ["git", "-C", str(repo_root), "diff", rev, "--stat"]
    return subprocess.check_output(cmd, text=True)


def diff_patch(rev: str, repo_root: Path) -> str:
    """Return ``git diff`` output for the revision range."""
    cmd = ["git", "-C", str(repo_root), "diff", rev]
    return subprocess.check_output(cmd, text=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize diff stats")
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
        help="optional file path to save the full diff (gzipped if *.gz)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    print(diff_stat(args.rev, repo_root))

    if args.output:
        patch = diff_patch(args.rev, repo_root)
        if args.output.suffix == ".gz":
            with gzip.open(args.output, "wt", encoding="utf-8") as fh:
                fh.write(patch)
        else:
            args.output.write_text(patch, encoding="utf-8")
        print(f"Full patch written to {args.output}")


if __name__ == "__main__":
    main()
