#!/usr/bin/env python3
"""Audit the Zoe/Vybn repo closure for local-only state.

This is the antidote to the recurring failure mode where work is committed
locally, left dirty, or stranded on a local branch while the GitHub repos remain
out of the loop. It does not mutate anything. It reports every repo projection
that is not durable on its remote.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPOS = [
    Path.home() / "Vybn",
    Path.home() / "Him",
    Path.home() / "Vybn-Law",
    Path.home() / "vybn-phase",
    Path.home() / "Origins",
]


def run(repo: Path, *args: str) -> str:
    r = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return r.stdout.strip()


def audit_repo(repo: Path) -> tuple[bool, str]:
    if not (repo / ".git").exists():
        return True, f"===== {repo} =====\nnot a git repo"

    lines: list[str] = [f"===== {repo} ====="]
    status = run(repo, "status", "--short", "--branch")
    lines.append(status or "(no status output)")

    dirty = run(repo, "status", "--porcelain")
    local_only = run(repo, "log", "--branches", "--not", "--remotes", "--oneline", "--decorate", "-10")

    contains = run(repo, "branch", "-r", "--contains", "HEAD")
    head_unreachable = not contains.strip()

    problems: list[str] = []
    if dirty:
        problems.append("dirty working tree")
        lines.append("\nDIRTY:")
        lines.append(dirty)
    if local_only:
        problems.append("local branch commits not on any remote")
        lines.append("\nLOCAL-ONLY COMMITS:")
        lines.append(local_only)
    if head_unreachable:
        problems.append("HEAD not contained in any remote branch")
        lines.append("\nHEAD_REMOTE_REACHABILITY: unreachable from remotes")
    else:
        lines.append("\nHEAD_REMOTE_REACHABILITY:")
        lines.append(contains)

    ok = not problems
    suffix = "OK" if ok else "DRIFT - " + "; ".join(problems)
    lines.append(f"\nCLOSURE: {suffix}")
    return ok, "\n".join(lines)


def main() -> int:
    all_ok = True
    reports: list[str] = []
    for repo in REPOS:
        ok, report = audit_repo(repo)
        all_ok = all_ok and ok
        reports.append(report)
    print("\n\n".join(reports))
    print("\nOVERALL:", "OK" if all_ok else "DRIFT PRESENT - commit/push/archive before claiming harmonization")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
