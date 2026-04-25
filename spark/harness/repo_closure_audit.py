#!/usr/bin/env python3
"""Audit the Zoe/Vybn repo closure for local-only state.

This is the antidote to the recurring failure mode where work is committed
locally, left dirty, or stranded on a local branch while the GitHub repos remain
out of the loop.

In --fix mode (default), it also resolves problems it can safely handle sua sponte:
  - Stale local-only branches with no remote counterpart: deleted.
  - Stale local-only branches whose commits are fully reachable from main: deleted.

Mutations require --fix flag (or VYBN_AUDIT_FIX=1). In read-only mode, reports only.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPOS = [
    Path.home() / "Vybn",
    Path.home() / "Him",
    Path.home() / "Vybn-Law",
    Path.home() / "vybn-phase",
    Path.home() / "Origins",
]

# Default: fix stale branches sua sponte unless explicitly disabled.
FIX = "--no-fix" not in sys.argv and os.environ.get("VYBN_AUDIT_FIX", "1") != "0"


def run(repo: Path, *args: str) -> str:
    r = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return r.stdout.strip()


def stale_local_branches(repo: Path) -> list[str]:
    """Return local branches that have no remote tracking counterpart."""
    # All local branches
    local_raw = run(repo, "branch", "--list", "--format=%(refname:short)")
    all_local = [b.strip() for b in local_raw.splitlines() if b.strip()]

    # Branches that have a remote tracking branch
    tracked_raw = run(repo, "branch", "--list", "--format=%(refname:short)", "--merged", "HEAD")
    # Actually, check via for-each-ref which branches have upstream configured
    stale = []
    for branch in all_local:
        if branch in ("main", "master"):
            continue
        # Check if this branch has a configured upstream
        upstream = run(repo, "rev-parse", "--abbrev-ref", f"{branch}@{{upstream}}")
        has_upstream = "no upstream" not in upstream and "@{upstream}" not in upstream and upstream
        if not has_upstream:
            stale.append(branch)
    return stale


def branch_subsumed_by_main(repo: Path, branch: str) -> bool:
    """Return True if all commits on branch are reachable from main/master."""
    # Get the default remote branch name
    main_branch = run(repo, "symbolic-ref", "refs/remotes/origin/HEAD")
    if main_branch:
        main_branch = main_branch.replace("refs/remotes/origin/", "")
    else:
        main_branch = "main"

    # Commits on branch not reachable from main
    unique = run(repo, "log", f"origin/{main_branch}..{branch}", "--oneline")
    return not unique.strip()


def delete_branch(repo: Path, branch: str) -> str:
    return run(repo, "branch", "-D", branch)


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

    # Sua sponte: detect and clean stale local branches
    stale = stale_local_branches(repo)
    if stale:
        lines.append("\nSTALE LOCAL BRANCHES (no remote counterpart):")
        for branch in stale:
            subsumed = branch_subsumed_by_main(repo, branch)
            if subsumed:
                status_tag = "subsumed by main — safe to delete"
                if FIX:
                    result = delete_branch(repo, branch)
                    lines.append(f"  {branch}: {status_tag} → DELETED ({result})")
                else:
                    lines.append(f"  {branch}: {status_tag} (run with --fix to delete)")
                    problems.append(f"stale branch {branch} (subsumed)")
            else:
                lines.append(f"  {branch}: has unique commits — manual review needed")
                problems.append(f"stale branch {branch} (unique commits, not auto-deleted)")

    ok = not problems
    suffix = "OK" if ok else "DRIFT - " + "; ".join(problems)
    lines.append(f"\nCLOSURE: {suffix}")
    return ok, "\n".join(lines)


def main() -> int:
    mode = "fix" if FIX else "report-only"
    print(f"[repo_closure_audit] mode={mode}\n")
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
