#!/usr/bin/env python3
"""Audit the Zoe/Vybn repo closure for hidden local/projection drift.

This is the antidote to the recurring failure mode where work is committed
locally, left dirty, stranded in a stash/local branch, or made invisible by a
narrow fetch refspec while GitHub and the local clone disagree about reality.

The key lesson: Git closure is not just "working tree clean." Remote-tracking
refs, fetch refspecs, upstreams, stashes, origin/HEAD, and live-process uptake
are separate projections. A branch can exist on GitHub and still be invisible to
``git log --branches --not --remotes`` if this clone only fetches ``main``.

In --fix mode (default), the audit safely normalizes projection state:
  - Ensures origin fetches all branch heads into refs/remotes/origin/*.
  - Fetches/prunes origin after normalizing the refspec.
  - Deletes stale local branches with no upstream only when their commits are
    already reachable from the active branch's upstream.
It does NOT auto-drop stashes or delete unique branch work.
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

PRIMARY_BRANCH_BY_REPO = {
    "Vybn": "main",
    "Him": "main",
    "Vybn-Law": "master",
    "vybn-phase": "main",
    "Origins": "gh-pages",
}

EXPECTED_FETCH_REFSPEC = "+refs/heads/*:refs/remotes/origin/*"

# Default: fix safe projection/stale-branch problems unless explicitly disabled.
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


def fetch_refspecs(repo: Path) -> list[str]:
    out = run(repo, "config", "--local", "--get-all", "remote.origin.fetch")
    return [line.strip() for line in out.splitlines() if line.strip()]


def fetch_refspec_is_complete(refspecs: list[str]) -> bool:
    return EXPECTED_FETCH_REFSPEC in refspecs


def normalize_fetch_refspec(repo: Path) -> str:
    run(repo, "config", "--local", "--unset-all", "remote.origin.fetch")
    run(repo, "config", "--local", "--add", "remote.origin.fetch", EXPECTED_FETCH_REFSPEC)
    fetched = run(repo, "fetch", "origin", "--prune")
    return fetched


def primary_branch_for(repo: Path) -> str:
    """Return the branch closure should end on for this repo."""
    return PRIMARY_BRANCH_BY_REPO.get(repo.name, "main")


def current_branch(repo: Path) -> str:
    return run(repo, "branch", "--show-current")


def upstream_for(repo: Path, branch: str) -> str:
    if not branch:
        return ""
    upstream = run(repo, "rev-parse", "--abbrev-ref", f"{branch}@{{upstream}}")
    if "no upstream" in upstream or "@{upstream}" in upstream:
        return ""
    return upstream.strip()


def origin_head(repo: Path) -> str:
    return run(repo, "symbolic-ref", "refs/remotes/origin/HEAD")


def stash_entries(repo: Path) -> list[str]:
    out = run(repo, "stash", "list")
    return [line for line in out.splitlines() if line.strip()]


def local_branches(repo: Path) -> list[str]:
    raw = run(repo, "branch", "--list", "--format=%(refname:short)")
    return [b.strip() for b in raw.splitlines() if b.strip()]


def stale_local_branches(repo: Path) -> list[str]:
    """Return non-active local branches that have no configured upstream."""
    active = current_branch(repo)
    stale: list[str] = []
    for branch in local_branches(repo):
        if branch == active:
            continue
        if not upstream_for(repo, branch):
            stale.append(branch)
    return stale


def primary_upstream_for(repo: Path) -> str:
    primary = primary_branch_for(repo)
    upstream = upstream_for(repo, primary)
    if upstream:
        return upstream
    candidate = f"origin/{primary}"
    if run(repo, "rev-parse", "--verify", "--quiet", candidate):
        return candidate
    return ""


def branch_unique_commits_against_primary(repo: Path, branch: str) -> str:
    """Return commits on branch not reachable from the repo's primary upstream."""
    base = primary_upstream_for(repo)
    if not base:
        return ""
    return run(repo, "log", f"{base}..{branch}", "--oneline", "--decorate", "-10")


def branch_subsumed_by_active_upstream(repo: Path, branch: str) -> bool:
    """True if ``branch`` has no commits beyond the primary branch's upstream."""
    return not branch_unique_commits_against_primary(repo, branch).strip()


def delete_branch(repo: Path, branch: str) -> str:
    return run(repo, "branch", "-D", branch)


def audit_repo(repo: Path) -> tuple[bool, str]:
    if not (repo / ".git").exists():
        return True, f"===== {repo} =====\nnot a git repo"

    lines: list[str] = [f"===== {repo} ====="]
    status = run(repo, "status", "--short", "--branch")
    lines.append(status or "(no status output)")

    problems: list[str] = []

    # Projection integrity: if this clone only fetches one branch, remote reality
    # can exist on GitHub while remaining invisible to closure checks.
    refspecs = fetch_refspecs(repo)
    if not fetch_refspec_is_complete(refspecs):
        lines.append("\nFETCH_REFSPEC:")
        lines.append("\n".join(refspecs) if refspecs else "(none)")
        if FIX:
            fetched = normalize_fetch_refspec(repo)
            lines.append(f"normalized -> {EXPECTED_FETCH_REFSPEC}")
            if fetched:
                lines.append(fetched)
            refspecs = fetch_refspecs(repo)
        if not fetch_refspec_is_complete(refspecs):
            problems.append("origin fetch refspec does not fetch all branches")

    origin_head_ref = origin_head(repo)
    lines.append("\nORIGIN_HEAD:")
    lines.append(origin_head_ref or "(missing / not symbolic)")

    active = current_branch(repo)
    primary = primary_branch_for(repo)
    active_upstream = upstream_for(repo, active)
    primary_upstream = primary_upstream_for(repo)
    lines.append("\nACTIVE_BRANCH:")
    lines.append(f"{active or '(detached)'} -> {active_upstream or '(no upstream)'}")
    lines.append(f"primary closure branch: {primary} -> {primary_upstream or '(missing upstream)'}")
    if active != primary:
        problems.append(f"active branch is {active or 'detached'}, not primary closure branch {primary}")
    if active and not active_upstream:
        problems.append(f"active branch {active} has no upstream")
    if not primary_upstream:
        problems.append(f"primary branch {primary} has no upstream")

    stashes = stash_entries(repo)
    if stashes:
        problems.append("stash entries present")
        lines.append("\nSTASHES:")
        lines.extend(stashes)

    dirty = run(repo, "status", "--porcelain")
    if dirty:
        problems.append("dirty working tree")
        lines.append("\nDIRTY:")
        lines.append(dirty)

    local_only = run(repo, "log", "--branches", "--not", "--remotes", "--oneline", "--decorate", "-10")
    if local_only:
        problems.append("local branch commits not on any remote")
        lines.append("\nLOCAL-ONLY COMMITS:")
        lines.append(local_only)

    contains = run(repo, "branch", "-r", "--contains", "HEAD")
    head_unreachable = not contains.strip()
    if head_unreachable:
        problems.append("HEAD not contained in any remote branch")
        lines.append("\nHEAD_REMOTE_REACHABILITY: unreachable from remotes")
    else:
        lines.append("\nHEAD_REMOTE_REACHABILITY:")
        lines.append(contains)

    # Sua sponte: detect local branch limbo. Closure means work is merged into
    # the primary branch or intentionally retired, not merely pushed somewhere.
    # Only auto-delete branches whose commits are already reachable from the
    # primary upstream. Unique topic-branch commits require merge/archive/retire.
    non_primary = [branch for branch in local_branches(repo) if branch != primary]
    if non_primary:
        lines.append("\nLOCAL NON-PRIMARY BRANCHES:")
        for branch in non_primary:
            unique = branch_unique_commits_against_primary(repo, branch)
            upstream = upstream_for(repo, branch)
            if unique.strip():
                lines.append(f"  {branch} -> {upstream or '(no upstream)'}: unique commits not merged to {primary}")
                lines.append(unique)
                problems.append(f"branch {branch} has unmerged work outside {primary}")
            else:
                status_tag = f"subsumed by {primary_upstream or primary} — safe to delete"
                if FIX:
                    result = delete_branch(repo, branch)
                    lines.append(f"  {branch}: {status_tag} → DELETED ({result})")
                else:
                    lines.append(f"  {branch}: {status_tag} (run with --fix to delete)")
                    problems.append(f"subsumed non-primary branch {branch} still present")

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
