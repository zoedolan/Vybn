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


def branch_subsumed_by_active_upstream(repo: Path, branch: str) -> bool:
    """True if ``branch`` has no commits beyond the active branch's upstream."""
    active = current_branch(repo)
    base = upstream_for(repo, active)
    if not base:
        # Fall back to common branch names only when the active branch has no
        # upstream. Do not trust origin/HEAD here; some repos intentionally work
        # from gh-pages while origin/HEAD points elsewhere.
        for candidate in ("origin/main", "origin/master", "origin/gh-pages"):
            if run(repo, "rev-parse", "--verify", "--quiet", candidate):
                base = candidate
                break
    if not base:
        return False
    unique = run(repo, "log", f"{base}..{branch}", "--oneline")
    return not unique.strip()


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
    active_upstream = upstream_for(repo, active)
    lines.append("\nACTIVE_BRANCH:")
    lines.append(f"{active or '(detached)'} -> {active_upstream or '(no upstream)'}")
    if active and not active_upstream:
        problems.append(f"active branch {active} has no upstream")

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

    # Sua sponte: detect and clean stale local branches. Only delete if the
    # branch has no upstream and no unique commits beyond the active upstream.
    stale = stale_local_branches(repo)
    if stale:
        lines.append("\nSTALE LOCAL BRANCHES (no upstream):")
        for branch in stale:
            subsumed = branch_subsumed_by_active_upstream(repo, branch)
            if subsumed:
                status_tag = "subsumed by active upstream — safe to delete"
                if FIX:
                    result = delete_branch(repo, branch)
                    lines.append(f"  {branch}: {status_tag} → DELETED ({result})")
                else:
                    lines.append(f"  {branch}: {status_tag} (run with --fix to delete)")
                    problems.append(f"stale branch {branch} (subsumed)")
            else:
                lines.append(f"  {branch}: has unique/unverified commits — manual review needed")
                problems.append(f"stale branch {branch} (unique or unverified commits)")

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
