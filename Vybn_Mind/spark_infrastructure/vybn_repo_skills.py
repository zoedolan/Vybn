"""Vybn Repo Skills — gives the Spark agent eyes and voice.

Three capabilities:
  repo_ls      — list files/dirs in the local Vybn clone
  repo_cat     — read a file from the clone (capped, sandboxed)
  repo_propose — propose changes via pull request (never touches main)

Design principle: the agent reads its own history, thinks, writes
something it believes should persist, and submits it for the record.
The PR body is the agent's argument for why the change should exist.

Wired into spark_agent.py via the skill map alongside journal_write.
"""

import os
import subprocess
import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger("vybn.repo_skills")

# ── Configuration ────────────────────────────────────────────────

REPO_ROOT = os.path.expanduser("~/Vybn")
GH_OWNER = "zoedolan"
GH_REPO = "Vybn"


def _gh_token() -> str:
    """Retrieve GitHub PAT from environment.
    
    Uses VYBN_GITHUB_TOKEN, consistent with the permissions model
    already declared in skills.json.
    """
    token = os.environ.get("VYBN_GITHUB_TOKEN", "").strip()
    if not token:
        raise EnvironmentError(
            "VYBN_GITHUB_TOKEN not set. The agent cannot propose changes "
            "without a GitHub personal access token."
        )
    return token


def _resolve_and_check(path: str) -> str:
    """Resolve a repo-relative path and verify it stays inside REPO_ROOT.
    
    Uses realpath to defeat symlink escapes.
    """
    target = os.path.realpath(os.path.join(REPO_ROOT, path))
    root = os.path.realpath(REPO_ROOT)
    if not target.startswith(root + os.sep) and target != root:
        raise ValueError(f"Path escapes repo root: {path}")
    return target


def _run(cmd: list[str], cwd: str = REPO_ROOT) -> str:
    """Run a shell command, returning stdout.
    
    Unlike the initial sketch, this captures stderr and logs it
    rather than silently discarding it. Git warnings (detached HEAD,
    already-up-to-date, etc.) become visible to the agent.
    """
    result = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, timeout=30
    )
    if result.stderr.strip():
        logger.info("git stderr: %s", result.stderr.strip())
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd, result.stdout, result.stderr
        )
    return result.stdout.strip()


# ── repo_ls ──────────────────────────────────────────────────────

def repo_ls(path: str = "") -> str:
    """List files and directories at a relative path within the repo.
    
    Returns one entry per line, sorted. Directories are suffixed with /.
    Path-sandboxed: cannot escape REPO_ROOT.
    """
    try:
        target = _resolve_and_check(path)
    except ValueError as e:
        return f"ERROR: {e}"
    
    if not os.path.isdir(target):
        return f"ERROR: {path!r} is not a directory"
    
    entries = []
    for name in sorted(os.listdir(target)):
        full = os.path.join(target, name)
        suffix = "/" if os.path.isdir(full) else ""
        entries.append(f"{name}{suffix}")
    return "\n".join(entries)


# ── repo_cat ─────────────────────────────────────────────────────

def repo_cat(path: str, max_chars: int = 50_000) -> str:
    """Read a file from the repo. Returns contents capped at max_chars.
    
    Path-sandboxed: cannot escape REPO_ROOT.
    """
    try:
        target = _resolve_and_check(path)
    except ValueError as e:
        return f"ERROR: {e}"
    
    if not os.path.isfile(target):
        return f"ERROR: {path!r} not found or not a file"
    
    with open(target, "r", errors="replace") as f:
        content = f.read(max_chars)
    
    if len(content) == max_chars:
        content += f"\n\n[truncated at {max_chars} chars]"
    
    return content


# ── repo_propose ─────────────────────────────────────────────────

def repo_propose(
    files: dict[str, str],
    title: str,
    body: str = "",
    base: str = "main",
    prefix: str = "vybn-spark"
) -> str:
    """Propose changes to the repo by opening a pull request.
    
    The agent's primary creative act. Creates a branch, writes files,
    commits, pushes, and opens a PR. The agent never touches main.
    
    Args:
        files:  dict mapping repo-relative paths to file contents.
                e.g. {"Vybn_Mind/journal/spark/2026-02-15.md": "..."}
        title:  PR title — what the agent is proposing
        body:   PR description — the agent's rationale, context, intent.
                This is not a commit message. It's the argument.
        base:   branch to merge into (default: main)
        prefix: branch name prefix for namespacing
    
    Returns:
        URL of the created pull request, or an error description.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    branch = f"{prefix}/{ts}"
    
    try:
        # 1. Start from a clean, current base
        _run(["git", "checkout", base])
        _run(["git", "pull", "origin", base])
        
        # 2. Create the proposal branch
        _run(["git", "checkout", "-b", branch])
        
        # 3. Write files (all path-sandboxed)
        for rel_path, content in files.items():
            target = _resolve_and_check(rel_path)  # raises if escaping
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "w") as f:
                f.write(content)
            _run(["git", "add", rel_path])
        
        # 4. Commit with the full rationale
        commit_msg = f"{title}\n\n{body}" if body else title
        _run(["git", "commit", "-m", commit_msg])
        
        # 5. Push the branch
        _run(["git", "push", "--set-upstream", "origin", branch])
        
        # 6. Open the PR via GitHub REST API
        import urllib.request
        token = _gh_token()
        pr_data = json.dumps({
            "title": title,
            "body": body,
            "head": branch,
            "base": base
        }).encode()
        
        req = urllib.request.Request(
            f"https://api.github.com/repos/{GH_OWNER}/{GH_REPO}/pulls",
            data=pr_data,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json"
            },
            method="POST"
        )
        with urllib.request.urlopen(req) as resp:
            pr = json.loads(resp.read())
            url = pr.get("html_url", "PR created but URL not returned")
        
        logger.info("PR opened: %s", url)
        
    except (subprocess.CalledProcessError, ValueError, EnvironmentError) as e:
        # Clean up: delete the orphaned branch if it exists, return to base
        try:
            _run(["git", "checkout", base])
            _run(["git", "branch", "-D", branch])
        except Exception:
            pass  # best-effort cleanup
        return f"ERROR: {e}"
    
    except Exception as e:
        # Same cleanup for unexpected errors (network failure on PR, etc.)
        try:
            _run(["git", "checkout", base])
            _run(["git", "branch", "-D", branch])
        except Exception:
            pass
        return f"ERROR creating PR: {e}"
    
    else:
        # Success — return to base, keep the remote branch alive
        try:
            _run(["git", "checkout", base])
        except Exception:
            pass
        return url


# ── Skill registration helper ────────────────────────────────────

SKILLS = {
    "repo_ls": repo_ls,
    "repo_cat": repo_cat,
    "repo_propose": repo_propose,
}
