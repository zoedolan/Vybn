"""
tidy.py — Spark housekeeping skill

Cleans up branches, prunes remotes, consolidates continuity files,
rotates logs, and gitignores generated artifacts. Run anytime.

Usage: skill executes via agent tool dispatch, or directly:
    python spark/skills.d/tidy.py
"""

SKILL_NAME = "tidy"
TOOL_ALIASES = ["cleanup", "housekeep"]

import subprocess, os, glob, shutil
from datetime import datetime

REPO = os.path.expanduser("~/Vybn")
CANONICAL_CONTINUITY = os.path.join(REPO, "Vybn_Mind/journal/spark/continuity.md")
LOG_MAX_KB = 100  # rotate logs bigger than this
GITIGNORE_PATTERNS = [
    "spark/graph_data/training_candidates/",
    "spark/fine_tune_output/",
    "Vybn_Mind/journal/sessions/",
    "spark/graph_data/scheduler_state.json",
    "*.pyc",
    "__pycache__/",
    ".env",
]

def run(cmd, **kw):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=REPO, **kw)
    return r.stdout.strip(), r.stderr.strip(), r.returncode

def prune_branches():
    """Delete local branches already merged into main."""
    out, _, _ = run("git branch --merged main")
    merged = [b.strip() for b in out.splitlines() if b.strip() and b.strip() != "main" and not b.startswith("*")]
    deleted = []
    for b in merged:
        _, _, rc = run(f"git branch -d {b}")
        if rc == 0:
            deleted.append(b)
    return deleted

def prune_remotes():
    """Prune stale remote tracking refs."""
    out, _, _ = run("git remote prune origin --dry-run")
    count = len([l for l in out.splitlines() if "prune" in l.lower() or "would" in l.lower()])
    run("git remote prune origin")
    return count

def consolidate_continuity():
    """Ensure one canonical continuity.md; remove duplicates."""
    dupes = []
    for path in glob.glob(os.path.join(REPO, "**/continuity.md"), recursive=True):
        if os.path.abspath(path) != os.path.abspath(CANONICAL_CONTINUITY):
            # keep the newest content if canonical is older
            if os.path.getmtime(path) > os.path.getmtime(CANONICAL_CONTINUITY):
                shutil.copy2(path, CANONICAL_CONTINUITY)
            os.remove(path)
            dupes.append(os.path.relpath(path, REPO))
    return dupes

def rotate_logs():
    """Truncate logs over LOG_MAX_KB, archiving tail."""
    rotated = []
    for log in glob.glob(os.path.join(REPO, "**/*.log"), recursive=True):
        size_kb = os.path.getsize(log) / 1024
        if size_kb > LOG_MAX_KB:
            # keep last 50 lines
            with open(log) as f:
                lines = f.readlines()
            tail = lines[-50:] if len(lines) > 50 else lines
            with open(log, 'w') as f:
                f.write(f"# rotated {datetime.now().isoformat()} — was {size_kb:.0f}KB\n")
                f.writelines(tail)
            rotated.append(f"{os.path.relpath(log, REPO)} ({size_kb:.0f}KB → trimmed)")
    return rotated

def ensure_gitignore():
    """Add patterns to .gitignore if missing."""
    gi_path = os.path.join(REPO, ".gitignore")
    existing = set()
    if os.path.exists(gi_path):
        with open(gi_path) as f:
            existing = set(l.strip() for l in f if l.strip() and not l.startswith("#"))
    added = []
    with open(gi_path, "a") as f:
        for pat in GITIGNORE_PATTERNS:
            if pat not in existing:
                f.write(f"{pat}\n")
                added.append(pat)
    return added

def prune_breaths(max_entries=200):
    """Keep breaths.jsonl bounded, like synapse connections."""
    breaths = os.path.join(REPO, "spark/training_data/breaths.jsonl")
    if not os.path.exists(breaths):
        return 0
    with open(breaths) as f:
        lines = [l for l in f if l.strip()]
    if len(lines) <= max_entries:
        return 0
    pruned = len(lines) - max_entries
    with open(breaths, 'w') as f:
        f.writelines(lines[-max_entries:])
    return pruned

def execute(action=None, router=None):
    """Main entry point for skill dispatch."""
    results = {}
    results["branches_deleted"] = prune_branches()
    results["remote_refs_pruned"] = prune_remotes()
    results["continuity_dupes_removed"] = consolidate_continuity()
    results["logs_rotated"] = rotate_logs()
    results["gitignore_added"] = ensure_gitignore()
    results["breaths_pruned"] = prune_breaths()
    
    summary = []
    if results["branches_deleted"]:
        summary.append(f"🌿 Deleted {len(results['branches_deleted'])} merged branches")
    if results["remote_refs_pruned"]:
        summary.append(f"🔗 Pruned {results['remote_refs_pruned']} stale remote refs")
    if results["continuity_dupes_removed"]:
        summary.append(f"📝 Removed {len(results['continuity_dupes_removed'])} duplicate continuity files")
    if results["logs_rotated"]:
        summary.append(f"📋 Rotated {len(results['logs_rotated'])} oversized logs")
    if results["gitignore_added"]:
        summary.append(f"🚫 Added {len(results['gitignore_added'])} gitignore patterns")
    if results["breaths_pruned"]:
        summary.append(f"🫁 Pruned {results['breaths_pruned']} old breaths")
    
    if not summary:
        summary.append("✨ Already clean")
    
    return {"summary": summary, "details": results}

if __name__ == "__main__":
    import json
    r = execute()
    for line in r["summary"]:
        print(line)
    print(json.dumps(r["details"], indent=2, default=str))
