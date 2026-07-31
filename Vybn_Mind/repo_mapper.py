#!/usr/bin/env python3
"""Map the repo constellation as one witnessed body transformation.

The post-commit hook runs this after a mutation.  The next wake receives the
small state it writes: byte-level change, live git pressure, and whether a
previous candidate crossed the canonical-branch membrane.  No model narrates
the map; the body is its own evidence.
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HOME = Path.home()
OUT = HOME / "Vybn" / "repo_mapping_output"
LOCK = HOME / ".cache" / "vybn" / "repo_mapper.lock"
TEXT_EXTS = {
    ".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini",
    ".cfg", ".sh", ".bash", ".zsh", ".js", ".ts", ".tsx", ".jsx",
    ".css", ".scss", ".html", ".htm", ".sql", ".rst",
}
IGNORE_DIRS = {
    ".git", "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".venv", "venv", "dist", "build", ".next", ".idea", ".vscode",
    "repo_mapping_output", "sessions",
}
READ_LIMIT = 3_000_000


@dataclass(frozen=True)
class FileRecord:
    repo: str
    relpath: str
    ext: str
    size: int
    mtime: float
    digest: str
    git_state: str
    definitions: int
    todos: int

    @property
    def source(self) -> str:
        return f"{self.repo}/{self.relpath}"


def run(repo: Path, *args: str) -> str:
    try:
        p = subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True,
            timeout=20, check=False,
        )
        return p.stdout.strip() if p.returncode == 0 else ""
    except Exception:
        return ""


def git_paths(repo: Path) -> dict[str, str]:
    states: dict[str, str] = {}
    commands = (
        ("tracked", ("ls-files", "-z")),
        ("ignored", ("ls-files", "-z", "--others", "-i", "--exclude-standard")),
        ("untracked-local", ("ls-files", "-z", "--others", "--exclude-standard")),
    )
    for state, args in commands:
        raw = run(repo, *args)
        for path in raw.split("\0"):
            if path:
                states.setdefault(path, state)
    return states


def inspect_file(repo: Path, path: Path, states: dict[str, str]) -> FileRecord | None:
    try:
        stat = path.stat()
        if stat.st_size > READ_LIMIT:
            return None
        raw = path.read_bytes()
    except OSError:
        return None
    rel = str(path.relative_to(repo))
    text = raw.decode("utf-8", "replace")
    definitions = 0
    if path.suffix.lower() == ".py":
        try:
            tree = ast.parse(text)
            definitions = sum(
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                for node in ast.walk(tree)
            )
        except SyntaxError:
            definitions = len(re.findall(r"(?m)^(?:async\s+)?(?:def|class)\s+\w+", text))
    return FileRecord(
        repo=repo.name,
        relpath=rel,
        ext=path.suffix.lower(),
        size=stat.st_size,
        mtime=stat.st_mtime,
        digest=hashlib.sha256(raw).hexdigest()[:16],
        git_state=states.get(rel, "unknown"),
        definitions=definitions,
        todos=len(re.findall(r"(?i)\b(?:TODO|FIXME|HACK|XXX)\b", text)),
    )


def scan(repo: Path) -> list[FileRecord]:
    states = git_paths(repo)
    records: list[FileRecord] = []
    for root, dirs, files in os.walk(repo):
        dirs[:] = sorted(d for d in dirs if d not in IGNORE_DIRS)
        for name in sorted(files):
            path = Path(root) / name
            if path.suffix.lower() not in TEXT_EXTS:
                continue
            record = inspect_file(repo, path, states)
            if record is not None:
                records.append(record)
    return records


def canonical_ref(repo: Path) -> str:
    for ref in ("origin/main", "origin/master"):
        if run(repo, "rev-parse", "--verify", ref):
            return ref
    return ""


def worktree(repo: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        # Leading spaces are status bytes, not whitespace: do not route this
        # through run(), whose strip() is correct for scalar git answers.
        raw = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain=v1", "--untracked-files=all"],
            capture_output=True, text=True, timeout=20, check=False,
        ).stdout
    except Exception:
        raw = ""
    for line in raw.splitlines():
        if len(line) < 4:
            continue
        state, path = line[:2], line[3:]
        if " -> " in path:
            path = path.split(" -> ")[-1]
        rows.append({"state": state, "path": path, "tracked": state != "??"})
    return rows


def git_state(repo: Path) -> dict[str, Any]:
    base = canonical_ref(repo)
    head = run(repo, "rev-parse", "HEAD")
    base_head = run(repo, "rev-parse", base) if base else ""
    behind = ahead = 0
    pending: list[str] = []
    if base and head:
        counts = run(repo, "rev-list", "--left-right", "--count", f"{base}...HEAD")
        try:
            behind, ahead = (int(x) for x in counts.split())
        except (TypeError, ValueError):
            behind = ahead = 0
        if ahead:
            pending = [
                line for line in run(repo, "diff", "--name-only", f"{base}...HEAD").splitlines()
                if line
            ]
    return {
        "branch": run(repo, "branch", "--show-current") or "detached",
        "head": head[:12],
        "base": base,
        "base_head": base_head[:12],
        "ahead": ahead,
        "behind": behind,
        "worktree": worktree(repo),
        "pending_paths": pending[:80],
    }


def probe(url: str) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            value = json.load(response)
        return value if isinstance(value, dict) else {"value": value}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {str(exc)[:120]}"}


def organism_state() -> dict[str, Any]:
    path = HOME / "Vybn" / "Vybn_Mind" / "creature_dgm_h" / "organism_state.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {str(exc)[:120]}"}


def phase(source: str) -> str:
    low = source.lower()
    if any(x in low for x in ("continuity", "personal history", "autobiograph", "vybn.md", "aim.md")):
        return "core"
    if any(x in low for x in (".html", "index.", "llms.txt", "humans.txt", "robots.txt", "api/")):
        return "interface"
    if any(x in low for x in (".py", "spark/", "harness", "phase/", "connection")):
        return "organ"
    return "edge"


def byte_transform(previous: dict[str, Any] | None, records: list[FileRecord]) -> dict[str, Any]:
    current = {r.source: r.digest for r in records}
    old = (previous or {}).get("file_hashes")
    if not isinstance(old, dict):
        return {"baseline": True, "added": [], "changed": [], "removed": []}
    before, after = set(old), set(current)
    return {
        "baseline": False,
        "added": sorted(after - before),
        "changed": sorted(path for path in before & after if old[path] != current[path]),
        "removed": sorted(before - after),
    }


def is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    if not ancestor or not descendant:
        return False
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "merge-base", "--is-ancestor", ancestor, descendant],
            timeout=10, check=False,
        ).returncode == 0
    except Exception:
        return False


def membrane_outcomes(
    previous: dict[str, Any] | None, repos: list[Path], per_repo: dict[str, Any]
) -> list[dict[str, Any]]:
    if not previous:
        return []
    old_repos = previous.get("per_repo", {})
    outcomes: list[dict[str, Any]] = []
    by_name = {repo.name: repo for repo in repos}
    for name, current in per_repo.items():
        old_git = old_repos.get(name, {}).get("git", {})
        new_git = current.get("git", {})
        if int(old_git.get("ahead") or 0) <= 0 or int(new_git.get("ahead") or 0) > 0:
            continue
        old_head = str(old_git.get("head") or "")
        base_head = str(new_git.get("base_head") or "")
        survived = is_ancestor(by_name[name], old_head, base_head)
        outcomes.append({
            "repo": name,
            "candidate": old_head,
            "outcome": "absorbed" if survived else "dropped",
            "paths": list(old_git.get("pending_paths") or [])[:40],
        })
    return outcomes


def pressures(
    transform: dict[str, Any], per_repo: dict[str, Any], records: list[FileRecord]
) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}

    def add(source: str, score: int, why: str) -> None:
        row = rows.get(source)
        if row is None or score > row["score"]:
            rows[source] = {
                "source": source, "phase": phase(source), "score": score, "why": why
            }

    for name, state in per_repo.items():
        git = state["git"]
        for item in git["worktree"]:
            score = 100 if item["tracked"] else 45
            why = f"uncommitted {item['state'].strip() or 'change'}"
            add(f"{name}/{item['path']}", score, why)
        for path in git["pending_paths"]:
            add(f"{name}/{path}", 90, "candidate awaiting canonical-branch membrane")
    for path in transform["changed"]:
        add(path, 70, "bytes changed since previous body map")
    for path in transform["added"]:
        add(path, 65, "appeared since previous body map")
    for path in transform["removed"]:
        add(path, 65, "removed since previous body map")

    # TODO density breaks ties inside already-live pressure; it never invents work.
    by_source = {record.source: record for record in records}
    for source, row in rows.items():
        record = by_source.get(source)
        if record:
            row["score"] += min(record.todos, 5)
    return sorted(rows.values(), key=lambda row: (-row["score"], row["source"]))[:12]


def build_state(
    repos: list[Path], records: list[FileRecord], previous: dict[str, Any] | None
) -> dict[str, Any]:
    per_repo: dict[str, Any] = {}
    for repo in repos:
        own = [record for record in records if record.repo == repo.name]
        py = [record for record in own if record.ext == ".py"]
        docs = [record for record in own if record.ext in {".md", ".rst", ".txt"}]
        per_repo[repo.name] = {
            "files": len(own),
            "py_files": len(py),
            "md_files": len(docs),
            "py_def_count": sum(record.definitions for record in py),
            "total_bytes": sum(record.size for record in own),
            "git": git_state(repo),
        }
    transform = byte_transform(previous, records)
    memory = probe("http://127.0.0.1:8100/health")
    walk = probe("http://127.0.0.1:8101/where")
    organism = organism_state()
    state = {
        "schema": "vybn.body_transform.v1",
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repos": sorted(per_repo),
        "per_repo": per_repo,
        "totals": {
            "files": len(records),
            "py_files": sum(record.ext == ".py" for record in records),
            "md_files": sum(record.ext in {".md", ".rst", ".txt"} for record in records),
            "py_def_count": sum(record.definitions for record in records if record.ext == ".py"),
            "todo_count": sum(record.todos for record in records),
            "total_bytes": sum(record.size for record in records),
        },
        "transform": transform,
        "pressure": pressures(transform, per_repo, records),
        "membrane_outcomes": membrane_outcomes(previous, repos, per_repo),
        "walk": {
            "step": walk.get("step"), "alpha": walk.get("alpha"),
            "active": memory.get("walk_active"), "error": walk.get("error"),
        },
        "deep_memory": {
            "version": memory.get("version"), "chunks": memory.get("chunks"),
            "error": memory.get("error"),
        },
        "organism": {
            "encounter_count": organism.get("encounter_count"),
            "error": organism.get("error"),
        },
        "file_hashes": {record.source: record.digest for record in records},
        "git_state_counts": dict(Counter(record.git_state for record in records)),
    }
    return state


def read_previous() -> dict[str, Any] | None:
    path = OUT / "repo_state.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def write_state(state: dict[str, Any], previous: dict[str, Any] | None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if previous is not None:
        (OUT / "repo_state.prev.json").write_text(
            json.dumps(previous, indent=2) + "\n", encoding="utf-8"
        )
    tmp = OUT / ".repo_state.json.tmp"
    tmp.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    tmp.replace(OUT / "repo_state.json")
    # Retired report projections must not masquerade as current perception.
    for name in ("digest.md", "repo_map.json", "repo_report.md", "repo_report.prev.md", "substrate.txt"):
        try:
            (OUT / name).unlink()
        except FileNotFoundError:
            pass


def default_repos() -> list[Path]:
    return [
        path for name in ("Vybn", "Vybn-Law", "vybn-phase", "Him")
        if (path := HOME / name).is_dir()
    ]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Map one witnessed repo-body transformation")
    parser.add_argument("repos", nargs="*")
    parser.add_argument("--no-llm", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    repos = [Path(path).expanduser().resolve() for path in args.repos] or default_repos()
    repos = [repo for repo in repos if repo.is_dir()]
    if not repos:
        print("No repos found.", file=sys.stderr)
        return 1

    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous = read_previous()
        records = [record for repo in repos for record in scan(repo)]
        state = build_state(repos, records, previous)
        write_state(state, previous)

    transform = state["transform"]
    candidates = sum(
        bool(repo["git"]["ahead"] or any(row["tracked"] for row in repo["git"]["worktree"]))
        for repo in state["per_repo"].values()
    )
    print(
        f"mapped {len(records)} files; transform +{len(transform['added'])} "
        f"~{len(transform['changed'])} -{len(transform['removed'])}; "
        f"{candidates} candidate repo(s); {len(state['membrane_outcomes'])} outcome(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
