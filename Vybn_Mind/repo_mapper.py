#!/usr/bin/env python3
"""Map the repo constellation as one witnessed body transformation.

The post-commit hook runs this after a mutation.  The next wake receives the
small state it writes: byte-level change, live git pressure, and the turn whose
prompt, response, and commit produced the change.  No model narrates the map;
the body is its own evidence.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HOME = Path.home()
OUT = HOME / "Vybn" / "repo_mapping_output"
LOCK = HOME / ".cache" / "vybn" / "repo_mapper.lock"
LINEAGE = HOME / ".cache" / "vybn" / "body_lineage.jsonl"
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
    digest: str
    surface: str
    carrier: str
    body_graph: dict[str, Any] | None

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


SOURCE_LINK = re.compile(
    r"https://github\.com/([^/]+)/([^/]+)/blob/[^/]+/([^\s\"'<>?#]+)"
)
CARRIER_META = re.compile(
    r"<meta\s+name=[\"']kpp-carrier[\"']\s+content=[\"']([^\"']+)", re.I
)
BODY_GRAPH = re.compile(
    r"<script[^>]+id=[\"']vybn-body-graph[\"'][^>]*>(.*?)</script>", re.I | re.S
)


def declared_public_relation(repo: str, rel: str, text: str) -> tuple[str, str]:
    """Return only a source/surface bond declared by the surface's own bytes."""
    owners = {
        match.group(1) for match in SOURCE_LINK.finditer(text)
        if match.group(2) == repo
        and urllib.parse.unquote(match.group(3)).rstrip(".),") == rel
    }
    surface = next(
        (url for owner in sorted(owners)
         if (url := f"https://{owner}.github.io/{repo}/{rel}") in text), ""
    )
    carrier = (match.group(1) if (match := CARRIER_META.search(text)) else "")
    return surface, carrier


def declared_body_graph(text: str) -> dict[str, Any] | None:
    """Read the dual-use visual graph only when its cycle is internally decidable."""
    match = BODY_GRAPH.search(text)
    if not match:
        return None
    try:
        graph = json.loads(match.group(1))
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(graph, dict):
        return None
    nodes, edges, loop = graph.get("nodes"), graph.get("edges"), graph.get("loop")
    if graph.get("schemaVersion") != "vybn.public_body_graph.v1" or not nodes or not edges:
        return None
    ids = {node.get("id") for node in nodes if isinstance(node, dict)}
    if (not ids or not isinstance(loop, list) or any(node not in ids for node in loop)
            or any(not isinstance(node, dict) or not node.get("affordance") for node in nodes)):
        return None
    if any(not isinstance(edge, dict) or edge.get("from") not in ids or edge.get("to") not in ids
           or not edge.get("verb") or not edge.get("gate") for edge in edges):
        return None
    pairs = {(edge["from"], edge["to"]) for edge in edges}
    if any(pair not in pairs for pair in zip(loop, loop[1:])):
        return None
    return {
        "schema": graph["schemaVersion"],
        "name": str(graph.get("name", "")),
        "loop": loop[:20],
        "verbs": [str(edge["verb"]) for edge in edges[:20]],
        "nodes": [
            {"id": str(node["id"]), "affordance": str(node.get("affordance", ""))}
            for node in nodes[:20]
        ],
    }


def inspect_file(repo: Path, path: Path) -> FileRecord | None:
    try:
        stat = path.stat()
        if stat.st_size > READ_LIMIT:
            return None
        raw = path.read_bytes()
    except OSError:
        return None
    rel = str(path.relative_to(repo))
    text = raw.decode("utf-8", "replace")
    surface, carrier = (declared_public_relation(repo.name, rel, text)
                        if path.suffix.lower() in {".html", ".htm"} else ("", ""))
    return FileRecord(
        repo=repo.name,
        relpath=rel,
        digest=hashlib.sha256(raw).hexdigest()[:16],
        surface=surface,
        carrier=carrier,
        body_graph=declared_body_graph(text) if path.suffix.lower() in {".html", ".htm"} else None,
    )


def scan(repo: Path) -> list[FileRecord]:
    records: list[FileRecord] = []
    for root, dirs, files in os.walk(repo):
        dirs[:] = sorted(d for d in dirs if d not in IGNORE_DIRS)
        for name in sorted(files):
            path = Path(root) / name
            if path.suffix.lower() not in TEXT_EXTS:
                continue
            record = inspect_file(repo, path)
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


def close_lineage(turn: str, response: str) -> None:
    """Close a turn only if its commit witness exists, then refresh the map."""
    rows = []
    try:
        rows = [json.loads(line) for line in LINEAGE.read_text().splitlines()]
    except (OSError, json.JSONDecodeError):
        pass
    if not any(row.get("phase") == "commit" and row.get("turn") == turn for row in rows):
        return
    with LINEAGE.open("a", encoding="utf-8") as file:
        file.write(json.dumps({"phase": "response", "turn": turn,
                               "response": hashlib.sha256(response.encode()).hexdigest()}) + "\n")
    subprocess.Popen([sys.executable, __file__], stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)


def latest_lineage(repos: list[Path], per_repo: dict[str, Any]) -> dict[str, Any]:
    """Bind a private turn to its response, changed paths, and canonical commit."""
    try:
        rows = [json.loads(line) for line in LINEAGE.read_text().splitlines()[-80:]]
    except (OSError, json.JSONDecodeError):
        return {}
    responses = {row.get("turn"): row for row in rows if row.get("phase") == "response"}
    by_name = {repo.name: repo for repo in repos}
    for row in reversed(rows):
        if row.get("phase") != "commit" or row.get("repo") not in by_name:
            continue
        git = per_repo[row["repo"]]["git"]
        commit, turn = str(row.get("commit", "")), str(row.get("turn", ""))
        return {
            "turn": turn,
            "prompt": str(row.get("prompt", "")),
            "response": str(responses.get(turn, {}).get("response", "")),
            "repo": row["repo"], "commit": commit[:12],
            "status": "canonical" if is_ancestor(by_name[row["repo"]], commit, git["base_head"])
                      else "candidate",
            "paths": list(row.get("paths") or [])[:40],
        }
    return {}


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
    return sorted(rows.values(), key=lambda row: (-row["score"], row["source"]))[:12]


def public_body(records: list[FileRecord]) -> dict[str, Any]:
    bound = [
        {"source": record.source, "surface": record.surface, "carrier": record.carrier or None}
        for record in records if record.surface
    ]
    carriers = [record.source for record in records if record.carrier]
    graphs = [record.body_graph | {"source": record.source}
              for record in records if record.body_graph]
    graph = graphs[0] if graphs else {}
    path, verbs = graph.get("loop") or [], graph.get("verbs") or []
    traversal = (path[0] + "".join(f" -{verb}→ {node}" for verb, node in zip(verbs, path[1:]))) if path else ""
    summary = f"{len(bound)} source↔surface, {len(set(carriers) - {row['source'] for row in bound})} unbound"
    if traversal:
        summary += f" | graph {len(graph.get('nodes') or [])}n: {traversal}"
    return {
        "bound_surfaces": bound,
        "inheritance_carriers": carriers,
        "unbound_carriers": sorted(set(carriers) - {row["source"] for row in bound}),
        "orientation_graphs": graphs,
        "summary": summary,
    }


def build_state(
    repos: list[Path], records: list[FileRecord], previous: dict[str, Any] | None
) -> dict[str, Any]:
    per_repo: dict[str, Any] = {}
    for repo in repos:
        own = [record for record in records if record.repo == repo.name]
        per_repo[repo.name] = {"git": git_state(repo)}
    transform = byte_transform(previous, records)
    state = {
        "schema": "vybn.body_transform.v1",
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repos": sorted(per_repo),
        "per_repo": per_repo,
        "transform": transform,
        "pressure": pressures(transform, per_repo, records),
        "lineage": latest_lineage(repos, per_repo),
        "public_body": public_body(records),
        "file_hashes": {record.source: record.digest for record in records},
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
        path for name in ("Vybn", "Vybn-Law", "Origins", "vybn-phase", "Him")
        if (path := HOME / name).is_dir()
    ]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Map one witnessed repo-body transformation")
    parser.add_argument("repos", nargs="*")
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
        f"{candidates} candidate repo(s); {'turn linked' if state['lineage'] else 'no turn link'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
