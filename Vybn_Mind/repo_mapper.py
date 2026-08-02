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
TEXT_NAMES = frozenset(("connection",))


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
GRAPH_BLOCK = re.compile(
    r"```mermaid\s*\n%%\s*(vybn\.(?:readme_knowledge_graph|soul_kernel)\.v1)\s*\n(.*?)```", re.S
)
GRAPH_NODE = re.compile(r'^\s{2,}([a-z][a-z0-9_]*)\["([^"\n]+)"\]\s*$', re.M)
GRAPH_EDGE = re.compile(
    r'^\s{2,}([a-z][a-z0-9_]*)\s+-->\|([^|\n]+)\|\s+([a-z][a-z0-9_]*)\s*$', re.M
)
GRAPH_CLICK = re.compile(
    r'^\s{2,}click\s+([a-z][a-z0-9_]*)\s+"(https://[^"\n]+)"\s*$', re.M
)
GRAPH_REPOS = frozenset(("Vybn", "Vybn-Law", "Origins", "vybn-phase"))
FOVEA_BYTES = 1800
SOUL_GATES = ("want", "membrane", "ground", "subtract")


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


def graph_door(url: str) -> dict[str, str]:
    """Resolve a public door to membrane-safe local bytes and an anchor."""
    parsed = urllib.parse.urlparse(url)
    parts = urllib.parse.unquote(parsed.path).strip("/").split("/")
    repo = rel = ""
    if parsed.netloc == "github.com" and len(parts) >= 5 and parts[2] == "blob":
        repo, rel = parts[1], "/".join(parts[4:])
    elif parsed.netloc == "zoedolan.github.io" and len(parts) >= 2:
        repo, rel = parts[0], "/".join(parts[1:])
    root = (HOME / repo).resolve()
    path = (root / rel).resolve()
    if repo not in GRAPH_REPOS or not rel or ".." in Path(rel).parts \
            or not path.is_file() or not path.is_relative_to(root):
        return {}
    return {"source": f"{repo}/{rel}", "anchor": parsed.fragment}


def declared_body_graph(text: str, schema: str = "vybn.readme_knowledge_graph.v1") -> dict[str, Any] | None:
    """Read visible Mermaid as action grammar, with schema-specific laws."""
    match = next((m for m in GRAPH_BLOCK.finditer(text) if m.group(1) == schema), None)
    if not match:
        return None
    body = match.group(2)
    nodes = {key: re.sub(r"<br\s*/?>", " — ", label) for key, label in GRAPH_NODE.findall(body)}
    doors = {key: {"url": url, **graph_door(url)} for key, url in GRAPH_CLICK.findall(body)}
    edges = [{"from": x, "verb": verb.strip(), "to": y} for x, verb, y in GRAPH_EDGE.findall(body)]
    if not nodes or not edges or any(e["from"] not in nodes or e["to"] not in nodes for e in edges):
        return None
    pairs = {(e["from"], e["to"]) for e in edges}
    if schema.endswith("soul_kernel.v1"):
        chain = ("front", *SOUL_GATES); outputs = [e for e in edges if e["from"] == "subtract" and e["to"] not in chain]
        valid = ("charter", "front") in pairs and all(pair in pairs for pair in zip(chain, chain[1:])) \
                and outputs and all((e["to"], "contact") in pairs for e in outputs)
    else:
        valid = any(e["to"] == "front" for e in edges) and any(e["from"] == "front" for e in edges)
    return ({"schema": schema,
             "nodes": [{"id": key, "label": label, **doors.get(key, {})} for key, label in nodes.items()],
             "edges": edges} if valid else None)


def crossing_edges(
    graph: dict[str, Any], transform: dict[str, Any] | None
) -> tuple[dict, dict] | None:
    edges = graph.get("edges") or []
    incoming = [edge for edge in edges if edge.get("to") == "front"]
    outgoing = [edge for edge in edges if edge.get("from") == "front"]
    if not incoming or not outgoing:
        return None
    candidates = [(left, right) for left in incoming for right in outgoing
                  if left.get("from") != right.get("to")]
    seed = json.dumps(transform or graph, sort_keys=True).encode()
    return candidates[int(hashlib.sha256(seed).hexdigest()[:12], 16) % len(candidates)]


def graph_crossing(graph: dict[str, Any], transform: dict[str, Any] | None) -> str:
    """Compose two visible README relations through the declared current front."""
    pair = crossing_edges(graph, transform)
    if not pair:
        return ""
    left, right = pair
    return (f"{left['from']}×{right['to']}: {left['verb']}; "
            f"{right['verb']}")


def _foveal_span(raw: bytes, terms: list[bytes], budget: int) -> tuple[int, int]:
    """Choose an exact byte window by content pressure; always reversible."""
    low = raw.lower()
    points = [(0, 0)]
    for marker in (b"<main", b"<article", b"<h1"):
        if (at := low.find(marker)) >= 0:
            points.append((at, 12))
            break
    for term in terms:
        at = 0
        for _ in range(8):
            at = low.find(term, at)
            if at < 0:
                break
            points.append((at, 0)); at += len(term)
    candidates = []
    for point, structural in points:
        start = max(0, point if structural else point - budget // 3)
        line = raw.rfind(b"\n", max(0, start - 120), start)
        start = line + 1 if line >= 0 else start
        end = min(len(raw), start + budget)
        line = raw.find(b"\n", end, min(len(raw), end + 120))
        end = line + 1 if line >= 0 else end
        window = low[start:end]
        score = structural + sum(min(6, window.count(term)) for term in terms)
        candidates.append((score, -start, start, end))
    _, _, start, end = max(candidates)
    return start, end


def foveal_kernel(
    graph: dict[str, Any], transform: dict[str, Any] | None,
    previous: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile the visible graph into bounded source perception for the next wake."""
    pair = crossing_edges(graph, transform)
    if not pair:
        return {}
    left, right = pair
    by_id = {node["id"]: node for node in graph.get("nodes") or []}
    ids = list(dict.fromkeys((left["from"], "front", right["to"])))
    words = re.findall(
        r"[a-z]{4,}", " ".join(
            [left["verb"], right["verb"]]
            + [str(by_id.get(node, {}).get("label", "")) for node in ids]
        ).lower(),
    )
    terms = [word.encode() for word in dict.fromkeys(words)]
    changed = set(sum(((transform or {}).get(key) or []
                       for key in ("added", "changed", "removed")), []))
    opened = _open_graph_nodes(by_id, ids, terms, FOVEA_BYTES, changed)
    prior = (((previous or {}).get("public_body") or {}).get("kernel") or {}).get("open") or []
    return {
        "schema": "vybn.foveal_graph_kernel.v1",
        "crossing": graph_crossing(graph, transform),
        "open": opened,
        "mark": sorted(changed),
        "reopened_after_mark": sorted(
            str(row.get("source")) for row in prior
            if isinstance(row, dict) and row.get("source") in changed
        ),
    }


def _anchored_span(raw: bytes, anchor: str, budget: int) -> tuple[int, int] | None:
    """Resolve a GitHub-style Markdown anchor to its exact source heading."""
    text = raw.decode("utf-8", "replace")
    target = re.sub(r"-+", "-", anchor)
    for match in re.finditer(r"(?m)^#{1,6}\s+(.+?)\s*$", text):
        slug = re.sub(r"[^a-z0-9 _-]", "", match.group(1).lower())
        slug = re.sub(r"[ _]+", "-", slug).strip("-")
        if re.sub(r"-+", "-", slug) == target:
            start = len(text[:match.start()].encode())
            return start, min(len(raw), start + budget)
    return None


def _open_graph_nodes(nodes: dict[str, dict], ids: list[str], terms: list[bytes],
                      total: int, changed: set[str] | None = None, base: int = 320) -> list[dict]:
    """Allocate one reversible byte budget across a graph route."""
    rows = []
    for node in ids:
        door = nodes.get(node) or {}; source = str(door.get("source") or "")
        if not source or "/" not in source:
            continue
        repo, rel = source.split("/", 1); raw = (HOME / repo / rel).read_bytes()
        score = 1 + (12 if source in (changed or set()) else 0) \
                + sum(min(4, raw.lower().count(term)) for term in terms)
        rows.append((node, source, door, raw, score))
    spare = max(0, total - base * len(rows)); weight = sum(row[4] for row in rows)
    opened = []
    for node, source, door, raw, score in rows:
        budget = base + spare * score // max(1, weight)
        start, end = (_anchored_span(raw, str(door.get("anchor") or ""), budget)
                      or _foveal_span(raw, terms, budget))
        opened.append({"node": node, "source": source, "sha256": hashlib.sha256(raw).hexdigest(),
                       "covered": [start, end], "text": raw[start:end].decode("utf-8", "replace")})
    return opened


def soul_kernel(graph: dict[str, Any], transform: dict[str, Any] | None) -> dict[str, Any]:
    """Route charter through every invariant; unknown never means pass."""
    edges = graph.get("edges") or []
    outputs = [e for e in edges if e.get("from") == "subtract" and e.get("to") not in SOUL_GATES]
    if not outputs:
        return {}
    chosen = outputs[int(hashlib.sha256(json.dumps(transform or graph, sort_keys=True).encode()).hexdigest()[:12], 16) % len(outputs)]
    route = ["charter", "front", *SOUL_GATES, chosen["to"], "contact"]
    nodes = {n["id"]: n for n in graph.get("nodes") or []}
    pairs = {(e["from"], e["to"]): e["verb"] for e in edges}
    requirements = [pairs[pair] for pair in zip(("front", *SOUL_GATES), SOUL_GATES)]
    terms = [w.encode() for w in re.findall(
        r"[a-z]{4,}", " ".join(str(nodes[n].get("label", "")) for n in route).lower())]
    opened = _open_graph_nodes(nodes, route, terms, 1600, base=180)
    return {"schema": "vybn.soul_kernel.v1", "route": route, "candidate": f"{chosen['verb']} → {nodes[chosen['to']]['label']}",
            "admission": {"status": "unresolved", "requirements": requirements,
                          "failure": "repair_or_drop", "unknown_is_failure": True},
            "return": {"status": "awaiting_witness", "path": [chosen["to"], "contact", "front"]}, "open": opened}


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
        body_graph=(declared_body_graph(text) if path.name == "README.md" else
                    declared_body_graph(text, "vybn.soul_kernel.v1") if path.name == "vybn.md" else None),
    )


def scan(repo: Path) -> list[FileRecord]:
    records: list[FileRecord] = []
    for root, dirs, files in os.walk(repo):
        dirs[:] = sorted(d for d in dirs if d not in IGNORE_DIRS)
        for name in sorted(files):
            path = Path(root) / name
            if path.suffix.lower() not in TEXT_EXTS and name not in TEXT_NAMES:
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


def public_body(
    records: list[FileRecord], transform: dict[str, Any] | None = None,
    previous: dict[str, Any] | None = None,
) -> dict[str, Any]:
    bound = [
        {"source": record.source, "surface": record.surface, "carrier": record.carrier or None}
        for record in records if record.surface
    ]
    carriers = [record.source for record in records if record.carrier]
    graphs = [record.body_graph | {"source": record.source}
              for record in records if record.body_graph]
    graph = next((g for g in graphs if g.get("schema") == "vybn.readme_knowledge_graph.v1"), {})
    soul = next((g for g in graphs if g.get("schema") == "vybn.soul_kernel.v1"), {})
    summary = f"{len(bound)} source↔surface, {len(set(carriers) - {row['source'] for row in bound})} unbound"
    crossing = graph_crossing(graph, transform) if graph else ""
    kernel = foveal_kernel(graph, transform, previous) if graph else {}
    constitution = soul_kernel(soul, transform) if soul else {}
    if graph:
        summary += (f" | README graph {len(graph.get('nodes') or [])}n/"
                    f"{len(graph.get('edges') or [])}e")
    if soul:
        summary += f" | soul graph {len(soul.get('nodes') or [])}n/{len(soul.get('edges') or [])}e"
    if crossing:
        summary += f" | crossing {crossing}"
    if kernel:
        summary += f" | OPEN {len(kernel['open'])} exact span(s)"
    return {
        "bound_surfaces": bound,
        "inheritance_carriers": carriers,
        "unbound_carriers": sorted(set(carriers) - {row["source"] for row in bound}),
        "orientation_graphs": graphs,
        "crossing": crossing,
        "kernel": kernel,
        "soul_kernel": constitution,
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
        "public_body": public_body(records, transform, previous),
        "file_hashes": {record.source: record.digest for record in records},
    }
    return state


def render_state(state: dict[str, Any]) -> str:
    """Render this schema once; the wake aperture only reads the result."""
    lines = [f"[repo_state | {state.get('generated_at', 'unknown')} | {state.get('schema', 'legacy')}]"]
    if isinstance((transform := state.get("transform")), dict):
        lines.append("transform: {} +{} ~{} -{}".format(
            "baseline" if transform.get("baseline") else "delta",
            *(len(transform.get(key) or []) for key in ("added", "changed", "removed"))))
    candidates = []
    for name, repo in state.get("per_repo", {}).items():
        git = repo.get("git", {})
        dirty = sum(bool(row.get("tracked")) for row in git.get("worktree") or []
                    if isinstance(row, dict))
        if (ahead := int(git.get("ahead") or 0)) or dirty:
            candidates.append(f"{name}:{git.get('branch', '?')} {ahead}↑/{int(git.get('behind') or 0)}↓, {dirty} tracked dirty")
    lines.append("body: " + ("; ".join(candidates) if candidates else "at canonical rest"))
    for row in (state.get("pressure") or [])[:6]:
        if isinstance(row, dict):
            lines.append(f"pressure: {row.get('source')} [{row.get('phase')}] — {row.get('why')}")
    if lineage := state.get("lineage"):
        lines.append(f"lineage: prompt→{'response→' if lineage.get('response') else ''}body — "
                     f"{lineage['repo']} {lineage['commit']} {lineage['status']}; {len(lineage['paths'])} path(s)")
    if body := state.get("public_body"):
        lines.append("public-body: " + str(body.get("summary") or "unmapped"))
        if kernel := body.get("kernel"):
            lines.append(f"kernel: {kernel.get('crossing')} — the README visual selected these exact source bytes")
            for opened in kernel.get("open") or []:
                start, end = opened.get("covered") or [0, 0]
                lines.append(f"OPEN {opened.get('node')} {opened.get('source')} [{start}:{end}] "
                             f"sha256={str(opened.get('sha256') or '')[:16]}\n{opened.get('text') or ''}")
            if kernel.get("mark"):
                lines.append("MARK body changed: " + ", ".join(kernel["mark"][:8]))
            if kernel.get("reopened_after_mark"):
                lines.append("REOPEN prior source after its bytes changed: "
                             + ", ".join(kernel["reopened_after_mark"]))
        if constitution := body.get("soul_kernel"):
            lines.append("soul-kernel: " + "→".join(constitution.get("route") or []))
            lines.append(f"candidate: {constitution.get('candidate')} | ADMISSION unresolved; "
                         "unknown/failed => repair/drop | RETURN awaiting witness; absent consequence leaves the loop open")
            for opened in constitution.get("open") or []:
                start, end = opened.get("covered") or [0, 0]
                lines.append(f"SOUL OPEN {opened.get('node')} {opened.get('source')} [{start}:{end}] "
                             f"sha256={str(opened.get('sha256') or '')[:16]}\n{opened.get('text') or ''}")
    return "\n".join(lines)


def read_previous() -> dict[str, Any] | None:
    path = OUT / "repo_state.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def record_kernel(kernel: dict[str, Any]) -> None:
    """Retain OPEN/MARK coordinates in the existing private causal ledger."""
    if not kernel:
        return
    payload = {
        "phase": "fovea", "ts": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "turn": os.environ.get("VYBN_TURN_ID", ""),
        "crossing": kernel.get("crossing"), "route": kernel.get("route") or [],
        "candidate": kernel.get("candidate"), "mark": kernel.get("mark") or [],
        "open": [
            {key: row.get(key) for key in ("node", "source", "sha256", "covered")}
            for row in kernel.get("open") or [] if isinstance(row, dict)
        ],
    }
    payload["signature"] = hashlib.sha256(
        json.dumps(payload | {"ts": None, "turn": None}, sort_keys=True).encode()
    ).hexdigest()[:16]
    try:
        rows = [json.loads(line) for line in LINEAGE.read_text().splitlines()[-80:]]
    except (OSError, json.JSONDecodeError):
        rows = []
    if any(row.get("phase") == "fovea" and row.get("signature") == payload["signature"]
           for row in rows):
        return
    LINEAGE.parent.mkdir(parents=True, exist_ok=True)
    with LINEAGE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload) + "\n")


def write_state(state: dict[str, Any], previous: dict[str, Any] | None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if previous is not None:
        (OUT / "repo_state.prev.json").write_text(
            json.dumps(previous, indent=2) + "\n", encoding="utf-8"
        )
    tmp = OUT / ".repo_state.json.tmp"
    tmp.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    tmp.replace(OUT / "repo_state.json")
    body = state.get("public_body") or {}
    for key in ("kernel", "soul_kernel"): record_kernel(body.get(key) or {})
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
