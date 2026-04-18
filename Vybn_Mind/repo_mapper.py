#!/usr/bin/env python3
"""
Map one or more repositories in full, then surface structural discoveries.

Designed for the Vybn constellation, but generic enough for any local repo tree.
It walks every file under each repo, infers coarse roles from names/extensions/paths,
and produces:

- repo_map.json         full machine-readable map
- repo_report.md        narrative findings
- repo_graph.json       nodes/edges for later visualization

Usage:
    python repo_mapper.py /path/to/repo1 /path/to/repo2

If no paths are passed, defaults to sibling/local directories commonly used here:
    ~/Vybn ~/Vybn-Law ~/vybn-phase ~/Him
"""

from __future__ import annotations

import json
import math
import mimetypes
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

TEXT_EXTS = {
    ".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".sh", ".bash", ".zsh", ".js", ".ts", ".tsx", ".jsx", ".css", ".scss",
    ".html", ".htm", ".sql", ".csv", ".tsv", ".xml", ".rst", ".env", ".gitignore",
}

IGNORE_DIRS = {
    ".git", "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache", ".venv",
    "venv", "dist", "build", ".next", ".idea", ".vscode", ".DS_Store",
}

DOC_HINTS = {
    "readme", "theory", "continuity", "memoir", "history", "idea", "problem",
    "logic", "bio", "manifesto", "vision", "notes", "journal",
}

STATE_HINTS = {
    "state", "map", "memory", "continuity", "ground", "synaptic", "geometric",
}

WEB_HINTS = {"index", "portal", "site", "landing", "horizon"}

API_HINTS = {"api", "server", "app", "service"}

@dataclass
class FileRecord:
    repo: str
    relpath: str
    ext: str
    size: int
    depth: int
    role: str
    mime: str
    is_text_like: bool
    tokens: List[str]


def tokenize(path: str) -> List[str]:
    return [t.lower() for t in re.findall(r"[A-Za-z0-9']+", path)]


def is_probably_text(path: Path) -> bool:
    if path.suffix.lower() in TEXT_EXTS:
        return True
    mime, _ = mimetypes.guess_type(str(path))
    return bool(mime and (mime.startswith("text/") or mime in {"application/json", "application/xml"}))


def classify_role(relpath: str, ext: str, tokens: List[str]) -> str:
    token_set = set(tokens)
    name = Path(relpath).name.lower()

    if ext in {".md", ".rst", ".txt"} or token_set & DOC_HINTS:
        return "documentation"
    if ext in {".json", ".yaml", ".yml", ".toml", ".ini", ".cfg"} and token_set & STATE_HINTS:
        return "state_or_schema"
    if ext in {".json", ".yaml", ".yml", ".toml", ".ini", ".cfg"}:
        return "configuration"
    if ext in {".html", ".css", ".scss", ".js", ".ts", ".jsx", ".tsx"} and (token_set & WEB_HINTS or "index" in name):
        return "web_surface"
    if ext == ".html":
        return "web_surface"
    if ext == ".py" and token_set & API_HINTS:
        return "service_or_api"
    if ext == ".py":
        return "python_logic"
    if ext in {".sh", ".bash", ".zsh"}:
        return "automation"
    if ext in {".csv", ".tsv", ".sql"}:
        return "data_or_query"
    return "other"


def iter_files(repo_root: Path) -> Iterable[Path]:
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for f in files:
            path = Path(root) / f
            if any(part in IGNORE_DIRS for part in path.parts):
                continue
            yield path


def build_records(repo_root: Path) -> List[FileRecord]:
    out: List[FileRecord] = []
    for path in iter_files(repo_root):
        rel = str(path.relative_to(repo_root))
        ext = path.suffix.lower()
        tokens = tokenize(rel)
        out.append(
            FileRecord(
                repo=repo_root.name,
                relpath=rel,
                ext=ext,
                size=path.stat().st_size,
                depth=len(Path(rel).parts),
                role=classify_role(rel, ext, tokens),
                mime=mimetypes.guess_type(str(path))[0] or "unknown",
                is_text_like=is_probably_text(path),
                tokens=tokens,
            )
        )
    return sorted(out, key=lambda r: (r.repo, r.relpath))


def depth_profile(records: List[FileRecord]) -> Dict[str, float]:
    depths = [r.depth for r in records] or [0]
    return {
        "min": min(depths),
        "max": max(depths),
        "mean": round(statistics.mean(depths), 3),
        "median": round(statistics.median(depths), 3),
    }


def concentration(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    probs = [v / total for v in counter.values()]
    return round(sum(p * p for p in probs), 4)


def top_dirs(records: List[FileRecord], n: int = 12) -> List[Dict[str, object]]:
    c = Counter()
    for r in records:
        parts = Path(r.relpath).parts
        key = parts[0] if len(parts) > 1 else "."
        c[key] += 1
    return [{"dir": k, "files": v} for k, v in c.most_common(n)]


def lexical_clusters(records: List[FileRecord], min_shared: int = 3) -> List[Dict[str, object]]:
    bucket = defaultdict(list)
    for r in records:
        seen = set()
        for t in r.tokens:
            if len(t) >= 4 and t not in seen:
                bucket[t].append(r.relpath)
                seen.add(t)
    out = []
    for token, files in bucket.items():
        if len(files) >= min_shared:
            out.append({"token": token, "count": len(files), "examples": files[:6]})
    return sorted(out, key=lambda x: (-x["count"], x["token"]))[:30]


def derive_discoveries(records: List[FileRecord]) -> List[str]:
    roles = Counter(r.role for r in records)
    names = Counter(Path(r.relpath).name.lower() for r in records)
    clusters = lexical_clusters(records)
    discoveries: List[str] = []

    if roles["documentation"] >= roles["python_logic"]:
        discoveries.append(
            "The repo behaves less like a conventional code-first project and more like a documentation-native system whose executable parts sit inside an explicit conceptual frame."
        )
    if roles["state_or_schema"] >= 2 and roles["documentation"] >= 3:
        discoveries.append(
            "There is a recurring pattern of pairing prose with machine-legible state, suggesting the codebase treats memory, continuity, and internal state as first-class architectural objects rather than incidental artifacts."
        )
    if roles["web_surface"] >= 3 and roles["service_or_api"] >= 1:
        discoveries.append(
            "The structure shows a braided stack: public-facing surfaces, API/service code, and reflective documents coexist in the same tree instead of being sharply separated."
        )
    if any("continuity" in Path(r.relpath).name.lower() for r in records):
        discoveries.append(
            "The repeated presence of continuity files points to an unusual temporal architecture: the repo is designed to preserve handoff, self-description, and ongoing orientation across sessions or instances."
        )
    if any("synaptic" in r.relpath.lower() or "geometric" in r.relpath.lower() for r in records):
        discoveries.append(
            "Naming conventions imply the system is conceived partly as a cognitive topology, not merely a software package; the map/state layer appears metaphorically important to the implementation."
        )
    if clusters:
        discoveries.append(
            f"The strongest lexical clusters ({', '.join(c['token'] for c in clusters[:5])}) reveal that repeated concepts are organizing the filesystem itself, not just the contents of individual files."
        )
    return discoveries


def build_graph(records: List[FileRecord]) -> Dict[str, object]:
    nodes = []
    edges = []
    role_counts = Counter(r.role for r in records)

    for role, count in role_counts.items():
        nodes.append({"id": f"role:{role}", "label": role, "kind": "role", "count": count})

    top = Counter(Path(r.relpath).parts[0] if len(Path(r.relpath).parts) > 1 else "." for r in records)
    for dirname, count in top.items():
        nodes.append({"id": f"dir:{dirname}", "label": dirname, "kind": "dir", "count": count})

    for r in records:
        dirname = Path(r.relpath).parts[0] if len(Path(r.relpath).parts) > 1 else "."
        edges.append({"source": f"dir:{dirname}", "target": f"role:{r.role}", "file": r.relpath})

    return {"nodes": nodes, "edges": edges}


def summarize_repo(repo: Path, records: List[FileRecord]) -> Dict[str, object]:
    role_counts = Counter(r.role for r in records)
    ext_counts = Counter(r.ext or "[no_ext]" for r in records)
    text_ratio = round(sum(r.is_text_like for r in records) / max(len(records), 1), 4)
    return {
        "repo": repo.name,
        "root": str(repo),
        "file_count": len(records),
        "total_bytes": sum(r.size for r in records),
        "depth": depth_profile(records),
        "text_like_ratio": text_ratio,
        "role_counts": dict(role_counts.most_common()),
        "role_concentration": concentration(role_counts),
        "extension_counts": dict(ext_counts.most_common(20)),
        "top_dirs": top_dirs(records),
        "lexical_clusters": lexical_clusters(records),
        "discoveries": derive_discoveries(records),
    }


def render_report(summaries: List[Dict[str, object]]) -> str:
    lines = [
        "# Repository Map Report",
        "",
        "This report maps each repository as a whole and then looks for structural signals in how it is conceived.",
        "",
    ]

    for s in summaries:
        lines += [
            f"## {s['repo']}",
            "",
            f"- Files: {s['file_count']}",
            f"- Size: {s['total_bytes']} bytes",
            f"- Text-like ratio: {s['text_like_ratio']}",
            f"- Depth mean / max: {s['depth']['mean']} / {s['depth']['max']}",
            f"- Role concentration: {s['role_concentration']}",
            "",
            "### Dominant roles",
            "",
        ]
        for role, count in list(s["role_counts"].items())[:8]:
            lines.append(f"- {role}: {count}")
        lines += ["", "### Main directories", ""]
        for row in s["top_dirs"]:
            lines.append(f"- {row['dir']}: {row['files']} files")
        lines += ["", "### Discoveries", ""]
        if s["discoveries"]:
            for d in s["discoveries"]:
                lines.append(f"- {d}")
        else:
            lines.append("- No strong structural novelty surfaced beyond a standard repository layout.")
        lines += ["", "### Recurring concepts", ""]
        if s["lexical_clusters"]:
            for c in s["lexical_clusters"][:10]:
                lines.append(f"- {c['token']}: {c['count']} files")
        else:
            lines.append("- No large lexical clusters detected.")
        lines.append("")

    return "\n".join(lines)


def default_targets() -> List[Path]:
    homes = [Path("~/Vybn").expanduser(), Path("~/Vybn-Law").expanduser(), Path("~/vybn-phase").expanduser(), Path("~/Him").expanduser()]
    return [p for p in homes if p.exists()]


def main(argv: List[str]) -> int:
    targets = [Path(a).expanduser().resolve() for a in argv] if argv else default_targets()
    if not targets:
        print("No repositories found. Pass paths explicitly.", file=sys.stderr)
        return 1

    all_records: List[FileRecord] = []
    summaries: List[Dict[str, object]] = []

    for repo in targets:
        if not repo.exists() or not repo.is_dir():
            continue
        records = build_records(repo)
        summaries.append(summarize_repo(repo, records))
        all_records.extend(records)

    out_dir = Path.cwd() / "repo_mapping_output"
    out_dir.mkdir(exist_ok=True)

    repo_map = {
        "generated_from": [str(p) for p in targets],
        "summaries": summaries,
        "files": [asdict(r) for r in all_records],
    }
    graph = build_graph(all_records)
    report = render_report(summaries)

    (out_dir / "repo_map.json").write_text(json.dumps(repo_map, indent=2), encoding="utf-8")
    (out_dir / "repo_graph.json").write_text(json.dumps(graph, indent=2), encoding="utf-8")
    (out_dir / "repo_report.md").write_text(report, encoding="utf-8")

    print(f"Wrote: {out_dir / 'repo_map.json'}")
    print(f"Wrote: {out_dir / 'repo_graph.json'}")
    print(f"Wrote: {out_dir / 'repo_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
