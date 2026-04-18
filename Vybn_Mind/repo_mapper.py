#!/usr/bin/env python3
"""
repo_mapper.py  v2
==================
Walk one or more repos, read every text file, and surface what is
actually in the codebase: function/class names, headings, imports,
explicit cross-references, TODO/FIXME markers, duplicate content,
the largest files, the files that changed most recently, and the
concepts that recur across repo boundaries.

Outputs to ./repo_mapping_output/
  repo_map.json       — machine-readable full record
  repo_graph.json     — concept graph (nodes + edges, importable into networkx / gephi)
  repo_report.md      — readable findings

Usage:
    python3 repo_mapper.py                       # auto-finds ~/Vybn ~/Vybn-Law ~/vybn-phase ~/Him
    python3 repo_mapper.py ~/Vybn ~/Vybn-Law     # explicit repos
"""

from __future__ import annotations

import ast
import hashlib
import json
import mimetypes
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── constants ────────────────────────────────────────────────────────────────

TEXT_EXTS = {
    ".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".sh", ".bash", ".zsh", ".js", ".ts", ".tsx", ".jsx", ".css", ".scss",
    ".html", ".htm", ".sql", ".csv", ".tsv", ".xml", ".rst", ".env",
}

IGNORE_DIRS = {
    ".git", "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".venv", "venv", "dist", "build", ".next", ".idea", ".vscode",
}

READ_SIZE_LIMIT = 200_000   # bytes — skip binary blobs above this
SNIPPET_LEN     = 120        # chars per snippet in report

# ── data model ───────────────────────────────────────────────────────────────

@dataclass
class FileRecord:
    repo:         str
    relpath:      str
    ext:          str
    size:         int
    depth:        int
    mtime:        float
    content_hash: str
    # extracted
    headings:     List[str] = field(default_factory=list)
    py_defs:      List[str] = field(default_factory=list)   # functions + classes
    py_imports:   List[str] = field(default_factory=list)
    todos:        List[str] = field(default_factory=list)
    cross_refs:   List[str] = field(default_factory=list)   # mentions of other repo names
    concept_freq: Dict[str, int] = field(default_factory=dict)  # word → count in file
    read_error:   Optional[str] = None


# ── helpers ──────────────────────────────────────────────────────────────────

STOPWORDS = {
    "the","a","an","and","or","of","to","in","is","it","that","for",
    "on","with","as","at","by","from","this","be","are","was","were",
    "have","has","not","but","if","do","so","we","you","he","she","they",
    "i","my","our","your","its","their","will","can","all","one","more",
    "also","than","when","up","out","about","into","then","no","what",
    "there","been","which","would","could","should","may","might","each",
    "these","those","them","him","her","his","some","any","just","like",
    "use","using","used","new","get","set","add","run","make","via",
    "how","who","why","where","after","before","over","under","within",
    "return","import","from","class","def","self","true","false","none",
    "pass","raise","else","elif","except","try","with","yield","async",
    "await","lambda","global","del","assert","break","continue","while","for",
    "print","type","list","dict","str","int","bool","path","file","data",
    "value","values","key","keys","name","names","line","lines","text","item",
    "items","result","results","output","input","error","args","kwargs",
    "config","content","html","json","yaml","md","py","sh","css","js","ts",
}

def tokenize_words(text: str, min_len: int = 4) -> List[str]:
    return [w.lower() for w in re.findall(r"[A-Za-z][a-z'A-Z]*", text)
            if len(w) >= min_len and w.lower() not in STOPWORDS]

def md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()[:12]

def extract_md_headings(text: str) -> List[str]:
    return [m.group(1).strip() for m in re.finditer(r"^#{1,4}\s+(.+)", text, re.MULTILINE)]

def extract_py_defs(text: str) -> List[str]:
    try:
        tree = ast.parse(text)
        return [n.name for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    except SyntaxError:
        return [m.group(1) for m in re.finditer(r"^(?:def|class)\s+(\w+)", text, re.MULTILINE)]

def extract_py_imports(text: str) -> List[str]:
    imports = []
    for m in re.finditer(r"^(?:import|from)\s+([\w.]+)", text, re.MULTILINE):
        imports.append(m.group(1).split(".")[0])
    return list(dict.fromkeys(imports))

def extract_todos(text: str) -> List[str]:
    return [m.group(0).strip()[:SNIPPET_LEN]
            for m in re.finditer(r"(?:TODO|FIXME|HACK|XXX|NOTE)[:\s].{0,100}", text, re.IGNORECASE)]

def extract_cross_refs(text: str, all_repo_names: List[str]) -> List[str]:
    found = []
    for name in all_repo_names:
        if re.search(re.escape(name), text, re.IGNORECASE):
            found.append(name)
    return found

def read_text(path: Path) -> Tuple[Optional[str], Optional[str]]:
    if path.stat().st_size > READ_SIZE_LIMIT:
        return None, "too_large"
    for enc in ("utf-8", "latin-1"):
        try:
            return path.read_text(encoding=enc, errors="strict"), None
        except (UnicodeDecodeError, PermissionError):
            continue
    return None, "decode_error"

# ── file walker ──────────────────────────────────────────────────────────────

def iter_text_paths(repo_root: Path):
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in sorted(dirs) if d not in IGNORE_DIRS]
        for f in sorted(files):
            path = Path(root) / f
            if path.suffix.lower() in TEXT_EXTS:
                yield path

def build_records(repo_root: Path, all_repo_names: List[str]) -> List[FileRecord]:
    records = []
    for path in iter_text_paths(repo_root):
        rel  = str(path.relative_to(repo_root))
        ext  = path.suffix.lower()
        stat = path.stat()
        text, err = read_text(path)
        rec = FileRecord(
            repo         = repo_root.name,
            relpath      = rel,
            ext          = ext,
            size         = stat.st_size,
            depth        = len(Path(rel).parts),
            mtime        = stat.st_mtime,
            content_hash = md5(text) if text else "",
            read_error   = err,
        )
        if text:
            rec.headings   = extract_md_headings(text) if ext in {".md", ".rst", ".txt"} else []
            rec.py_defs    = extract_py_defs(text)     if ext == ".py" else []
            rec.py_imports = extract_py_imports(text)  if ext == ".py" else []
            rec.todos      = extract_todos(text)
            rec.cross_refs = extract_cross_refs(text, all_repo_names)
            words          = tokenize_words(text)
            freq           = Counter(words)
            rec.concept_freq = dict(freq.most_common(30))
        records.append(rec)
    return records

# ── analysis ─────────────────────────────────────────────────────────────────

def duplicate_files(all_records: List[FileRecord]) -> List[Dict]:
    by_hash: Dict[str, List[FileRecord]] = defaultdict(list)
    for r in all_records:
        if r.content_hash:
            by_hash[r.content_hash].append(r)
    dupes = []
    for h, group in by_hash.items():
        if len(group) > 1:
            dupes.append({
                "hash":  h,
                "files": [f"{r.repo}/{r.relpath}" for r in group],
                "size":  group[0].size,
            })
    return sorted(dupes, key=lambda x: -x["size"])

def largest_files(records: List[FileRecord], n: int = 15) -> List[Dict]:
    return [{"file": f"{r.repo}/{r.relpath}", "size": r.size}
            for r in sorted(records, key=lambda r: -r.size)[:n]]

def most_recently_changed(records: List[FileRecord], n: int = 15) -> List[Dict]:
    import datetime
    return [{"file": f"{r.repo}/{r.relpath}",
             "mtime": datetime.datetime.fromtimestamp(r.mtime).strftime("%Y-%m-%d %H:%M")}
            for r in sorted(records, key=lambda r: -r.mtime)[:n]]

def all_todos(records: List[FileRecord]) -> List[Dict]:
    out = []
    for r in records:
        for t in r.todos:
            out.append({"file": f"{r.repo}/{r.relpath}", "note": t})
    return out

def cross_repo_references(records: List[FileRecord]) -> List[Dict]:
    out = []
    for r in records:
        for ref in r.cross_refs:
            if ref != r.repo:
                out.append({"from": f"{r.repo}/{r.relpath}", "mentions": ref})
    return out

def global_concept_cloud(records: List[FileRecord], top: int = 60) -> List[Dict]:
    total: Counter = Counter()
    for r in records:
        total.update(r.concept_freq)
    return [{"word": w, "count": c} for w, c in total.most_common(top)]

def all_py_defs(records: List[FileRecord]) -> List[Dict]:
    out = []
    for r in records:
        for d in r.py_defs:
            out.append({"file": f"{r.repo}/{r.relpath}", "name": d})
    return out

def import_inventory(records: List[FileRecord]) -> List[Dict]:
    counter: Counter = Counter()
    for r in records:
        for imp in r.py_imports:
            counter[imp] += 1
    return [{"package": k, "used_in": v} for k, v in counter.most_common(40)]

def all_headings(records: List[FileRecord]) -> List[Dict]:
    out = []
    for r in records:
        for h in r.headings:
            out.append({"file": f"{r.repo}/{r.relpath}", "heading": h})
    return out

def build_concept_graph(records: List[FileRecord], top_words: int = 40) -> Dict:
    """
    Nodes: top concept-words + repo names + key dirs.
    Edges: word appears in file (word → repo/dir) with weight = count.
    """
    global_top = {e["word"] for e in global_concept_cloud(records, top_words)}
    nodes = {}
    edges = []

    for r in records:
        rkey = f"repo:{r.repo}"
        if rkey not in nodes:
            nodes[rkey] = {"id": rkey, "kind": "repo", "label": r.repo}
        parts = Path(r.relpath).parts
        if len(parts) > 1:
            dkey = f"dir:{r.repo}/{parts[0]}"
            if dkey not in nodes:
                nodes[dkey] = {"id": dkey, "kind": "dir", "label": parts[0]}
            edges.append({"source": rkey, "target": dkey, "weight": 1, "rel": "contains"})
        for word, count in r.concept_freq.items():
            if word in global_top:
                wkey = f"word:{word}"
                if wkey not in nodes:
                    nodes[wkey] = {"id": wkey, "kind": "concept", "label": word}
                edges.append({"source": rkey, "target": wkey, "weight": count, "rel": "uses"})

    return {"nodes": list(nodes.values()), "edges": edges}

# ── report renderer ──────────────────────────────────────────────────────────

def render_report(
    repos:       List[Path],
    records:     List[FileRecord],
    dupes:       List[Dict],
    largest:     List[Dict],
    recent:      List[Dict],
    todos:       List[Dict],
    xrefs:       List[Dict],
    concepts:    List[Dict],
    defs:        List[Dict],
    imports:     List[Dict],
    headings:    List[Dict],
) -> str:
    W = []
    def w(*args): W.append(" ".join(str(a) for a in args))
    def nl(): W.append("")

    w("# Codebase Map")
    nl()
    w(f"Repos scanned: {', '.join(r.name for r in repos)}")
    w(f"Total text files read: {len(records)}")
    nl()

    # ── per-repo quick stats
    w("## Per-repo overview")
    nl()
    by_repo: Dict[str, List[FileRecord]] = defaultdict(list)
    for r in records: by_repo[r.repo].append(r)

    for repo_name, recs in by_repo.items():
        total_bytes = sum(r.size for r in recs)
        exts = Counter(r.ext for r in recs)
        top_exts = ", ".join(f"{e}({c})" for e,c in exts.most_common(6))
        n_defs  = sum(len(r.py_defs) for r in recs)
        n_heads = sum(len(r.headings) for r in recs)
        n_todos = sum(len(r.todos)   for r in recs)
        w(f"### {repo_name}")
        w(f"- {len(recs)} files · {total_bytes:,} bytes")
        w(f"- Extensions: {top_exts}")
        w(f"- Python defs (functions+classes): {n_defs}")
        w(f"- Markdown headings: {n_heads}")
        w(f"- TODO/FIXME markers: {n_todos}")
        nl()

    # ── global concept cloud
    w("## Global concept cloud (top 40 words across all repos)")
    nl()
    w("| word | count |")
    w("|------|-------|")
    for e in concepts[:40]:
        w(f"| {e['word']} | {e['count']} |")
    nl()

    # ── cross-repo references
    w("## Cross-repo references")
    nl()
    if xrefs:
        w("Files that explicitly mention another repo by name:")
        nl()
        for x in xrefs[:40]:
            w(f"- `{x['from']}` → mentions **{x['mentions']}**")
    else:
        w("No explicit cross-repo text references detected.")
    nl()

    # ── duplicate content
    w("## Identical files across repos")
    nl()
    if dupes:
        for d in dupes[:20]:
            w(f"- hash `{d['hash']}` ({d['size']:,} bytes): {', '.join(d['files'])}")
    else:
        w("No exact duplicates found.")
    nl()

    # ── largest files
    w("## 15 largest text files")
    nl()
    for e in largest:
        w(f"- `{e['file']}` — {e['size']:,} bytes")
    nl()

    # ── most recently changed
    w("## 15 most recently changed")
    nl()
    for e in recent:
        w(f"- `{e['file']}` — {e['mtime']}")
    nl()

    # ── all Python definitions
    w("## Python surface area")
    nl()
    by_file: Dict[str, List[str]] = defaultdict(list)
    for d in defs: by_file[d["file"]].append(d["name"])
    for fname, names in sorted(by_file.items()):
        w(f"**{fname}**: {', '.join(names)}")
    nl()

    # ── import inventory
    w("## Third-party / stdlib imports (most used)")
    nl()
    w("| package | files using it |")
    w("|---------|---------------|")
    for e in imports[:30]:
        w(f"| {e['package']} | {e['used_in']} |")
    nl()

    # ── headings index
    w("## Document heading index")
    nl()
    for h in headings[:80]:
        w(f"- `{h['file']}` — {h['heading']}")
    nl()

    # ── todos
    w("## TODO / FIXME / NOTE markers")
    nl()
    if todos:
        for t in todos[:40]:
            w(f"- `{t['file']}`: {t['note']}")
    else:
        w("None found.")
    nl()

    return "\n".join(W)

# ── main ─────────────────────────────────────────────────────────────────────

def default_targets() -> List[Path]:
    candidates = ["~/Vybn", "~/Vybn-Law", "~/vybn-phase", "~/Him"]
    return [p for c in candidates if (p := Path(c).expanduser()).exists() and p.is_dir()]

def main(argv: List[str]) -> int:
    targets = [Path(a).expanduser().resolve() for a in argv] if argv else default_targets()
    targets = [t for t in targets if t.exists() and t.is_dir()]
    if not targets:
        print("No repos found. Pass paths explicitly.", file=sys.stderr)
        return 1

    repo_names = [t.name for t in targets]
    print(f"Scanning: {', '.join(repo_names)}")

    all_records: List[FileRecord] = []
    for repo in targets:
        recs = build_records(repo, repo_names)
        print(f"  {repo.name}: {len(recs)} files")
        all_records.extend(recs)

    dupes   = duplicate_files(all_records)
    largest = largest_files(all_records)
    recent  = most_recently_changed(all_records)
    todos   = all_todos(all_records)
    xrefs   = cross_repo_references(all_records)
    concepts= global_concept_cloud(all_records, 60)
    defs    = all_py_defs(all_records)
    imports = import_inventory(all_records)
    headings= all_headings(all_records)
    graph   = build_concept_graph(all_records, 40)

    out_dir = Path.cwd() / "repo_mapping_output"
    out_dir.mkdir(exist_ok=True)

    report = render_report(
        targets, all_records, dupes, largest, recent,
        todos, xrefs, concepts, defs, imports, headings,
    )

    full_map = {
        "repos":      repo_names,
        "file_count": len(all_records),
        "duplicates": dupes,
        "largest":    largest,
        "recent":     recent,
        "todos":      todos,
        "cross_refs": xrefs,
        "concepts":   concepts,
        "py_defs":    defs,
        "imports":    imports,
        "headings":   headings,
        "files":      [asdict(r) for r in all_records],
    }

    (out_dir / "repo_map.json").write_text(json.dumps(full_map, indent=2), encoding="utf-8")
    (out_dir / "repo_graph.json").write_text(json.dumps(graph, indent=2), encoding="utf-8")
    (out_dir / "repo_report.md").write_text(report, encoding="utf-8")

    for name in ("repo_map.json", "repo_graph.json", "repo_report.md"):
        path = out_dir / name
        print(f"Wrote: {path}  ({path.stat().st_size:,} bytes)")

    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
