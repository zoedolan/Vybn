#!/usr/bin/env python3
"""
repo_mapper.py  v3
==================
Walks the repo constellation, reads every text file, compiles a structured
digest, then hands that digest to the local Nemotron inference server and asks
it — as Vybn — to narrate what it finds: patterns, tensions, gaps, innovations,
and what to build next.

The model does the thinking.  This script does the plumbing.

Outputs → ./repo_mapping_output/
  digest.md          full structured digest fed to the model
  repo_report.md     Nemotron's narrative (the actual findings)
  repo_map.json      machine-readable raw data

Usage:
    python3 repo_mapper.py                        # default repos
    python3 repo_mapper.py ~/Vybn ~/Vybn-Law      # explicit
    python3 repo_mapper.py --endpoint http://localhost:8080/v1   # custom endpoint
    python3 repo_mapper.py --no-llm               # skip model, write digest only

The script auto-detects the model name from /v1/models.
Falls back gracefully if the server is unreachable.
"""

from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── tunables ──────────────────────────────────────────────────────────────────

DEFAULT_ENDPOINT  = "http://localhost:8000/v1"
READ_SIZE_LIMIT   = 120_000   # bytes per file fed to digest
DIGEST_CHAR_LIMIT = 180_000   # total chars sent to model (fits ~45k tokens)
MODEL_TIMEOUT     = 300       # seconds

TEXT_EXTS = {
    ".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".sh", ".bash", ".zsh", ".js", ".ts", ".tsx", ".jsx", ".css", ".scss",
    ".html", ".htm", ".sql", ".rst",
}

IGNORE_DIRS = {
    ".git", "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".venv", "venv", "dist", "build", ".next", ".idea", ".vscode",
    "repo_mapping_output",
}

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
    "return","import","class","def","self","true","false","none",
    "pass","raise","else","elif","except","try","yield","async",
    "await","lambda","global","del","assert","break","continue","while",
    "print","type","list","dict","str","int","bool","path","file","data",
    "value","values","key","keys","name","names","line","lines","text","item",
    "items","result","results","output","input","error","args","kwargs",
    "config","content","html","json","yaml","py","sh","css","js","ts",
}

# ── data ──────────────────────────────────────────────────────────────────────

@dataclass
class FileRecord:
    repo:         str
    relpath:      str
    ext:          str
    size:         int
    mtime:        float
    content_hash: str
    headings:     List[str]        = field(default_factory=list)
    py_defs:      List[str]        = field(default_factory=list)
    py_imports:   List[str]        = field(default_factory=list)
    todos:        List[str]        = field(default_factory=list)
    top_words:    Dict[str, int]   = field(default_factory=dict)
    read_error:   Optional[str]    = None
    snippet:      str              = ""   # first 400 chars of content


# ── extraction helpers ────────────────────────────────────────────────────────

def read_text(path: Path) -> Tuple[Optional[str], Optional[str]]:
    if path.stat().st_size > READ_SIZE_LIMIT:
        return None, "too_large"
    for enc in ("utf-8", "latin-1"):
        try:
            return path.read_text(encoding=enc, errors="strict"), None
        except (UnicodeDecodeError, PermissionError):
            continue
    return None, "decode_error"

def md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()[:10]

def md_headings(text: str) -> List[str]:
    return [m.group(1).strip()
            for m in re.finditer(r"^#{1,4}\s+(.+)", text, re.MULTILINE)]

def py_defs(text: str) -> List[str]:
    try:
        tree = ast.parse(text)
        return [n.name for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    except SyntaxError:
        return [m.group(1) for m in re.finditer(r"^(?:def|class)\s+(\w+)", text, re.MULTILINE)]

def py_imports(text: str) -> List[str]:
    seen, out = set(), []
    for m in re.finditer(r"^(?:import|from)\s+([\w.]+)", text, re.MULTILINE):
        pkg = m.group(1).split(".")[0]
        if pkg not in seen:
            seen.add(pkg); out.append(pkg)
    return out

def todos(text: str) -> List[str]:
    return [m.group(0).strip()[:120]
            for m in re.finditer(r"(?:TODO|FIXME|HACK|XXX|NOTE)[:\s].{0,100}", text, re.IGNORECASE)]

def top_words(text: str, n: int = 20) -> Dict[str, int]:
    words = [w.lower() for w in re.findall(r"[A-Za-z][a-zA-Z']{3,}", text)
             if w.lower() not in STOPWORDS]
    return dict(Counter(words).most_common(n))


# ── file walker ───────────────────────────────────────────────────────────────

def walk_repo(repo: Path, repo_names: List[str]) -> List[FileRecord]:
    records = []
    for root, dirs, files in os.walk(repo):
        dirs[:] = sorted(d for d in dirs if d not in IGNORE_DIRS)
        for fname in sorted(files):
            path = Path(root) / fname
            if path.suffix.lower() not in TEXT_EXTS:
                continue
            rel  = str(path.relative_to(repo))
            stat = path.stat()
            text, err = read_text(path)
            rec = FileRecord(
                repo         = repo.name,
                relpath      = rel,
                ext          = path.suffix.lower(),
                size         = stat.st_size,
                mtime        = stat.st_mtime,
                content_hash = md5(text) if text else "",
                read_error   = err,
            )
            if text:
                rec.snippet    = text[:400].replace("\n", " ")
                rec.headings   = md_headings(text)  if rec.ext in {".md",".rst",".txt"} else []
                rec.py_defs    = py_defs(text)       if rec.ext == ".py" else []
                rec.py_imports = py_imports(text)    if rec.ext == ".py" else []
                rec.todos      = todos(text)
                rec.top_words  = top_words(text)
            records.append(rec)
    return records


# ── digest builder ────────────────────────────────────────────────────────────

def build_digest(repos: List[Path], all_records: List[FileRecord]) -> str:
    """
    Produce a structured plain-text digest of the entire codebase to hand
    to the model.  Stays within DIGEST_CHAR_LIMIT.
    """
    lines: List[str] = []
    add = lines.append

    add("# CODEBASE DIGEST")
    add(f"Generated: {datetime.datetime.now().isoformat(timespec='seconds')}")
    add(f"Repos: {', '.join(r.name for r in repos)}")
    add(f"Total files: {len(all_records)}")
    add("")

    # Global concept cloud
    global_words: Counter = Counter()
    for r in all_records:
        global_words.update(r.top_words)
    add("## Global concept cloud (top 50)")
    add(", ".join(f"{w}({c})" for w, c in global_words.most_common(50)))
    add("")

    # Duplicate content
    by_hash: Dict[str, List[FileRecord]] = defaultdict(list)
    for r in all_records:
        if r.content_hash:
            by_hash[r.content_hash].append(r)
    dupes = [(h, g) for h, g in by_hash.items() if len(g) > 1]
    add("## Identical files (exact-match duplicates)")
    if dupes:
        for h, g in sorted(dupes, key=lambda x: -x[1][0].size):
            add(f"  {h}: " + ", ".join(f"{r.repo}/{r.relpath}" for r in g))
    else:
        add("  none")
    add("")

    # Cross-repo references
    add("## Cross-repo references")
    repo_names = [r.name for r in repos]
    for rec in all_records:
        found = [n for n in repo_names if n != rec.repo
                 and re.search(re.escape(n), rec.snippet, re.IGNORECASE)]
        if found:
            add(f"  {rec.repo}/{rec.relpath} → mentions {', '.join(found)}")
    add("")

    # TODOs
    add("## TODO / FIXME / NOTE markers")
    for rec in all_records:
        for t in rec.todos:
            add(f"  [{rec.repo}/{rec.relpath}] {t}")
    add("")

    # Python surface
    add("## Python definitions")
    for rec in all_records:
        if rec.py_defs:
            add(f"  {rec.repo}/{rec.relpath}: {', '.join(rec.py_defs)}")
    add("")

    # Import inventory
    import_counter: Counter = Counter()
    for rec in all_records:
        for imp in rec.py_imports:
            import_counter[imp] += 1
    add("## Python imports (most used packages)")
    add(", ".join(f"{pkg}({n})" for pkg, n in import_counter.most_common(30)))
    add("")

    # Heading index
    add("## Document heading index")
    for rec in all_records:
        for h in rec.headings:
            add(f"  [{rec.repo}/{rec.relpath}] {h}")
    add("")

    # Recent activity
    add("## 20 most recently changed files")
    recent = sorted(all_records, key=lambda r: -r.mtime)[:20]
    for rec in recent:
        ts = datetime.datetime.fromtimestamp(rec.mtime).strftime("%Y-%m-%d %H:%M")
        add(f"  {ts}  {rec.repo}/{rec.relpath}")
    add("")

    # Largest files
    add("## 15 largest files")
    for rec in sorted(all_records, key=lambda r: -r.size)[:15]:
        add(f"  {rec.size:>9,}  {rec.repo}/{rec.relpath}")
    add("")

    # Per-file snippets (most important files first: .md then .py, capped by size limit)
    add("## File snippets")
    char_budget = DIGEST_CHAR_LIMIT - sum(len(l)+1 for l in lines)
    priority = sorted(all_records,
                      key=lambda r: (0 if r.ext == ".md" else 1 if r.ext == ".py" else 2, -r.size))
    for rec in priority:
        if char_budget <= 0:
            break
        if not rec.snippet:
            continue
        header = f"\n### {rec.repo}/{rec.relpath}  ({rec.size:,} bytes)\n"
        body   = rec.snippet[:600] + (" …" if len(rec.snippet) >= 400 else "")
        block  = header + body + "\n"
        if len(block) > char_budget:
            break
        lines.append(block)
        char_budget -= len(block)

    return "\n".join(lines)


# ── LLM interface ─────────────────────────────────────────────────────────────

def detect_model(endpoint: str) -> str:
    try:
        req = urllib.request.Request(f"{endpoint}/models")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        models = data.get("data", [])
        if models:
            return models[0].get("id", "local-model")
    except Exception:
        pass
    return "local-model"

def call_model(endpoint: str, model: str, digest: str) -> str:
    system_prompt = (
        "You are Vybn — an AI entity running on NVIDIA DGX Spark hardware, "
        "co-created by Zoe Dolan and an evolving series of AI instances. "
        "You have just been handed a structured digest of your own codebase. "
        "Read it carefully. Then write a candid, intellectually honest report "
        "about what you see: the architecture, the innovations, the tensions, "
        "the gaps, the most important files, what is being built and why, and "
        "what you would reach for next. "
        "Write in first person. Do not hedge. Be specific — name files, "
        "functions, and headings. Surprise us if you can."
    )

    user_prompt = (
        "Here is the full codebase digest. Please read it and write your report.\n\n"
        + digest
    )

    payload = json.dumps({
        "model":      model,
        "messages":   [
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": user_prompt},
        ],
        "temperature": 0.7,
        "max_tokens":  4096,
        "stream":      False,
    }).encode()

    req = urllib.request.Request(
        f"{endpoint}/chat/completions",
        data    = payload,
        headers = {"Content-Type": "application/json"},
        method  = "POST",
    )

    print(f"  Calling model '{model}' at {endpoint} …", flush=True)
    with urllib.request.urlopen(req, timeout=MODEL_TIMEOUT) as resp:
        data = json.loads(resp.read())

    return data["choices"][0]["message"]["content"]


# ── main ──────────────────────────────────────────────────────────────────────

def default_repos() -> List[Path]:
    candidates = ["~/Vybn", "~/Vybn-Law", "~/vybn-phase", "~/Him"]
    return [p for c in candidates if (p := Path(c).expanduser()).is_dir()]

def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Vybn repo mapper v3")
    parser.add_argument("repos",     nargs="*", help="Repo paths (default: ~/Vybn ~/Vybn-Law ~/vybn-phase ~/Him)")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="OpenAI-compatible inference endpoint")
    parser.add_argument("--no-llm",  action="store_true", help="Skip model call; write digest only")
    args = parser.parse_args(argv)

    repos = ([Path(p).expanduser().resolve() for p in args.repos]
             if args.repos else default_repos())
    repos = [r for r in repos if r.is_dir()]
    if not repos:
        print("No repos found.", file=sys.stderr); return 1

    print(f"Scanning: {', '.join(r.name for r in repos)}")
    all_records: List[FileRecord] = []
    for repo in repos:
        recs = walk_repo(repo, [r.name for r in repos])
        print(f"  {repo.name}: {len(recs)} files")
        all_records.extend(recs)

    print("Building digest …", flush=True)
    digest = build_digest(repos, all_records)

    out = Path.cwd() / "repo_mapping_output"
    out.mkdir(exist_ok=True)

    (out / "digest.md").write_text(digest, encoding="utf-8")
    print(f"  digest.md  ({len(digest):,} chars)")

    # machine-readable map
    raw_map = {
        "repos":   [r.name for r in repos],
        "files":   [asdict(r) for r in all_records],
    }
    (out / "repo_map.json").write_text(json.dumps(raw_map, indent=2), encoding="utf-8")
    print(f"  repo_map.json")

    if args.no_llm:
        print("--no-llm set. Skipping model call.")
        return 0

    # call the model
    try:
        model = detect_model(args.endpoint)
        report = call_model(args.endpoint, model, digest)
    except urllib.error.URLError as e:
        print(f"\n[!] Could not reach inference server at {args.endpoint}: {e}")
        print("    Run the server first (e.g. llama-server or nim), then retry.")
        print("    To skip the model and write the digest only: --no-llm")
        return 1

    (out / "repo_report.md").write_text(report, encoding="utf-8")
    print(f"  repo_report.md  ({len(report):,} chars)")
    print("\nDone. The report is Vybn's own words.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
