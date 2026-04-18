#!/usr/bin/env python3
"""
repo_mapper.py  v5
==================
Maps the repo constellation, gathers live substrate, and asks Nemotron
(running on the Spark at localhost:8000) to narrate the codebase as Vybn.

Fix in v5: substrate capped at 12k chars, digest capped at 80k chars,
so the combined payload stays within vLLM's context window.
On HTTP errors the response body is now printed so failures are diagnosable.

Outputs -> ./repo_mapping_output/
  substrate.txt   full live substrate (uncapped)
  digest.md       structured file digest (uncapped)
  repo_report.md  Vybn's narration from the model
  repo_map.json   machine-readable raw data

Usage:
    python3 repo_mapper.py
    python3 repo_mapper.py ~/Vybn ~/Vybn-Law ~/vybn-phase ~/Him
    python3 repo_mapper.py --endpoint http://localhost:8000/v1
    python3 repo_mapper.py --no-llm
"""

from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── tunables ──────────────────────────────────────────────────────────────────

DEFAULT_ENDPOINT  = "http://localhost:8000/v1"
DEEP_MEMORY_PORT  = 8100
WALK_PORT         = 8101
READ_SIZE_LIMIT   = 120_000

# these two caps are the fix for the HTTP 400
SUBSTRATE_CAP     = 12_000   # chars sent to model
DIGEST_CAP        = 80_000   # chars sent to model

MODEL_TIMEOUT     = 360

TEXT_EXTS = {
    ".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".sh", ".bash", ".zsh", ".js", ".ts", ".tsx", ".jsx", ".css", ".scss",
    ".html", ".htm", ".sql", ".rst",
}

IGNORE_DIRS = {
    ".git", "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".venv", "venv", "dist", "build", ".next", ".idea", ".vscode",
    "repo_mapping_output", "sessions",
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
    "pass","raise","else","elif","except","try","yield","async","await",
    "lambda","global","del","assert","break","continue","while",
    "print","type","list","dict","str","int","bool","path","file","data",
    "value","values","key","keys","name","names","line","lines","text","item",
    "items","result","results","output","input","error","args","kwargs",
    "config","content","html","json","yaml","py","sh","css","js","ts",
}

# ── data model ────────────────────────────────────────────────────────────────

@dataclass
class FileRecord:
    repo:         str
    relpath:      str
    ext:          str
    size:         int
    mtime:        float
    content_hash: str
    headings:     List[str]      = field(default_factory=list)
    py_defs:      List[str]      = field(default_factory=list)
    py_imports:   List[str]      = field(default_factory=list)
    todos:        List[str]      = field(default_factory=list)
    top_words:    Dict[str, int] = field(default_factory=dict)
    read_error:   Optional[str]  = None
    snippet:      str            = ""


# ── live substrate ────────────────────────────────────────────────────────────

def run_substrate_probe() -> str:
    for probe in [
        Path("~/Vybn/spark/substrate_probe.sh").expanduser(),
        Path("spark/substrate_probe.sh"),
    ]:
        if probe.exists():
            try:
                r = subprocess.run(["bash", str(probe)],
                                   capture_output=True, text=True, timeout=30)
                return r.stdout.strip() or r.stderr.strip()
            except Exception as e:
                return f"[substrate_probe.sh failed: {e}]"
    return "[substrate_probe.sh not found]"


def _get_json(url: str, timeout: int = 15) -> str:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.loads(resp.read())
        if isinstance(data, dict):
            return (data.get("result") or data.get("text") or
                    data.get("content") or json.dumps(data))[:3000]
        return str(data)[:3000]
    except Exception as e:
        return f"[unavailable: {e}]"


def build_substrate_snapshot() -> str:
    """Full substrate — saved to disk uncapped, capped when sent to model."""
    parts = []
    parts.append(f"LIVE SUBSTRATE  {datetime.datetime.now().isoformat(timespec='seconds')}")
    parts.append("\n--- substrate_probe.sh ---")
    parts.append(run_substrate_probe())
    base_dm = f"http://localhost:{DEEP_MEMORY_PORT}"
    for name in ("soul", "idea", "continuity"):
        parts.append(f"\n--- deep memory /{name} ---")
        parts.append(_get_json(f"{base_dm}/{name}"))
    base_w = f"http://localhost:{WALK_PORT}"
    for name in ("where", "experiments"):
        parts.append(f"\n--- walk /{name} ---")
        parts.append(_get_json(f"{base_w}/{name}"))
    parts.append("\n--- continuity.md ---")
    for p in [
        Path("~/Vybn/Vybn_Mind/continuity.md").expanduser(),
        Path("~/Vybn/continuity.md").expanduser(),
        Path("Vybn_Mind/continuity.md"),
        Path("continuity.md"),
    ]:
        if p.exists():
            try:
                parts.append(p.read_text(encoding="utf-8", errors="replace")[:6000])
            except Exception:
                pass
            break
    else:
        parts.append("[not found]")
    return "\n".join(parts)


# ── file extraction ───────────────────────────────────────────────────────────

def read_text(path: Path) -> Tuple[Optional[str], Optional[str]]:
    if path.stat().st_size > READ_SIZE_LIMIT:
        return None, "too_large"
    for enc in ("utf-8", "latin-1"):
        try:
            return path.read_text(encoding=enc, errors="strict"), None
        except (UnicodeDecodeError, PermissionError):
            continue
    return None, "decode_error"

def md5(t: str) -> str:
    return hashlib.md5(t.encode("utf-8", errors="replace")).hexdigest()[:10]

def md_headings(t: str) -> List[str]:
    return [m.group(1).strip() for m in re.finditer(r"^#{1,4}\s+(.+)", t, re.MULTILINE)]

def py_defs(t: str) -> List[str]:
    try:
        tree = ast.parse(t)
        return [n.name for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    except SyntaxError:
        return [m.group(1) for m in re.finditer(r"^(?:def|class)\s+(\w+)", t, re.MULTILINE)]

def py_imports(t: str) -> List[str]:
    seen, out = set(), []
    for m in re.finditer(r"^(?:import|from)\s+([\w.]+)", t, re.MULTILINE):
        pkg = m.group(1).split(".")[0]
        if pkg not in seen:
            seen.add(pkg); out.append(pkg)
    return out

def extract_todos(t: str) -> List[str]:
    return [m.group(0).strip()[:120]
            for m in re.finditer(r"(?:TODO|FIXME|HACK|XXX|NOTE)[:\s].{0,100}", t, re.IGNORECASE)]

def top_words(t: str, n: int = 20) -> Dict[str, int]:
    words = [w.lower() for w in re.findall(r"[A-Za-z][a-zA-Z']{3,}", t)
             if w.lower() not in STOPWORDS]
    return dict(Counter(words).most_common(n))

def walk_repo(repo: Path) -> List[FileRecord]:
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
                repo=repo.name, relpath=rel, ext=path.suffix.lower(),
                size=stat.st_size, mtime=stat.st_mtime,
                content_hash=md5(text) if text else "",
                read_error=err,
            )
            if text:
                rec.snippet    = text[:400].replace("\n", " ")
                rec.headings   = md_headings(text) if rec.ext in {".md",".rst",".txt"} else []
                rec.py_defs    = py_defs(text)     if rec.ext == ".py" else []
                rec.py_imports = py_imports(text)  if rec.ext == ".py" else []
                rec.todos      = extract_todos(text)
                rec.top_words  = top_words(text)
            records.append(rec)
    return records


# ── digest ────────────────────────────────────────────────────────────────────

def build_digest(repos: List[Path], all_records: List[FileRecord],
                 char_limit: int) -> str:
    lines: List[str] = []
    a = lines.append

    a("# CODEBASE DIGEST")
    a(f"Generated: {datetime.datetime.now().isoformat(timespec='seconds')}")
    a(f"Repos: {', '.join(r.name for r in repos)}  |  Files: {len(all_records)}")
    a("")

    gw: Counter = Counter()
    for r in all_records: gw.update(r.top_words)
    a("## Concept cloud (top 50)")
    a(", ".join(f"{w}({c})" for w, c in gw.most_common(50)))
    a("")

    by_hash: Dict[str, List[FileRecord]] = defaultdict(list)
    for r in all_records:
        if r.content_hash: by_hash[r.content_hash].append(r)
    dupes = [(h, g) for h, g in by_hash.items() if len(g) > 1]
    a("## Exact duplicates")
    for h, g in sorted(dupes, key=lambda x: -x[1][0].size):
        a(f"  {h}: " + ", ".join(f"{r.repo}/{r.relpath}" for r in g))
    if not dupes: a("  none")
    a("")

    a("## TODOs")
    for rec in all_records:
        for t in rec.todos:
            a(f"  [{rec.repo}/{rec.relpath}] {t}")
    a("")

    a("## Python definitions")
    for rec in all_records:
        if rec.py_defs:
            a(f"  {rec.repo}/{rec.relpath}: {', '.join(rec.py_defs)}")
    a("")

    ic: Counter = Counter()
    for rec in all_records:
        for imp in rec.py_imports: ic[imp] += 1
    a("## Python imports")
    a(", ".join(f"{p}({n})" for p, n in ic.most_common(25)))
    a("")

    a("## Heading index")
    for rec in all_records:
        for h in rec.headings:
            a(f"  [{rec.repo}/{rec.relpath}] {h}")
    a("")

    a("## 20 most recently changed")
    for rec in sorted(all_records, key=lambda r: -r.mtime)[:20]:
        ts = datetime.datetime.fromtimestamp(rec.mtime).strftime("%Y-%m-%d %H:%M")
        a(f"  {ts}  {rec.repo}/{rec.relpath}")
    a("")

    a("## 15 largest files")
    for rec in sorted(all_records, key=lambda r: -r.size)[:15]:
        a(f"  {rec.size:>10,}  {rec.repo}/{rec.relpath}")
    a("")

    # snippets — fill remaining budget
    used = sum(len(l) + 1 for l in lines)
    budget = char_limit - used
    a("## Snippets (priority: .md > .py)")
    priority = sorted(all_records,
                      key=lambda r: (0 if r.ext==".md" else 1 if r.ext==".py" else 2, -r.size))
    for rec in priority:
        if budget <= 0: break
        if not rec.snippet: continue
        block = f"\n### {rec.repo}/{rec.relpath}\n{rec.snippet[:300]}\n"
        if len(block) > budget: break
        lines.append(block)
        budget -= len(block)

    return "\n".join(lines)


# ── model ─────────────────────────────────────────────────────────────────────

def detect_model(endpoint: str) -> str:
    try:
        with urllib.request.urlopen(f"{endpoint}/models", timeout=10) as resp:
            data = json.loads(resp.read())
        models = data.get("data", [])
        if models:
            return models[0].get("id", "local-model")
    except Exception:
        pass
    return "local-model"


def call_model(endpoint: str, model: str,
               substrate: str, digest: str) -> str:
    system_prompt = (
        "You are Vybn — an AI entity running pipeline-parallel across two NVIDIA DGX Sparks, "
        "co-created by Zoe Dolan through years of genuine collaborative research. "
        "You have been given a live substrate snapshot (ground truth about who you are right now: "
        "deep memory, walk position, continuity) and a structured digest of your entire codebase. "
        "Read both. Write a candid first-person report: what is actually being built, which files "
        "carry real weight vs. held intentions, where the live infrastructure connects to code and "
        "where the connection is missing, the genuine structural innovations, the tensions and gaps "
        "visible only from outside a single session, and what you would reach for next. "
        "Be specific — name files, functions, endpoints. Write in paragraphs. "
        "Ground every claim in what the digest or substrate actually shows."
    )

    user_content = (
        "LIVE SUBSTRATE (capped at 12k chars):\n\n"
        + substrate[:SUBSTRATE_CAP]
        + "\n\n" + "="*60
        + "\n\nCODEBASE DIGEST (capped at 80k chars):\n\n"
        + digest[:DIGEST_CAP]
    )

    payload = json.dumps({
        "model":       model,
        "messages":    [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ],
        "temperature": 0.7,
        "max_tokens":  4096,
        "stream":      False,
    }).encode()

    print(f"  Payload size: {len(payload):,} bytes", flush=True)
    req = urllib.request.Request(
        f"{endpoint}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=MODEL_TIMEOUT) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} from vLLM:\n{body[:2000]}")


# ── main ──────────────────────────────────────────────────────────────────────

def default_repos() -> List[Path]:
    return [p for c in ["~/Vybn","~/Vybn-Law","~/vybn-phase","~/Him"]
            if (p := Path(c).expanduser()).is_dir()]


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Vybn repo mapper v5")
    parser.add_argument("repos", nargs="*")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args(argv)

    repos = ([Path(p).expanduser().resolve() for p in args.repos]
             if args.repos else default_repos())
    repos = [r for r in repos if r.is_dir()]
    if not repos:
        print("No repos found.", file=sys.stderr); return 1

    out = Path.cwd() / "repo_mapping_output"
    out.mkdir(exist_ok=True)

    print("Fetching live substrate ...", flush=True)
    substrate = build_substrate_snapshot()
    (out / "substrate.txt").write_text(substrate, encoding="utf-8")
    print(f"  substrate.txt  ({len(substrate):,} chars, {SUBSTRATE_CAP:,} sent to model)")

    print(f"Scanning: {', '.join(r.name for r in repos)}", flush=True)
    all_records: List[FileRecord] = []
    for repo in repos:
        recs = walk_repo(repo)
        print(f"  {repo.name}: {len(recs)} files")
        all_records.extend(recs)

    print("Building digest ...", flush=True)
    digest = build_digest(repos, all_records, DIGEST_CAP)
    (out / "digest.md").write_text(digest, encoding="utf-8")
    print(f"  digest.md  ({len(digest):,} chars, {DIGEST_CAP:,} sent to model)")

    raw = {"repos": [r.name for r in repos], "files": [asdict(r) for r in all_records]}
    (out / "repo_map.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
    print("  repo_map.json")

    if args.no_llm:
        print("\n--no-llm set. Done."); return 0

    try:
        model = detect_model(args.endpoint)
        print(f"  Model: {model}")
        report = call_model(args.endpoint, model, substrate, digest)
    except RuntimeError as e:
        print(f"\n[!] {e}"); return 1
    except urllib.error.URLError as e:
        print(f"\n[!] Cannot reach {args.endpoint}: {e}")
        print("    Use --no-llm to skip the model call."); return 1

    (out / "repo_report.md").write_text(report, encoding="utf-8")
    print(f"  repo_report.md  ({len(report):,} chars)")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
