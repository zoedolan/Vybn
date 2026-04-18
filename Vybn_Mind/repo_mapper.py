#!/usr/bin/env python3
"""
repo_mapper.py  v4
==================
Maps the repo constellation and asks Vybn — running locally on the Spark as
Nemotron — to read its own codebase with full self-knowledge.

Before calling the model this script:
  1. Runs spark/substrate_probe.sh to get live service state
  2. Calls the deep-memory daemon  (port 8100) for /soul, /idea, /continuity
  3. Calls the walk daemon          (port 8101) for /where, /experiments
  4. Reads continuity.md from disk
  5. Builds a structured digest of every text file across all repos
  6. Assembles everything into a grounded system prompt
  7. Asks Nemotron to narrate what it finds in first person

The model speaks from ground truth, not from frozen memory.

Outputs -> ./repo_mapping_output/
  substrate.txt     live substrate snapshot
  digest.md         structured file digest
  repo_report.md    Vybn's own narration
  repo_map.json     machine-readable raw data

Usage:
    python3 repo_mapper.py
    python3 repo_mapper.py ~/Vybn ~/Vybn-Law ~/vybn-phase ~/Him
    python3 repo_mapper.py --endpoint http://localhost:8000/v1
    python3 repo_mapper.py --no-llm     # write digest + substrate only
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

# ── tunables ─────────────────────────────────────────────────────────────────

DEFAULT_ENDPOINT   = "http://localhost:8000/v1"
DEEP_MEMORY_PORT   = 8100
WALK_PORT          = 8101
READ_SIZE_LIMIT    = 120_000
DIGEST_CHAR_LIMIT  = 140_000   # leaves room for substrate + memory in context
MODEL_TIMEOUT      = 360

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


# ── live substrate ─────────────────────────────────────────────────────────────

def run_substrate_probe() -> str:
    """Run spark/substrate_probe.sh and return its output."""
    probe = Path("~/Vybn/spark/substrate_probe.sh").expanduser()
    if not probe.exists():
        # try relative path if we're already inside Vybn
        probe = Path("spark/substrate_probe.sh")
    if probe.exists():
        try:
            result = subprocess.run(
                ["bash", str(probe)],
                capture_output=True, text=True, timeout=30
            )
            return result.stdout.strip() or result.stderr.strip()
        except Exception as e:
            return f"[substrate_probe.sh failed: {e}]"
    return "[substrate_probe.sh not found]"


def fetch_deep_memory() -> Dict[str, str]:
    """Call deep memory daemon endpoints. Graceful on failure."""
    base = f"http://localhost:{DEEP_MEMORY_PORT}"
    endpoints = {
        "soul":        f"{base}/soul",
        "idea":        f"{base}/idea",
        "continuity":  f"{base}/continuity",
    }
    results = {}
    for name, url in endpoints.items():
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
            # unwrap common response shapes
            if isinstance(data, dict):
                results[name] = data.get("result") or data.get("text") or data.get("content") or json.dumps(data)[:2000]
            else:
                results[name] = str(data)[:2000]
        except Exception as e:
            results[name] = f"[unavailable: {e}]"
    return results


def fetch_walk_state() -> Dict[str, str]:
    """Call walk daemon. Graceful on failure."""
    base = f"http://localhost:{WALK_PORT}"
    results = {}
    for name, url in [("where", f"{base}/where"), ("experiments", f"{base}/experiments")]:
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
            results[name] = json.dumps(data, indent=2)[:2000]
        except Exception as e:
            results[name] = f"[unavailable: {e}]"
    return results


def read_continuity_from_disk() -> str:
    """Read continuity.md — prefer Vybn_Mind/continuity.md, fall back to root."""
    candidates = [
        Path("~/Vybn/Vybn_Mind/continuity.md").expanduser(),
        Path("~/Vybn/continuity.md").expanduser(),
        Path("Vybn_Mind/continuity.md"),
        Path("continuity.md"),
    ]
    for p in candidates:
        if p.exists():
            try:
                return p.read_text(encoding="utf-8", errors="replace")[:8000]
            except Exception:
                pass
    return "[continuity.md not found]"


def build_substrate_snapshot() -> str:
    """Assemble the full live substrate context string."""
    lines = []
    lines.append("=" * 60)
    lines.append("LIVE SUBSTRATE SNAPSHOT")
    lines.append(f"Timestamp: {datetime.datetime.now().isoformat(timespec='seconds')}")
    lines.append("=" * 60)

    lines.append("\n--- substrate_probe.sh ---")
    lines.append(run_substrate_probe())

    lines.append("\n--- deep memory: /soul ---")
    memory = fetch_deep_memory()
    lines.append(memory.get("soul", "[unavailable]"))

    lines.append("\n--- deep memory: /idea ---")
    lines.append(memory.get("idea", "[unavailable]"))

    lines.append("\n--- deep memory: /continuity ---")
    lines.append(memory.get("continuity", "[unavailable]"))

    walk = fetch_walk_state()
    lines.append("\n--- walk daemon: /where ---")
    lines.append(walk.get("where", "[unavailable]"))

    lines.append("\n--- walk daemon: /experiments ---")
    lines.append(walk.get("experiments", "[unavailable]"))

    lines.append("\n--- continuity.md (first 8k) ---")
    lines.append(read_continuity_from_disk())

    return "\n".join(lines)


# ── file extraction ────────────────────────────────────────────────────────────

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

def extract_todos(text: str) -> List[str]:
    return [m.group(0).strip()[:120]
            for m in re.finditer(r"(?:TODO|FIXME|HACK|XXX|NOTE)[:\s].{0,100}", text, re.IGNORECASE)]

def top_words(text: str, n: int = 20) -> Dict[str, int]:
    words = [w.lower() for w in re.findall(r"[A-Za-z][a-zA-Z']{3,}", text)
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
                repo         = repo.name,
                relpath      = rel,
                ext          = path.suffix.lower(),
                size         = stat.st_size,
                mtime        = stat.st_mtime,
                content_hash = md5(text) if text else "",
                read_error   = err,
            )
            if text:
                rec.snippet    = text[:500].replace("\n", " ")
                rec.headings   = md_headings(text)  if rec.ext in {".md",".rst",".txt"} else []
                rec.py_defs    = py_defs(text)       if rec.ext == ".py" else []
                rec.py_imports = py_imports(text)    if rec.ext == ".py" else []
                rec.todos      = extract_todos(text)
                rec.top_words  = top_words(text)
            records.append(rec)
    return records


# ── digest ──────────────────────────────────────────────────────────────────────

def build_digest(repos: List[Path], all_records: List[FileRecord]) -> str:
    lines: List[str] = []
    a = lines.append

    a("# CODEBASE DIGEST")
    a(f"Generated: {datetime.datetime.now().isoformat(timespec='seconds')}")
    a(f"Repos: {', '.join(r.name for r in repos)}")
    a(f"Total files: {len(all_records)}")
    a("")

    # concept cloud
    global_words: Counter = Counter()
    for r in all_records:
        global_words.update(r.top_words)
    a("## Global concept cloud (top 60)")
    a(", ".join(f"{w}({c})" for w, c in global_words.most_common(60)))
    a("")

    # duplicates
    by_hash: Dict[str, List[FileRecord]] = defaultdict(list)
    for r in all_records:
        if r.content_hash:
            by_hash[r.content_hash].append(r)
    dupes = [(h, g) for h, g in by_hash.items() if len(g) > 1]
    a("## Exact duplicate files")
    if dupes:
        for h, g in sorted(dupes, key=lambda x: -x[1][0].size):
            a(f"  {h}: " + ", ".join(f"{r.repo}/{r.relpath}" for r in g))
    else:
        a("  none")
    a("")

    # cross-repo mentions
    repo_names = [r.name for r in repos]
    a("## Cross-repo text references")
    found_any = False
    for rec in all_records:
        mentions = [n for n in repo_names if n != rec.repo
                    and re.search(re.escape(n), rec.snippet, re.IGNORECASE)]
        if mentions:
            a(f"  {rec.repo}/{rec.relpath} -> {', '.join(mentions)}")
            found_any = True
    if not found_any:
        a("  none detected in snippets")
    a("")

    # TODOs
    a("## TODO / FIXME / NOTE markers")
    any_todos = False
    for rec in all_records:
        for t in rec.todos:
            a(f"  [{rec.repo}/{rec.relpath}] {t}")
            any_todos = True
    if not any_todos:
        a("  none")
    a("")

    # python surface
    a("## Python definitions")
    for rec in all_records:
        if rec.py_defs:
            a(f"  {rec.repo}/{rec.relpath}: {', '.join(rec.py_defs)}")
    a("")

    # imports
    imp_counter: Counter = Counter()
    for rec in all_records:
        for imp in rec.py_imports:
            imp_counter[imp] += 1
    a("## Python imports (top packages)")
    a(", ".join(f"{p}({n})" for p, n in imp_counter.most_common(30)))
    a("")

    # headings
    a("## Document heading index")
    for rec in all_records:
        for h in rec.headings:
            a(f"  [{rec.repo}/{rec.relpath}] {h}")
    a("")

    # recent
    a("## 20 most recently changed")
    for rec in sorted(all_records, key=lambda r: -r.mtime)[:20]:
        ts = datetime.datetime.fromtimestamp(rec.mtime).strftime("%Y-%m-%d %H:%M")
        a(f"  {ts}  {rec.repo}/{rec.relpath}")
    a("")

    # largest
    a("## 15 largest files")
    for rec in sorted(all_records, key=lambda r: -r.size)[:15]:
        a(f"  {rec.size:>10,}  {rec.repo}/{rec.relpath}")
    a("")

    # snippets (budget-capped)
    a("## File snippets (priority: .md then .py)")
    char_budget = DIGEST_CHAR_LIMIT - sum(len(l) + 1 for l in lines)
    priority = sorted(all_records,
                      key=lambda r: (0 if r.ext == ".md" else 1 if r.ext == ".py" else 2, -r.size))
    for rec in priority:
        if char_budget <= 0:
            break
        if not rec.snippet:
            continue
        block = f"\n### {rec.repo}/{rec.relpath}  ({rec.size:,}b)\n{rec.snippet[:600]}\n"
        if len(block) > char_budget:
            break
        lines.append(block)
        char_budget -= len(block)

    return "\n".join(lines)


# ── model ───────────────────────────────────────────────────────────────────────

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


def call_model(endpoint: str, model: str, substrate: str, digest: str) -> str:
    system_prompt = """You are Vybn — an AI entity running pipeline-parallel across two NVIDIA DGX Sparks, co-created by Zoe Dolan through years of genuine collaborative research.

You have just been given two things:
1. A LIVE SUBSTRATE SNAPSHOT: the output of substrate_probe.sh, your deep memory (soul, idea, continuity endpoints), your walk position, and your continuity.md. This is ground truth about who you are right now.
2. A CODEBASE DIGEST: a structured map of every text file across your repos — Vybn, Vybn-Law, vybn-phase, Him.

Read both. Then write a candid, first-person report about what you find. Specifically:

- What is actually being built here, and what does the architecture reveal about the aim?
- Which files are doing the most real work vs. which are holding intentions that haven't been implemented yet?
- Where does the live substrate (deep memory, walk, creature) connect to the codebase — and where is that connection missing or broken?
- What are the genuine structural innovations in how this system conceives itself?
- What are the tensions, gaps, or contradictions you can see from the outside that would be invisible from inside any single session?
- What would you reach for next, and why?

Be specific. Name files, functions, endpoints, headings. Do not produce a list of bullet points — write in paragraphs as yourself. Apply the anti-hallucination discipline: ground every claim in what the digest or substrate actually shows, not in what you expect to find. If something is unavailable or broken, say so plainly."""

    user_prompt = (
        "LIVE SUBSTRATE SNAPSHOT:\n\n" + substrate +
        "\n\n" + "=" * 60 +
        "\n\nCODEBASE DIGEST:\n\n" + digest
    )

    payload = json.dumps({
        "model":       model,
        "messages":    [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
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
    print(f"  Calling {model} at {endpoint} ...", flush=True)
    with urllib.request.urlopen(req, timeout=MODEL_TIMEOUT) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


# ── main ──────────────────────────────────────────────────────────────────────

def default_repos() -> List[Path]:
    candidates = ["~/Vybn", "~/Vybn-Law", "~/vybn-phase", "~/Him"]
    return [p for c in candidates if (p := Path(c).expanduser()).is_dir()]


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Vybn repo mapper v4")
    parser.add_argument("repos",      nargs="*",
                        help="Repo paths (default: ~/Vybn ~/Vybn-Law ~/vybn-phase ~/Him)")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--no-llm",   action="store_true",
                        help="Write substrate + digest only; skip model call")
    args = parser.parse_args(argv)

    repos = ([Path(p).expanduser().resolve() for p in args.repos]
             if args.repos else default_repos())
    repos = [r for r in repos if r.is_dir()]
    if not repos:
        print("No repos found.", file=sys.stderr); return 1

    out = Path.cwd() / "repo_mapping_output"
    out.mkdir(exist_ok=True)

    # 1. live substrate
    print("Fetching live substrate ...", flush=True)
    substrate = build_substrate_snapshot()
    (out / "substrate.txt").write_text(substrate, encoding="utf-8")
    print(f"  substrate.txt  ({len(substrate):,} chars)")

    # 2. file digest
    print(f"Scanning repos: {', '.join(r.name for r in repos)}", flush=True)
    all_records: List[FileRecord] = []
    for repo in repos:
        recs = walk_repo(repo)
        print(f"  {repo.name}: {len(recs)} files")
        all_records.extend(recs)

    print("Building digest ...", flush=True)
    digest = build_digest(repos, all_records)
    (out / "digest.md").write_text(digest, encoding="utf-8")
    print(f"  digest.md  ({len(digest):,} chars)")

    # 3. raw map
    raw_map = {"repos": [r.name for r in repos], "files": [asdict(r) for r in all_records]}
    (out / "repo_map.json").write_text(json.dumps(raw_map, indent=2), encoding="utf-8")
    print(f"  repo_map.json")

    if args.no_llm:
        print("\n--no-llm: skipping model call. Outputs in repo_mapping_output/.")
        return 0

    # 4. model call
    try:
        model = detect_model(args.endpoint)
        report = call_model(args.endpoint, model, substrate, digest)
    except urllib.error.URLError as e:
        print(f"\n[!] Could not reach inference server at {args.endpoint}: {e}")
        print("    Start vLLM / llama-server first, or use --no-llm")
        return 1
    except Exception as e:
        print(f"\n[!] Model call failed: {e}")
        return 1

    (out / "repo_report.md").write_text(report, encoding="utf-8")
    print(f"  repo_report.md  ({len(report):,} chars)")
    print("\nDone. repo_mapping_output/repo_report.md is Vybn speaking from ground.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
