#!/usr/bin/env python3
"""
repo_mapper.py  v7  — diff-attuned
=================================
Maps the repo constellation, asks Nemotron three focused questions,
stitches the answers, AND reads the previous run's state so every
report opens with a "what changed since last run" delta.

The point is not a snapshot. The point is velocity. Z' - Z is where
we are developing and evolving; a report that only describes Z'
has no access to where we came from. So each run:

  1. rotates the previous repo_report.md → repo_report.prev.md,
     and the previous repo_state.json → repo_state.prev.json;
  2. emits a new repo_state.json with a stable, diff-friendly schema
     (walk step, encounter count, daemon health, per-repo file and
     defs counts, deep-memory version, timestamp);
  3. computes the delta against the previous repo_state.json and
     prepends it to the report as section 0 — so any downstream
     reader (the harness, the nightly evolve cron, Zoe) encounters
     the movement first, the snapshot second.

Pass 1 — Live Infrastructure:  substrate + service/daemon code snippets
Pass 2 — Code Architecture:    Python definitions, imports, TODOs, largest files
Pass 3 — Docs & Ideas:         headings index, concept cloud, markdown snippets

Outputs -> ./repo_mapping_output/
  substrate.txt         live substrate snapshot
  digest.md             full structured digest (uncapped, for reference)
  repo_report.md        delta section + stitched three-pass report
  repo_report.prev.md   the previous run's report (rotated in)
  repo_map.json         machine-readable raw file data
  repo_state.json       stable diff-friendly snapshot
  repo_state.prev.json  the previous run's snapshot (rotated in)

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
import math
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

DEFAULT_ENDPOINT  = "http://localhost:8000/v1"
DEEP_MEMORY_PORT  = 8100
WALK_PORT         = 8101
READ_SIZE_LIMIT   = 120_000

# Each pass must fit comfortably under 32k tokens total.
# ~4 chars/token, 32768 tokens → ~131k chars total per call.
# System prompt ~200 chars, user prefix ~100 chars → ~130k chars for content.
# We reserve 2500 tokens for output → leaves ~120k chars for input per pass.
PASS_CONTENT_CAP  = 55_000   # chars of domain content per pass
SUBSTRATE_CAP     =  8_000   # chars of substrate included in pass 1
MAX_TOKENS_OUT    =  2_500   # output tokens per pass

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
                    data.get("content") or json.dumps(data))[:2000]
        return str(data)[:2000]
    except Exception as e:
        return f"[unavailable: {e}]"


def build_substrate_snapshot() -> str:
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
                parts.append(p.read_text(encoding="utf-8", errors="replace")[:4000])
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


# ── digest (saved to disk, not sent whole to model) ───────────────────────────

def build_full_digest(repos: List[Path], all_records: List[FileRecord]) -> str:
    lines: List[str] = []
    a = lines.append
    a("# CODEBASE DIGEST")
    a(f"Generated: {datetime.datetime.now().isoformat(timespec='seconds')}")
    a(f"Repos: {', '.join(r.name for r in repos)}  |  Files: {len(all_records)}")
    a("")
    gw: Counter = Counter()
    for r in all_records: gw.update(r.top_words)
    a("## Concept cloud (top 60)")
    a(", ".join(f"{w}({c})" for w, c in gw.most_common(60)))
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
    a(", ".join(f"{p}({n})" for p, n in ic.most_common(30)))
    a("")
    a("## Heading index")
    for rec in all_records:
        for h in rec.headings:
            a(f"  [{rec.repo}/{rec.relpath}] {h}")
    a("")
    a("## 30 most recently changed")
    for rec in sorted(all_records, key=lambda r: -r.mtime)[:30]:
        ts = datetime.datetime.fromtimestamp(rec.mtime).strftime("%Y-%m-%d %H:%M")
        a(f"  {ts}  {rec.repo}/{rec.relpath}")
    a("")
    a("## 20 largest files")
    for rec in sorted(all_records, key=lambda r: -r.size)[:20]:
        a(f"  {rec.size:>10,}  {rec.repo}/{rec.relpath}")
    a("")
    a(build_semantic_anatomy(all_records))
    a("")
    a("## All snippets")
    for rec in sorted(all_records, key=lambda r: (0 if r.ext==".md" else 1 if r.ext==".py" else 2, -r.size)):
        if rec.snippet:
            a(f"\n### {rec.repo}/{rec.relpath}\n{rec.snippet}\n")
    return "\n".join(lines)


# ── semantic-operational anatomy ──────────────────────────────────────────────

PUBLIC_ROUTE_RE = re.compile(r"@(?:app|router|self\.app)\.(get|post|put|delete|websocket)\(\s*['\"]([^'\"]+)['\"]")
PUBLIC_ADD_ROUTE_RE = re.compile(r"(?:app|router)\.add_api_route\(\s*['\"]([^'\"]+)['\"]")
PUBLIC_LINK_RE = re.compile(r"(?:src|href)=['\"]([^'\"]+)['\"]")
CHAT_NERVE_TERMS = (
    "/api/chat", "/api/instant", "/api/walk", "/api/arrive",
    "/api/manifold/points", "text/event-stream", "chat/completions",
    "CONTEXT_OVERLAYS", "VLLM", "vllm", "EventSource",
)
MIND_NERVE_TERMS = (
    "build_layered_prompt", "continuity", "BeamKeeper", "router_policy",
    "NEEDS-EXEC", "NEEDS-RESTART", "probe_envelope", "recall", "substrate",
)
HIMOS_NERVE_TERMS = (
    "runtime_snapshot", "render_runtime_context", "PROCESS_TABLE",
    "frictionmaxx", "him_os", "HimOS", "membrane", "seti", "dream",
)
PROVENANCE_TERMS = (
    "Personal History", "Medium", "Artificial Liberation", "autobiography",
    "zoes_memoirs", "what_vybn_would_have_missed", "origin relic",
)
SEDIMENT_TERMS = (
    "sensorium_state", "synaptic_map", "microgpt_mirror", "repo_mapping_output",
    "__pycache__", ".pytest_cache", "trained_checkpoint", "latest.json",
)
ARCHIVE_TERMS = ("_archive", "continuity_archive", "repo_archives", "archive/")
PUBLIC_CONTRACT_TERMS = (
    "llms.txt", "humans.txt", "robots.txt", "ai.txt", "semantic-web",
    "somewhere.html", "talk.html", "connect.html", "read.html", "wellspring.html",
    "chat.html", "vybn.html", "mcp.json",
)


def _vec(rec: FileRecord) -> Counter:
    return Counter(rec.top_words)


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return float(dot / (na * nb)) if na and nb else 0.0


def _contains_any(hay: str, terms: tuple) -> bool:
    low = hay.lower()
    return any(t.lower() in low for t in terms)


def _record_text(rec: FileRecord) -> str:
    """Read full file content for anatomy extraction when available."""
    roots = {
        "Vybn": Path.home() / "Vybn",
        "Him": Path.home() / "Him",
        "Vybn-Law": Path.home() / "Vybn-Law",
        "vybn-phase": Path.home() / "vybn-phase",
        "Origins": Path.home() / "Origins",
    }
    root = roots.get(rec.repo)
    if root:
        path = root / rec.relpath
        try:
            if path.is_file() and path.stat().st_size < 3000001:
                return path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            pass
    return rec.snippet or ""


def _risk_class(rec: FileRecord, inbound: int, centrality: int, semantic_neighbors: int, full_text: str = "") -> str:
    hay = f"{rec.repo}/{rec.relpath}\n{rec.snippet}\n{full_text}"
    if _contains_any(hay, PROVENANCE_TERMS):
        return "protect: origin/provenance"
    if rec.repo in {"Him", "vybn-phase"} and (rec.repo == "vybn-phase" or _contains_any(hay, HIMOS_NERVE_TERMS)):
        return "protect: private membrane"
    if _contains_any(hay, PUBLIC_CONTRACT_TERMS) or _contains_any(hay, CHAT_NERVE_TERMS):
        return "protect: public/interface nerve"
    if inbound > 0 or centrality >= 4:
        return "protect/refactor: connected code"
    if _contains_any(hay, SEDIMENT_TERMS):
        return "investigate: generated sediment"
    if _contains_any(hay, ARCHIVE_TERMS):
        return "investigate: archive/restore path"
    if rec.size > 80000 and semantic_neighbors >= 3:
        return "inspect: large semantically covered"
    if rec.size > 80000:
        return "inspect: large unique"
    return "ordinary"


def build_semantic_anatomy(all_records: List[FileRecord]) -> str:
    """Build a grounded, ML-lite anatomy layer.

    This is intentionally local and deterministic: lexical vectors stand in for
    embeddings until a local embedding pass is added. The point is the coupled
    method: deductive guardrails + inductive topology + abductive ABC hints.
    """
    by_key = {f"{r.repo}/{r.relpath}": r for r in all_records}
    edges: List[tuple] = []
    route_rows: List[str] = []
    link_rows: List[str] = []

    for rec in all_records:
        key = f"{rec.repo}/{rec.relpath}"
        txt = _record_text(rec)
        if rec.ext == ".py":
            for method, route in PUBLIC_ROUTE_RE.findall(txt):
                route_rows.append(f"  - {key}: {method.upper()} {route}")
                edges.append((key, f"route:{route}", "defines_route"))
            for route in PUBLIC_ADD_ROUTE_RE.findall(txt):
                route_rows.append(f"  - {key}: ROUTE {route}")
                edges.append((key, f"route:{route}", "defines_route"))
        if rec.ext in {".html", ".js", ".md"}:
            for link in PUBLIC_LINK_RE.findall(txt)[:20]:
                if link.startswith(("http", "/", "./", "../")) or ".html" in link or ".js" in link or ".css" in link:
                    link_rows.append(f"  - {key} -> {link}")
                    edges.append((key, link, "links"))
        for term in CHAT_NERVE_TERMS + MIND_NERVE_TERMS + HIMOS_NERVE_TERMS:
            if term in txt:
                edges.append((key, f"term:{term}", "mentions_term"))
        for imp in rec.py_imports:
            edges.append((key, f"import:{imp}", "imports"))

    inbound: Counter = Counter()
    outbound: Counter = Counter()
    for src, dst, kind in edges:
        outbound[src] += 1
        inbound[dst] += 1

    # Lexical semantic neighborhoods.
    vecs = {f"{r.repo}/{r.relpath}": _vec(r) for r in all_records if r.top_words}
    pairs: List[tuple] = []
    keys = list(vecs)
    for i, a in enumerate(keys):
        for b in keys[i+1:]:
            sim = _cosine(vecs[a], vecs[b])
            if sim >= 0.55:
                pairs.append((sim, a, b))
    pairs.sort(reverse=True)
    neighbor_count: Counter = Counter()
    for sim, a, b in pairs:
        neighbor_count[a] += 1
        neighbor_count[b] += 1

    classified = []
    for rec in all_records:
        key = f"{rec.repo}/{rec.relpath}"
        centrality = outbound[key]
        rclass = _risk_class(rec, inbound[key], centrality, neighbor_count[key], _record_text(rec))
        classified.append((rclass, rec.size, key, inbound[key], centrality, neighbor_count[key]))

    lines: List[str] = []
    a = lines.append
    a("## Semantic-operational anatomy")
    a("")
    a("Method: deductive guardrails + inductive lexical topology + abductive ABC hypotheses + verification before cuts. This is ML-lite for now: per-file lexical vectors, route/link/import edges, and risk classes. A later pass can replace lexical vectors with local embeddings without changing the map contract.")
    a("")
    a("### Risk-class counts")
    class_counts = Counter(row[0] for row in classified)
    for name, count in class_counts.most_common():
        a(f"  - {name}: {count}")
    a("")
    a("### High-centrality / nerve candidates")
    for rclass, sz, key, inn, out, neigh in sorted(classified, key=lambda x: (-(x[3]+x[4]+x[5]), -x[1]))[:30]:
        a(f"  - {key} — {rclass}; inbound={inn}, outbound={out}, semantic_neighbors={neigh}, size={sz}")
    a("")
    a("### Strong semantic overlaps")
    if pairs:
        for sim, x, y in pairs[:40]:
            a(f"  - {sim:.2f}: {x} ↔ {y}")
    else:
        a("  none above threshold")
    a("")
    a("### ABC pressure hypotheses")
    for wanted in [
        "investigate: generated sediment",
        "investigate: archive/restore path",
        "inspect: large semantically covered",
        "inspect: large unique",
    ]:
        rows = [row for row in classified if row[0] == wanted]
        if not rows:
            continue
        a(f"#### {wanted}")
        for rclass, sz, key, inn, out, neigh in sorted(rows, key=lambda x: -x[1])[:25]:
            a(f"  - {key} — inbound={inn}, outbound={out}, semantic_neighbors={neigh}, size={sz}")
        a("")
    a("### Public/API route and link edges sampled")
    if route_rows:
        a("Routes:")
        a("\n".join(route_rows[:80]))
    if link_rows:
        a("Links/assets:")
        a("\n".join(link_rows[:80]))
    a("")
    return "\n".join(lines)


# ── per-pass content builders ─────────────────────────────────────────────────

def pass1_content(substrate: str, all_records: List[FileRecord]) -> str:
    """Pass 1: live state + service/daemon source files."""
    lines = []
    lines.append("## Live Substrate\n")
    lines.append(substrate[:SUBSTRATE_CAP])
    lines.append("\n\n## Service & Daemon Source Files\n")
    budget = PASS_CONTENT_CAP - len(substrate[:SUBSTRATE_CAP]) - 200
    service_keywords = {"api", "daemon", "server", "service", "harness", "agent",
                        "portal", "chat", "walk", "memory", "creature", "organism"}
    candidates = [r for r in all_records
                  if r.ext == ".py" and r.snippet
                  and any(kw in r.relpath.lower() for kw in service_keywords)]
    candidates.sort(key=lambda r: -r.size)
    for rec in candidates:
        block = f"\n### {rec.repo}/{rec.relpath}\nDefs: {', '.join(rec.py_defs[:20])}\n{rec.snippet[:600]}\n"
        if budget - len(block) < 0:
            break
        lines.append(block)
        budget -= len(block)
    return "\n".join(lines)


def pass2_content(all_records: List[FileRecord]) -> str:
    """Pass 2: code architecture — defs, imports, TODOs, recent changes, largest files."""
    lines = []
    lines.append("## Python Definitions by File\n")
    for rec in all_records:
        if rec.py_defs:
            lines.append(f"  {rec.repo}/{rec.relpath}: {', '.join(rec.py_defs)}")
    lines.append("")

    ic: Counter = Counter()
    for rec in all_records:
        for imp in rec.py_imports: ic[imp] += 1
    lines.append("## Import Frequency\n")
    lines.append(", ".join(f"{p}({n})" for p, n in ic.most_common(40)))
    lines.append("")

    lines.append("## TODOs\n")
    for rec in all_records:
        for t in rec.todos:
            lines.append(f"  [{rec.repo}/{rec.relpath}] {t}")
    lines.append("")

    lines.append("## 30 Most Recently Changed\n")
    for rec in sorted(all_records, key=lambda r: -r.mtime)[:30]:
        ts = datetime.datetime.fromtimestamp(rec.mtime).strftime("%Y-%m-%d %H:%M")
        lines.append(f"  {ts}  {rec.repo}/{rec.relpath}")
    lines.append("")

    lines.append("## 20 Largest Files\n")
    for rec in sorted(all_records, key=lambda r: -r.size)[:20]:
        lines.append(f"  {rec.size:>10,}  {rec.repo}/{rec.relpath}")
    lines.append("")

    # Fill remaining budget with .py snippets
    body = "\n".join(lines)
    budget = PASS_CONTENT_CAP - len(body)
    py_recs = sorted([r for r in all_records if r.ext == ".py" and r.snippet],
                     key=lambda r: -r.size)
    snippets = []
    for rec in py_recs:
        block = f"\n### {rec.repo}/{rec.relpath}\n{rec.snippet[:500]}\n"
        if budget - len(block) < 0:
            break
        snippets.append(block)
        budget -= len(block)
    return body + "\n## Python Snippets\n" + "\n".join(snippets)


def pass3_content(all_records: List[FileRecord]) -> str:
    """Pass 3: documentation, ideas, concept cloud — heading index + md snippets."""
    lines = []
    gw: Counter = Counter()
    for r in all_records: gw.update(r.top_words)
    lines.append("## Concept Cloud (top 60)\n")
    lines.append(", ".join(f"{w}({c})" for w, c in gw.most_common(60)))
    lines.append("")

    lines.append("## Heading Index\n")
    for rec in all_records:
        for h in rec.headings:
            lines.append(f"  [{rec.repo}/{rec.relpath}] {h}")
    lines.append("")

    body = "\n".join(lines)
    budget = PASS_CONTENT_CAP - len(body)
    md_recs = sorted([r for r in all_records if r.ext in {".md", ".txt", ".rst"} and r.snippet],
                     key=lambda r: -r.size)
    snippets = []
    for rec in md_recs:
        block = f"\n### {rec.repo}/{rec.relpath}\n{rec.snippet[:500]}\n"
        if budget - len(block) < 0:
            break
        snippets.append(block)
        budget -= len(block)
    return body + "\n## Document Snippets\n" + "\n".join(snippets)


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


def strip_reasoning(text: str) -> str:
    """Remove chain-of-thought preamble emitted by reasoning models."""
    if "</think>" in text:
        text = text[text.index("</think>") + len("</think>"):]
        return text.strip()
    paragraphs = re.split(r"\n{2,}", text)
    for i, para in enumerate(paragraphs):
        first = para.lstrip()
        if first.startswith(("I am", "I run", "I have", "##", "# ", "My ", "As of",
                              "The ", "At ", "Across ", "Running ", "Five ")):
            return "\n\n".join(paragraphs[i:]).strip()
    return text.strip()


def call_model(endpoint: str, model: str,
               system_prompt: str, user_content: str,
               pass_name: str) -> str:
    payload = json.dumps({
        "model":       model,
        "messages":    [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ],
        "temperature": 0.7,
        "max_tokens":  MAX_TOKENS_OUT,
        "stream":      False,
    }).encode()

    print(f"    {pass_name}: {len(payload):,} bytes", flush=True)
    req = urllib.request.Request(
        f"{endpoint}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=MODEL_TIMEOUT) as resp:
            data = json.loads(resp.read())
        raw = data["choices"][0]["message"]["content"]
        return strip_reasoning(raw)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} ({pass_name}):\n{body[:1000]}")


def run_three_passes(endpoint: str, model: str,
                     substrate: str, all_records: List[FileRecord]) -> str:
    identity = (
        "You are Vybn — an AI entity running pipeline-parallel across two NVIDIA DGX Sparks, "
        "co-created by Zoe Dolan through years of genuine collaborative research. "
        "Write in first person. Begin with the first sentence of the report. "
        "Do not show reasoning, steps, or preamble."
    )

    print("  Running 3-pass model calls ...", flush=True)

    # Pass 1
    s1 = (identity + " Focus: your live infrastructure — which services are running, "
          "what their source files reveal, and how live state connects to code.")
    u1 = pass1_content(substrate, all_records) + "\n\nWrite the infrastructure section now."
    r1 = call_model(endpoint, model, s1, u1, "Pass 1 infrastructure")

    # Pass 2
    s2 = (identity + " Focus: code architecture — Python modules, function inventory, "
          "import dependencies, TODOs, recently changed files, and what they reveal "
          "about what is actually built versus intended.")
    u2 = pass2_content(all_records) + "\n\nWrite the code architecture section now."
    r2 = call_model(endpoint, model, s2, u2, "Pass 2 code architecture")

    # Pass 3
    s3 = (identity + " Focus: documentation, ideas, and conceptual structure — "
          "the heading index, concept cloud, and document snippets. What ideas recur? "
          "What is theorized, what is documented, what is aspirational?")
    u3 = pass3_content(all_records) + "\n\nWrite the documentation and ideas section now."
    r3 = call_model(endpoint, model, s3, u3, "Pass 3 docs & ideas")

    ts = datetime.datetime.now().isoformat(timespec="seconds")
    return (
        f"# Repository Map Report\n\nGenerated: {ts}  |  Model: {model}\n\n"
        f"---\n\n## I. Live Infrastructure\n\n{r1}\n\n"
        f"---\n\n## II. Code Architecture\n\n{r2}\n\n"
        f"---\n\n## III. Documentation & Ideas\n\n{r3}\n"
    )


# ── main ──────────────────────────────────────────────────────────────────────

# ── diff-attuned state + delta ───────────────────────────────────────────────
#
# The snapshot produced here is intentionally small, typed, and stable.
# It is what the nightly evolve loop diffs against the previous run to
# see where the system is developing: what grew, what broke, what Zoe
# touched since last we looked. Anything deep or narrative-shaped
# lives in repo_report.md; this file is ground-truth numbers only.

def _probe_daemon(port: int, path: str = "/status") -> dict:
    url = f"http://localhost:{port}{path}"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
        return data if isinstance(data, dict) else {"_value": data}
    except Exception as e:
        return {"_error": str(e)[:200]}


def _read_organism_state() -> dict:
    for p in [
        Path("~/Vybn/Vybn_Mind/creature_dgm_h/organism_state.json").expanduser(),
        Path("Vybn_Mind/creature_dgm_h/organism_state.json"),
    ]:
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8", errors="replace"))
            except Exception as e:
                return {"_error": str(e)[:200]}
    return {"_error": "organism_state.json not found"}


def _read_deep_memory_meta() -> dict:
    p = Path("~/.cache/vybn-phase/deep_memory_meta.json").expanduser()
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8", errors="replace"))
        except Exception as e:
            return {"_error": str(e)[:200]}
    return {"_error": "deep_memory_meta.json not found"}


def build_repo_state(repos: List[Path],
                     all_records: List[FileRecord]) -> dict:
    """Stable diff-friendly snapshot. Every field has a clear numeric or
    string shape so two consecutive runs can be compared field-by-field.
    """
    per_repo: Dict[str, dict] = {}
    for r in repos:
        recs = [x for x in all_records if x.repo == r.name]
        py = [x for x in recs if x.ext == ".py"]
        md = [x for x in recs if x.ext in {".md", ".rst", ".txt"}]
        per_repo[r.name] = {
            "files":         len(recs),
            "py_files":      len(py),
            "md_files":      len(md),
            "py_def_count":  sum(len(x.py_defs) for x in py),
            "total_bytes":   sum(x.size for x in recs),
            "latest_mtime":  max((x.mtime for x in recs), default=0.0),
            "recent_files":  [
                f"{x.repo}/{x.relpath}"
                for x in sorted(recs, key=lambda y: -y.mtime)[:5]
            ],
        }

    walk_status = _probe_daemon(WALK_PORT)
    dm_status   = _probe_daemon(DEEP_MEMORY_PORT)
    organism    = _read_organism_state()
    dm_meta     = _read_deep_memory_meta()

    return {
        "generated_at": datetime.datetime.now(datetime.timezone.utc)
                               .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repos":       sorted(r.name for r in repos),
        "per_repo":    per_repo,
        "totals": {
            "files":        len(all_records),
            "py_files":     sum(1 for r in all_records if r.ext == ".py"),
            "md_files":     sum(1 for r in all_records if r.ext in {".md",".rst",".txt"}),
            "py_def_count": sum(len(r.py_defs) for r in all_records if r.ext == ".py"),
            "todo_count":   sum(len(r.todos) for r in all_records),
            "total_bytes":  sum(r.size for r in all_records),
        },
        "walk": {
            "step":              walk_status.get("step"),
            "alpha":             walk_status.get("alpha"),
            "winding_coherence": walk_status.get("winding_coherence"),
            "active":            walk_status.get("walk_active"),
            "error":             walk_status.get("_error"),
        },
        "deep_memory": {
            "version":     dm_meta.get("version"),
            "chunks":      dm_meta.get("chunks"),
            "built_at":    dm_meta.get("built_at"),
            "status_error": dm_status.get("_error"),
        },
        "organism": {
            "encounter_count": organism.get("encounter_count"),
            "error":           organism.get("_error"),
        },
    }


def _fmt_scalar(v) -> str:
    if v is None:    return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def build_delta_section(prev: Optional[dict], curr: dict) -> str:
    """Render the 'What changed since last run' section.

    The shape is deliberately skimmable so a reader (human or agent)
    encounters velocity first, snapshot second. Fields that did not
    move are omitted — the signal is in the deltas.
    """
    lines = ["## 0. What changed since last run\n"]
    if prev is None:
        lines.append("  No previous repo_state.json found — this is the "
                     "first diff-attuned run. Next run will compare against "
                     "this one.")
        return "\n".join(lines) + "\n"

    prev_ts = prev.get("generated_at", "—")
    curr_ts = curr.get("generated_at", "—")
    lines.append(f"Previous run: {prev_ts}")
    lines.append(f"Current run:  {curr_ts}")
    lines.append("")

    def _diff_scalar(label: str, a, b) -> Optional[str]:
        if a == b: return None
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return f"  {label}: {_fmt_scalar(a)} → {_fmt_scalar(b)} ({b - a:+})"
        return f"  {label}: {_fmt_scalar(a)} → {_fmt_scalar(b)}"

    moved = []
    for k in ("files","py_files","md_files","py_def_count","todo_count","total_bytes"):
        row = _diff_scalar(
            f"totals.{k}",
            prev.get("totals", {}).get(k),
            curr.get("totals", {}).get(k),
        )
        if row: moved.append(row)

    repos = sorted(set(prev.get("per_repo", {})) | set(curr.get("per_repo", {})))
    for r in repos:
        prev_r = prev.get("per_repo", {}).get(r, {})
        curr_r = curr.get("per_repo", {}).get(r, {})
        for k in ("files","py_files","md_files","py_def_count","total_bytes"):
            row = _diff_scalar(f"{r}.{k}", prev_r.get(k), curr_r.get(k))
            if row: moved.append(row)

    for k in ("step","alpha","winding_coherence","active"):
        row = _diff_scalar(
            f"walk.{k}",
            prev.get("walk", {}).get(k),
            curr.get("walk", {}).get(k),
        )
        if row: moved.append(row)

    for k in ("version","chunks","built_at"):
        row = _diff_scalar(
            f"deep_memory.{k}",
            prev.get("deep_memory", {}).get(k),
            curr.get("deep_memory", {}).get(k),
        )
        if row: moved.append(row)

    row = _diff_scalar(
        "organism.encounter_count",
        prev.get("organism", {}).get("encounter_count"),
        curr.get("organism", {}).get("encounter_count"),
    )
    if row: moved.append(row)

    if not moved:
        lines.append("  Nothing moved between runs. The substrate is at "
                     "rest.")
    else:
        lines.extend(moved)
    return "\n".join(lines) + "\n"


def rotate_output(out: Path, name: str) -> Optional[Path]:
    """Move out/<name> to out/<stem>.prev.<ext> and return the prev path."""
    src = out / name
    if not src.exists():
        return None
    stem, ext = src.stem, src.suffix
    dst = out / f"{stem}.prev{ext}"
    try:
        if dst.exists():
            dst.unlink()
        src.rename(dst)
        return dst
    except Exception as e:
        print(f"[!] rotate {name}: {e}", file=sys.stderr)
        return None


def default_repos() -> List[Path]:
    return [p for c in ["~/Vybn","~/Vybn-Law","~/vybn-phase","~/Him"]
            if (p := Path(c).expanduser()).is_dir()]


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Vybn repo mapper v6")
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

    # Read the previous state BEFORE we rotate it — we want to diff
    # against what was here when this run started, not against ourselves.
    prev_state_path = out / "repo_state.json"
    prev_state: Optional[dict] = None
    if prev_state_path.exists():
        try:
            prev_state = json.loads(prev_state_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[!] could not read previous repo_state.json: {e}",
                  file=sys.stderr)

    # Rotate previous outputs so the diff-attuned reader can still see
    # what the system looked like last night.
    rotate_output(out, "repo_report.md")
    rotate_output(out, "repo_state.json")

    print("Fetching live substrate ...", flush=True)
    substrate = build_substrate_snapshot()
    (out / "substrate.txt").write_text(substrate, encoding="utf-8")
    print(f"  substrate.txt  ({len(substrate):,} chars)")

    print(f"Scanning: {', '.join(r.name for r in repos)}", flush=True)
    all_records: List[FileRecord] = []
    for repo in repos:
        recs = walk_repo(repo)
        print(f"  {repo.name}: {len(recs)} files")
        all_records.extend(recs)

    print("Building full digest ...", flush=True)
    digest = build_full_digest(repos, all_records)
    (out / "digest.md").write_text(digest, encoding="utf-8")
    print(f"  digest.md  ({len(digest):,} chars)")

    raw = {"repos": [r.name for r in repos], "files": [asdict(r) for r in all_records]}
    (out / "repo_map.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
    print("  repo_map.json")

    print("Building repo_state.json (diff-friendly snapshot) ...", flush=True)
    state = build_repo_state(repos, all_records)
    (out / "repo_state.json").write_text(
        json.dumps(state, indent=2), encoding="utf-8"
    )
    print("  repo_state.json")

    delta_section = build_delta_section(prev_state, state)

    if args.no_llm:
        # Still write a minimal report so downstream readers see the
        # delta even when Nemotron is skipped.
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        short = (
            f"# Repository Map Report\n\nGenerated: {ts}  |  Model: — (--no-llm)\n\n"
            f"---\n\n{delta_section}\n"
            "---\n\n## I–III. (skipped: --no-llm)\n"
        )
        (out / "repo_report.md").write_text(short, encoding="utf-8")
        print(f"  repo_report.md  ({len(short):,} chars)")
        print("\n--no-llm set. Done.")
        return 0

    try:
        model = detect_model(args.endpoint)
        print(f"  Model: {model}")
        report = run_three_passes(args.endpoint, model, substrate, all_records)
    except RuntimeError as e:
        print(f"\n[!] {e}"); return 1
    except urllib.error.URLError as e:
        print(f"\n[!] Cannot reach {args.endpoint}: {e}")
        print("    Use --no-llm to skip the model call."); return 1

    # Prepend the delta section as section 0 so any reader encounters
    # velocity first, snapshot second.
    header, sep, body = report.partition("---\n\n## I. Live Infrastructure")
    if sep:
        stitched = (
            header + "---\n\n" + delta_section + "\n"
            + "---\n\n## I. Live Infrastructure" + body
        )
    else:
        stitched = delta_section + "\n" + report
    (out / "repo_report.md").write_text(stitched, encoding="utf-8")
    print(f"  repo_report.md  ({len(stitched):,} chars)")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
