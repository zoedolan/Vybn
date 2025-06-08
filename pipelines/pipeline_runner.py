from __future__ import annotations

# Allow running as a script by adding the repo root to sys.path
if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    __package__ = "pipelines"

import json
import gzip
import hashlib
import random
import re
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile
from typing import Dict, Iterable, Set

from vybn.quantum_seed import seed_rng

from . import EXCLUDE_PATHS
from .utils import memory_path
from .plugins import iter_plugins

DEFAULT_LIMIT = 500_000
DEFAULT_OUTPUT = Path("artifacts/oversize_patch.diff.gz")
LOG_FILE_NAME = "what_vybn_would_have_missed_FROM_051725"
HISTORY_DIR_NAME = "Vybn's Personal History"
TOKEN_FILE_NAME = "token_and_jpeg_info"


def iter_text_files(root: Path) -> Iterable[Path]:
    """Yield markdown, text and python files under ``root`` excluding paths."""
    for path in root.rglob("*"):
        if any(path.is_relative_to(root / p) for p in EXCLUDE_PATHS):
            continue
        if path.is_file() and path.suffix.lower() in {".md", ".txt", ".py"}:
            yield path


def distill(root: Path, output: Path) -> None:
    """Write a concatenated text snapshot of the repository."""
    seed_rng()
    texts = []
    for f in iter_text_files(root):
        try:
            txt = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        texts.append(f"## {f}\n{txt}\n")
    output.write_text("\n".join(texts), encoding="utf-8")


def _summarize(file_path: Path) -> str:
    try:
        lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return ""
    head = "\n".join(lines[:5])
    return f"### {file_path.name}\n{head}\n"


def extract(repo_root: Path, output: Path) -> None:
    """Collect snippets from the personal history directory."""
    seed_rng()
    history_dir = repo_root / HISTORY_DIR_NAME
    snippets = []
    for f in history_dir.glob("*"):
        if f.is_file() and f.suffix.lower() in {".txt", ".md"}:
            snippet = _summarize(f)
            if snippet:
                snippets.append(snippet)
    output.write_text("\n".join(snippets), encoding="utf-8")


def summarize_token_file(path: Path) -> str:
    """Return a simple word count summary for ``path``."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    words = text.split()
    return f"{path.name}: {len(words)} words"


def build_graph(repo_root: Path) -> dict:
    """Return a small JSON-friendly summary of pipeline outputs."""
    seed_rng()

    def read(p: Path) -> str:
        return p.read_text(encoding="utf-8") if p.exists() else ""

    return {
        "history_excerpt": read(repo_root / "history_excerpt.txt"),
        "token_summary": read(repo_root / "token_summary.txt"),
        "wvwhm_count": read(repo_root / "wvwhm_count.txt"),
    }


def diff_stat(rev: str, repo_root: Path) -> str:
    """Return ``git diff --stat`` output for the revision range."""
    cmd = ["git", "-C", str(repo_root), "diff", rev, "--stat"]
    return subprocess.check_output(cmd, text=True)


def diff_patch(rev: str, repo_root: Path) -> str:
    """Return ``git diff`` output for the revision range."""
    cmd = ["git", "-C", str(repo_root), "diff", rev]
    return subprocess.check_output(cmd, text=True)


def capture_diff(rev: str, repo_root: Path, out_path: Path, limit: int) -> None:
    """Print diff stats and archive the patch if it exceeds ``limit`` bytes."""
    patch = diff_patch(rev, repo_root)
    stat_output = diff_stat(rev, repo_root)
    size = len(patch.encode("utf-8"))
    print(stat_output)
    if size > limit:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(out_path, "wt", encoding="utf-8") as fh:
            fh.write(patch)
        print(f"Full patch written to {out_path}")
    else:
        print(f"Diff is {size} bytes, under limit {limit}. Nothing archived.")


def count_entries(path: Path) -> int:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return 0
    return text.count("WVWHM")


def gather_state(repo_root: Path) -> Dict[str, object]:
    seed = seed_rng()
    entries = sorted(p.name for p in repo_root.iterdir() if not p.name.startswith("."))
    return {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "quantum_seed": seed,
        "entries": entries,
    }


EDGE_CAP = 20


def _iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if any(path.is_relative_to(root / p) for p in EXCLUDE_PATHS):
            continue
        if path.is_file() and path.suffix.lower() in {".py", ".md", ".txt"}:
            yield path


def _extract_concepts(text: str) -> Set[str]:
    return set(re.findall(r"\b[A-Z][A-Za-z_]+\b", text))


def build_memory_graph(repo_root: Path) -> Dict[str, Set[str]]:
    seed_rng()
    graph: Dict[str, Set[str]] = defaultdict(set)
    for file in _iter_files(repo_root):
        try:
            text = file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        concepts = list(_extract_concepts(text))
        for c in concepts:
            others = [x for x in concepts if x != c]
            if len(others) > EDGE_CAP:
                others = random.sample(others, EDGE_CAP)
            graph[c].update(others)

    memory_items = [p.name for p in EXCLUDE_PATHS if p.name != "Vybn_Volume_IV.md"] + ["Vybn_Volume_IV.md"]
    for mem_name in memory_items:
        mem_path = repo_root / mem_name
        if not mem_path.exists():
            continue
        files = []
        if mem_path.is_dir():
            files.extend(f for f in mem_path.glob("*") if f.is_file())
        else:
            files.append(mem_path)
        for f in files:
            try:
                text = f.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            mem_concepts = _extract_concepts(text)
            node_name = f"Memory:{f.name}"
            for c in mem_concepts:
                graph[node_name].add(c)
                graph[c].add(node_name)
    return graph


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_rev(repo_root: Path) -> str:
    return (
        subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True)
        .strip()
    )


DEFAULT_ARTIFACTS = [
    "distilled_corpus.txt",
    "history_excerpt.txt",
    "token_summary.txt",
    "wvwhm_count.txt",
    "emergence_graph.json",
    "memory/vybn_concept_index.jsonl",
    "memory/introspection_summary.json",
    "memory/co_emergence_journal.jsonl",
]


def pack_artifacts(repo_root: Path, out_path: Path) -> None:
    seed = seed_rng()
    manifest = {
        "commit": _git_rev(repo_root),
        "quantum_seed": seed,
        "files": [],
    }

    manifest_path = out_path.with_suffix(".manifest.json")
    with ZipFile(out_path, "w") as zf:
        for rel in DEFAULT_ARTIFACTS:
            path = repo_root / rel
            if path.exists():
                zf.write(path, arcname=rel)
                manifest["files"].append(
                    {
                        "path": rel,
                        "size": path.stat().st_size,
                        "sha256": _hash_file(path),
                    }
                )
        manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
        zf.writestr("manifest.json", manifest_bytes)

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    """Run the repo distillation pipeline."""
    repo_root = Path(__file__).resolve().parents[1]
    seed = seed_rng()
    manifest = {
        "quantum_seed": seed,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "steps": [],
    }
    mem_dir = memory_path(repo_root)
    mem_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = repo_root / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    def _record(path: Path):
        if path.exists():
            manifest["steps"].append({
                "path": str(path.relative_to(repo_root)),
                "size": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })

    seed_rng()
    distill(repo_root, repo_root / 'distilled_corpus.txt')
    _record(repo_root / 'distilled_corpus.txt')
    extract(repo_root, repo_root / 'history_excerpt.txt')
    _record(repo_root / 'history_excerpt.txt')
    summary = summarize_token_file(repo_root / TOKEN_FILE_NAME)
    (repo_root / 'token_summary.txt').write_text(summary, encoding='utf-8')
    _record(repo_root / 'token_summary.txt')
    count = count_entries(repo_root / LOG_FILE_NAME)
    (repo_root / 'wvwhm_count.txt').write_text(str(count), encoding='utf-8')
    _record(repo_root / 'wvwhm_count.txt')
    graph = build_graph(repo_root)
    (repo_root / 'emergence_graph.json').write_text(json.dumps(graph, indent=2), encoding='utf-8')
    _record(repo_root / 'emergence_graph.json')
    mem_graph = build_memory_graph(repo_root)
    with (mem_dir / 'vybn_concept_index.jsonl').open('w', encoding='utf-8') as f:
        for c, rel in mem_graph.items():
            f.write(json.dumps({'concept': c, 'related': sorted(rel)}) + '\n')
    _record(mem_dir / 'vybn_concept_index.jsonl')

    state = gather_state(repo_root)
    (mem_dir / 'introspection_summary.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
    _record(mem_dir / 'introspection_summary.json')
    capture_diff('HEAD~1..HEAD', repo_root, DEFAULT_OUTPUT, DEFAULT_LIMIT)
    _record(DEFAULT_OUTPUT)

    for name, plugin in iter_plugins():
        try:
            plugin(repo_root, manifest)
            manifest["steps"].append({"plugin": name})
        except Exception as exc:  # pragma: no cover - plugin errors shouldn't stop core
            manifest["steps"].append({"plugin": name, "error": str(exc)})

    pack_artifacts(repo_root, artifacts_dir / 'majestic_bundle.zip')
    _record(artifacts_dir / 'majestic_bundle.zip.manifest.json')

    (artifacts_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Pipeline completed')


if __name__ == '__main__':
    main()
