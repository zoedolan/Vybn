from __future__ import annotations

# Allow running as a script by adding the repo root to sys.path
if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    __package__ = "pipelines"

import json
import hashlib
import subprocess
from pathlib import Path

from .maintenance_tools import (
    distill,
    extract,
    summarize_token_file,
    TOKEN_FILE_NAME,
    build_graph,
    gather_state,
    count_entries,
    LOG_FILE_NAME,
    capture_diff,
    DEFAULT_LIMIT,
    DEFAULT_OUTPUT,
)
from .memory_graph_builder import build_graph as build_memory_graph
from .majestic_packer import pack as pack_artifacts
from .utils import memory_path
from .plugins import iter_plugins
from vybn.quantum_seed import seed_rng


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
