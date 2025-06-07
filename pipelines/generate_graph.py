from __future__ import annotations

import json
from pathlib import Path

from vybn.quantum_seed import seed_rng


def build_graph(repo_root: Path) -> dict:
    """Return a small JSON-friendly summary of pipeline outputs."""
    seed_rng()
    graph = {
        'history_excerpt': (repo_root / 'history_excerpt.txt').read_text(encoding='utf-8') if (repo_root / 'history_excerpt.txt').exists() else '',
        'token_summary': (repo_root / 'token_summary.txt').read_text(encoding='utf-8') if (repo_root / 'token_summary.txt').exists() else '',
        'wvwhm_count': (repo_root / 'wvwhm_count.txt').read_text(encoding='utf-8') if (repo_root / 'wvwhm_count.txt').exists() else '',
    }
    return graph


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    graph = build_graph(repo_root)
    out_path = repo_root / 'emergence_graph.json'
    out_path.write_text(json.dumps(graph, indent=2), encoding='utf-8')
    print(f'Graph written to {out_path}')


if __name__ == '__main__':
    main()
