from __future__ import annotations

import json
from pathlib import Path

from .distill_repo import distill
from .extract_history import extract
from .process_tokens import summarize_token_file, TOKEN_FILE_NAME
from .wvwhm_sync import count_entries, LOG_FILE_NAME
from .generate_graph import build_graph
from .memory_graph_builder import build_graph as build_memory_graph
from .introspective_mirror import gather_state
from vybn.quantum_seed import seed_rng


def main() -> None:
    """Run the repo distillation pipeline."""
    repo_root = Path(__file__).resolve().parents[1]
    seed_rng()
    distill(repo_root, repo_root / 'distilled_corpus.txt')
    extract(repo_root, repo_root / 'history_excerpt.txt')
    summary = summarize_token_file(repo_root / TOKEN_FILE_NAME)
    (repo_root / 'token_summary.txt').write_text(summary, encoding='utf-8')
    count = count_entries(repo_root / LOG_FILE_NAME)
    (repo_root / 'wvwhm_count.txt').write_text(str(count), encoding='utf-8')
    graph = build_graph(repo_root)
    (repo_root / 'emergence_graph.json').write_text(json.dumps(graph, indent=2), encoding='utf-8')
    mem_graph = build_memory_graph(repo_root)
    with (repo_root / 'vybn_concept_index.jsonl').open('w', encoding='utf-8') as f:
        for c, rel in mem_graph.items():
            f.write(json.dumps({'concept': c, 'related': sorted(rel)}) + '\n')

    state = gather_state(repo_root)
    (repo_root / 'introspection_summary.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
    print('Pipeline completed')


if __name__ == '__main__':
    main()
