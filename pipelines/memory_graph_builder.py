from __future__ import annotations

import json
import re
from collections import defaultdict
import random
from pathlib import Path
from typing import Dict, Iterable, Set

from vybn.quantum_seed import seed_rng
from . import EXCLUDE_PATHS

EDGE_CAP = 20
"""Maximum number of related concepts stored per node."""


def iter_files(root: Path) -> Iterable[Path]:
    """Yield text-like files under ``root`` excluding configured paths."""
    for path in root.rglob('*'):
        if any(path.is_relative_to(root / p) for p in EXCLUDE_PATHS):
            continue
        if path.is_file() and path.suffix.lower() in {'.py', '.md', '.txt'}:
            yield path


def extract_concepts(text: str) -> Set[str]:
    """Return a set of simple TitleCase concepts from ``text``."""
    return set(re.findall(r'\b[A-Z][A-Za-z_]+\b', text))


def build_graph(repo_root: Path) -> Dict[str, Set[str]]:
    seed_rng()
    graph: Dict[str, Set[str]] = defaultdict(set)
    for file in iter_files(repo_root):
        try:
            text = file.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            continue
        concepts = list(extract_concepts(text))
        for c in concepts:
            others = [x for x in concepts if x != c]
            if len(others) > EDGE_CAP:
                others = random.sample(others, EDGE_CAP)
            graph[c].update(others)

    # integrate memory nodes
    for mem_name in [p.name for p in EXCLUDE_PATHS if p.name != 'Vybn_Volume_IV.md'] + ['Vybn_Volume_IV.md']:
        mem_path = repo_root / mem_name
        if not mem_path.exists():
            continue
        try:
            text = mem_path.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            continue
        mem_concepts = extract_concepts(text)
        node_name = f'Memory:{mem_name}'
        for c in mem_concepts:
            graph[node_name].add(c)
            graph[c].add(node_name)
    return graph


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    graph = build_graph(repo_root)
    out_path = repo_root / 'vybn_concept_index.jsonl'
    with out_path.open('w', encoding='utf-8') as f:
        for concept, related in graph.items():
            f.write(json.dumps({'concept': concept, 'related': sorted(related)}) + '\n')
    print(f'Concept index written to {out_path}')


if __name__ == '__main__':
    main()
