from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from pipelines.utils import memory_path

from .vybn_recursive_emergence import compute_co_emergence_score

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GRAPH = REPO_ROOT / 'early_codex_experiments' / 'scripts' / 'self_assembly' / 'integrated_graph.json'
JOURNAL_PATH = memory_path(REPO_ROOT) / 'co_emergence_journal.jsonl'


def log_score(graph_path: str | Path = DEFAULT_GRAPH, journal_path: str | Path = JOURNAL_PATH) -> dict:
    """Append current co-emergence score to ``journal_path`` and return the entry."""
    graph_path = Path(graph_path)
    journal_path = Path(journal_path)
    score = compute_co_emergence_score(str(graph_path))
    entry = {
        'timestamp': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
        'score': round(score, 3),
    }
    with journal_path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(entry) + '\n')
    return entry


def main() -> None:
    entry = log_score()
    print(json.dumps(entry, indent=2))


if __name__ == '__main__':
    main()
