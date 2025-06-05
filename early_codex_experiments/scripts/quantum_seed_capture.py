from __future__ import annotations
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from vybn.quantum_seed import seed_rng

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JOURNAL = REPO_ROOT / 'co_emergence_journal.jsonl'


def capture_seed(journal_path: str | Path = DEFAULT_JOURNAL) -> dict:
    """Record the quantum seed used for the run."""
    journal_path = Path(journal_path)
    env_seed = os.environ.get('QUANTUM_SEED')
    file_seed = Path('/tmp/quantum_seed') if env_seed is None else None
    source = 'generated'
    if env_seed is not None:
        source = 'QUANTUM_SEED'
    elif file_seed is not None and file_seed.exists():
        source = '/tmp/quantum_seed'
    qrand = seed_rng()
    entry = {
        'timestamp': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
        'seed': int(qrand),
        'source': source,
    }
    with journal_path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(entry) + '\n')
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description='Log quantum seed to journal')
    parser.add_argument('--journal', default=str(DEFAULT_JOURNAL), help='path to journal file')
    args = parser.parse_args()
    entry = capture_seed(args.journal)
    print(json.dumps(entry, indent=2))


if __name__ == '__main__':
    main()
