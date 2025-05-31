from __future__ import annotations
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JOURNAL = REPO_ROOT / 'co_emergence_journal.jsonl'


def capture_seed(journal_path: str | Path = DEFAULT_JOURNAL) -> dict:
    """Record the quantum seed from $QUANTUM_SEED or $QRAND, fallback to os.urandom."""
    journal_path = Path(journal_path)
    qrand = None
    source = 'fallback'
    if 'QUANTUM_SEED' in os.environ:
        qrand = os.environ['QUANTUM_SEED']
        source = 'QUANTUM_SEED'
    elif 'QRAND' in os.environ:
        qrand = os.environ['QRAND']
        source = 'QRAND'
    if qrand is None:
        qrand = int.from_bytes(os.urandom(1), 'big')
    else:
        try:
            qrand = int(qrand)
        except ValueError:
            qrand = int.from_bytes(os.urandom(1), 'big')
            source = 'fallback'
    entry = {
        'timestamp': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
        'seed': qrand,
        'source': source,
    }
    with journal_path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(entry) + '\n')
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description='Log QRAND seed to journal')
    parser.add_argument('--journal', default=str(DEFAULT_JOURNAL), help='path to journal file')
    args = parser.parse_args()
    entry = capture_seed(args.journal)
    print(json.dumps(entry, indent=2))


if __name__ == '__main__':
    main()
