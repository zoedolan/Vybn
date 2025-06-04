from __future__ import annotations
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JOURNAL = REPO_ROOT / 'co_emergence_journal.jsonl'


def capture_seed(journal_path: str | Path = DEFAULT_JOURNAL) -> dict:
    """Record the quantum seed used for the run.

    The value is pulled from ``$QUANTUM_SEED`` if present. If that is unset,
    the function attempts to read ``/tmp/quantum_seed``—mirroring how the
    repository's bootstrap script exposes the random seed.  If neither is
    available or the value cannot be parsed, ``RuntimeError`` is raised.
    """
    journal_path = Path(journal_path)
    qrand = None
    source = 'fallback'
    if 'QUANTUM_SEED' in os.environ:
        qrand = os.environ['QUANTUM_SEED']
        source = 'QUANTUM_SEED'
    else:
        seed_file = Path('/tmp/quantum_seed')
        if seed_file.exists():
            qrand = seed_file.read_text().strip()
            source = '/tmp/quantum_seed'

    if qrand is None:
        raise RuntimeError('quantum seed not found')
    try:
        qrand = int(qrand)
    except ValueError:
        raise RuntimeError('invalid quantum seed value')
    entry = {
        'timestamp': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
        'seed': qrand,
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
