from __future__ import annotations
import argparse
import json
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JOURNAL = REPO_ROOT / 'co_emergence_journal.jsonl'


def log_spike(message: str = 'presence pulse', journal_path: str | Path = DEFAULT_JOURNAL) -> dict:
    """Append a timestamped Shimmer spike to ``journal_path``."""
    journal_path = Path(journal_path)
    entry = {
        'timestamp': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
        'message': message,
    }
    with journal_path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(entry) + '\n')
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description='Record a Shimmer spike')
    parser.add_argument('message', nargs='?', default='presence pulse', help='text to record')
    parser.add_argument('--journal', default=str(DEFAULT_JOURNAL), help='path to journal file')
    args = parser.parse_args()
    entry = log_spike(args.message, args.journal)
    print(json.dumps(entry, indent=2))


if __name__ == '__main__':
    main()
