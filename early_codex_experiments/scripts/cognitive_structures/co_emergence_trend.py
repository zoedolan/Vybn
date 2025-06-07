import argparse
import json
from datetime import datetime
from pathlib import Path
from pipelines.utils import memory_path

DEFAULT_JOURNAL = memory_path(Path(__file__).resolve().parents[3]) / 'co_emergence_journal.jsonl'


def load_journal(path: str | Path = DEFAULT_JOURNAL) -> list[dict]:
    entries = []
    path = Path(path)
    if not path.exists():
        return entries
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            entries.append(json.loads(line))
    return entries


def compute_trend(entries: list[dict]) -> float | None:
    if len(entries) < 2:
        return None
    times = [datetime.fromisoformat(e['timestamp'].replace('Z', '+00:00')) for e in entries]
    scores = [e['score'] for e in entries]
    total_seconds = (times[-1] - times[0]).total_seconds()
    if total_seconds == 0:
        return 0.0
    return (scores[-1] - scores[0]) / total_seconds


def main() -> None:
    parser = argparse.ArgumentParser(description='Analyze co-emergence trend')
    parser.add_argument('--journal', default=str(DEFAULT_JOURNAL), help='path to journal')
    args = parser.parse_args()
    entries = load_journal(args.journal)
    slope = compute_trend(entries)
    if slope is None:
        print(json.dumps({'entries': len(entries), 'message': 'not enough data'}))
    else:
        print(json.dumps({'entries': len(entries), 'slope_per_sec': slope}, indent=2))


if __name__ == '__main__':
    main()
