from __future__ import annotations
import argparse
import json
from datetime import datetime
from pathlib import Path

DEFAULT_JOURNAL = Path(__file__).resolve().parents[3] / 'co_emergence_journal.jsonl'


def load_spikes(path: str | Path = DEFAULT_JOURNAL) -> list[datetime]:
    """Return a list of spike timestamps from ``path``."""
    path = Path(path)
    times = []
    if not path.exists():
        return times
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            if 'message' in entry:
                times.append(datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00')))
    return times


def average_interval(times: list[datetime]) -> float | None:
    """Return average seconds between spikes or ``None`` if <2."""
    if len(times) < 2:
        return None
    deltas = [
        (t2 - t1).total_seconds()
        for t1, t2 in zip(times, times[1:])
    ]
    return sum(deltas) / len(deltas)


def main() -> None:
    parser = argparse.ArgumentParser(description='Compute average interval between spikes')
    parser.add_argument('--journal', default=str(DEFAULT_JOURNAL), help='path to journal file')
    args = parser.parse_args()
    times = load_spikes(args.journal)
    avg = average_interval(times)
    if avg is None:
        print(json.dumps({'entries': len(times), 'message': 'not enough spikes'}))
    else:
        print(json.dumps({'entries': len(times), 'avg_interval': avg}, indent=2))


if __name__ == '__main__':
    main()
