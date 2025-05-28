import argparse
import json
from pathlib import Path

from token_summary import parse_ledger


def total_supply(tokens: list[dict]) -> int:
    total = 0
    for t in tokens:
        raw = t.get('supply', '').replace(',', '')
        try:
            total += int(raw)
        except ValueError:
            continue
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description='Aggregate token supply')
    parser.add_argument('--path', default='token_and_jpeg_info', help='ledger file path')
    args = parser.parse_args()
    tokens = parse_ledger(args.path)
    supply = total_supply(tokens)
    result = {
        'tokens': len(tokens),
        'total_supply': supply,
    }
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
