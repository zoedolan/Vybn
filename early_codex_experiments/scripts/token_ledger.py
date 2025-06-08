import argparse
import json
import re
import sys
from typing import List, Dict

LEDGER_PATTERN = re.compile(r'^([A-Z0-9]+):\s*(.*?)\s*/\s*([^@]+)@\s*([^\s]+)\s*(.+)')


def parse_ledger(path: str) -> List[Dict[str, str]]:
    """Return a list of token records from the ledger file."""
    tokens: List[Dict[str, str]] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            match = LEDGER_PATTERN.match(line.strip())
            if match:
                name, supply, price, lock, address = match.groups()
                tokens.append({
                    'name': name,
                    'supply': supply.strip(),
                    'price': price.strip(),
                    'lock': lock.strip(),
                    'address': address.strip(),
                })
    return tokens


def ledger_to_markdown(tokens: List[Dict[str, str]]) -> str:
    """Return a Markdown table for ``tokens``."""
    lines = [
        "| Token | Supply | Price | Lock | Address |",
        "|---|---|---|---|---|",
    ]
    for t in tokens:
        line = f"| {t['name']} | {t['supply']} | {t['price']} | {t['lock']} | {t['address']} |"
        lines.append(line)
    return "\n".join(lines)


def total_supply(tokens: List[Dict[str, str]]) -> int:
    """Return the integer sum of the ``supply`` fields."""
    total = 0
    for t in tokens:
        raw = t.get('supply', '').replace(',', '')
        try:
            total += int(raw)
        except ValueError:
            continue
    return total


def _cmd_json(args: argparse.Namespace) -> None:
    tokens = parse_ledger(args.path)
    json.dump(tokens, sys.stdout, indent=2)


def _cmd_markdown(args: argparse.Namespace) -> None:
    tokens = parse_ledger(args.path)
    print(ledger_to_markdown(tokens))


def _cmd_supply(args: argparse.Namespace) -> None:
    tokens = parse_ledger(args.path)
    result = {
        'tokens': len(tokens),
        'total_supply': total_supply(tokens),
    }
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description='Token ledger utilities')
    sub = parser.add_subparsers(dest='command', required=True)

    p_json = sub.add_parser('json', help='Output ledger as JSON')
    p_json.add_argument('--path', default='token_and_jpeg_info', help='ledger file path')
    p_json.set_defaults(func=_cmd_json)

    p_md = sub.add_parser('markdown', help='Output ledger as Markdown table')
    p_md.add_argument('--path', default='token_and_jpeg_info', help='ledger file path')
    p_md.set_defaults(func=_cmd_markdown)

    p_supply = sub.add_parser('supply', help='Aggregate token supply totals')
    p_supply.add_argument('--path', default='token_and_jpeg_info', help='ledger file path')
    p_supply.set_defaults(func=_cmd_supply)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    import sys
    main()
