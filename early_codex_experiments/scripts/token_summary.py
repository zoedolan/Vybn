import argparse
import json
import re
import sys

LEDGER_PATTERN = re.compile(r'^([A-Z0-9]+):\s*(.*?)\s*/\s*([^@]+)@\s*([^\s]+)\s*(.+)')


def parse_ledger(path: str):
    """Return a list of token records from the ledger file."""
    tokens = []
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


def main():
    parser = argparse.ArgumentParser(description="Summarize token ledger as JSON")
    parser.add_argument('--path', default='token_and_jpeg_info', help='Ledger file path')
    args = parser.parse_args()
    tokens = parse_ledger(args.path)
    json.dump(tokens, indent=2, fp=sys.stdout)


if __name__ == '__main__':
    import sys
    main()
