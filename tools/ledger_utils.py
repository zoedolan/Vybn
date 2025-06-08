from __future__ import annotations

import json
import re
from typing import Dict, List

LEDGER_PATTERN = re.compile(r'^([A-Z0-9]+):\s*(.*?)\s*/\s*([^@]+)@\s*([^\s]+)\s*(.+)')


def parse_ledger(path: str) -> List[Dict[str, str]]:
    """Return a list of token records from ``path``."""
    tokens: List[Dict[str, str]] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            match = LEDGER_PATTERN.match(line.strip())
            if match:
                name, supply, price, lock, address = match.groups()
                tokens.append(
                    {
                        'name': name,
                        'supply': supply.strip(),
                        'price': price.strip(),
                        'lock': lock.strip(),
                        'address': address.strip(),
                    }
                )
    return tokens


def ledger_to_markdown(tokens: List[Dict[str, str]]) -> str:
    """Return a Markdown table representing ``tokens``."""
    lines = ["| Token | Supply | Price | Lock | Address |", "|---|---|---|---|---|"]
    for t in tokens:
        line = f"| {t['name']} | {t['supply']} | {t['price']} | {t['lock']} | {t['address']} |"
        lines.append(line)
    return "\n".join(lines)


def total_supply(tokens: List[Dict[str, str]]) -> int:
    """Return the integer sum of the supply fields."""
    total = 0
    for t in tokens:
        raw = t.get('supply', '').replace(',', '')
        try:
            total += int(raw)
        except ValueError:
            continue
    return total
