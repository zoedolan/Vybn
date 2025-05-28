import argparse

try:
    # When used as a module within the `scripts` package
    from .token_summary import parse_ledger
except ImportError:  # pragma: no cover - fallback for direct execution
    # Support running as a standalone script
    from token_summary import parse_ledger


def ledger_to_markdown(tokens: list[dict]) -> str:
    """Return a Markdown table for the ledger tokens."""
    lines = [
        "| Token | Supply | Price | Lock | Address |",
        "|---|---|---|---|---|",
    ]
    for t in tokens:
        line = f"| {t['name']} | {t['supply']} | {t['price']} | {t['lock']} | {t['address']} |"
        lines.append(line)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render token ledger as Markdown table")
    parser.add_argument('--path', default='token_and_jpeg_info', help='ledger file path')
    args = parser.parse_args()
    tokens = parse_ledger(args.path)
    table = ledger_to_markdown(tokens)
    print(table)


if __name__ == '__main__':
    main()
