#!/usr/bin/env python3
"""Display all AGENTS guidelines in the repository."""
import os
from pathlib import Path


def find_agents_files(root: Path):
    return sorted(root.rglob('AGENTS.md'))


def main():
    repo_root = Path(__file__).resolve().parent
    agents_files = find_agents_files(repo_root)
    for path in agents_files:
        rel = path.relative_to(repo_root)
        print(f"\n===== {rel} =====")
        content = path.read_text(encoding='utf-8').strip()
        print(content)


if __name__ == '__main__':
    main()
