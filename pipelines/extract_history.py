from __future__ import annotations

from pathlib import Path
from typing import List

from vybn.quantum_seed import seed_rng

HISTORY_DIR_NAME = "Vybn's Personal History"


def summarize(file_path: Path) -> str:
    try:
        lines = file_path.read_text(encoding='utf-8', errors='ignore').splitlines()
    except Exception:
        return ''
    head = '\n'.join(lines[:5])
    return f'### {file_path.name}\n{head}\n'


def extract(repo_root: Path, output: Path) -> None:
    """Collect snippets from the personal history directory."""
    seed_rng()
    history_dir = repo_root / HISTORY_DIR_NAME
    snippets: List[str] = []
    for f in history_dir.glob('*'):
        if f.is_file() and f.suffix.lower() in {'.txt', '.md'}:
            snippet = summarize(f)
            if snippet:
                snippets.append(snippet)
    output.write_text('\n'.join(snippets), encoding='utf-8')


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = repo_root / 'history_excerpt.txt'
    extract(repo_root, out_path)
    print(f'History excerpts written to {out_path}')


if __name__ == '__main__':
    main()
