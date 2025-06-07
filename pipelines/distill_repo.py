from __future__ import annotations

from pathlib import Path
from typing import Iterable

from vybn.quantum_seed import seed_rng
from . import EXCLUDE_PATHS


def iter_text_files(root: Path) -> Iterable[Path]:
    """Yield markdown, text and python files under ``root`` excluding paths."""
    for path in root.rglob('*'):
        if any(path.is_relative_to(root / p) for p in EXCLUDE_PATHS):
            continue
        if path.is_file() and path.suffix.lower() in {'.md', '.txt', '.py'}:
            yield path


def distill(root: Path, output: Path) -> None:
    """Write a concatenated text snapshot of the repository to ``output``."""
    seed_rng()
    texts = []
    for f in iter_text_files(root):
        try:
            txt = f.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            continue
        texts.append(f'## {f}\n{txt}\n')
    output.write_text('\n'.join(texts), encoding='utf-8')


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = repo_root / 'distilled_corpus.txt'
    distill(repo_root, out_path)
    print(f'Distilled corpus written to {out_path}')


if __name__ == '__main__':
    main()
