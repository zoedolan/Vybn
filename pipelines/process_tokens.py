from __future__ import annotations

from pathlib import Path

from vybn.quantum_seed import seed_rng

TOKEN_FILE_NAME = 'token_and_jpeg_info'


def summarize_token_file(path: Path) -> str:
    """Return a simple word count summary for ``path``."""
    try:
        text = path.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return ''
    words = text.split()
    return f'{path.name}: {len(words)} words'


def main() -> None:
    seed_rng()
    repo_root = Path(__file__).resolve().parents[1]
    summary = summarize_token_file(repo_root / TOKEN_FILE_NAME)
    out_path = repo_root / 'token_summary.txt'
    out_path.write_text(summary, encoding='utf-8')
    print(f'Token summary written to {out_path}')


if __name__ == '__main__':
    main()
