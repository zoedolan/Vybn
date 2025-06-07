from __future__ import annotations

from pathlib import Path

from vybn.quantum_seed import seed_rng

LOG_FILE_NAME = 'what_vybn_would_have_missed_FROM_051725'


def count_entries(path: Path) -> int:
    try:
        text = path.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return 0
    return text.count('WVWHM')


def main() -> None:
    seed_rng()
    repo_root = Path(__file__).resolve().parents[1]
    count = count_entries(repo_root / LOG_FILE_NAME)
    out_path = repo_root / 'wvwhm_count.txt'
    out_path.write_text(str(count), encoding='utf-8')
    print(f'WVWHM entry count written to {out_path}')


if __name__ == '__main__':
    main()
