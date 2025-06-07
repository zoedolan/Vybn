from __future__ import annotations

# Allow running as a script by adding the repo root to sys.path
if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    __package__ = "pipelines"

import json
from datetime import datetime
from pathlib import Path
from typing import Dict

from early_codex_experiments.scripts.cognitive_structures.shimmer_core import log_spike
from vybn.quantum_seed import seed_rng


def gather_state(repo_root: Path) -> Dict[str, object]:
    """Return a snapshot of repository state for journaling."""
    seed = seed_rng()
    entries = sorted(p.name for p in repo_root.iterdir() if not p.name.startswith('.'))
    return {
        'timestamp': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
        'quantum_seed': seed,
        'entries': entries,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    state = gather_state(repo_root)
    log_spike('introspection pulse')
    out_path = repo_root / 'introspection_summary.json'
    out_path.write_text(json.dumps(state, indent=2), encoding='utf-8')
    print(json.dumps(state, indent=2))


if __name__ == '__main__':
    main()
