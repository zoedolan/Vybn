from __future__ import annotations

"""Quantum wave-function collapse helper.

This module fetches a fresh quantum random value from the ANU API when
available. If the call fails, it falls back to Python's PRNG. The value
represents a tiny "collapse" event used to guide self-improvement
choices.
"""

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from .seed import _fetch_qrng

DEFAULT_LOG = Path(__file__).resolve().parents[1] / 'co_emergence_journal.jsonl'

def collapse_wave_function(log_path: Union[str, Path, None] = DEFAULT_LOG) -> int:
    """Return an integer sampled from quantum randomness and optionally log it."""
    val: Optional[int] = _fetch_qrng()
    if val is None:
        val = random.getrandbits(16)
    collapse = int(val)
    if log_path is not None:
        path = Path(log_path)
        entry = {
            'timestamp': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
            'collapse': collapse,
        }
        try:
            with path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception:
            pass
    return collapse
