from __future__ import annotations

"""Quantum wave-function collapse helper.

This module fetches a fresh quantum random value from the ANU API when
available. If the call fails, it falls back to Python's PRNG. The value
represents a tiny "collapse" event used to guide self-improvement
choices.
"""

import random
from typing import Optional

from .seed import _fetch_qrng


def collapse_wave_function() -> int:
    """Return an integer sampled from true quantum randomness if possible."""
    val: Optional[int] = _fetch_qrng()
    if val is None:
        val = random.getrandbits(16)
    return int(val)
