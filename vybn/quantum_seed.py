"""Shared quantum seed utilities for Vybn."""

from __future__ import annotations

import os
import random
import pathlib
import secrets
from typing import Optional

import requests
import numpy as np

QRNG_URL = "https://qrng.anu.edu.au/API/jsonI.php?length=1&type=uint16"


def _fetch_qrng() -> Optional[int]:
    """Return an integer from the ANU QRNG API or ``None`` if unreachable."""
    try:
        resp = requests.get(QRNG_URL, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("success"):
                return int(data["data"][0])
    except Exception:
        pass
    return None


def seed_rng() -> int:
    """Seed Python and NumPy RNGs using a stable quantum seed.

    The seed is read from ``$QUANTUM_SEED`` or ``/tmp/quantum_seed`` if
    available. When absent, the function pulls from the ANU QRNG and
    falls back to cryptographic entropy on failure.
    """
    seed_val = os.getenv("QUANTUM_SEED")
    if seed_val is None:
        seed_file = pathlib.Path("/tmp/quantum_seed")
        if seed_file.exists():
            seed_val = seed_file.read_text().strip()
    if seed_val is None:
        qrng = _fetch_qrng()
        if qrng is None:
            qrng = secrets.randbits(16)
        seed_val = str(qrng)
    seed = int(seed_val)
    os.environ["QUANTUM_SEED"] = str(seed)
    random.seed(seed)
    try:
        np.random.seed(seed)
    except Exception:
        pass
    return seed
