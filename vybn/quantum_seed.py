"""Shared quantum seed utilities for Vybn."""

from __future__ import annotations

import os
import random
import pathlib
import secrets
from typing import Optional

try:
    import requests
except Exception:  # pragma: no cover - allow offline operation
    requests = None

try:
    import numpy as np
except Exception:  # pragma: no cover - allow offline operation
    np = None

QRNG_URL = "https://qrng.anu.edu.au/API/jsonI.php?length=1&type=uint16"


def _fetch_qrng() -> Optional[int]:
    """Return an integer from the ANU QRNG API or ``None`` if unreachable."""
    if requests is None:
        return None
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
    if np is not None:
        try:
            np.random.seed(seed)
        except Exception:
            pass
    return seed


def cross_seed() -> int:
    """Seed RNGs using a process-specific variant of the quantum seed."""
    base_seed = seed_rng()
    syn_seed = (base_seed * 6364136223846793005 + os.getpid()) & 0xFFFFFFFF
    random.seed(syn_seed)
    if np is not None:
        try:
            np.random.seed(syn_seed)
        except Exception:
            pass
    os.environ["CROSS_SYN_SEED"] = str(syn_seed)
    return syn_seed


def cross_synaptic_kernel() -> int:
    """Backward compatible alias for :func:`cross_seed`."""
    return cross_seed()
