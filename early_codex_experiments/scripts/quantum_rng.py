import os
import random
from pathlib import Path
from typing import Optional

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy may be unavailable
    np = None


def _get_seed() -> int:
    """Return an integer seed from ``$QUANTUM_SEED`` or ``/tmp/quantum_seed``.

    Raises
    ------
    RuntimeError
        If neither source is available or the value cannot be parsed as an
        integer.
    """
    val: Optional[str] = os.environ.get("QUANTUM_SEED")
    if val is None:
        seed_file = Path("/tmp/quantum_seed")
        if seed_file.exists():
            val = seed_file.read_text().strip()
    if val is not None:
        try:
            return int(val)
        except ValueError:
            raise RuntimeError("invalid quantum seed value")
    raise RuntimeError("quantum seed not found")


def seed_random() -> int:
    """Seed Python and NumPy RNGs using the quantum seed.

    Returns the seed used. Raises ``RuntimeError`` if no quantum seed is
    available.
    """
    seed = _get_seed()
    random.seed(seed)
    if np is not None:
        try:
            np.random.seed(seed)
        except Exception:
            pass
    return seed

