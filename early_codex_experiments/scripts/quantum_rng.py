import os
import random
from typing import Optional

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy may be unavailable
    np = None


def _get_seed() -> int:
    """Return an integer seed from the environment or os.urandom."""
    val: Optional[str] = os.environ.get("QUANTUM_SEED")
    if val is None:
        val = os.environ.get("QRAND")
    if val is not None:
        try:
            return int(val)
        except ValueError:
            pass
    return int.from_bytes(os.urandom(2), "big")


def seed_random() -> int:
    """Seed Python and NumPy RNGs using `$QUANTUM_SEED` or `$QRAND`.

    Returns the seed used, falling back to OS randomness when the env vars are
    unset or invalid.
    """
    seed = _get_seed()
    random.seed(seed)
    if np is not None:
        try:
            np.random.seed(seed)
        except Exception:
            pass
    return seed

