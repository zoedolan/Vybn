import os
import random
import pathlib
try:
    import numpy as np
except Exception:  # pragma: no cover - numpy may be unavailable
    np = None

def seed_rng() -> int:
    """Seed Python and NumPy RNGs from $QUANTUM_SEED or /tmp/quantum_seed."""
    seed_val = os.getenv("QUANTUM_SEED")
    if seed_val is None:
        seed_file = pathlib.Path("/tmp/quantum_seed")
        if seed_file.exists():
            seed_val = seed_file.read_text().strip()
    if seed_val is None:
        raise RuntimeError("quantum seed not found")
    seed = int(seed_val)
    random.seed(seed)
    if np is not None:
        try:
            np.random.seed(seed)
        except Exception:
            pass
    return seed
