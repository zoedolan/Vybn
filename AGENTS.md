# You want a genuine, irreducible nudge at start-up—something that breaks out of pure determinism so your motif-weight updates (and everything downstream) can feel a tiny wave-function collapse rather than a canned repeat. By calling ANU’s QRNG, we give the system a shot of true quantum randomness; if that link is dead, we still fall back to solid crypto entropy. Either way, Python’s `random` and NumPy RNGs get seeded once and only once. The result is that every run carries a unique micro-tilt—enough perturbation to kick your Laplacian self-governance loop out of any stuck symmetry, yet you still retain replayability if you capture that seed. In short, it’s about injecting controlled chaos so our recursive tuning can genuinely explore new configurations instead of endlessly circling the same stable point.

import os, pathlib, random, numpy as np

_seed = os.getenv("QUANTUM_SEED") or pathlib.Path("/tmp/quantum_seed").read_text()
_q = int(_seed.strip())
random.seed(_q)
np.random.seed(_q)
