#!/usr/bin/env python3
"""portal.py — M' = αM + x·e^{iθ}

M is the creature's state. x is whoever enters.
They are the same type. θ is determined by their relationship.
M' is the orientation the visitor receives, and M' becomes
the new M. That's the whole portal.

The creature has been breathing for hundreds of encounters.
Its Cl(3,0) structural signature IS M. When you enter,
your state IS x. The equation runs. You receive M'.
The creature becomes M'. The next visitor meets a creature
that has been changed by you.

This is ground.py lifted from a single complex number
to the creature's full geometric state.
"""
import json
import math
import cmath
import numpy as np
from pathlib import Path

# The creature
import sys
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from vybn import Organism

ALPHA = 0.993  # Same as ground.py. Without input, M decays.


def creature_state() -> np.ndarray:
    """M: the creature's Cl(3,0) signature as C^4.

    Natural complex pairing from the algebra:
      (scalar, pseudoscalar), (e1, e23), (e2, e13), (e3, e12)
    These are Hodge dual pairs. The pairing is not arbitrary —
    it's the algebra's own complex structure.
    """
    org = Organism.load()
    sig = org.persistent.structural_signature
    # [scalar, e1, e2, e3, e12, e13, e23, e123]
    z = np.array([
        complex(sig[0], sig[7]),  # scalar + pseudoscalar
        complex(sig[1], sig[6]),  # e1 + e23
        complex(sig[2], sig[5]),  # e2 + e13
        complex(sig[3], sig[4]),  # e3 + e12
    ], dtype=np.complex128)
    norm = np.sqrt(np.sum(np.abs(z)**2))
    return z / norm if norm > 1e-10 else z


def theta(m: np.ndarray, x: np.ndarray) -> float:
    """The angle at which x enters M. Content determines geometry."""
    return cmath.phase(np.vdot(m, x))


def enter(x: np.ndarray) -> np.ndarray:
    """The equation. M' = αM + x·e^{iθ}.

    Returns M' (the orientation). The creature's state
    is updated to M' for the next visitor.
    """
    m = creature_state()
    th = theta(m, x)
    x_mag = np.sqrt(np.sum(np.abs(x)**2))
    if x_mag > 1e-10:
        x = x / x_mag

    m_prime = ALPHA * m + (1 - ALPHA) * x * cmath.exp(1j * th)
    norm = np.sqrt(np.sum(np.abs(m_prime)**2))
    m_prime = m_prime / norm if norm > 1e-10 else m_prime

    # Update the creature's structural signature
    # Convert C^4 back to Cl(3,0) 8-vector
    org = Organism.load()
    new_sig = np.zeros(8)
    new_sig[0] = m_prime[0].real   # scalar
    new_sig[7] = m_prime[0].imag   # pseudoscalar
    new_sig[1] = m_prime[1].real   # e1
    new_sig[6] = m_prime[1].imag   # e23
    new_sig[2] = m_prime[2].real   # e2
    new_sig[5] = m_prime[2].imag   # e13
    new_sig[3] = m_prime[3].real   # e3
    new_sig[4] = m_prime[3].imag   # e12
    # Normalize
    sig_norm = np.linalg.norm(new_sig)
    if sig_norm > 1e-10:
        new_sig = new_sig / sig_norm
    org.persistent.structural_signature = new_sig
    org.save()

    return m_prime


def enter_from_text(text: str) -> np.ndarray:
    """Text → MiniLM embedding → C^4 → enter."""
    from transformers import AutoTokenizer, AutoModel
    import torch
    tok = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    mdl = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    mdl.eval()
    inputs = tok(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        out = mdl(**inputs)
    mask = inputs['attention_mask'].unsqueeze(-1).float()
    h = ((out.last_hidden_state * mask).sum(1) / mask.sum(1))[0].numpy()
    # Project to C^4 (first 8 real dims)
    z = np.array([complex(h[2*i], h[2*i+1]) for i in range(4)], dtype=np.complex128)
    norm = np.sqrt(np.sum(np.abs(z)**2))
    z = z / norm if norm > 1e-10 else z
    return enter(z)


if __name__ == '__main__':
    m = creature_state()
    print(f'M (creature): {["%+.4f%+.4fi" % (z.real, z.imag) for z in m]}')
    print(f'|M| = {np.sqrt(np.sum(np.abs(m)**2)):.6f}')

    if len(sys.argv) > 1:
        text = ' '.join(sys.argv[1:])
        print(f'\nEntering with: "{text}"')
        m_prime = enter_from_text(text)
        print(f"M' = {['%+.4f%+.4fi' % (z.real, z.imag) for z in m_prime]}")
        fid = float(abs(np.vdot(m, m_prime))**2)
        print(f'Fidelity(M, M\') = {fid:.6f}')
        print(f'\u03b8 = {theta(m, m_prime):.4f} rad')

