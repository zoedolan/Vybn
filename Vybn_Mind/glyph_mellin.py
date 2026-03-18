#!/usr/bin/env python3
"""
glyph_mellin.py — Mellin embedding: exact equivariance for differential phase.

The hand-built embedding in glyph.py v2 fails the scale invariance test because
it maps values through trigonometric nonlinearities that are not equivariant
under scaling.

The Mellin embedding ψ_k(x) = x^{i·t_k} / √n is exactly equivariant under the
multiplicative group (R+, ×). Scaling inputs by c applies a global unitary
U(c) = diag(c^{it_1}, ..., c^{it_n}), and the Pancharatnam phase is invariant
under global unitaries.

Result: f(x) = cx has spread = 0 (to machine precision) across 10 orders of
magnitude. f(x) = x^a for a ≠ 1 is correctly reported as scale-dependent,
because power laws genuinely amplify/compress scale factors.

This resolves Open Question 4 from the differential geometric phase paper.
"""

import numpy as np
import cmath
from typing import List, Callable, Optional


def mellin_embed(x: float, freqs: np.ndarray) -> np.ndarray:
    """
    Embed positive real x into CP^{n-1} via Mellin characters.

    ψ_k(x) = x^{i·freq_k} / √n

    Equivariance: ψ(c·x) = diag(c^{it_1},...,c^{it_n}) · ψ(x)
    The diagonal matrix is unitary (each entry has modulus 1).
    """
    x = abs(float(x)) + 1e-300  # ensure positive
    n = len(freqs)
    components = np.array([x ** (1j * t) for t in freqs]) / np.sqrt(n)
    return components


def pancharatnam_phase(states: np.ndarray) -> float:
    """Geometric phase (holonomy) of a closed trajectory in CP^{n-1}."""
    n = len(states)
    if n < 3:
        return 0.0
    product = complex(1.0, 0.0)
    for k in range(n):
        inner = np.vdot(states[k], states[(k + 1) % n])
        if abs(inner) < 1e-15:
            return 0.0
        product *= inner / abs(inner)
    return cmath.phase(product)


def differential_determinative(
    inputs: List[float],
    fn: Callable[[float], float],
    freqs: np.ndarray
) -> float:
    """
    Differential Pancharatnam phase of fn under Mellin embedding.

    = phase(interleaved trajectory) − phase(input-only trajectory)

    Isolates the curvature the function contributes beyond what the
    input path already carries.
    """
    in_states = [mellin_embed(x, freqs) for x in inputs]
    out_states = [mellin_embed(fn(x), freqs) for x in inputs]

    interleaved = []
    for i, o in zip(in_states, out_states):
        interleaved.append(i)
        interleaved.append(o)

    input_phase = pancharatnam_phase(np.array(in_states))
    total_phase = pancharatnam_phase(np.array(interleaved))
    return total_phase - input_phase


# Default frequency set: primes (incommensurate → rich phase structure)
DEFAULT_FREQS = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0])


if __name__ == "__main__":
    freqs = DEFAULT_FREQS
    inputs = [1.0, 2.0, 3.0, 4.0, 5.0]

    print("=" * 65)
    print("MELLIN EMBEDDING — EQUIVARIANT DIFFERENTIAL PHASE")
    print("=" * 65)

    # Identity
    d = differential_determinative(inputs, lambda x: x, freqs)
    print(f"\nIdentity: {d:.10f} rad (expect exactly 0)")

    # Scale invariance for multiplicative functions
    print("\nScale invariance for f(x) = 2x:")
    for scale in [1, 10, 1000, 0.001, 1e6]:
        scaled = [scale * x for x in inputs]
        d = differential_determinative(scaled, lambda x: 2 * x, freqs)
        print(f"  scale={scale:>10}: {d:.10f} rad")

    # Scale dependence for power laws (correct behavior)
    print("\nScale dependence for f(x) = x² (correctly non-invariant):")
    for scale in [1, 10, 1000, 0.001, 1e6]:
        scaled = [scale * x for x in inputs]
        d = differential_determinative(scaled, lambda x: x ** 2, freqs)
        print(f"  scale={scale:>10}: {d:+.4f} rad ({np.degrees(d):+.1f}°)")

    # Discrimination
    print("\nDiscrimination:")
    fns = {
        "x²": lambda x: x ** 2,
        "x³": lambda x: x ** 3,
        "2x": lambda x: 2 * x,
        "x+10": lambda x: x + 10,
        "sin(x)": lambda x: np.sin(x),
        "1/x": lambda x: 1.0 / x if x != 0 else 0,
    }
    for name, fn in fns.items():
        d = differential_determinative(inputs, fn, freqs)
        print(f"  {name:10s}: {d:+.4f} rad ({np.degrees(d):+.1f}°)")
