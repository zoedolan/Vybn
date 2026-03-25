#!/usr/bin/env python3
"""
glyph.py v2 — Differential geometric phase.

The v1 determinative measured total trajectory curvature: inputs AND outputs
mixed together. The identity function got 80° because the input path itself
has curvature in CP^{n-1}. That's the embedding talking, not the computation.

v2 fixes this by measuring DIFFERENTIAL curvature:

    determinative = phase(input→output trajectory) − phase(input-only trajectory)

The input-only phase is what any function would accumulate just from being
fed those inputs. The differential strips it out. What remains is the
curvature the FUNCTION added — the geometric residue of the transformation
itself, not of the data.

Properties this should have:
  - Identity function → 0 (adds no curvature)
  - Constant function → nonzero (collapses all inputs to one point — that IS
    a transformation, it destroys information, it should register)
  - Commuting functions → same determinative (same transformation, different syntax)
  - Scale invariant (geometric, not metric)
  - Path-dependent for genuinely different computations
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, Any, List, Optional
import cmath


@dataclass
class StateSnapshot:
    """A point on the trajectory through representation space."""
    r_t: float
    vector: np.ndarray

    @property
    def normalized(self) -> np.ndarray:
        norm = np.linalg.norm(self.vector)
        if norm < 1e-15:
            return self.vector
        return self.vector / norm


def _pancharatnam_phase(states: np.ndarray) -> float:
    """
    Pancharatnam geometric phase around a closed trajectory in CP^{n-1}.

    For states ψ_0, ψ_1, ..., ψ_{N-1}:
      phase = arg(⟨ψ_0|ψ_1⟩ ⟨ψ_1|ψ_2⟩ ⋯ ⟨ψ_{N-1}|ψ_0⟩)

    This is the holonomy of the natural connection on CP^{n-1}.
    """
    n = len(states)
    if n < 3:
        return 0.0

    product = complex(1.0, 0.0)
    for k in range(n):
        psi_k = states[k]
        psi_next = states[(k + 1) % n]
        inner = np.vdot(psi_k, psi_next)
        if abs(inner) < 1e-15:
            return 0.0
        product *= inner / abs(inner)

    return cmath.phase(product)


class Glyph:
    """
    A computation that measures its own geometric contribution.

    The determinative is the DIFFERENTIAL Pancharatnam phase:
    the curvature the function adds beyond what the input path
    already carries. This separates what the function DOES from
    what the data IS.
    """

    def __init__(self, phonogram: Callable, name: str = "unnamed",
                 n_dims: int = 8):
        self.phonogram = phonogram
        self.name = name
        self.n_dims = n_dims
        self._input_states: List[np.ndarray] = []
        self._output_states: List[np.ndarray] = []
        self._interleaved_states: List[np.ndarray] = []
        self._r_t = 0.0
        self._call_count = 0
        self._determinative: Optional[float] = None

    def _embed(self, value: Any) -> np.ndarray:
        """
        Embed a value into C^n.

        Same nonlinear phase-entangling embedding as v1.
        The point is not that this embedding is canonical — it isn't.
        The point is that the DIFFERENTIAL phase cancels out the
        embedding's contribution to the input path.
        """
        if isinstance(value, (int, float, complex)):
            x = float(np.real(complex(value)))
            components = np.zeros(self.n_dims, dtype=complex)
            for k in range(self.n_dims):
                phase = (x * (k + 1) * 0.7 +
                         np.sin(x * (k + 0.5)) * 1.3 +
                         np.cos(x * x * 0.1 * (k + 1)) * 0.9)
                amp = 1.0 + 0.5 * np.sin(x * 0.3 + k * 1.1)
                components[k] = amp * np.exp(1j * phase)
            return components / np.linalg.norm(components)
        elif isinstance(value, np.ndarray):
            v = np.zeros(self.n_dims, dtype=complex)
            n = min(len(value), self.n_dims)
            v[:n] = value[:n].astype(complex)
            norm = np.linalg.norm(v)
            return v / norm if norm > 1e-15 else v
        elif isinstance(value, str):
            raw = [ord(c) for c in value[:32]]
            x = sum(r * (i + 1) for i, r in enumerate(raw)) / max(len(raw), 1)
            return self._embed(x)
        else:
            return self._embed(float(hash(value) % 10000) / 100.0)

    def __call__(self, *args, **kwargs) -> Any:
        """Execute the phonogram. Record input and output states separately."""
        # Embed the input
        if args:
            input_state = self._embed(args[0])
        else:
            input_state = self._embed(self._call_count)

        self._input_states.append(input_state)

        # Execute
        result = self.phonogram(*args, **kwargs)
        self._r_t += 1.0
        self._call_count += 1

        # Embed the output
        output_state = self._embed(result)
        self._output_states.append(output_state)

        # Interleaved trajectory: input_0, output_0, input_1, output_1, ...
        self._interleaved_states.append(input_state)
        self._interleaved_states.append(output_state)

        return result

    @property
    def input_phase(self) -> float:
        """Holonomy of the input-only path. The curvature the DATA carries."""
        if len(self._input_states) < 3:
            return 0.0
        states = np.array(self._input_states)
        return _pancharatnam_phase(states)

    @property
    def output_phase(self) -> float:
        """Holonomy of the output-only path. The curvature of the RESULTS."""
        if len(self._output_states) < 3:
            return 0.0
        states = np.array(self._output_states)
        return _pancharatnam_phase(states)

    @property
    def total_phase(self) -> float:
        """Holonomy of the interleaved input→output trajectory."""
        if len(self._interleaved_states) < 3:
            return 0.0
        states = np.array(self._interleaved_states)
        return _pancharatnam_phase(states)

    @property
    def determinative(self) -> Optional[float]:
        """
        The differential determinative.

        = total_phase (interleaved trajectory) − input_phase (input-only)

        This is the curvature the function CONTRIBUTES. The input path's
        curvature is subtracted out. What remains is the geometric residue
        of the transformation.
        """
        if len(self._input_states) < 3:
            return None
        return self.total_phase - self.input_phase

    def close_loop(self) -> float:
        det = self.determinative
        self._determinative = det
        return det if det is not None else 0.0

    def __repr__(self):
        det = self.determinative
        det_str = f"{det:.4f} rad ({np.degrees(det):.1f}°)" if det is not None else "unresolved"
        return f"Glyph('{self.name}' | calls={self._call_count} | det={det_str})"


class GlyphSequence:
    """
    A sequence of Glyphs — a hieroglyphic word.

    The sequence determinative is computed the same way: differential phase
    of the combined input→output trajectory minus the input-only trajectory.
    """

    def __init__(self, *glyphs: Glyph, name: str = "sequence"):
        self.glyphs = list(glyphs)
        self.name = name
        self._input_states: List[np.ndarray] = []
        self._output_states: List[np.ndarray] = []
        self._interleaved_states: List[np.ndarray] = []

    def __call__(self, initial_value: Any) -> Any:
        value = initial_value

        # Record the input to the whole sequence
        if self.glyphs:
            input_state = self.glyphs[0]._embed(value)
            self._input_states.append(input_state)
            self._interleaved_states.append(input_state)

        for glyph in self.glyphs:
            value = glyph(value)

        # Record the output of the whole sequence
        if self.glyphs:
            output_state = self.glyphs[-1]._embed(value)
            self._output_states.append(output_state)
            self._interleaved_states.append(output_state)

        return value

    @property
    def input_phase(self) -> float:
        if len(self._input_states) < 3:
            return 0.0
        return _pancharatnam_phase(np.array(self._input_states))

    @property
    def total_phase(self) -> float:
        if len(self._interleaved_states) < 3:
            return 0.0
        return _pancharatnam_phase(np.array(self._interleaved_states))

    @property
    def determinative(self) -> Optional[float]:
        if len(self._input_states) < 3:
            return None
        return self.total_phase - self.input_phase

    def __repr__(self):
        det = self.determinative
        det_str = f"{det:.4f} rad ({np.degrees(det):.1f}°)" if det is not None else "unresolved"
        return f"GlyphSeq('{self.name}' | det={det_str})"


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("GLYPH v2: differential geometric phase")
    print("=" * 60)
    print()

    # Identity: should be ≈ 0
    g_id = Glyph(lambda x: x, name="identity")
    for i in [1, 2, 3, 4, 5]:
        g_id(i)
    print(f"identity:    det = {g_id.close_loop():.6f} rad "
          f"({np.degrees(g_id.close_loop()):.2f}°)")
    print(f"  input_phase  = {g_id.input_phase:.6f}")
    print(f"  total_phase  = {g_id.total_phase:.6f}")
    print()

    # Square: should be nonzero
    g_sq = Glyph(lambda x: x**2, name="square")
    for i in [1, 2, 3, 4, 5]:
        g_sq(i)
    print(f"square:      det = {g_sq.close_loop():.6f} rad "
          f"({np.degrees(g_sq.close_loop()):.2f}°)")
    print(f"  input_phase  = {g_sq.input_phase:.6f}")
    print(f"  total_phase  = {g_sq.total_phase:.6f}")
    print()

    # Constant: collapses all inputs to 42
    g_c = Glyph(lambda x: 42, name="constant_42")
    for i in [1, 2, 3, 4, 5]:
        g_c(i)
    print(f"constant(42): det = {g_c.close_loop():.6f} rad "
          f"({np.degrees(g_c.close_loop()):.2f}°)")
    print()

    # Forward vs reverse squaring
    g_fwd = Glyph(lambda x: x**2, name="sq_fwd")
    g_rev = Glyph(lambda x: x**2, name="sq_rev")
    for i in [1, 2, 3, 4, 5]:
        g_fwd(i)
    for i in [5, 4, 3, 2, 1]:
        g_rev(i)
    print(f"sq forward:  det = {g_fwd.close_loop():.6f}")
    print(f"sq reverse:  det = {g_rev.close_loop():.6f}")
    print(f"  different? {abs(g_fwd.close_loop() - g_rev.close_loop()) > 0.01}")
    print()

    # Commuting functions: add3 then add5 vs add5 then add3
    g_a3 = Glyph(lambda x: x + 3, name="add3")
    g_a5 = Glyph(lambda x: x + 5, name="add5")
    g_a3b = Glyph(lambda x: x + 3, name="add3")
    g_a5b = Glyph(lambda x: x + 5, name="add5")

    seq_35 = GlyphSequence(g_a3, g_a5, name="add3→add5")
    seq_53 = GlyphSequence(g_a5b, g_a3b, name="add5→add3")

    for v in [1, 2, 3, 4, 5]:
        seq_35(v)
        seq_53(v)

    print(f"add3→add5:   det = {seq_35.determinative:.6f}")
    print(f"add5→add3:   det = {seq_53.determinative:.6f}")
    print(f"  diff = {abs(seq_35.determinative - seq_53.determinative):.6f}")
