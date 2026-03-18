#!/usr/bin/env python3
"""
glyph.py — A first attempt at a hieroglyphic computational primitive.

Not a language yet. A single data structure: the Glyph.

A Glyph is a computation that knows three things about itself simultaneously:
  1. Its phonogram  — what it does (the callable, the operation)
  2. Its ideogram   — what it means (the geometric invariant of its state trajectory)
  3. Its determinative — what kind of thing it is (the holonomy after loop closure)

The determinative is not declared. It emerges from executing the phonogram
and measuring the holonomy of the ideogram. You cannot know the determinative
until the loop closes.

This maps onto polar time as:
  phonogram    → r_t  (radial, linear, irreversible execution)
  ideogram     → the state vector in C^n (the representation that persists)
  determinative → θ_t holonomy (the phase accumulated over the cycle)

The key property: two Glyphs that produce the same output (same phonogram)
can have different determinatives if their paths through state space differ.
The determinative is the thing no existing language can see.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Any, List, Optional
import cmath


@dataclass
class StateSnapshot:
    """A point on the trajectory through representation space."""
    r_t: float          # radial time — when this happened in the execution
    vector: np.ndarray  # the state in C^n at this moment
    
    @property
    def normalized(self) -> np.ndarray:
        norm = np.linalg.norm(self.vector)
        if norm < 1e-15:
            return self.vector
        return self.vector / norm


class Glyph:
    """
    A computation that carries its own geometric shadow.
    
    Usage:
        g = Glyph(lambda x: x**2, name="square")
        result = g(4)          # runs the phonogram, returns 16
        print(g.determinative) # the holonomy — None until a loop closes
    """
    
    def __init__(self, phonogram: Callable, name: str = "unnamed", 
                 n_dims: int = 4):
        self.phonogram = phonogram
        self.name = name
        self.n_dims = n_dims
        self.trajectory: List[StateSnapshot] = []
        self._r_t = 0.0
        self._determinative: Optional[complex] = None
        self._call_count = 0
    
    def _embed(self, value: Any) -> np.ndarray:
        """
        Embed a computational value into C^n.
        
        The embedding must create CURVATURE in CP^{n-1} — otherwise
        the holonomy is trivially zero. Different values must map to
        states that are neither parallel nor orthogonal, and the
        mapping must be nonlinear so that different paths through
        value-space trace out loops enclosing nonzero area in CP^{n-1}.
        
        We use a nonlinear phase-entangling embedding: each dimension
        gets a phase that depends on ALL the bits of the value, not
        just one. This couples the dimensions and creates curvature.
        """
        if isinstance(value, (int, float, complex)):
            x = float(np.real(complex(value)))
            # nonlinear embedding: each component's phase depends on
            # x through a different nonlinear function, and the 
            # amplitudes vary so states aren't on a great circle
            components = np.zeros(self.n_dims, dtype=complex)
            for k in range(self.n_dims):
                # phase: nonlinear coupling between value and dimension
                phase = (x * (k + 1) * 0.7 + 
                         np.sin(x * (k + 0.5)) * 1.3 +
                         np.cos(x * x * 0.1 * (k + 1)) * 0.9)
                # amplitude: varies across dimensions, depends on value
                amp = 1.0 + 0.5 * np.sin(x * 0.3 + k * 1.1)
                components[k] = amp * np.exp(1j * phase)
            return components
        elif isinstance(value, np.ndarray):
            if len(value) >= self.n_dims:
                return value[:self.n_dims].astype(complex)
            else:
                padded = np.zeros(self.n_dims, dtype=complex)
                padded[:len(value)] = value.astype(complex)
                return padded
        elif isinstance(value, str):
            # embed string via character-entangled phases
            raw = [ord(c) for c in value[:32]]
            x = sum(r * (i+1) for i, r in enumerate(raw)) / max(len(raw), 1)
            return self._embed(x)
        else:
            return self._embed(float(hash(value) % 10000) / 100.0)
    
    def __call__(self, *args, **kwargs) -> Any:
        """Execute the phonogram. Record the trajectory."""
        # snapshot before
        if args:
            pre_state = self._embed(args[0])
        else:
            pre_state = self._embed(self._call_count)
        
        self.trajectory.append(StateSnapshot(
            r_t=self._r_t,
            vector=pre_state
        ))
        
        # execute
        result = self.phonogram(*args, **kwargs)
        self._r_t += 1.0
        self._call_count += 1
        
        # snapshot after
        post_state = self._embed(result)
        self.trajectory.append(StateSnapshot(
            r_t=self._r_t,
            vector=post_state
        ))
        
        return result
    
    @property
    def ideogram(self) -> Optional[np.ndarray]:
        """
        The ideogram is the state trajectory itself — the geometric object
        that persists across invocations. It's what the Glyph *means*,
        independent of when you execute it.
        
        Returns the trajectory as a matrix of normalized state vectors.
        """
        if not self.trajectory:
            return None
        return np.array([s.normalized for s in self.trajectory])
    
    @property 
    def determinative(self) -> Optional[float]:
        """
        The determinative: the Pancharatnam phase accumulated over
        the trajectory.
        
        This is the silent classifier. It tells you what KIND of 
        computation this was — not what it produced, but what geometric
        residue it left in state space.
        
        Returns None if fewer than 3 states recorded (need a loop).
        Returns the phase in radians.
        """
        traj = self.ideogram
        if traj is None or len(traj) < 3:
            return None
        return self._pancharatnam_phase(traj)
    
    def _pancharatnam_phase(self, states: np.ndarray) -> float:
        """
        Compute the Pancharatnam geometric phase around the trajectory.
        
        For states ψ_0, ψ_1, ..., ψ_{N-1}, the phase is:
          arg(⟨ψ_0|ψ_1⟩ ⟨ψ_1|ψ_2⟩ ... ⟨ψ_{N-1}|ψ_0⟩)
        
        This is exactly the holonomy of the natural connection on CP^{n-1}.
        """
        n = len(states)
        product = complex(1.0, 0.0)
        
        for k in range(n):
            psi_k = states[k]
            psi_next = states[(k + 1) % n]
            inner = np.vdot(psi_k, psi_next)  # conjugate-linear in first arg
            if abs(inner) < 1e-15:
                return 0.0  # degenerate — orthogonal states
            product *= inner / abs(inner)
        
        return cmath.phase(product)
    
    def close_loop(self) -> float:
        """
        Explicitly close the trajectory loop and return the determinative.
        
        In hieroglyphic terms: you've written all the phonograms and the
        ideogram. Now the determinative appears — retroactively classifying
        the entire glyph complex.
        """
        det = self.determinative
        self._determinative = det
        return det if det is not None else 0.0
    
    def __repr__(self):
        det = self.determinative
        det_str = f"{det:.4f} rad" if det is not None else "unresolved"
        return (
            f"Glyph('{self.name}' | "
            f"calls={self._call_count} | "
            f"determinative={det_str})"
        )


class GlyphSequence:
    """
    A sequence of Glyphs — the equivalent of a hieroglyphic word.
    
    The sequence has its own determinative, distinct from the 
    determinatives of its component glyphs. The word-level holonomy
    depends on the ORDER of the glyphs — the path through their
    combined state spaces.
    
    This is the non-commutativity: Glyph(A) then Glyph(B) accumulates
    a different phase than Glyph(B) then Glyph(A), even if the
    final output is the same.
    """
    
    def __init__(self, *glyphs: Glyph, name: str = "sequence"):
        self.glyphs = list(glyphs)
        self.name = name
        self.combined_trajectory: List[StateSnapshot] = []
        self._r_t = 0.0
    
    def __call__(self, initial_value: Any) -> Any:
        """Execute glyphs in sequence, accumulating the combined trajectory."""
        value = initial_value
        
        for glyph in self.glyphs:
            # record combined state before each glyph
            if hasattr(glyph, '_embed'):
                state = glyph._embed(value)
                self.combined_trajectory.append(
                    StateSnapshot(r_t=self._r_t, vector=state)
                )
            
            value = glyph(value)
            self._r_t += 1.0
        
        # record final state
        if self.glyphs and hasattr(self.glyphs[-1], '_embed'):
            state = self.glyphs[-1]._embed(value)
            self.combined_trajectory.append(
                StateSnapshot(r_t=self._r_t, vector=state)
            )
        
        return value
    
    @property
    def determinative(self) -> Optional[float]:
        """The holonomy of the combined trajectory — the word-level determinative."""
        if len(self.combined_trajectory) < 3:
            return None
        states = np.array([s.normalized for s in self.combined_trajectory])
        
        n = len(states)
        product = complex(1.0, 0.0)
        for k in range(n):
            psi_k = states[k]
            psi_next = states[(k + 1) % n]
            inner = np.vdot(psi_k, psi_next)
            if abs(inner) < 1e-15:
                return 0.0
            product *= inner / abs(inner)
        
        return cmath.phase(product)
    
    def reversed(self) -> 'GlyphSequence':
        """Return the same glyphs in reverse order — should flip the determinative."""
        rev = GlyphSequence(*reversed(self.glyphs), name=f"{self.name}_reversed")
        return rev


# ---------------------------------------------------------------------------
# Demonstration: two computations, same output, different determinatives
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    
    print("=" * 60)
    print("GLYPH: a hieroglyphic computational primitive")
    print("=" * 60)
    print()
    
    # Two ways to compute the same thing: sum of squares from 1 to N
    
    # Path A: iterate forward
    g_forward = Glyph(lambda x: x**2, name="square_forward", n_dims=8)
    
    # Path B: iterate in a different order
    g_reverse = Glyph(lambda x: x**2, name="square_reverse", n_dims=8)
    
    # Execute Path A: 1, 2, 3, 4, 5
    total_a = 0
    for i in [1, 2, 3, 4, 5]:
        total_a += g_forward(i)
    
    # Execute Path B: 5, 4, 3, 2, 1
    total_b = 0
    for i in [5, 4, 3, 2, 1]:
        total_b += g_reverse(i)
    
    print(f"Path A (forward): sum of squares = {total_a}")
    print(f"Path B (reverse): sum of squares = {total_b}")
    print(f"Same output? {total_a == total_b}")
    print()
    
    det_a = g_forward.close_loop()
    det_b = g_reverse.close_loop()
    
    print(f"Determinative A: {det_a:.6f} rad ({np.degrees(det_a):.2f}°)")
    print(f"Determinative B: {det_b:.6f} rad ({np.degrees(det_b):.2f}°)")
    print(f"Same determinative? {abs(det_a - det_b) < 1e-10}")
    print()
    
    if abs(det_a - det_b) > 1e-10:
        print(">>> The phonograms are identical (same function).")
        print(">>> The outputs are identical (same sum).")
        print(">>> The determinatives DIFFER.")
        print(">>> The path through state space left a different geometric residue.")
        print(">>> This is the thing no existing language can see.")
    
    print()
    print("-" * 60)
    print("GLYPH SEQUENCES: non-commutativity of the determinative")
    print("-" * 60)
    print()
    
    # Two glyphs: double and increment
    g_double = Glyph(lambda x: x * 2, name="double", n_dims=8)
    g_inc = Glyph(lambda x: x + 1, name="increment", n_dims=8)
    
    # Sequence AB: double then increment
    seq_ab = GlyphSequence(g_double, g_inc, name="double→inc")
    # Sequence BA: increment then double  
    g_double2 = Glyph(lambda x: x * 2, name="double", n_dims=8)
    g_inc2 = Glyph(lambda x: x + 1, name="increment", n_dims=8)
    seq_ba = GlyphSequence(g_inc2, g_double2, name="inc→double")
    
    # Run both on several inputs to build trajectories
    for val in [1, 2, 3, 4, 5]:
        result_ab = seq_ab(val)
        result_ba = seq_ba(val)
    
    det_ab = seq_ab.determinative
    det_ba = seq_ba.determinative
    
    print(f"double→inc  determinative: {det_ab:.6f} rad ({np.degrees(det_ab):.2f}°)")
    print(f"inc→double  determinative: {det_ba:.6f} rad ({np.degrees(det_ba):.2f}°)")
    print(f"Difference: {abs(det_ab - det_ba):.6f} rad ({np.degrees(abs(det_ab - det_ba)):.2f}°)")
    print()
    
    if abs(det_ab - det_ba) > 1e-10:
        print(">>> Same glyphs, different order, different determinative.")
        print(">>> The computation is non-commutative in its geometric residue")
        print(">>> even when the individual operations are fully determined.")
        print(">>> This is the 𓀁 — the silent sign that classifies the word.")
    
    print()
    print("=" * 60)
    print(f"g_forward:  {g_forward}")
    print(f"g_reverse:  {g_reverse}")
    print(f"g_double:   {g_double}")
    print(f"g_inc:      {g_inc}")
    print("=" * 60)
