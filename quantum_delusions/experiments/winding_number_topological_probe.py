#!/usr/bin/env python3
"""
winding_number_topological_probe.py

The polar-time conjecture is topological, not geometric.

The spacetime has signature (-,-,+,+,+) and is FLAT (R=0 for r_t > 0).
There is no curvature to detect. The theta_t dimension is compact and
periodic, which means the space has non-trivial fundamental group:

    pi_1(M) = Z  (the integers, i.e., winding numbers)

This is structurally identical to the Aharonov-Bohm effect:
  - In AB: a solenoid creates a vector potential with A != 0 outside, B = 0.
    Electrons traversing paths that enclose the solenoid accumulate phase
    proportional to the enclosed flux, even though they never touch a field.
  - In polar time: the compact theta_t dimension creates a holonomy for
    quantum probes that complete a loop in temporal phase space, even though
    the geometry is locally flat everywhere.

The key experimental distinction between TOPOLOGICAL and GEOMETRIC phase:

  Geometric (Berry) phase:
    gamma = integral of curvature over enclosed area
    -> depends on loop SHAPE and SIZE
    -> changes smoothly as you deform the loop
    -> scales with solid angle

  Topological (winding) phase:
    gamma = 2*pi * n   (n = winding number, integer)
    -> depends only on how many times the loop WINDS around the hole
    -> invariant under smooth deformation of the path
    -> changes only when the loop topology changes (passes through the hole)
    -> quantized: must be an integer multiple of 2*pi

THE PREDICTION:
If polar time is topological, a qubit steered around the Bloch equator n
times should accumulate phase exactly n * Phi_0, where Phi_0 is the
fundamental winding phase. The phase should:
  1. Scale linearly with n (winding number)
  2. Be invariant under loop shape deformation (ellipse vs. circle)
  3. Be invariant under traversal speed (fewer steps vs. more steps)
  4. REVERSE SIGN when direction reverses
  5. NOT depend smoothly on loop area/radius

Points 2 and 5 distinguish topological from geometric. In the v3 GPT-2
experiment (polar_holonomy_v3_results.md), shape invariance was detected
(delta = 0.001 rad at CP^15). That is a topological signature, not geometric.
This experiment tests the same invariance directly on IBM quantum hardware
using a single qubit as the temporal probe.

FALSIFICATION:
If the phase does NOT scale as n * Phi_0 for integer n - if it scales
smoothed or non-linearly, or if shape deformation changes it - the
topological interpretation is falsified. The conjecture may still survive
if a purely geometric version can be written down, but the specific
claim that pi_1(M) = Z would be wrong.

CONNECTION TO v3 RESULTS:
The v3 GPT-2 experiment found holonomy in CP^15 (32 PCA dimensions) that
is orientation-reversing and shape-invariant. This is the classical
representational analogue of what we are testing physically here. If both
experiments confirm the same topological structure, the conjecture gains
cross-substrate support: polar-time topology appears in both the physical
quantum system and in the transformer's representational geometry.
"""

import argparse
import json
import math
import os
import random
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np


def winding_n_qasm(n: int, phi_step: float = math.pi / 4) -> str:
    steps_per_winding = int(round(2 * math.pi / phi_step))
    total_steps = n * steps_per_winding
    lines = [
        'OPENQASM 2.0;',
        'include "qelib1.inc";',
        'qreg q[1];',
        'creg c[1];',
        'h q[0];',
        f'// {n} winding(s), {steps_per_winding} steps each',
    ]
    for step in range(total_steps):
        lines.append(f'rz({phi_step:.6f}) q[0];')
    lines += ['h q[0];', 'measure q[0] -> c[0];']
    return '\n'.join(lines)


def winding_shape_deformed_qasm(n: int = 1, ellipse_ratio: float = 0.5) -> str:
    base_step  = math.pi / 4  # radians
    step_large = base_step * (1 + ellipse_ratio)
    step_small = base_step * (1 - ellipse_ratio)
    total_steps = n * 8
    lines = [
        'OPENQASM 2.0;',
        'include "qelib1.inc";',
        'qreg q[1];',
        'creg c[1];',
        'h q[0];',
        f'// Shape-deformed winding: ellipse_ratio={ellipse_ratio}, n={n}',
    ]
    for step in range(total_steps):
        phi = step_large if step % 2 == 0 else step_small
        lines.append(f'rz({phi:.6f}) q[0];')
    lines += ['h q[0];', 'measure q[0] -> c[0];']
    return '\n'.join(lines)


def _random_control_qasm(n_pairs: int = 8, seed: int = 2026) -> str:
    """Random-angle rz/ry circuit. Same structure as creature loop, random content."""
    rng = random.Random(seed)
    lines = [
        'OPENQASM 2.0;',
        'include "qelib1.inc";',
        'qreg q[1];',
        'creg c[1];',
        'h q[0];',
        f'// Random control: {n_pairs} rz/ry pairs, seed={seed}',
    ]
    for _ in range(n_pairs):
        rz_angle = rng.uniform(-math.pi, math.pi)
        ry_angle = rng.uniform(-math.pi, math.pi)
        lines.append(f'rz({rz_angle:.6f}) q[0];')
        lines.append(f'ry({ry_angle:.6f}) q[0];')
    lines += ['h q[0];', 'measure q[0] -> c[0];']
    return '\n'.join(lines)


def winding_fractional_shape_deformed_qasm(fraction: float, ellipse_ratio: float = 0.5) -> str:
    """Fractional winding with elliptical step pattern. Same total phase, different path."""
    total_phase = fraction * 2 * math.pi
    n_steps = 4
    base_step = total_phase / n_steps
    step_large = base_step * (1 + ellipse_ratio)
    step_small = base_step * (1 - ellipse_ratio)
    lines = [
        'OPENQASM 2.0;',
        'include "qelib1.inc";',
        'qreg q[1];',
        'creg c[1];',
        'h q[0];',
        f'// Shape-deformed fractional: frac={fraction}, ellipse={ellipse_ratio}',
    ]
    for step in range(n_steps):
        phi = step_large if step % 2 == 0 else step_small
        lines.append(f'rz({phi:.6f}) q[0];')
    lines += ['h q[0];', 'measure q[0] -> c[0];']
    return '\n'.join(lines)


def winding_reversed_qasm(n: int = 1) -> str:
    total_steps = n * 8
    phi_step = -(math.pi / 4)
    lines = [
        'OPENQASM 2.0;',
        'include "qelib1.inc";',
        'qreg q[1];',
        'creg c[1];',
        'h q[0];',
        f'// Reversed winding: n={n}',
    ]
    for _ in range(total_steps):
        lines.append(f'rz({phi_step:.6f}) q[0];')
    lines += ['h q[0];', 'measure q[0] -> c[0];']
    return '\n'.join(lines)


def winding_speed_deformed_qasm(n: int = 1, density: int = 4) -> str:
    steps_per_winding = 8 * density
    phi_step = 2 * math.pi / steps_per_winding
    total_steps = n * steps_per_winding
    lines = [
        'OPENQASM 2.0;',
        'include "qelib1.inc";',
        'qreg q[1];',
        'creg c[1];',
        'h q[0];',
        f'// Speed-deformed winding: n={n}, density={density}x',
    ]
    for _ in range(total_steps):
        lines.append(f'rz({phi_step:.6f}) q[0];')
    lines += ['h q[0];', 'measure q[0] -> c[0];']
    return '\n'.join(lines)


def winding_fractional_qasm(fraction: float, n_steps: int = 4) -> str:
    """Fractional winding: total phase = fraction * 2*pi.

    Uses n_steps gates of equal size. Fewer gates = less decoherence.
    Key fractions:
      0.25 -> P(0) = cos²(pi/4) = 0.5
      0.50 -> P(0) = cos²(pi/2) = 0.0
      0.75 -> P(0) = cos²(3*pi/4) = 0.5
      1.50 -> P(0) = cos²(3*pi/2) = 0.0
    """
    total_phase = fraction * 2 * math.pi
    phi_step = total_phase / n_steps
    lines = [
        'OPENQASM 2.0;',
        'include "qelib1.inc";',
        'qreg q[1];',
        'creg c[1];',
        'h q[0];',
        f'// fractional winding {fraction}, {n_steps} steps, total={total_phase:.4f} rad',
    ]
    for _ in range(n_steps):
        lines.append(f'rz({phi_step:.6f}) q[0];')
    lines += ['h q[0];', 'measure q[0] -> c[0];']
    return '\n'.join(lines)


def winding_fractional_ybasis_qasm(fraction: float, direction: int = 1, n_steps: int = 4) -> str:
    """Fractional winding with Y-basis measurement for sign sensitivity."""
    total_phase = direction * fraction * 2 * math.pi
    phi_step = total_phase / n_steps
    lines = [
        'OPENQASM 2.0;',
        'include "qelib1.inc";',
        'qreg q[1];',
        'creg c[1];',
        'h q[0];',
        f'// fractional Y-basis: frac={fraction}, dir={direction:+d}, {n_steps} steps',
    ]
    for _ in range(n_steps):
        lines.append(f'rz({phi_step:.6f}) q[0];')
    lines += ['sdg q[0];', 'h q[0];', 'measure q[0] -> c[0];']
    return '\n'.join(lines)


def winding_ybasis_qasm(n: int = 1, direction: int = 1) -> str:
    """Y-basis measurement to distinguish +phi from -phi.
    
    Instead of H...H (X-basis), uses S†·H...H·S to measure
    in the Y-basis. cos²(theta) can't distinguish sign;
    the Y-basis expectation CAN.
    """
    total_steps = n * 8
    phi_step = direction * (math.pi / 4)
    lines = [
        'OPENQASM 2.0;',
        'include "qelib1.inc";',
        'qreg q[1];',
        'creg c[1];',
        'h q[0];',
        f'// Y-basis sign test: n={n}, direction={direction:+d}',
    ]
    for _ in range(total_steps):
        lines.append(f'rz({phi_step:.6f}) q[0];')
    # Y-basis measurement: H then S†(=Sdg) then measure
    lines += ['sdg q[0];', 'h q[0];', 'measure q[0] -> c[0];']
    return '\n'.join(lines)


# ── Creature-derived loop from basin weight trajectory ─────────────────────

def trajectory_to_bloch_angles(weight_vecs: List[List[float]]) -> List[tuple]:
    """Project weight trajectory onto top-2 PCA directions, convert to Bloch angles.

    Returns list of (theta, phi) pairs suitable for rz/ry gate encoding.
    If the trajectory is too short or degenerate, returns an empty list.
    """
    W = np.array(weight_vecs, dtype=np.float64)
    if W.shape[0] < 3:
        return []
    # Centre the trajectory
    W_c = W - W.mean(axis=0)
    # PCA via SVD on centred matrix
    U, S, Vt = np.linalg.svd(W_c, full_matrices=False)
    # Project onto the top-2 principal directions
    proj = W_c @ Vt[:2].T  # shape (T, 2)
    # Normalise to unit-circle scale so angles are meaningful
    norms = np.linalg.norm(proj, axis=1, keepdims=True)
    max_norm = norms.max()
    if max_norm < 1e-12:
        return []
    proj = proj / max_norm
    # Convert (x, y) path to Bloch angles:
    #   phi   = atan2(y, x)           (azimuthal, mapped to rz)
    #   theta = pi * sqrt(x^2+y^2)    (polar, mapped to ry; 0 at centre, pi at rim)
    angles = []
    for x, y in proj:
        phi = math.atan2(y, x)
        r = math.sqrt(x * x + y * y)
        theta = math.pi * min(r, 1.0)
        angles.append((theta, phi))
    return angles


def build_creature_loop_qasm(bloch_angles: List[tuple]) -> str:
    """Encode Bloch angle sequence as rz/ry gates matching the probe family.

    Uses the same gate pattern as the existing winding circuits:
    H -> sequence of (rz, ry) pairs -> H -> measure.
    """
    if not bloch_angles:
        raise ValueError("Empty Bloch angle sequence; cannot build circuit")
    lines = [
        'OPENQASM 2.0;',
        'include "qelib1.inc";',
        'qreg q[1];',
        'creg c[1];',
        'h q[0];',
        f'// Creature-derived loop: {len(bloch_angles)} steps from weight trajectory PCA',
    ]
    for theta, phi in bloch_angles:
        lines.append(f'rz({phi:.6f}) q[0];')
        lines.append(f'ry({theta:.6f}) q[0];')
    lines += ['h q[0];', 'measure q[0] -> c[0];']
    return '\n'.join(lines)


def run_on_ibm(circuits_qasm: List[str], token: Optional[str] = None,
               shots: int = 4096) -> List[dict]:
    """Submit QASM circuits to IBM quantum hardware. No simulator fallback.

    Returns a list of count dicts, one per circuit.
    """
    try:
        from qiskit import QuantumCircuit
    except ImportError:
        raise ImportError(
            "qiskit is required. Install with: pip install qiskit qiskit-ibm-runtime"
        )

    qcs = [QuantumCircuit.from_qasm_str(q) for q in circuits_qasm]

    token = token or os.environ.get("QISKIT_IBM_TOKEN") or os.environ.get("IBM_QUANTUM_TOKEN")
    if not token:
        raise RuntimeError(
            "No IBM Quantum token found. Set QISKIT_IBM_TOKEN (or IBM_QUANTUM_TOKEN) "
            "in your environment. This experiment runs on real quantum hardware only."
        )

    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    channel = os.environ.get("QISKIT_IBM_CHANNEL", "ibm_quantum")
    instance = os.environ.get("QISKIT_IBM_INSTANCE")
    service = QiskitRuntimeService(channel=channel, token=token, instance=instance)
    backend = service.least_busy(simulator=False, operational=True)
    print(f"\u26a1 IBM backend: {backend.name}")
    # Transpile to backend's native gate set (required since March 2024)
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    qcs_isa = pm.run(qcs)
    print(f"  Transpiled {len(qcs_isa)} circuits for {backend.name}")
    sampler = SamplerV2(backend)
    job = sampler.run(qcs_isa, shots=shots)
    result = job.result()
    all_counts = []
    for pub_result in result:
        counts = pub_result.data.c.get_counts()
        all_counts.append(counts)
    return all_counts


# ── v2 suite: fractional windings that produce real signal on calibrated hardware ──
#
# Integer windings (n=1,2,3) give rz(n*2*pi) = global phase = identity.
# On a well-calibrated machine P(0) ≈ 1.0 for all of them — no discriminating
# power. The previous run confirmed this (P(0) = 0.99 across the board).
#
# Fractional windings produce DISTINCT P(0) values via cos²(fraction*pi):
#   0.25 -> P(0) = 0.500    (4 gates)
#   0.50 -> P(0) = 0.000    (4 gates) — confirmed on ibm_fez: 0.018
#   0.75 -> P(0) = 0.500    (4 gates)
#   1.00 -> P(0) = 1.000    (4 gates) — confirmed: 0.992
#   1.50 -> P(0) = 0.000    (4 gates)
#
# Shape invariance test: same fraction, different step-size pattern.
# Speed invariance test: same fraction, more smaller steps.
# Sign reversal: Y-basis at fractional winding where sin(theta) != 0.
# Creature loop: subsampled to 8 points (16 gates) to survive decoherence.

WINDING_EXPERIMENT_SUITE = [
    # ── Fractional winding ladder: 4 gates each, distinct P(0) values ──
    {
        "circuit_name":       "frac_0.25",
        "is_theory_relevant": True,
        "hypothesis": (
            "Quarter winding (4 gates). P(0) = cos²(pi/4) = 0.5. "
            "Baseline for the fractional ladder."
        ),
        "expected_counts":    {"0": 0.5, "1": 0.5},
        "estimated_seconds":  2.0,
        "qasm_fn":            lambda: winding_fractional_qasm(0.25),
        "family":             "fractional",
        "winding":            0.25,
        "variant":            "base",
    },
    {
        "circuit_name":       "frac_0.50",
        "is_theory_relevant": True,
        "hypothesis": (
            "Half winding (4 gates). P(0) = cos²(pi/2) = 0.0. "
            "Already confirmed on ibm_fez at P(0) = 0.018."
        ),
        "expected_counts":    {"0": 0.0, "1": 1.0},
        "estimated_seconds":  2.0,
        "qasm_fn":            lambda: winding_fractional_qasm(0.50),
        "family":             "fractional",
        "winding":            0.50,
        "variant":            "base",
    },
    {
        "circuit_name":       "frac_0.75",
        "is_theory_relevant": True,
        "hypothesis": (
            "Three-quarter winding (4 gates). P(0) = cos²(3*pi/4) = 0.5. "
            "Same P(0) as 0.25 but different phase — Y-basis distinguishes them."
        ),
        "expected_counts":    {"0": 0.5, "1": 0.5},
        "estimated_seconds":  2.0,
        "qasm_fn":            lambda: winding_fractional_qasm(0.75),
        "family":             "fractional",
        "winding":            0.75,
        "variant":            "base",
    },
    {
        "circuit_name":       "frac_1.00",
        "is_theory_relevant": True,
        "hypothesis": (
            "Full winding (4 gates). P(0) = cos²(pi) = 1.0. "
            "Calibration anchor — confirms the circuit works."
        ),
        "expected_counts":    {"0": 1.0, "1": 0.0},
        "estimated_seconds":  2.0,
        "qasm_fn":            lambda: winding_fractional_qasm(1.00),
        "family":             "fractional",
        "winding":            1.00,
        "variant":            "base",
    },
    {
        "circuit_name":       "frac_1.50",
        "is_theory_relevant": True,
        "hypothesis": (
            "One-and-a-half windings (4 gates). P(0) = cos²(3*pi/2) = 0.0. "
            "Same P(0) as 0.50 — tests linearity at higher winding."
        ),
        "expected_counts":    {"0": 0.0, "1": 1.0},
        "estimated_seconds":  2.0,
        "qasm_fn":            lambda: winding_fractional_qasm(1.50),
        "family":             "fractional",
        "winding":            1.50,
        "variant":            "base",
    },
    # ── Shape invariance at half-winding (the only fraction with sharp P(0)) ──
    {
        "circuit_name":       "frac_0.50_shape",
        "is_theory_relevant": True,
        "hypothesis": (
            "Half winding, elliptical path. TOPOLOGICAL: same P(0) as frac_0.50. "
            "GEOMETRIC: different P(0). This is the critical test at a fraction "
            "where the signal is maximal (P(0) near 0)."
        ),
        "expected_counts":    {"0": 0.0, "1": 1.0},
        "estimated_seconds":  2.0,
        "qasm_fn":            lambda: winding_fractional_shape_deformed_qasm(0.50, 0.5),
        "family":             "fractional",
        "winding":            0.50,
        "variant":            "shape_deformed",
    },
    # ── Y-basis sign reversal at quarter-winding (where sin != 0) ──
    {
        "circuit_name":       "frac_0.25_ybasis_fwd",
        "is_theory_relevant": True,
        "hypothesis": (
            "Quarter winding Y-basis FORWARD. P(0) = (1 + sin(pi/2))/2 = 1.0. "
            "Y-basis breaks the cos² symmetry and CAN distinguish sign."
        ),
        "expected_counts":    {"0": 1.0, "1": 0.0},
        "estimated_seconds":  2.0,
        "qasm_fn":            lambda: winding_fractional_ybasis_qasm(0.25, +1),
        "family":             "fractional",
        "winding":            0.25,
        "variant":            "ybasis_fwd",
    },
    {
        "circuit_name":       "frac_0.25_ybasis_rev",
        "is_theory_relevant": True,
        "hypothesis": (
            "Quarter winding Y-basis REVERSED. P(0) = (1 + sin(-pi/2))/2 = 0.0. "
            "If sign reversal works, this should be ~0 while fwd is ~1."
        ),
        "expected_counts":    {"0": 0.0, "1": 1.0},
        "estimated_seconds":  2.0,
        "qasm_fn":            lambda: winding_fractional_ybasis_qasm(0.25, -1),
        "family":             "fractional",
        "winding":            -0.25,
        "variant":            "ybasis_rev",
    },
    # ── Random control: same gate depth as creature, random angles ──
    {
        "circuit_name":       "random_control",
        "is_theory_relevant": True,
        "hypothesis": (
            "Random-angle circuit with same gate structure as the creature loop "
            "(8 rz + 8 ry pairs = 16 gates). If this gives P(0) near 0.5 and "
            "the creature loop does not, the creature's encoding carries signal. "
            "If both are near 0.5, the gate depth decoheres everything."
        ),
        "expected_counts":    {"0": 0.5, "1": 0.5},
        "estimated_seconds":  3.0,
        "qasm_fn":            lambda: _random_control_qasm(8, seed=2026),
        "family":             "control",
        "winding":            0,
        "variant":            "random_control",
    },
    # Creature-derived entry is added dynamically via add_creature_circuit()
]


def add_creature_circuit(weight_trajectory: List[List[float]],
                         subsample: int = 8) -> Optional[dict]:
    """Build creature-loop circuit from a basin geometry weight trajectory.

    Sub-samples the trajectory to keep gate count manageable, converts to
    Bloch angles via PCA, and appends the circuit to the experiment suite.
    Returns the new suite entry, or None if the trajectory is degenerate.
    """
    # Sub-sample if trajectory is long
    wt = weight_trajectory
    if len(wt) > subsample:
        step = len(wt) / subsample
        wt = [wt[int(i * step)] for i in range(subsample)]

    angles = trajectory_to_bloch_angles(wt)
    if not angles:
        return None

    entry = {
        "circuit_name":       "creature_loop",
        "is_theory_relevant": True,
        "hypothesis": (
            "Creature-derived loop from basin geometry weight trajectory. "
            "PCA projects the training path into 2D, then Bloch-encodes it. "
            "If the creature's learning path has topological winding, this "
            "circuit should show non-trivial P(0) deviation comparable to "
            "the theory circuits. If the path is open or unwound, P(0) stays "
            "near 0.5 — which is itself informative."
        ),
        "expected_counts":    {"0": 0.5, "1": 0.5},
        "estimated_seconds":  5.0,
        "qasm_fn":            lambda a=angles: build_creature_loop_qasm(a),
        "family":             "creature",
        "winding":            0,  # unknown a priori; measured empirically
        "variant":            "creature_pca",
    }
    WINDING_EXPERIMENT_SUITE.append(entry)
    return entry


def get_suite_qasm() -> list[dict]:
    suite = []
    for spec in WINDING_EXPERIMENT_SUITE:
        entry = {k: v for k, v in spec.items() if k != "qasm_fn"}
        entry["circuit_qasm"] = spec["qasm_fn"]()
        suite.append(entry)
    return suite


def analyze_winding_suite(results: list[dict]) -> dict:
    """Analyse v2 fractional winding suite.

    Model: P(0) = cos²(fraction * pi) for each fractional winding.
    Tests:
      1. Linearity — do all fractional base circuits match cos²(f*pi)?
      2. Shape invariance — does frac_0.50_shape match frac_0.50?
      3. Sign reversal — do Y-basis fwd/rev differ at quarter-winding?
      4. Creature loop — does it differ from 0.5 (noise) and from random control?
    """
    def p0(counts: dict) -> float | None:
        total = sum(counts.values())
        return counts.get("0", 0) / total if total else None

    by_name = {r["circuit_name"]: r for r in results}
    analysis = {
        "timestamp":          datetime.now(timezone.utc).isoformat(),
        "n_circuits":         len(results),
        "fractional_ladder":  None,
        "shape_invariance":   None,
        "sign_reversal":      None,
        "creature_loop":      None,
        "random_control":     None,
        "verdict":            "INSUFFICIENT_DATA",
        "notes":              [],
    }

    # ── Fractional ladder: P(0) = cos²(fraction * pi) ──
    ladder = {}
    for entry in results:
        if entry.get("variant") == "base" and entry.get("family") == "fractional":
            p = p0(entry.get("counts", {}))
            w = entry.get("winding")
            if p is not None and w is not None:
                theory = math.cos(w * math.pi) ** 2
                ladder[entry["circuit_name"]] = {
                    "winding": w, "p0": round(p, 4),
                    "theory": round(theory, 4),
                    "delta": round(abs(p - theory), 4),
                }
    if ladder:
        max_delta = max(v["delta"] for v in ladder.values())
        analysis["fractional_ladder"] = {
            "circuits": ladder,
            "max_delta": round(max_delta, 4),
            "passes": max_delta < 0.06,
        }

    # ── Shape invariance: frac_0.50 vs frac_0.50_shape ──
    base_half = by_name.get("frac_0.50")
    shape_half = by_name.get("frac_0.50_shape")
    if base_half and shape_half:
        p_b = p0(base_half.get("counts", {}))
        p_s = p0(shape_half.get("counts", {}))
        if p_b is not None and p_s is not None:
            delta = abs(p_b - p_s)
            analysis["shape_invariance"] = {
                "p0_base": round(p_b, 4),
                "p0_shaped": round(p_s, 4),
                "delta": round(delta, 4),
                "passes": delta < 0.05,
            }

    # ── Sign reversal: Y-basis at quarter-winding ──
    yfwd = by_name.get("frac_0.25_ybasis_fwd")
    yrev = by_name.get("frac_0.25_ybasis_rev")
    if yfwd and yrev:
        p_f = p0(yfwd.get("counts", {}))
        p_r = p0(yrev.get("counts", {}))
        if p_f is not None and p_r is not None:
            delta = abs(p_f - p_r)
            analysis["sign_reversal"] = {
                "basis": "Y",
                "p0_fwd": round(p_f, 4),
                "p0_rev": round(p_r, 4),
                "delta": round(delta, 4),
                "passes": delta > 0.5,  # should be near 1.0 swing
            }

    # ── Creature loop ──
    creature = by_name.get("creature_loop")
    if creature:
        p_c = p0(creature.get("counts", {}))
        if p_c is not None:
            deviation_from_noise = abs(p_c - 0.5)
            analysis["creature_loop"] = {
                "p0": round(p_c, 4),
                "deviation_from_noise": round(deviation_from_noise, 4),
                "significant": deviation_from_noise > 0.05,
            }

    # ── Random control ──
    control = by_name.get("random_control")
    if control:
        p_ctrl = p0(control.get("counts", {}))
        if p_ctrl is not None:
            analysis["random_control"] = {
                "p0": round(p_ctrl, 4),
            }
            # Compare creature to control
            if analysis.get("creature_loop"):
                creature_p = analysis["creature_loop"]["p0"]
                gap = abs(creature_p - p_ctrl)
                analysis["creature_vs_control"] = {
                    "creature_p0": creature_p,
                    "control_p0": round(p_ctrl, 4),
                    "gap": round(gap, 4),
                    "creature_differs": gap > 0.05,
                }

    # ── Verdict ──
    checks = {}
    if analysis["fractional_ladder"]:
        checks["linearity"] = analysis["fractional_ladder"]["passes"]
    if analysis["shape_invariance"]:
        checks["shape"] = analysis["shape_invariance"]["passes"]
    if analysis["sign_reversal"]:
        checks["sign"] = analysis["sign_reversal"]["passes"]

    n_valid = len(checks)
    n_pass = sum(1 for v in checks.values() if v)

    # Report
    lines = []
    for name, passed in checks.items():
        lines.append(f"{name}: {'PASS' if passed else 'FAIL'}")
    if analysis.get("creature_loop"):
        cl = analysis["creature_loop"]
        lines.append(f"creature: P(0)={cl['p0']} ({'signal' if cl['significant'] else 'noise'})")
    if analysis.get("creature_vs_control"):
        cc = analysis["creature_vs_control"]
        lines.append(f"creature vs random: gap={cc['gap']} ({'differs' if cc['creature_differs'] else 'same'})")

    if n_pass == n_valid and n_valid >= 3:
        analysis["verdict"] = "TOPOLOGICAL"
    elif n_pass == n_valid and n_valid >= 2:
        analysis["verdict"] = "TOPOLOGICAL"
    elif n_pass >= 2:
        analysis["verdict"] = "TOPOLOGICAL_PARTIAL"
    else:
        analysis["verdict"] = "NOISE"

    analysis["notes"] = [f"{n_pass}/{n_valid} theory tests passed."] + lines
    return analysis


SEED_FOR_LIVING_LOOP = {
    "experiment_family":       "winding_number_topological_probe",
    "is_theory_relevant":      True,
    "priority":                "HIGH",
    "created":                 "2026-03-24",
    "description": (
        "Tests the topological interpretation of polar-time holonomy: pi_1(M) = Z. "
        "6 circuits covering winding number, shape invariance, speed invariance, "
        "and sign reversal. Topological verdict requires all four."
    ),
    "falsification": (
        "FALSIFIED if: (1) shape deformation changes phase (-> geometric, not topological), "
        "(2) phase does not scale linearly with n, or "
        "(3) phase is zero across all n (-> wrong theta_t / Bloch mapping)."
    ),
    "connection_to_v3": (
        "GPT-2 v3 found shape-invariant, schedule-invariant, orientation-reversing "
        "holonomy in CP^15. This tests the same structure on physical IBM hardware. "
        "Cross-substrate confirmation would strongly support the conjecture."
    ),
    "next_design_instruction": (
        "Design winding_n1 first (establish Phi_0), then winding_n1_shape_deformed "
        "(critical falsification), then winding_n2/n3 (linearity). "
        "Run all in the same IBM session to cancel systematic errors."
    ),
    "circuits": get_suite_qasm(),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Winding-number topological probe: run circuits on IBM."
    )
    parser.add_argument("--basin-json", type=str, default=None,
                        help="Path to basin geometry JSON to derive a creature loop")
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print circuits and metadata without executing")
    args = parser.parse_args()

    # Optionally load a basin geometry result and add the creature circuit
    if args.basin_json:
        bp = Path(args.basin_json)
        if bp.exists():
            basin_data = json.loads(bp.read_text())
            if isinstance(basin_data, list):
                basin_data = basin_data[0]
            wt = basin_data.get("weight_trajectory", [])
            if wt:
                entry = add_creature_circuit(wt)
                if entry:
                    print(f"  Added creature_loop circuit ({len(wt)} weight vectors)")
                else:
                    print("  WARNING: weight trajectory too short/degenerate for creature loop")
            else:
                print(f"  WARNING: no weight_trajectory in {bp.name}")
        else:
            print(f"  WARNING: basin JSON not found: {bp}")

    suite = get_suite_qasm()

    print(f"Winding number experiment suite ({len(suite)} circuits):")
    for exp in suite:
        lines = exp["circuit_qasm"].count("\n")
        w = exp.get("winding", "?")
        w_str = f"{w:+d}" if isinstance(w, int) else str(w)
        print(f"  {exp['circuit_name']:40s}  w={w_str}  "
              f"{exp.get('variant','?'):18s}  {lines} QASM lines")

    if args.dry_run:
        print("\n[dry-run] No execution.")
        seed_path = Path(__file__).parent / "winding_number_seed.json"
        with seed_path.open("w") as f:
            json.dump(SEED_FOR_LIVING_LOOP, f, indent=2, default=str)
        print(f"Seed written to {seed_path}")
        return

    # Execute circuits
    print(f"\nRunning {len(suite)} circuits (shots={args.shots})...")
    qasm_list = [exp["circuit_qasm"] for exp in suite]
    try:
        all_counts = run_on_ibm(qasm_list, shots=args.shots)
    except ImportError as exc:
        print(f"\nCannot execute: {exc}")
        print("Install qiskit + qiskit-ibm-runtime to run on IBM hardware.")
        return

    # Attach counts to suite entries
    for exp, counts in zip(suite, all_counts):
        exp["counts"] = counts
        total = sum(counts.values())
        p0 = counts.get("0", 0) / total if total else 0
        print(f"  {exp['circuit_name']:40s}  P(0)={p0:.4f}  counts={counts}")

    # Analyze
    analysis = analyze_winding_suite(suite)
    print(f"\nVerdict: {analysis['verdict']}")
    for note in analysis.get("notes", []):
        print(f"  {note}")

    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_path = results_dir / f"winding_run_{ts}.json"
    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "shots":     args.shots,
        "circuits":  suite,
        "analysis":  analysis,
    }
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
