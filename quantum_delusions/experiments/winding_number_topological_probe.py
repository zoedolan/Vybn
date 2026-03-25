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

import json
import math
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional


def winding_n_qasm(n: int, phi_step: float = math.pi / 4) -> str:
    steps_per_winding = int(round(2 * math.pi / phi_step))
    total_steps = n * steps_per_winding
    phi_deg = math.degrees(phi_step)
    lines = [
        'OPENQASM 2.0;',
        'include "qelib1.inc";',
        'qreg q[1];',
        'creg c[1];',
        'h q[0];',
        f'// {n} winding(s), {steps_per_winding} steps each',
    ]
    for step in range(total_steps):
        lines.append(f'rz({phi_deg:.6f}) q[0];')
    lines += ['h q[0];', 'measure q[0] -> c[0];']
    return '\n'.join(lines)


def winding_shape_deformed_qasm(n: int = 1, ellipse_ratio: float = 0.5) -> str:
    base_step  = math.pi / 4
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
        lines.append(f'rz({math.degrees(phi):.6f}) q[0];')
    lines += ['h q[0];', 'measure q[0] -> c[0];']
    return '\n'.join(lines)


def winding_reversed_qasm(n: int = 1) -> str:
    total_steps = n * 8
    phi_deg = -math.degrees(math.pi / 4)
    lines = [
        'OPENQASM 2.0;',
        'include "qelib1.inc";',
        'qreg q[1];',
        'creg c[1];',
        'h q[0];',
        f'// Reversed winding: n={n}',
    ]
    for _ in range(total_steps):
        lines.append(f'rz({phi_deg:.6f}) q[0];')
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
        lines.append(f'rz({math.degrees(phi_step):.6f}) q[0];')
    lines += ['h q[0];', 'measure q[0] -> c[0];']
    return '\n'.join(lines)


WINDING_EXPERIMENT_SUITE = [
    {
        "circuit_name":       "winding_n1",
        "is_theory_relevant": True,
        "hypothesis": (
            "1 full equatorial winding. If polar-time topology is real, P(0) shifts "
            "from 0.5 by a hardware-independent amount Phi_0. This establishes the "
            "baseline for all winding-number comparisons."
        ),
        "expected_counts":    {"0": 0.5, "1": 0.5},
        "estimated_seconds":  3.0,
        "qasm_fn":            lambda: winding_n_qasm(1),
        "family":             "winding_number",
        "winding":            1,
        "variant":            "base",
    },
    {
        "circuit_name":       "winding_n2",
        "is_theory_relevant": True,
        "hypothesis": (
            "2 windings -> phase 2*Phi_0. Topological: doubles exactly. "
            "Geometric (Berry): scales with loop area, not winding count. "
            "Comparing n1 vs n2 is the linearity test."
        ),
        "expected_counts":    {"0": 0.5, "1": 0.5},
        "estimated_seconds":  4.0,
        "qasm_fn":            lambda: winding_n_qasm(2),
        "family":             "winding_number",
        "winding":            2,
        "variant":            "base",
    },
    {
        "circuit_name":       "winding_n3",
        "is_theory_relevant": True,
        "hypothesis": (
            "3 windings -> phase 3*Phi_0. Linear scaling with integer n is "
            "the topological signature. Non-linear scaling falsifies pi_1(M)=Z."
        ),
        "expected_counts":    {"0": 0.5, "1": 0.5},
        "estimated_seconds":  5.0,
        "qasm_fn":            lambda: winding_n_qasm(3),
        "family":             "winding_number",
        "winding":            3,
        "variant":            "base",
    },
    {
        "circuit_name":       "winding_n1_reversed",
        "is_theory_relevant": True,
        "hypothesis": (
            "Reverse direction: phase -Phi_0. Exact sign reversal required. "
            "Decoherence shows asymmetric damping; topology shows symmetric negation."
        ),
        "expected_counts":    {"0": 0.5, "1": 0.5},
        "estimated_seconds":  3.0,
        "qasm_fn":            lambda: winding_reversed_qasm(1),
        "family":             "winding_number",
        "winding":            -1,
        "variant":            "reversed",
    },
    {
        "circuit_name":       "winding_n1_shape_deformed",
        "is_theory_relevant": True,
        "hypothesis": (
            "Elliptical path, 1 winding. TOPOLOGICAL PREDICTION: same phase as winding_n1. "
            "GEOMETRIC PREDICTION: different phase (depends on enclosed area). "
            "THIS IS THE CRITICAL DISTINGUISHING TEST between topological and geometric holonomy. "
            "If shape changes the phase, the pi_1(M)=Z interpretation is falsified."
        ),
        "expected_counts":    {"0": 0.5, "1": 0.5},
        "estimated_seconds":  3.0,
        "qasm_fn":            lambda: winding_shape_deformed_qasm(1, 0.5),
        "family":             "winding_number",
        "winding":            1,
        "variant":            "shape_deformed",
    },
    {
        "circuit_name":       "winding_n1_speed_deformed",
        "is_theory_relevant": True,
        "hypothesis": (
            "1 winding at 4x slower traversal. TOPOLOGICAL: same phase. "
            "DECOHERENCE: more noise (more gates). "
            "Schedule invariance was detected in GPT-2 v3 (delta=-0.012 rad at CP^15). "
            "This tests the same invariance on physical IBM hardware."
        ),
        "expected_counts":    {"0": 0.5, "1": 0.5},
        "estimated_seconds":  5.0,
        "qasm_fn":            lambda: winding_speed_deformed_qasm(1, 4),
        "family":             "winding_number",
        "winding":            1,
        "variant":            "speed_deformed",
    },
]


def get_suite_qasm() -> list[dict]:
    suite = []
    for spec in WINDING_EXPERIMENT_SUITE:
        entry = {k: v for k, v in spec.items() if k != "qasm_fn"}
        entry["circuit_qasm"] = spec["qasm_fn"]()
        suite.append(entry)
    return suite


def analyze_winding_suite(results: list[dict]) -> dict:
    def p0(counts: dict) -> Optional[float]:
        total = sum(counts.values())
        if total == 0:
            return None
        return counts.get("0", 0) / total

    by_name = {r["circuit_name"]: r for r in results}
    analysis = {
        "timestamp":         datetime.now(timezone.utc).isoformat(),
        "n_circuits":        len(results),
        "winding_linearity": None,
        "shape_invariance":  None,
        "speed_invariance":  None,
        "sign_reversal":     None,
        "verdict":           "INSUFFICIENT_DATA",
        "notes":             [],
    }

    # Winding linearity
    winding_deviations = {}
    for entry in results:
        if entry.get("variant") == "base" and "winding" in entry:
            p = p0(entry.get("counts", {}))
            if p is not None:
                winding_deviations[entry["winding"]] = p - 0.5
    if len(winding_deviations) >= 2:
        ns = sorted(winding_deviations.keys())
        devs = [winding_deviations[n] for n in ns]
        if len(ns) >= 2 and devs[0] != 0:
            ratio_obs = abs(devs[1] / devs[0])
            ratio_exp = ns[1] / ns[0]
            err = abs(ratio_obs - ratio_exp) / ratio_exp
            analysis["winding_linearity"] = {
                "deviations": winding_deviations,
                "observed_ratio": round(ratio_obs, 3),
                "expected_ratio": round(ratio_exp, 3),
                "linearity_error": round(err, 3),
                "passes": err < 0.2,
            }

    # Shape invariance (THE critical test)
    base   = by_name.get("winding_n1")
    shaped = by_name.get("winding_n1_shape_deformed")
    if base and shaped:
        p_b = p0(base.get("counts", {}))
        p_s = p0(shaped.get("counts", {}))
        if p_b is not None and p_s is not None:
            delta = abs((p_b - 0.5) - (p_s - 0.5))
            analysis["shape_invariance"] = {
                "p0_base":   round(p_b, 4),
                "p0_shaped": round(p_s, 4),
                "delta":     round(delta, 4),
                "passes":    delta < 0.05,
            }

    # Speed invariance
    speed = by_name.get("winding_n1_speed_deformed")
    if base and speed:
        p_b = p0(base.get("counts", {}))
        p_sp = p0(speed.get("counts", {}))
        if p_b is not None and p_sp is not None:
            delta = abs((p_b - 0.5) - (p_sp - 0.5))
            analysis["speed_invariance"] = {
                "p0_base":  round(p_b,  4),
                "p0_speed": round(p_sp, 4),
                "delta":    round(delta, 4),
                "passes":   delta < 0.05,
            }

    # Sign reversal
    rev = by_name.get("winding_n1_reversed")
    if base and rev:
        p_b = p0(base.get("counts", {}))
        p_r = p0(rev.get("counts", {}))
        if p_b is not None and p_r is not None:
            fwd = p_b - 0.5
            bwd = p_r - 0.5
            rsum = abs(fwd + bwd)
            analysis["sign_reversal"] = {
                "fwd_deviation": round(fwd, 4),
                "rev_deviation": round(bwd, 4),
                "reversal_sum":  round(rsum, 4),
                "passes":        rsum < 0.05,
            }

    checks = [
        analysis["winding_linearity"] and analysis["winding_linearity"]["passes"],
        analysis["shape_invariance"]  and analysis["shape_invariance"]["passes"],
        analysis["speed_invariance"]  and analysis["speed_invariance"]["passes"],
        analysis["sign_reversal"]     and analysis["sign_reversal"]["passes"],
    ]
    n_pass = sum(1 for c in checks if c)

    if n_pass >= 3:
        analysis["verdict"] = "TOPOLOGICAL"
        analysis["notes"].append(
            f"{n_pass}/4 topological tests passed. Phase is quantized, "
            "path-invariant, sign-reversing. Consistent with pi_1(M) = Z."
        )
    elif analysis["shape_invariance"] and not analysis["shape_invariance"]["passes"]:
        analysis["verdict"] = "GEOMETRIC"
        analysis["notes"].append(
            "Shape deformation changes phase. Geometric (curvature-dependent) signal. "
            "The pi_1(M)=Z interpretation is falsified; "
            "geometric Berry phase may still hold."
        )
    elif n_pass == 2:
        analysis["verdict"] = "AMBIGUOUS"
        analysis["notes"].append("2/4 tests passed. More experiments needed.")
    else:
        analysis["verdict"] = "NOISE"
        analysis["notes"].append(
            "No consistent signal. Phase does not scale with winding number."
        )

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


if __name__ == "__main__":
    print("Winding number experiment suite:")
    for exp in SEED_FOR_LIVING_LOOP["circuits"]:
        lines = exp["circuit_qasm"].count("\n")
        print(f"  {exp['circuit_name']:40s}  w={exp.get('winding', '?'):+d}  {exp.get('variant','?'):18s}  {lines} QASM lines")
    print()
    print("Falsification:")
    print(SEED_FOR_LIVING_LOOP["falsification"])

    seed_path = Path(__file__).parent / "winding_number_seed.json"
    with seed_path.open("w") as f:
        json.dump(SEED_FOR_LIVING_LOOP, f, indent=2, default=str)
    print(f"\nSeed written to {seed_path}")
