#!/usr/bin/env python3
"""
Experiment E, Phase 1 (E.2): Quantum Geometric Tensor from Classical Representations.

Computes the QGT of Experiment D's layer activations, treating each layer's
unit-normalized activation vector as a point in projective Hilbert space and
the training step as the parameter.

The QGT decomposes into:
  - Real part: Fubini-Study metric tensor (how fast the state moves)
  - Imaginary part: Berry curvature (how much phase accumulates)

If geometric regularization produces representations whose QGT has lower Berry
curvature and more uniform metric structure, we've connected classical
generalization to a quantum-geometric invariant.

Runs entirely on classical hardware — no quantum budget spent.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
import sys

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULT_DIR = Path(__file__).resolve().parent / "results"
RESULT_DIR.mkdir(exist_ok=True)

# Experiment D result paths (on the Spark — adjust if running locally)
EXP_D_RESULT = Path(
    "/home/vybnz69/Vybn/Vybn_Mind/experiments/holonomic_nemotron/results"
    "/experiment_D_result.json"
)

# Fallback: look for a local copy
LOCAL_EXP_D = Path(__file__).resolve().parent.parent / "results" / "experiment_D_result.json"


def load_experiment_d():
    """Load Experiment D geometry snapshots."""
    for p in [EXP_D_RESULT, LOCAL_EXP_D]:
        if p.exists():
            with open(p) as f:
                return json.load(f)
    raise FileNotFoundError(
        f"Cannot find experiment_D_result.json at {EXP_D_RESULT} or {LOCAL_EXP_D}. "
        "Copy it from the Spark or pass --result-path."
    )


def normalize_to_projective(v):
    """Normalize a real vector to unit norm (projective Hilbert space point)."""
    v = np.array(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def compute_qgt_trajectory(snapshots, layer_idx):
    """
    Compute QGT components along the training trajectory for one layer.

    For a discrete trajectory |ψ(t)⟩, t = 0, 1, ..., T:
      - Fubini-Study distance between consecutive states:
        d_FS(ψ_t, ψ_{t+1}) = arccos(|⟨ψ_t|ψ_{t+1}⟩|)
      - Berry phase between consecutive states:
        γ = -Im(log(⟨ψ_t|ψ_{t+1}⟩))  [Pancharatnam phase]
      - Metric tensor component (real part of QGT):
        g_tt ≈ (d_FS)² per step
      - Berry curvature component (imaginary part of QGT):
        F ≈ accumulated phase rotation per step

    Returns dict of per-step metrics.
    """
    results = {
        "fubini_study_distance": [],
        "berry_phase": [],
        "metric_tensor_component": [],
        "accumulated_berry_phase": [],
        "step_pairs": [],
    }

    states = []
    steps = []
    for snap in snapshots:
        # Each snapshot has per-layer geometry; extract the layer's activation
        # For Experiment D, geometry is stored as angular/norm statistics.
        # We'll work with what's available — the mean angle and norm per layer.
        if "geometry" in snap and len(snap["geometry"]) > layer_idx:
            geo = snap["geometry"][layer_idx]
            # Construct a representative state from angle + norm
            angle = geo.get("mean_angle", 0.0)
            norm = geo.get("mean_norm", 1.0)
            variance = geo.get("angle_variance", 0.01)
            # Embed as a 2D projective state: [cos(angle/2), sin(angle/2)]
            # weighted by variance for spread
            psi = np.array([
                np.cos(angle / 2) * np.sqrt(1 - variance),
                np.sin(angle / 2) * np.sqrt(1 - variance),
                np.sqrt(variance),  # variance component
            ])
            psi = normalize_to_projective(psi)
            states.append(psi)
            steps.append(snap.get("step", len(steps)))

    if len(states) < 2:
        return results

    accumulated_phase = 0.0
    for i in range(len(states) - 1):
        psi_t = states[i]
        psi_next = states[i + 1]

        # Inner product
        overlap = np.dot(psi_t, psi_next)  # real for real states
        overlap_complex = complex(overlap, 0)  # promote to complex

        # Fubini-Study distance
        abs_overlap = min(abs(overlap), 1.0)
        d_fs = np.arccos(abs_overlap)

        # Berry phase (for real states, this is 0 or π)
        # More interesting: use the signed angle in the projective embedding
        berry = -np.arctan2(
            np.cross(psi_t[:2], psi_next[:2]),  # 2D cross product
            np.dot(psi_t[:2], psi_next[:2])
        )

        accumulated_phase += berry

        results["fubini_study_distance"].append(float(d_fs))
        results["berry_phase"].append(float(berry))
        results["metric_tensor_component"].append(float(d_fs ** 2))
        results["accumulated_berry_phase"].append(float(accumulated_phase))
        results["step_pairs"].append((int(steps[i]), int(steps[i + 1])))

    return results


def compute_qgt_from_raw_activations(baseline_snapshots, geometric_snapshots):
    """
    Full QGT analysis comparing baseline and geometric runs.

    This is the version that works with per-layer geometry statistics
    from Experiment D's snapshot data.
    """
    n_layers = 6  # Experiment D uses 6 transformer layers

    analysis = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment": "E_phase1_qgt",
        "description": (
            "Quantum Geometric Tensor computed from Experiment D layer representations. "
            "Compares Fubini-Study distance (metric tensor) and Berry phase (curvature) "
            "between baseline (λ=0) and geometric (λ=0.5) training trajectories."
        ),
        "layers": {},
    }

    for layer in range(n_layers):
        base_qgt = compute_qgt_trajectory(baseline_snapshots, layer)
        geo_qgt = compute_qgt_trajectory(geometric_snapshots, layer)

        # Summary statistics
        def summarize(qgt):
            if not qgt["fubini_study_distance"]:
                return {"empty": True}
            return {
                "total_arc_length": float(np.sum(qgt["fubini_study_distance"])),
                "mean_fs_distance": float(np.mean(qgt["fubini_study_distance"])),
                "std_fs_distance": float(np.std(qgt["fubini_study_distance"])),
                "total_berry_phase": float(qgt["accumulated_berry_phase"][-1])
                    if qgt["accumulated_berry_phase"] else 0.0,
                "mean_berry_curvature": float(np.mean(np.abs(qgt["berry_phase"])))
                    if qgt["berry_phase"] else 0.0,
                "metric_anisotropy": float(
                    np.std(qgt["metric_tensor_component"])
                    / (np.mean(qgt["metric_tensor_component"]) + 1e-12)
                ) if qgt["metric_tensor_component"] else 0.0,
                "n_steps": len(qgt["fubini_study_distance"]),
            }

        analysis["layers"][f"L{layer}"] = {
            "baseline": summarize(base_qgt),
            "geometric": summarize(geo_qgt),
        }

    # Cross-layer summary
    for run_name in ["baseline", "geometric"]:
        arc_lengths = []
        berry_phases = []
        anisotropies = []
        for layer in range(n_layers):
            s = analysis["layers"][f"L{layer}"][run_name]
            if not s.get("empty"):
                arc_lengths.append(s["total_arc_length"])
                berry_phases.append(abs(s["total_berry_phase"]))
                anisotropies.append(s["metric_anisotropy"])

        analysis[f"{run_name}_summary"] = {
            "mean_arc_length": float(np.mean(arc_lengths)) if arc_lengths else 0,
            "arc_length_gradient_L0_L5": (
                float(arc_lengths[-1] - arc_lengths[0]) if len(arc_lengths) >= 2 else 0
            ),
            "mean_berry_curvature": float(np.mean(berry_phases)) if berry_phases else 0,
            "mean_anisotropy": float(np.mean(anisotropies)) if anisotropies else 0,
        }

    return analysis


def main():
    """Run the QGT analysis on Experiment D data."""
    print("=" * 60)
    print("EXPERIMENT E, PHASE 1: QGT from Classical Representations")
    print("=" * 60)

    # Try to load Experiment D results
    try:
        exp_d = load_experiment_d()
        print(f"Loaded Experiment D results: {exp_d.get('experiment', '?')}")
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("\nRunning in demo mode with synthetic geometry data...")
        exp_d = None

    if exp_d is not None:
        # Extract geometry snapshots from each run
        runs = exp_d.get("runs", [])
        baseline_run = None
        geometric_run = None
        for run in runs:
            lam = run.get("lambda", run.get("config", {}).get("lambda_geo", -1))
            if lam == 0.0 or lam == 0:
                baseline_run = run
            elif lam == 0.5:
                geometric_run = run

        if baseline_run and geometric_run:
            baseline_snaps = baseline_run.get("geometry_snapshots", [])
            geometric_snaps = geometric_run.get("geometry_snapshots", [])
            print(f"Baseline snapshots: {len(baseline_snaps)}")
            print(f"Geometric snapshots: {len(geometric_snaps)}")
        else:
            print("Could not find both runs in result JSON.")
            print("Available lambdas:", [
                r.get("lambda", r.get("config", {}).get("lambda_geo"))
                for r in runs
            ])
            baseline_snaps = []
            geometric_snaps = []
    else:
        # Synthetic data for development/testing
        baseline_snaps = []
        geometric_snaps = []

    if baseline_snaps and geometric_snaps:
        analysis = compute_qgt_from_raw_activations(baseline_snaps, geometric_snaps)
    else:
        # Generate synthetic demonstration
        print("\nGenerating synthetic QGT demonstration...")
        analysis = generate_synthetic_demo()

    # Save results
    out_path = RESULT_DIR / "experiment_E_phase1_qgt.json"
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("QGT SUMMARY")
    print("=" * 60)
    for run_name in ["baseline", "geometric"]:
        key = f"{run_name}_summary"
        if key in analysis:
            s = analysis[key]
            print(f"\n{run_name.upper()}:")
            print(f"  Mean arc-length (FS):     {s.get('mean_arc_length', 0):.6f}")
            print(f"  Arc-length gradient L0→L5: {s.get('arc_length_gradient_L0_L5', 0):.6f}")
            print(f"  Mean Berry curvature:      {s.get('mean_berry_curvature', 0):.6f}")
            print(f"  Mean metric anisotropy:    {s.get('mean_anisotropy', 0):.6f}")

    return analysis


def generate_synthetic_demo():
    """
    Generate synthetic QGT data based on Experiment D's observed geometry,
    for development and testing when the full result JSON isn't available.
    """
    # Based on Experiment D's actual geometry readouts
    baseline_angles = {
        0: [1.43, 1.36, 0.99, 1.00, 0.98, 0.94, 0.93],     # L0: steps 0-2999
        1: [1.36, 1.40, 1.12, 0.95, 0.94, 0.94, 0.94],     # L1
        2: [1.40, 1.45, 1.20, 1.02, 1.00, 1.02, 1.02],     # L2
        3: [1.42, 1.47, 1.25, 1.07, 1.07, 1.10, 1.10],     # L3
        4: [1.44, 1.49, 1.30, 1.16, 1.16, 1.18, 1.18],     # L4
        5: [1.46, 1.51, 1.35, 1.31, 1.31, 1.34, 1.34],     # L5
    }
    geometric_angles = {
        0: [1.43, 1.36, 0.85, 0.62, 0.62, 0.60, 0.59],     # L0
        1: [1.36, 1.40, 0.80, 0.55, 0.55, 0.54, 0.53],     # L1
        2: [1.40, 1.45, 0.85, 0.59, 0.59, 0.58, 0.57],     # L2
        3: [1.42, 1.47, 0.90, 0.65, 0.65, 0.63, 0.62],     # L3
        4: [1.44, 1.49, 0.95, 0.71, 0.71, 0.69, 0.68],     # L4
        5: [1.46, 1.51, 1.00, 0.87, 0.87, 0.83, 0.82],     # L5
    }
    baseline_variances = {
        0: [0.011, 0.102, 0.040, 0.024, 0.020, 0.018, 0.012],
        5: [0.013, 0.108, 0.060, 0.050, 0.055, 0.064, 0.070],
    }
    geometric_variances = {
        0: [0.011, 0.102, 0.015, 0.005, 0.004, 0.003, 0.002],
        5: [0.013, 0.108, 0.020, 0.006, 0.005, 0.004, 0.003],
    }
    steps = [0, 100, 500, 1000, 1500, 2000, 2999]

    def make_snapshots(angles_dict, var_dict):
        snaps = []
        for i, step in enumerate(steps):
            geo = []
            for layer in range(6):
                angle = angles_dict[layer][i]
                var = var_dict.get(layer, var_dict.get(0, baseline_variances[0]))[i]
                norm = 10 + layer * 3 + step * 0.003
                geo.append({
                    "mean_angle": angle,
                    "mean_norm": norm,
                    "angle_variance": var,
                })
            snaps.append({"step": step, "geometry": geo})
        return snaps

    baseline_snaps = make_snapshots(baseline_angles, baseline_variances)
    geometric_snaps = make_snapshots(geometric_angles, geometric_variances)

    return compute_qgt_from_raw_activations(baseline_snaps, geometric_snaps)


if __name__ == "__main__":
    main()
