#!/usr/bin/env python3
"""closure_bundle_from_exp_d.py — Build the closure bundle from Experiment D v3 data.

This bridges the gap between the closure bundle framework and the existing
experiment pipeline. It reads the D_v3 result (384-dim centroid trajectories
for both baseline and geometric runs) and constructs closure bundles for each,
then computes their Chern classes.

This is the first measurement of the closure bundle over a REAL training
trajectory — not a synthetic demo, not a static concept sweep.

The D_v3 data has:
  - 6 layers × 13 snapshots (steps 0, 250, 500, ..., 3000)
  - Per-layer centroid_unit vectors (384-dim, unit-normalized)
  - Both baseline (λ=0.0) and geometric (λ=0.5) runs
  - The geometric run shows 1.2% better generalization and prevents angular scatter

The closure bundle measurement asks:
  1. What is the Chern class of each run's bundle?
  2. Does the geometric run have higher Chern number? (More topological structure)
  3. Does the sign stratification persist across training?
  4. Is the founding curvature (L0→L1 concentration) preserved or destroyed?

Run on the Spark:
    cd /home/vybnz69/Vybn
    /home/vybnz69/.venv/spark/bin/python3 closure_bundle_from_exp_d.py

Authors: Vybn & Zoe Dolan
Date: March 23, 2026
"""

import json
import math
import cmath
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "spark" / "growth"))

from closure_bundle import (
    Closure, SortOperatorProfile, EmbeddingContext,
    ClosureBundle, ChernClassMeasurement,
)


# ── Path to Experiment D v3 results ──
EXP_D_V3 = Path(__file__).resolve().parent / "Vybn_Mind" / "experiments" / "holonomic_nemotron" / "results" / "experiment_D_v3_result.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "Vybn_Mind" / "experiments" / "closure_bundle_results"


def load_d_v3():
    if not EXP_D_V3.exists():
        raise FileNotFoundError(
            f"Need {EXP_D_V3} — run run_D_v3.py first.\n"
            f"This file contains the 384-dim centroid trajectories from Experiment D."
        )
    with open(EXP_D_V3) as f:
        return json.load(f)


def extract_run_data(run_data, n_layers=6):
    """Extract per-snapshot, per-layer data from a D_v3 run.

    Returns list of dicts, each with:
      step: int
      layer_centroids: dict[int, np.ndarray]  (layer_idx -> unit centroid)
      layer_angles: dict[int, float]  (layer_idx -> mean angle, if available)
    """
    snaps = run_data.get("snapshots", {})
    result = []

    for step_key, snap in snaps.items():
        if step_key == "final":
            step = 99999
        else:
            try:
                step = int(step_key)
            except ValueError:
                continue

        centroids = {}
        angles = {}
        for i in range(n_layers):
            layer_key = f"layer_{i}"
            if layer_key in snap:
                centroid = snap[layer_key].get("centroid_unit")
                if centroid is not None:
                    centroids[i] = np.array(centroid, dtype=np.float64)
                angle = snap[layer_key].get("mean_angle")
                if angle is not None:
                    angles[i] = float(angle)

        if centroids:
            result.append({
                "step": step,
                "centroids": centroids,
                "angles": angles,
                "val_loss": snap.get("val_loss"),
            })

    result.sort(key=lambda x: x["step"])
    return result


def build_bundle_from_run(run_snapshots, run_label, n_layers=6):
    """Build a ClosureBundle from a sequence of D_v3 snapshots.

    Each snapshot becomes a fiber (Closure). The layer phase profile
    is computed from consecutive centroids using the Pancharatnam phase.
    """
    bundle = ClosureBundle()

    for idx, snap in enumerate(run_snapshots):
        step = snap["step"]
        centroids = snap["centroids"]
        angles = snap["angles"]

        # Compute layer phase profile: Pancharatnam phase between
        # consecutive layer centroids at this snapshot.
        # This is the cross-layer phase (spatial structure).
        layer_phases = []
        for l in range(n_layers - 1):
            if l in centroids and (l + 1) in centroids:
                v1 = centroids[l]
                v2 = centroids[l + 1]
                # For real vectors: inner product gives cos(angle)
                # Pancharatnam phase for real vectors is 0 or π
                # But we use the signed overlap as a continuous proxy
                overlap = float(np.dot(v1, v2))
                # Map to a phase: overlap ∈ [-1, 1] → phase ∈ [-π, π]
                # via arccos of absolute overlap for magnitude,
                # and sign for direction
                magnitude = math.acos(min(abs(overlap), 1.0))
                phase = magnitude if overlap >= 0 else math.pi - magnitude
                layer_phases.append(phase)
            else:
                layer_phases.append(0.0)

        # Sort operator profile: L0→L1 phase and sign
        founding_phase = layer_phases[0] if layer_phases else 0.0
        max_rest = max(abs(p) for p in layer_phases[1:]) if len(layer_phases) > 1 else 1e-10

        # Build the closure
        closure = Closure(
            checkpoint_id=f"{run_label}_step_{step}",
            training_step=step,
            timestamp=datetime.now(timezone.utc).isoformat(),
            sort_profile=SortOperatorProfile(
                concept_phases={"training": founding_phase},
                sign_stratification={"training": 1 if founding_phase >= 0 else -1},
                founding_curvature=abs(founding_phase),
                curvature_concentration=abs(founding_phase) / max(max_rest, 1e-10),
            ),
            embedding_context=EmbeddingContext(
                d_model=len(next(iter(centroids.values()))) if centroids else 0,
                mean_embedding_norm=float(np.mean([np.linalg.norm(v) for v in centroids.values()])),
                effective_dimension=0.0,  # would need full weight matrix
                isotropy=0.0,
            ),
            semantic_holonomy=0.0,  # would need generated text
            layer_phases=layer_phases,
            param_norm=0.0,  # not available from centroid data
        )
        bundle.add_fiber(closure)

    return bundle


def compute_temporal_berry_phases(run_snapshots, n_layers=6):
    """Compute Berry phases along the TRAINING trajectory for each layer.

    This is the temporal holonomy — how the representation at each layer
    evolves across training steps. This is where the nontrivial topology
    should live (per the closure bundle paper's prediction).

    For each layer, we have a sequence of unit centroids ψ_0, ψ_1, ..., ψ_T.
    The Berry phase increment between consecutive steps is:
        γ_t = arg(⟨ψ_t | ψ_{t+1}⟩)

    For real vectors, this is 0 (same hemisphere) or π (sign flip).
    The total Berry phase is Σ γ_t. If it's near 2πn for integer n,
    the Chern number is n.
    """
    results = {}

    for layer_idx in range(n_layers):
        vecs = []
        steps = []
        for snap in run_snapshots:
            if layer_idx in snap["centroids"]:
                vecs.append(snap["centroids"][layer_idx])
                steps.append(snap["step"])

        if len(vecs) < 3:
            continue

        # Berry phase increments
        phases = []
        overlaps = []
        for i in range(len(vecs) - 1):
            overlap = float(np.dot(vecs[i], vecs[i + 1]))
            overlaps.append(overlap)
            # For real vectors, Berry phase is 0 or π
            phase = 0.0 if overlap >= 0 else math.pi
            phases.append(phase)

        # Triplet holonomy (Bargmann invariant)
        triplet_bargmanns = []
        for i in range(len(vecs) - 2):
            o12 = np.dot(vecs[i], vecs[i + 1])
            o23 = np.dot(vecs[i + 1], vecs[i + 2])
            o31 = np.dot(vecs[i + 2], vecs[i])
            triplet_bargmanns.append(float(o12 * o23 * o31))

        total_phase = sum(phases)
        c1_raw = total_phase / (2 * math.pi)

        # Fubini-Study arc lengths
        fs_distances = [math.acos(min(abs(o), 1.0)) for o in overlaps]

        results[f"L{layer_idx}"] = {
            "n_snapshots": len(vecs),
            "steps": steps,
            "total_berry_phase": total_phase,
            "c1_raw": c1_raw,
            "c1_quantized": round(c1_raw),
            "n_sign_flips": sum(1 for o in overlaps if o < 0),
            "mean_overlap": float(np.mean(overlaps)),
            "min_overlap": float(np.min(overlaps)),
            "total_arc_length": float(np.sum(fs_distances)),
            "mean_fs_distance": float(np.mean(fs_distances)),
            "n_negative_bargmann": sum(1 for b in triplet_bargmanns if b < 0),
            "mean_bargmann": float(np.mean(triplet_bargmanns)) if triplet_bargmanns else 0.0,
        }

        print(f"    L{layer_idx}: total Berry phase = {total_phase:.4f} rad "
              f"(c₁ ≈ {c1_raw:.3f}), "
              f"arc-length = {np.sum(fs_distances):.6f}, "
              f"sign flips = {sum(1 for o in overlaps if o < 0)}, "
              f"neg Bargmann = {sum(1 for b in triplet_bargmanns if b < 0)}")

    return results


def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  CLOSURE BUNDLE FROM EXPERIMENT D v3                        ║")
    print("║  First measurement over a real training trajectory          ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    data = load_d_v3()
    print(f"Loaded: {data['experiment']} — {data['description']}")
    print(f"Config: n_embd={data['config']['n_embd']}, n_layers={data['config']['n_layer']}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    full_results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "experiment_D_v3",
        "runs": {},
    }

    runs = data["runs"]

    for run in runs:
        lam = run["lambda_geo"]
        label = f"baseline" if lam == 0.0 else f"geometric_lambda_{lam}"
        print(f"\n{'=' * 60}")
        print(f"  Run: {label} (λ = {lam})")
        print(f"{'=' * 60}")

        snapshots = extract_run_data(run)
        print(f"  Extracted {len(snapshots)} snapshots")

        # Build the cross-layer closure bundle
        print(f"\n  Cross-layer bundle (spatial structure):")
        bundle = build_bundle_from_run(snapshots, label)
        chern = bundle.compute_chern_class()
        print(f"    c₁ = {chern.c1:.4f} (verdict: {chern.verdict})")
        print(f"    Total Berry phase = {chern.total_berry_phase:.4f} rad")
        print(f"    Sign persistence: {bundle.sign_persistence()}")

        slope, trend = bundle.founding_curvature_trend()
        print(f"    Founding curvature trend: {trend} (slope={slope:.6f})")

        # Save the bundle
        bundle_path = OUTPUT_DIR / f"bundle_{label}.jsonl"
        bundle.save(bundle_path)

        # Temporal Berry phases (the training-direction topology)
        print(f"\n  Temporal holonomy (training trajectory topology):")
        temporal = compute_temporal_berry_phases(snapshots)

        # Val loss trajectory
        val_losses = [(s["step"], s.get("val_loss")) for s in snapshots if s.get("val_loss") is not None]

        run_result = {
            "lambda": lam,
            "label": label,
            "n_snapshots": len(snapshots),
            "cross_layer_bundle": {
                "chern_class": chern.to_dict(),
                "sign_persistence": bundle.sign_persistence(),
                "founding_curvature_trend": trend,
            },
            "temporal_holonomy": temporal,
            "val_loss_trajectory": val_losses,
        }
        full_results["runs"][label] = run_result

    # Cross-run comparison
    print(f"\n{'=' * 60}")
    print("COMPARISON: Baseline vs Geometric")
    print(f"{'=' * 60}")

    base = full_results["runs"].get("baseline", {}).get("temporal_holonomy", {})
    geo_key = [k for k in full_results["runs"] if k.startswith("geometric")]
    geo = full_results["runs"].get(geo_key[0], {}).get("temporal_holonomy", {}) if geo_key else {}

    if base and geo:
        print(f"\n  {'Layer':<8} {'Base c₁':<12} {'Geo c₁':<12} {'Base ArcLen':<14} {'Geo ArcLen':<14} {'Base Flips':<12} {'Geo Flips':<12}")
        print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*14} {'-'*14} {'-'*12} {'-'*12}")
        for i in range(6):
            key = f"L{i}"
            if key in base and key in geo:
                b, g = base[key], geo[key]
                print(f"  L{i:<6} {b['c1_raw']:<12.4f} {g['c1_raw']:<12.4f} "
                      f"{b['total_arc_length']:<14.6f} {g['total_arc_length']:<14.6f} "
                      f"{b['n_sign_flips']:<12d} {g['n_sign_flips']:<12d}")

        # The key prediction: geometric run should have MORE topological
        # structure (higher |c₁|) despite LOWER arc-length (smoother trajectory)
        print(f"\n  PREDICTION: geometric run = smoother path + richer topology")
        print(f"  (Lower arc-length but higher |c₁| would confirm the theory)")

    # Save
    result_path = OUTPUT_DIR / "closure_bundle_exp_d_results.json"
    with open(result_path, "w") as f:
        json.dump(full_results, f, indent=2, default=str)
    print(f"\nResults saved to {result_path}")


if __name__ == "__main__":
    main()
