#!/usr/bin/env python3
"""
Experiment E, Phase 1 (E.2): Quantum Geometric Tensor from REAL Activations.

This version uses the 384-dim centroid vectors saved by run_D_v3.py —
no embedding choice, no information loss. The QGT is computed directly
on the actual projective geometry of the network's representation trajectory.

For a sequence of unit-normalized centroids ψ_0, ψ_1, ..., ψ_T ∈ S^{383}:
  - Fubini-Study distance:  d_FS(ψ_t, ψ_{t+1}) = arccos(|⟨ψ_t|ψ_{t+1}⟩|)
  - Berry phase (Pancharatnam): γ_t = arg(⟨ψ_t|ψ_{t+1}⟩)
  - Total holonomy: Φ = sum of Berry phases around the trajectory

The prediction from the DESIGN.md:
  - Geometric run (λ=0.5): lower Berry curvature, more uniform metric, lower anisotropy
  - Baseline run (λ=0.0): increasing anisotropy over training, higher Berry curvature
  - If the QGT's Berry curvature correlates with the generalization gap, we've connected
    classical generalization to a topological invariant.

Run:
    /home/vybnz69/.venv/spark/bin/python3 experiment_E/qgt_from_centroids.py
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

RESULT_DIR = Path(__file__).resolve().parent / "results"
RESULT_DIR.mkdir(exist_ok=True)

EXP_D_V3 = Path(__file__).resolve().parent.parent.parent / "results" / "experiment_D_v3_result.json"


def load_data():
    if not EXP_D_V3.exists():
        raise FileNotFoundError(f"Need {EXP_D_V3} — run run_D_v3.py first")
    with open(EXP_D_V3) as f:
        return json.load(f)


def extract_trajectory(run_data, n_layers=6):
    """Extract per-layer trajectories of unit-normalized centroids.
    
    Returns: dict[layer_idx] -> list of (step, centroid_unit) tuples, sorted by step.
    """
    snaps = run_data["snapshots"]
    trajectories = {i: [] for i in range(n_layers)}
    
    for step_key, snap in snaps.items():
        if step_key == "final":
            step = 99999  # sort to end
        else:
            step = int(step_key)
        for i in range(n_layers):
            layer_key = f"layer_{i}"
            centroid_unit = snap[layer_key].get("centroid_unit")
            if centroid_unit is not None:
                trajectories[i].append((step, np.array(centroid_unit, dtype=np.float64)))
    
    for i in range(n_layers):
        trajectories[i].sort(key=lambda x: x[0])
    
    return trajectories


def compute_qgt_along_trajectory(trajectory):
    """Compute QGT components for a trajectory of unit vectors in R^d.
    
    Since the centroids are real-valued (not complex), the Berry phase
    per step is either 0 or π (from sign flips in the inner product).
    But the Fubini-Study distance captures the full projective geometry.
    
    For the Berry curvature to be non-trivial, we compute the holonomy
    around triangles of consecutive triplets — the discrete analog of
    the curvature 2-form.
    """
    steps = [s for s, _ in trajectory]
    vecs = [v for _, v in trajectory]
    n = len(vecs)
    
    if n < 2:
        return None
    
    result = {
        "steps": steps,
        "n_points": n,
        "dim": len(vecs[0]),
        # Per-step metrics
        "fs_distances": [],          # Fubini-Study distance per step
        "overlaps": [],              # |⟨ψ_t|ψ_{t+1}⟩| per step
        "signed_overlaps": [],       # ⟨ψ_t|ψ_{t+1}⟩ per step (can be negative)
        "step_pairs": [],
        # Curvature: holonomy around consecutive triplets
        "triplet_phases": [],        # arg(⟨ψ_t|ψ_{t+1}⟩⟨ψ_{t+1}|ψ_{t+2}⟩⟨ψ_{t+2}|ψ_t⟩)
        "triplet_steps": [],
    }
    
    # Consecutive pair metrics
    for i in range(n - 1):
        overlap = float(np.dot(vecs[i], vecs[i+1]))
        abs_overlap = min(abs(overlap), 1.0)
        d_fs = float(np.arccos(abs_overlap))
        
        result["fs_distances"].append(d_fs)
        result["overlaps"].append(abs_overlap)
        result["signed_overlaps"].append(float(overlap))
        result["step_pairs"].append((steps[i], steps[i+1]))
    
    # Triplet holonomy — the real test of curvature
    # For real vectors, the Bargmann invariant of a triangle is:
    #   Δ_3 = ⟨ψ_1|ψ_2⟩⟨ψ_2|ψ_3⟩⟨ψ_3|ψ_1⟩
    # Its argument is the solid angle (geometric phase) of the triangle
    # on the projective space. For real vectors this is 0 or π, but the
    # SIGN of the triple product carries the curvature information.
    for i in range(n - 2):
        o12 = np.dot(vecs[i], vecs[i+1])
        o23 = np.dot(vecs[i+1], vecs[i+2])
        o31 = np.dot(vecs[i+2], vecs[i])
        bargmann = o12 * o23 * o31
        # For real vectors: phase is 0 if bargmann > 0, π if bargmann < 0
        # The magnitude |bargmann| measures how far from degenerate the triangle is
        phase = 0.0 if bargmann >= 0 else np.pi
        result["triplet_phases"].append({
            "phase": float(phase),
            "bargmann_invariant": float(bargmann),
            "abs_bargmann": float(abs(bargmann)),
        })
        result["triplet_steps"].append((steps[i], steps[i+1], steps[i+2]))
    
    # Summary statistics
    fs = np.array(result["fs_distances"])
    result["summary"] = {
        "total_arc_length": float(np.sum(fs)),
        "mean_fs_distance": float(np.mean(fs)),
        "std_fs_distance": float(np.std(fs)),
        "max_fs_distance": float(np.max(fs)),
        "min_fs_distance": float(np.min(fs)),
        "arc_length_anisotropy": float(np.std(fs) / (np.mean(fs) + 1e-12)),
        # Curvature summary
        "n_sign_flips": int(sum(1 for o in result["signed_overlaps"] if o < 0)),
        "mean_overlap": float(np.mean(result["overlaps"])),
        "n_negative_bargmann": int(sum(
            1 for t in result["triplet_phases"] if t["bargmann_invariant"] < 0
        )),
        "mean_abs_bargmann": float(np.mean([
            t["abs_bargmann"] for t in result["triplet_phases"]
        ])) if result["triplet_phases"] else 0.0,
    }
    
    return result


def main():
    print("=" * 70)
    print("EXPERIMENT E, PHASE 1: QGT from 384-dim Activation Centroids")
    print("=" * 70)
    
    data = load_data()
    print(f"Loaded: {data['experiment']} — {data['description']}")
    n_embd = data["config"]["n_embd"]
    print(f"Activation dimension: {n_embd}")
    
    runs = data["runs"]
    analysis = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment": "E_phase1_qgt_real",
        "source": "experiment_D_v3 (384-dim centroids)",
        "activation_dim": n_embd,
        "runs": {},
    }
    
    for run in runs:
        lam = run["lambda_geo"]
        tag = f"lambda_{lam}"
        print(f"\n{'='*50}")
        print(f"  Run: λ = {lam}")
        print(f"{'='*50}")
        
        trajectories = extract_trajectory(run)
        run_analysis = {}
        
        for layer_idx in range(6):
            traj = trajectories[layer_idx]
            print(f"\n  Layer {layer_idx}: {len(traj)} snapshots, dim={len(traj[0][1]) if traj else '?'}")
            
            qgt = compute_qgt_along_trajectory(traj)
            if qgt is None:
                continue
            
            s = qgt["summary"]
            print(f"    Total arc-length (FS):    {s['total_arc_length']:.6f}")
            print(f"    Mean FS distance/step:    {s['mean_fs_distance']:.6f}")
            print(f"    Arc-length anisotropy:    {s['arc_length_anisotropy']:.6f}")
            print(f"    Mean overlap:             {s['mean_overlap']:.6f}")
            print(f"    Sign flips:               {s['n_sign_flips']}")
            print(f"    Negative Bargmann:        {s['n_negative_bargmann']}")
            print(f"    Mean |Bargmann|:          {s['mean_abs_bargmann']:.6f}")
            
            run_analysis[f"L{layer_idx}"] = qgt
        
        analysis["runs"][tag] = run_analysis
    
    # Cross-run comparison
    print(f"\n{'='*70}")
    print("CROSS-RUN COMPARISON: Baseline vs Geometric")
    print(f"{'='*70}")
    
    base = analysis["runs"].get("lambda_0.0", {})
    geo = analysis["runs"].get("lambda_0.5", {})
    
    print(f"\n  {'Layer':<8} {'Base ArcLen':<14} {'Geo ArcLen':<14} {'Base Aniso':<14} {'Geo Aniso':<14} {'Base |Barg|':<14} {'Geo |Barg|':<14}")
    print(f"  {'-'*8} {'-'*14} {'-'*14} {'-'*14} {'-'*14} {'-'*14} {'-'*14}")
    
    comparison = {}
    for i in range(6):
        key = f"L{i}"
        if key in base and key in geo:
            bs = base[key]["summary"]
            gs = geo[key]["summary"]
            print(f"  L{i:<6} {bs['total_arc_length']:<14.6f} {gs['total_arc_length']:<14.6f} "
                  f"{bs['arc_length_anisotropy']:<14.6f} {gs['arc_length_anisotropy']:<14.6f} "
                  f"{bs['mean_abs_bargmann']:<14.6f} {gs['mean_abs_bargmann']:<14.6f}")
            comparison[key] = {
                "arc_length_ratio": gs['total_arc_length'] / (bs['total_arc_length'] + 1e-12),
                "anisotropy_ratio": gs['arc_length_anisotropy'] / (bs['arc_length_anisotropy'] + 1e-12),
                "bargmann_ratio": gs['mean_abs_bargmann'] / (bs['mean_abs_bargmann'] + 1e-12),
            }
    
    analysis["comparison"] = comparison
    
    # Prediction check
    print(f"\n  PREDICTION CHECK:")
    print(f"  Design.md predicted geometric run should show:")
    print(f"    ✓/✗ Lower Berry curvature (fewer sign flips, fewer negative Bargmann)")
    print(f"    ✓/✗ More uniform metric structure (lower anisotropy)")
    print(f"    ✓/✗ The effect should be strongest at deep layers (L4, L5)")
    
    if base and geo:
        # Check predictions
        for i in range(6):
            key = f"L{i}"
            if key in base and key in geo:
                bs = base[key]["summary"]
                gs = geo[key]["summary"]
                aniso_lower = gs["arc_length_anisotropy"] < bs["arc_length_anisotropy"]
                barg_lower = gs["n_negative_bargmann"] <= bs["n_negative_bargmann"]
                mark_a = "✓" if aniso_lower else "✗"
                mark_b = "✓" if barg_lower else "✗"
                print(f"    L{i}: anisotropy {mark_a} (geo={gs['arc_length_anisotropy']:.4f} vs base={bs['arc_length_anisotropy']:.4f}), "
                      f"curvature {mark_b} (geo neg_barg={gs['n_negative_bargmann']} vs base={bs['n_negative_bargmann']})")
    
    # Save
    out_path = RESULT_DIR / "experiment_E_phase1_qgt_real.json"
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nResults saved to {out_path}")
    
    return analysis


if __name__ == "__main__":
    main()
