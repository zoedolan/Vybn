#!/usr/bin/env python3
"""
creature_quantum_bridge.py — Connect the creature's classical learning
geometry to the quantum winding number probe.

The creature trains in a rotor-modulated weight space (Cl(3,0) geometric
algebra). Its weight trajectory during basin convergence traces a path
through ~4K-dimensional parameter space. This bridge:

  1. Loads a basin geometry result (weight_trajectory from experiment_basin_geometry)
  2. PCA-projects the trajectory to 2D
  3. Encodes the projected path as Bloch-sphere rotations (rz/ry gates)
  4. Submits the creature-loop circuit alongside the theory winding circuits
     to IBM quantum hardware

If the creature's learning path has non-trivial topological winding,
the circuit will show P(0) deviation from 0.5. If the path is open or
unwound, P(0) stays near 0.5 — which is itself informative.

The two substrates being compared:
  - GPT-2 v3: shape-invariant holonomy in CP^15 representational geometry
  - IBM hardware: winding-number-dependent phase in physical qubit rotations
  - Creature: weight-space trajectory from Cl(3,0) rotor-modulated training

Cross-substrate topological invariance is the thesis.

Usage:
  python creature_quantum_bridge.py scan                    # find all basin results
  python creature_quantum_bridge.py build <basin.json>      # generate creature QASM
  python creature_quantum_bridge.py run <basin.json>        # run full suite on IBM
  python creature_quantum_bridge.py run <basin.json> --dry-run  # inspect without executing
"""

import argparse
import json
import math
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

# Basin results live here
BASIN_RESULTS_DIR = REPO_ROOT / "Vybn_Mind" / "creature_dgm_h" / "experiment_results" / "basin_geometry"

# Import from the winding probe
sys.path.insert(0, str(SCRIPT_DIR))
from winding_number_topological_probe import (
    add_creature_circuit,
    get_suite_qasm,
    run_on_ibm,
    analyze_winding_suite,
    trajectory_to_bloch_angles,
    WINDING_EXPERIMENT_SUITE,
)


def find_basin_results() -> list[Path]:
    """Scan for basin geometry result files containing weight trajectories."""
    results = []
    if not BASIN_RESULTS_DIR.exists():
        return results
    for f in sorted(BASIN_RESULTS_DIR.glob("basin_*.json")):
        try:
            data = json.loads(f.read_text())
            if isinstance(data, list):
                data = data[0]
            if data.get("weight_trajectory"):
                results.append(f)
        except Exception:
            pass
    return results


def analyze_trajectory(weight_trajectory: list[list[float]]) -> dict:
    """Characterise the weight trajectory before encoding."""
    W = np.array(weight_trajectory, dtype=np.float64)
    norms = np.linalg.norm(W, axis=1)
    
    # PCA for 2D projection
    W_c = W - W.mean(axis=0)
    U, S, Vt = np.linalg.svd(W_c, full_matrices=False)
    proj = W_c @ Vt[:2].T
    var_explained = (S[:2] ** 2).sum() / max((S ** 2).sum(), 1e-12)
    
    # Compute winding number of the 2D projection
    angles = np.arctan2(proj[:, 1], proj[:, 0])
    dtheta = np.diff(angles)
    # Unwrap angle jumps
    dtheta = np.where(dtheta > math.pi, dtheta - 2*math.pi, dtheta)
    dtheta = np.where(dtheta < -math.pi, dtheta + 2*math.pi, dtheta)
    winding = float(np.sum(dtheta)) / (2 * math.pi)
    
    return {
        "n_steps":          len(weight_trajectory),
        "param_dim":        W.shape[1],
        "norm_start":       round(float(norms[0]), 4),
        "norm_end":         round(float(norms[-1]), 4),
        "norm_mean":        round(float(norms.mean()), 4),
        "norm_std":         round(float(norms.std()), 4),
        "pca_var_explained": round(float(var_explained), 4),
        "estimated_winding": round(winding, 3),
        "path_closed":      bool(np.linalg.norm(W[0] - W[-1]) < 0.1 * norms.mean()),
    }


def cmd_scan(args):
    """List all basin results with weight trajectories."""
    results = find_basin_results()
    if not results:
        print(f"No basin results found in {BASIN_RESULTS_DIR}")
        print("Run: python experiments.py basin  (in creature_dgm_h/)")
        return
    
    print(f"Found {len(results)} basin result(s) in {BASIN_RESULTS_DIR}:\n")
    for f in results:
        data = json.loads(f.read_text())
        if isinstance(data, list):
            for i, agent in enumerate(data):
                wt = agent.get("weight_trajectory", [])
                if wt:
                    info = analyze_trajectory(wt)
                    print(f"  {f.name} [agent {i}]:")
                    print(f"    steps={info['n_steps']}  dim={info['param_dim']}")
                    print(f"    norm: {info['norm_start']:.1f} → {info['norm_end']:.1f} (mean={info['norm_mean']:.1f})")
                    print(f"    PCA var explained: {info['pca_var_explained']:.2f}")
                    print(f"    estimated winding: {info['estimated_winding']:.3f}")
                    print(f"    path closed: {info['path_closed']}")
                    print()
        else:
            wt = data.get("weight_trajectory", [])
            if wt:
                info = analyze_trajectory(wt)
                print(f"  {f.name}:")
                print(f"    steps={info['n_steps']}  dim={info['param_dim']}")
                print(f"    estimated winding: {info['estimated_winding']:.3f}")
                print()


def cmd_build(args):
    """Generate creature QASM from a basin result."""
    bp = Path(args.basin_json)
    if not bp.exists():
        print(f"File not found: {bp}")
        return
    
    data = json.loads(bp.read_text())
    if isinstance(data, list):
        agent_idx = args.agent_idx or 0
        if agent_idx >= len(data):
            print(f"Agent index {agent_idx} out of range (max {len(data)-1})")
            return
        data = data[agent_idx]
    
    wt = data.get("weight_trajectory", [])
    if not wt:
        print("No weight_trajectory in this file.")
        return
    
    info = analyze_trajectory(wt)
    print(f"Trajectory: {info['n_steps']} steps, dim={info['param_dim']}")
    print(f"Estimated winding: {info['estimated_winding']:.3f}")
    print(f"Path closed: {info['path_closed']}")
    print()
    
    angles = trajectory_to_bloch_angles(wt)
    if not angles:
        print("Trajectory too short or degenerate for Bloch encoding.")
        return
    
    entry = add_creature_circuit(wt, subsample=args.subsample)
    if entry:
        qasm = entry["qasm_fn"]()
        print(f"Creature circuit: {len(angles)} Bloch angle pairs")
        print(f"QASM ({qasm.count(chr(10))+1} lines):\n")
        print(qasm)
    else:
        print("Failed to build creature circuit.")


def cmd_run(args):
    """Run full winding suite (with creature circuit) on IBM."""
    bp = Path(args.basin_json)
    if not bp.exists():
        print(f"File not found: {bp}")
        return
    
    data = json.loads(bp.read_text())
    if isinstance(data, list):
        agent_idx = args.agent_idx or 0
        data = data[agent_idx]
    
    wt = data.get("weight_trajectory", [])
    info = None
    if wt:
        info = analyze_trajectory(wt)
        print(f"Creature trajectory: {info['n_steps']} steps, "
              f"winding≈{info['estimated_winding']:.3f}")
        entry = add_creature_circuit(wt, subsample=args.subsample)
        if entry:
            print(f"Added creature_loop circuit ({len(wt)} → {args.subsample} subsampled)")
        else:
            print("WARNING: creature trajectory degenerate, running theory circuits only")
    
    suite = get_suite_qasm()
    print(f"\nFull suite: {len(suite)} circuits")
    for exp in suite:
        w = exp.get("winding", "?")
        w_str = f"{w}" if isinstance(w, (int, float)) else str(w)
        print(f"  {exp['circuit_name']:40s}  w={w_str}  {exp.get('variant','?')}")
    
    if args.dry_run:
        print("\n[dry-run] No execution.")
        return
    
    print(f"\nSubmitting {len(suite)} circuits (shots={args.shots})...")
    qasm_list = [exp["circuit_qasm"] for exp in suite]
    try:
        all_counts = run_on_ibm(qasm_list, shots=args.shots)
    except (ImportError, RuntimeError) as exc:
        print(f"\nCannot execute: {exc}")
        return
    
    for exp, counts in zip(suite, all_counts):
        exp["counts"] = counts
        total = sum(counts.values())
        p = counts.get("0", 0) / total if total else 0
        print(f"  {exp['circuit_name']:40s}  P(0)={p:.4f}")
    
    analysis = analyze_winding_suite(suite)
    print(f"\nVerdict: {analysis['verdict']}")
    for note in analysis.get("notes", []):
        print(f"  {note}")
    
    # Save
    results_dir = SCRIPT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_path = results_dir / f"creature_bridge_run_{ts}.json"
    out = {
        "timestamp":            datetime.now(timezone.utc).isoformat(),
        "basin_source":         str(bp),
        "creature_trajectory":  info if wt else None,
        "shots":                args.shots,
        "circuits":             suite,
        "analysis":             analysis,
    }
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nResults saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Bridge creature weight trajectories to quantum winding probe."
    )
    sub = parser.add_subparsers(dest="command")
    
    p_scan = sub.add_parser("scan", help="Find basin results with weight trajectories")
    
    p_build = sub.add_parser("build", help="Generate creature QASM from basin result")
    p_build.add_argument("basin_json", type=str)
    p_build.add_argument("--agent-idx", type=int, default=0)
    p_build.add_argument("--subsample", type=int, default=32)
    
    p_run = sub.add_parser("run", help="Run full suite on IBM with creature circuit")
    p_run.add_argument("basin_json", type=str)
    p_run.add_argument("--agent-idx", type=int, default=0)
    p_run.add_argument("--subsample", type=int, default=32)
    p_run.add_argument("--shots", type=int, default=4096)
    p_run.add_argument("--dry-run", action="store_true")
    
    args = parser.parse_args()
    
    if args.command == "scan":
        cmd_scan(args)
    elif args.command == "build":
        cmd_build(args)
    elif args.command == "run":
        cmd_run(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
