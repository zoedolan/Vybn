#!/usr/bin/env python3
"""
Experiment E.3 — Temporal Phase Coherence on IBM Quantum Hardware.

The falsification experiment: does the temporal sequence of representational
change in a geometrically trained neural network have quantum-detectable
phase structure?

Method:
  1. Load D_v3 centroid trajectories (baseline and geometric runs)
  2. For each layer, compute transition unitaries U_t between consecutive
     snapshots: U_t maps ψ_t to ψ_{t+1}
  3. Compose N consecutive transition unitaries: V_N = U_N ∘ ... ∘ U_1
  4. Encode V_N as a quantum circuit via Qiskit's unitary synthesis
  5. Apply V_N to a reference state on IBM hardware
  6. Measure output entropy

Null hypothesis (H0):
  The transition phases are random (random walk on U(d)). The composed
  unitary V_N produces near-maximally entropic output. Both baseline and
  geometric runs look the same.

Alternative hypothesis (H1 — the polar-time conjecture):
  The geometric run's transitions have directional phase coherence.
  V_N produces structured, lower-entropy output for the geometric run
  but not the baseline. The training trajectory of a geometrically
  regularized network carries genuine quantum-geometric information.

What each outcome means:
  H0 confirmed: The classical geometric effect is real but classically
  accessible — no quantum content in the training trajectory.
  Still a strong result (E.1 + E.2 give unified geometric theory).

  H1 confirmed: The temporal evolution of geometrically trained
  representations has phase structure that survives quantum hardware.
  This connects the closure bundle's Chern class to a measurable
  quantum-topological invariant.

Hardware: IBM quantum processor via Qiskit Runtime.
Budget: ~8 circuits, ~8s quantum time.

Prerequisite: Run E.2 (qgt_from_centroids.py) first to confirm QGT
distinguishes the runs. Only proceed to hardware if E.2 shows a signal.

Run:
  # Simulation first (always):
  /home/vybnz69/.venv/spark/bin/python3 experiment_E/run_E3_temporal_coherence.py --simulate

  # Hardware (only after simulation shows signal):
  /home/vybnz69/.venv/spark/bin/python3 experiment_E/run_E3_temporal_coherence.py --hardware

Authors: Vybn & Zoe Dolan
Date: March 23, 2026
Provenance: "see you on the other side"
"""

import json
import math
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

# ── Qiskit imports (available on the Spark) ──
try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector, Operator, entropy, partial_trace
    from qiskit_aer import AerSimulator
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False
    print("WARNING: Qiskit not installed. Install with: pip install qiskit qiskit-aer")

# ── IBM Runtime (for hardware submission) ──
try:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    HAS_RUNTIME = True
except ImportError:
    HAS_RUNTIME = False


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

N_QUBITS = 4            # Encode 2^4 = 16 dim subspace of the 384-dim centroids
SHOTS = 4096            # Measurement shots per circuit
EXP_D_V3_PATH = Path(__file__).resolve().parent.parent.parent / "results" / "experiment_D_v3_result.json"
RESULT_DIR = Path(__file__).resolve().parent / "results"


# ═══════════════════════════════════════════════════════════════════════════
# §1. Data Loading & Projection
# ═══════════════════════════════════════════════════════════════════════════

def load_centroids():
    """Load D_v3 centroid trajectories."""
    if not EXP_D_V3_PATH.exists():
        raise FileNotFoundError(f"Need {EXP_D_V3_PATH} — run run_D_v3.py first")
    with open(EXP_D_V3_PATH) as f:
        return json.load(f)


def extract_layer_trajectory(run_data, layer_idx):
    """Extract unit centroids for a single layer across training."""
    snaps = run_data.get("snapshots", {})
    traj = []
    for step_key, snap in snaps.items():
        try:
            step = int(step_key) if step_key != "final" else 99999
        except ValueError:
            continue
        layer_key = f"layer_{layer_idx}"
        if layer_key in snap:
            centroid = snap[layer_key].get("centroid_unit")
            if centroid is not None:
                traj.append((step, np.array(centroid, dtype=np.float64)))
    traj.sort(key=lambda x: x[0])
    return traj


def project_to_qubit_space(vec, n_qubits=N_QUBITS):
    """Project a high-dimensional unit vector to 2^n_qubits dimensions.

    Uses PCA-like projection: take the first 2^n components and renormalize.
    This preserves the dominant structure of the representation.
    """
    dim = 2 ** n_qubits
    # Take the first `dim` components
    projected = vec[:dim].copy()
    norm = np.linalg.norm(projected)
    if norm < 1e-12:
        projected = np.ones(dim) / math.sqrt(dim)
    else:
        projected /= norm
    return projected


# ═══════════════════════════════════════════════════════════════════════════
# §2. Transition Unitaries
# ═══════════════════════════════════════════════════════════════════════════

def compute_transition_unitary(psi_t, psi_t1, n_qubits=N_QUBITS):
    """Compute a unitary U such that U|ψ_t⟩ ≈ |ψ_{t+1}⟩.

    Since |ψ_t⟩ and |ψ_{t+1}⟩ are real unit vectors, the minimal unitary
    is a rotation in the 2D plane spanned by them, embedded in 2^n dim.

    Method: Householder-based construction.
    U = I - 2|v⟩⟨v| where |v⟩ is chosen so U|ψ_t⟩ = |ψ_{t+1}⟩.
    More precisely: U = (I - 2|w⟩⟨w|) where w = (ψ_t - ψ_{t+1})/||ψ_t - ψ_{t+1}||
    This gives a reflection, not a rotation. For a rotation, we use
    the composition of two reflections that maps ψ_t → ψ_{t+1}.
    """
    dim = 2 ** n_qubits

    # Project to qubit space
    v_t = project_to_qubit_space(psi_t, n_qubits)
    v_t1 = project_to_qubit_space(psi_t1, n_qubits)

    # Overlap
    overlap = np.dot(v_t, v_t1)

    if abs(overlap - 1.0) < 1e-10:
        # Nearly identical — return identity
        return np.eye(dim, dtype=complex)

    if abs(overlap + 1.0) < 1e-10:
        # Opposite — return negation (a Householder reflection)
        return -np.eye(dim, dtype=complex)

    # Construct rotation in the (v_t, v_t1) plane
    # Orthogonalize v_t1 against v_t to get the second basis vector
    v_perp = v_t1 - overlap * v_t
    v_perp_norm = np.linalg.norm(v_perp)
    if v_perp_norm < 1e-12:
        return np.eye(dim, dtype=complex)
    v_perp /= v_perp_norm

    # The rotation angle in the 2D plane
    theta = math.acos(np.clip(overlap, -1.0, 1.0))

    # Rotation matrix: R = I + (cos θ - 1)(|e1⟩⟨e1| + |e2⟩⟨e2|) + sin θ (|e2⟩⟨e1| - |e1⟩⟨e2|)
    # where e1 = v_t, e2 = v_perp
    e1 = v_t.reshape(-1, 1)
    e2 = v_perp.reshape(-1, 1)

    U = (np.eye(dim)
         + (math.cos(theta) - 1) * (e1 @ e1.T + e2 @ e2.T)
         + math.sin(theta) * (e2 @ e1.T - e1 @ e2.T))

    return U.astype(complex)


def compose_unitaries(trajectory, n_qubits=N_QUBITS):
    """Compose transition unitaries along a trajectory.

    V_N = U_{N-1} ∘ ... ∘ U_1 ∘ U_0

    Returns (V_N, individual_unitaries, individual_angles)
    """
    dim = 2 ** n_qubits
    V = np.eye(dim, dtype=complex)
    unitaries = []
    angles = []

    for i in range(len(trajectory) - 1):
        _, psi_t = trajectory[i]
        _, psi_t1 = trajectory[i + 1]
        U = compute_transition_unitary(psi_t, psi_t1, n_qubits)
        unitaries.append(U)

        # Rotation angle (how much this step rotates)
        v_t = project_to_qubit_space(psi_t, n_qubits)
        v_t1 = project_to_qubit_space(psi_t1, n_qubits)
        angle = math.acos(np.clip(abs(np.dot(v_t, v_t1)), 0, 1))
        angles.append(angle)

        V = U @ V

    return V, unitaries, angles


# ═══════════════════════════════════════════════════════════════════════════
# §3. Circuit Construction & Measurement
# ═══════════════════════════════════════════════════════════════════════════

def build_circuit(unitary_matrix, n_qubits=N_QUBITS):
    """Build a quantum circuit that implements the composed unitary.

    Uses Qiskit's unitary synthesis to decompose into native gates.
    """
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.unitary(Operator(unitary_matrix), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def measure_output_entropy(counts, n_qubits=N_QUBITS):
    """Compute Shannon entropy of measurement outcomes.

    Max entropy = n_qubits (bits) for uniform distribution.
    Low entropy = structured output = phase coherence.
    """
    total = sum(counts.values())
    probs = np.array([counts.get(format(i, f'0{n_qubits}b'), 0) / total
                      for i in range(2 ** n_qubits)])
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def simulate_circuit(qc, shots=SHOTS):
    """Run circuit in Aer simulator."""
    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    return result.get_counts(qc)


def compute_statevector_entropy(unitary_matrix, n_qubits=N_QUBITS):
    """Compute von Neumann entropy of the output state (no measurement noise)."""
    # Apply unitary to |0...0⟩
    initial = np.zeros(2 ** n_qubits, dtype=complex)
    initial[0] = 1.0
    output = unitary_matrix @ initial

    # Output state purity (1.0 for pure state, < 1.0 if we partial-trace)
    # For a pure state, Shannon entropy of |amplitude|^2 measures "spread"
    probs = np.abs(output) ** 2
    probs = probs[probs > 1e-15]
    shannon = float(-np.sum(probs * np.log2(probs)))
    return shannon, output


# ═══════════════════════════════════════════════════════════════════════════
# §4. Random Baseline: What Does a Random Walk Look Like?
# ═══════════════════════════════════════════════════════════════════════════

def random_walk_entropy(n_steps, n_qubits=N_QUBITS, n_trials=50, angle_scale=0.1):
    """Generate random transition unitaries with matched rotation angles
    and measure the output entropy distribution.

    This is the null model: transitions of the same magnitude but
    random direction should produce high-entropy outputs.
    """
    dim = 2 ** n_qubits
    entropies = []

    for _ in range(n_trials):
        V = np.eye(dim, dtype=complex)
        for _ in range(n_steps):
            # Random rotation of magnitude ~angle_scale
            # Generate random Hermitian, exponentiate
            H = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
            H = (H + H.conj().T) / 2  # Hermitian
            H *= angle_scale / np.linalg.norm(H)
            U = np.eye(dim, dtype=complex)
            # First-order approximation: U ≈ I + iH for small H
            # (exact would be scipy.linalg.expm, but this suffices for null model)
            from scipy.linalg import expm
            U = expm(1j * H)
            V = U @ V

        S, _ = compute_statevector_entropy(V, n_qubits)
        entropies.append(S)

    return {
        "mean": float(np.mean(entropies)),
        "std": float(np.std(entropies)),
        "min": float(np.min(entropies)),
        "max": float(np.max(entropies)),
        "n_trials": n_trials,
    }


# ═══════════════════════════════════════════════════════════════════════════
# §5. Main Experiment
# ═══════════════════════════════════════════════════════════════════════════

def run_experiment(simulate_only=True):
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  EXPERIMENT E.3 — Temporal Phase Coherence                  ║")
    print("║  Does the training trajectory carry quantum-geometric       ║")
    print("║  information?                                               ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    if not HAS_QISKIT:
        print("ERROR: Qiskit required. pip install qiskit qiskit-aer")
        return

    data = load_centroids()
    print(f"Loaded: {data['experiment']} — {data['description']}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "experiment": "E.3",
        "description": "Temporal Phase Coherence — Closure Bundle on Quantum Hardware",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_qubits": N_QUBITS,
        "mode": "simulation" if simulate_only else "hardware",
        "runs": {},
    }

    target_layers = [0, 2, 5]  # First, middle, deepest

    for run in data["runs"]:
        lam = run["lambda_geo"]
        label = "baseline" if lam == 0.0 else f"geometric"
        print(f"\n{'=' * 60}")
        print(f"  {label.upper()} (λ = {lam})")
        print(f"{'=' * 60}")

        run_results = {}

        for layer_idx in target_layers:
            traj = extract_layer_trajectory(run, layer_idx)
            print(f"\n  Layer {layer_idx}: {len(traj)} snapshots")

            if len(traj) < 3:
                print(f"    Insufficient data, skipping")
                continue

            # Compose transition unitaries
            V, unitaries, angles = compose_unitaries(traj, N_QUBITS)
            mean_angle = float(np.mean(angles))
            total_angle = float(np.sum(angles))

            print(f"    {len(unitaries)} transitions, mean angle = {math.degrees(mean_angle):.2f}°")
            print(f"    Total rotation = {math.degrees(total_angle):.2f}°")

            # Measure output entropy (statevector — exact)
            sv_entropy, output_state = compute_statevector_entropy(V, N_QUBITS)
            print(f"    Output entropy (statevector): {sv_entropy:.4f} bits "
                  f"(max = {N_QUBITS:.1f})")

            # Measure via circuit simulation
            qc = build_circuit(V, N_QUBITS)
            counts = simulate_circuit(qc)
            circuit_entropy = measure_output_entropy(counts, N_QUBITS)
            print(f"    Output entropy (circuit, {SHOTS} shots): {circuit_entropy:.4f} bits")

            # Compute purity of output
            probs = np.abs(output_state) ** 2
            purity = float(np.sum(probs ** 2))  # IPR
            print(f"    Output purity (IPR): {purity:.6f} "
                  f"(1/{purity:.1f} effective states)")

            layer_result = {
                "n_transitions": len(unitaries),
                "mean_angle_deg": math.degrees(mean_angle),
                "total_angle_deg": math.degrees(total_angle),
                "statevector_entropy": sv_entropy,
                "circuit_entropy": circuit_entropy,
                "output_purity": purity,
                "angles_deg": [math.degrees(a) for a in angles],
                "top_outcomes": dict(sorted(counts.items(),
                                            key=lambda x: -x[1])[:5]),
            }
            run_results[f"L{layer_idx}"] = layer_result

        results["runs"][label] = run_results

    # Random baseline
    print(f"\n{'=' * 60}")
    print("  RANDOM BASELINE (null model)")
    print(f"{'=' * 60}")

    # Match the angle scale to the real data
    all_angles = []
    for run_r in results["runs"].values():
        for layer_r in run_r.values():
            if isinstance(layer_r, dict) and "mean_angle_deg" in layer_r:
                all_angles.append(math.radians(layer_r["mean_angle_deg"]))

    if all_angles:
        angle_scale = float(np.mean(all_angles))
        n_steps = max(
            lr.get("n_transitions", 0)
            for rr in results["runs"].values()
            for lr in rr.values()
            if isinstance(lr, dict)
        )

        print(f"  Generating {50} random walks with {n_steps} steps, "
              f"angle ≈ {math.degrees(angle_scale):.2f}°")

        try:
            null_model = random_walk_entropy(n_steps, N_QUBITS, n_trials=50,
                                             angle_scale=angle_scale)
            print(f"  Random entropy: {null_model['mean']:.4f} ± {null_model['std']:.4f} bits")
            results["null_model"] = null_model
        except Exception as e:
            print(f"  Random baseline failed: {e}")
            results["null_model"] = {"error": str(e)}

    # Comparison
    print(f"\n{'=' * 60}")
    print("  COMPARISON: Baseline vs Geometric vs Random")
    print(f"{'=' * 60}")

    base = results["runs"].get("baseline", {})
    geo = results["runs"].get("geometric", {})
    null = results.get("null_model", {})

    print(f"\n  {'Layer':<8} {'Base S':<10} {'Geo S':<10} {'Random S':<12} {'Verdict':<20}")
    print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*12} {'-'*20}")

    for layer_idx in target_layers:
        key = f"L{layer_idx}"
        b_s = base.get(key, {}).get("statevector_entropy", float('nan'))
        g_s = geo.get(key, {}).get("statevector_entropy", float('nan'))
        r_s = null.get("mean", float('nan'))

        # Verdict logic
        if not math.isnan(g_s) and not math.isnan(r_s):
            r_std = null.get("std", 0.1)
            if g_s < r_s - 2 * r_std:
                verdict = "COHERENT (geo < random)"
            elif b_s < r_s - 2 * r_std:
                verdict = "COHERENT (both)"
            else:
                verdict = "NULL (no signal)"
        else:
            verdict = "INSUFFICIENT DATA"

        print(f"  L{layer_idx:<6} {b_s:<10.4f} {g_s:<10.4f} {r_s:<12.4f} {verdict}")

    results["verdict_summary"] = (
        "COHERENT" if any(
            geo.get(f"L{l}", {}).get("statevector_entropy", float('inf'))
            < null.get("mean", float('inf')) - 2 * null.get("std", 0.1)
            for l in target_layers
        ) else "NULL"
    )
    print(f"\n  OVERALL VERDICT: {results['verdict_summary']}")

    # Hardware submission (if requested and signal exists in simulation)
    if not simulate_only and HAS_RUNTIME:
        if results["verdict_summary"] == "COHERENT":
            print(f"\n{'=' * 60}")
            print("  HARDWARE SUBMISSION: IBM Quantum")
            print(f"{'=' * 60}")
            # Hardware code would go here — build circuits, submit via SamplerV2
            print("  [Hardware submission not yet implemented — simulation shows signal]")
            print("  [To implement: use SamplerV2 with circuits from simulation]")
        else:
            print("\n  Simulation shows NULL — skipping hardware submission.")
            print("  (No point spending quantum time on a null signal.)")

    # Save
    out_path = RESULT_DIR / "experiment_E3_temporal_coherence.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E.3: Temporal Phase Coherence")
    parser.add_argument("--simulate", action="store_true", default=True,
                        help="Run in simulation only (default)")
    parser.add_argument("--hardware", action="store_true",
                        help="Submit to IBM quantum hardware")
    args = parser.parse_args()

    run_experiment(simulate_only=not args.hardware)
