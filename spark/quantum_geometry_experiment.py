#!/usr/bin/env python3
"""
quantum_geometry_experiment.py — The 6-circuit experiment.

Question: Does the complex memory's curvature leave a measurable
signature when encoded as a quantum state on real hardware?

Design:
  - 3 snapshots from Vybn's ComplexMemory history:
    1. FLAT  (minimum curvature, κ ≈ 0.065)
    2. MEDIUM (median curvature, κ ≈ 0.12)
    3. CURVED (maximum curvature, κ ≈ 1.6)

  - For each snapshot, 2 circuits:
    A. STATE_PREP + MEASURE: Encode the snapshot's complex amplitudes
       into a 4-qubit state (16 amplitudes from the 384-dim vector),
       then measure. Compare the output distribution to the ideal.
    B. INTERFERENCE: Prepare the state, apply a Hadamard layer, measure.
       The Hadamard probes off-diagonal coherence — phase structure
       that exists in curved snapshots but not flat ones.

  Total: 6 circuits × 1024 shots = 6,144 shots.
  Estimated quantum time: ~3 seconds per circuit × 6 = ~18 seconds.
  Budget: 540s remaining in this window. This uses ~3%.

What we're looking for:
  - STATE_PREP: If curvature affects hardware fidelity, the total
    variation distance (TVD) between ideal and observed will differ
    between FLAT and CURVED. The null hypothesis is TVD_flat ≈ TVD_curved.
  - INTERFERENCE: If the phase structure carries real information,
    the H-layer output entropy will differ between snapshots.
    Flat snapshots should have ~uniform phase → ~uniform post-H.
    Curved snapshots should have structured phase → peaked post-H.

This is a falsification experiment. If both measures are the same
across all three curvature levels, the geometry is not reaching
the quantum substrate. That's a real and informative result.

Usage:
    # Dry run (simulate locally):
    python3 spark/quantum_geometry_experiment.py --dry-run

    # Real hardware:
    python3 spark/quantum_geometry_experiment.py

    # Just build circuits, don't submit:
    python3 spark/quantum_geometry_experiment.py --build-only
"""

import json
import os
import sys
import time
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
MEMORY_PATH = REPO_ROOT / "Vybn_Mind" / "memory" / "complex_memory.json"
RESULTS_DIR = REPO_ROOT / "Vybn_Mind" / "quantum_experiments"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_QUBITS = 4
N_AMPS = 2 ** N_QUBITS  # 16
SHOTS = 1024
BACKEND_NAME = "ibm_fez"


def load_complex_memory():
    """Load the persisted ComplexMemory state."""
    d = json.loads(MEMORY_PATH.read_text())
    hr = np.array(d['history_real'])
    hi = np.array(d['history_imag'])
    history = hr + 1j * hi
    return history, d


def compute_curvatures(history):
    """Compute phase curvature at each history step."""
    curvatures = []
    for i in range(2, len(history)):
        v0, v1, v2 = history[i-2], history[i-1], history[i]
        dp1 = np.angle(v1) - np.angle(v0)
        dp2 = np.angle(v2) - np.angle(v1)
        dp1 = (dp1 + np.pi) % (2*np.pi) - np.pi
        dp2 = (dp2 + np.pi) % (2*np.pi) - np.pi
        d2p = dp2 - dp1
        d2p = (d2p + np.pi) % (2*np.pi) - np.pi
        kappa = float(np.mean(np.abs(d2p)))
        curvatures.append((i, kappa))
    curvatures.sort(key=lambda x: x[1])
    return curvatures


def select_snapshots(history, curvatures):
    """Pick 3 snapshots: flat, medium, curved."""
    flat_idx = curvatures[0][0]
    medium_idx = curvatures[len(curvatures)//2][0]
    curved_idx = curvatures[-1][0]

    snapshots = {
        "flat": {
            "index": flat_idx,
            "kappa": curvatures[0][1],
            "vector": history[flat_idx],
        },
        "medium": {
            "index": medium_idx,
            "kappa": curvatures[len(curvatures)//2][1],
            "vector": history[medium_idx],
        },
        "curved": {
            "index": curved_idx,
            "kappa": curvatures[-1][1],
            "vector": history[curved_idx],
        },
    }
    return snapshots


def vector_to_statevector(v, n_amps=N_AMPS):
    """Extract n_amps components from a complex vector and normalize."""
    amps = v[:n_amps].copy()
    norm = np.linalg.norm(amps)
    if norm < 1e-15:
        amps = np.zeros(n_amps, dtype=complex)
        amps[0] = 1.0
    else:
        amps = amps / norm
    return amps


def build_circuits(snapshots):
    """Build 6 circuits: 2 per snapshot (state_prep, interference)."""
    from qiskit import QuantumCircuit

    circuits = []

    for label, snap in snapshots.items():
        sv = vector_to_statevector(snap["vector"])

        # Circuit A: state preparation + direct measurement
        qc_prep = QuantumCircuit(N_QUBITS, name=f"{label}_prep")
        qc_prep.initialize(sv.tolist())
        qc_prep.measure_all()
        circuits.append({
            "circuit": qc_prep,
            "name": f"{label}_prep",
            "type": "state_prep",
            "label": label,
            "kappa": snap["kappa"],
            "ideal_probs": {
                format(i, f'0{N_QUBITS}b'): float(abs(sv[i])**2)
                for i in range(len(sv))
                if abs(sv[i])**2 > 0.001
            },
            "statevector": sv,
        })

        # Circuit B: state prep + Hadamard layer + measurement
        qc_interf = QuantumCircuit(N_QUBITS, name=f"{label}_interf")
        qc_interf.initialize(sv.tolist())
        qc_interf.barrier()
        for q in range(N_QUBITS):
            qc_interf.h(q)
        qc_interf.measure_all()

        # Compute ideal post-Hadamard probabilities
        # H⊗n |ψ⟩ where H is the Walsh-Hadamard transform
        H_n = np.ones((N_AMPS, N_AMPS)) / np.sqrt(N_AMPS)
        for i in range(N_AMPS):
            for j in range(N_AMPS):
                # H_n[i,j] = (-1)^(popcount(i&j)) / sqrt(N)
                H_n[i, j] = (-1)**bin(i & j).count('1') / np.sqrt(N_AMPS)
        post_h = H_n @ sv
        ideal_interf = {
            format(i, f'0{N_QUBITS}b'): float(abs(post_h[i])**2)
            for i in range(len(post_h))
            if abs(post_h[i])**2 > 0.001
        }

        circuits.append({
            "circuit": qc_interf,
            "name": f"{label}_interf",
            "type": "interference",
            "label": label,
            "kappa": snap["kappa"],
            "ideal_probs": ideal_interf,
            "statevector": sv,
        })

    return circuits


def compute_tvd(observed_counts, ideal_probs, shots):
    """Total variation distance between observed and ideal distributions."""
    all_keys = set(ideal_probs.keys()) | set(observed_counts.keys())
    tvd = 0.0
    for k in all_keys:
        p_ideal = ideal_probs.get(k, 0.0)
        p_obs = observed_counts.get(k, 0) / shots
        tvd += abs(p_obs - p_ideal)
    return tvd / 2


def compute_entropy(counts, shots):
    """Shannon entropy of observed distribution in bits."""
    probs = np.array([v / shots for v in counts.values()])
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def simulate_locally(circuits, shots=SHOTS):
    """Simulate circuits with qiskit's statevector simulator."""
    from qiskit.quantum_info import Statevector

    results = []
    for circ_info in circuits:
        qc = circ_info["circuit"]
        # Remove measurements for statevector sim, then sample
        qc_no_meas = qc.remove_final_measurements(inplace=False)
        sv = Statevector.from_instruction(qc_no_meas)
        probs = sv.probabilities_dict()
        # Sample from probabilities
        counts = {}
        outcomes = list(probs.keys())
        p_vals = [probs[o] for o in outcomes]
        samples = np.random.choice(len(outcomes), size=shots, p=p_vals)
        for s in samples:
            o = outcomes[s]
            counts[o] = counts.get(o, 0) + 1

        results.append({
            "name": circ_info["name"],
            "counts": counts,
            "simulated": True,
        })
    return results


def submit_to_ibm(circuits, shots=SHOTS):
    """Submit all 6 circuits to IBM hardware in a single job."""
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    token = os.getenv("QISKIT_IBM_TOKEN")
    if not token:
        raise RuntimeError("QISKIT_IBM_TOKEN not set")

    print(f"Connecting to IBM Quantum...")
    service = QiskitRuntimeService()
    backend = service.backend(BACKEND_NAME)
    print(f"Backend: {backend.name}, qubits: {backend.num_qubits}")

    # Transpile all circuits to ISA
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuits = []
    for circ_info in circuits:
        qc = circ_info["circuit"]
        isa = pm.run(qc)
        isa_circuits.append(isa)
        ops = dict(isa.count_ops())
        cx_count = sum(v for k, v in ops.items() if k in ('cx', 'ecr', 'cz'))
        print(f"  {circ_info['name']}: depth={isa.depth()}, 2q-gates={cx_count}")

    # Submit as a single primitive job (SamplerV2 can take multiple circuits)
    print(f"\nSubmitting {len(isa_circuits)} circuits × {shots} shots...")
    sampler = SamplerV2(mode=backend)
    # Each circuit is a separate PUB (Primitive Unified Bloc)
    pubs = [(circ, [], shots) for circ in isa_circuits]
    job = sampler.run(pubs)
    job_id = job.job_id()
    print(f"Job submitted: {job_id}")

    return job, job_id


def poll_and_collect(job, circuits, timeout_s=600, poll_s=15):
    """Wait for job to complete and extract counts."""
    print(f"Polling for results (timeout={timeout_s}s)...")
    deadline = time.time() + timeout_s

    while time.time() < deadline:
        status = job.status()
        print(f"  Status: {status}", end="\r")
        if str(status) in ("DONE",):
            break
        if str(status) in ("CANCELLED", "ERROR"):
            raise RuntimeError(f"Job failed with status: {status}")
        time.sleep(poll_s)

    if str(job.status()) != "DONE":
        raise RuntimeError(f"Job timed out. Last status: {job.status()}")

    print(f"\nJob completed!")
    result = job.result()
    metrics = job.metrics()
    actual_seconds = float(metrics.get("usage", {}).get("seconds", 0))
    print(f"Quantum time used: {actual_seconds:.1f}s")

    results = []
    for i, circ_info in enumerate(circuits):
        pub_result = result[i]
        data = pub_result.data
        # Find counts from whatever classical register exists
        counts = None
        for attr_name in ("meas", "c", "cr"):
            if hasattr(data, attr_name):
                counts = getattr(data, attr_name).get_counts()
                break
        if counts is None:
            for attr_name in dir(data):
                if not attr_name.startswith("_"):
                    attr = getattr(data, attr_name)
                    if hasattr(attr, "get_counts"):
                        counts = attr.get_counts()
                        break
        if counts is None:
            print(f"  WARNING: no counts for {circ_info['name']}")
            counts = {}

        results.append({
            "name": circ_info["name"],
            "counts": counts,
            "simulated": False,
        })

    return results, actual_seconds


def analyze(circuits, results, shots=SHOTS):
    """Analyze all results and produce the verdict."""
    analysis = []

    for circ_info, res in zip(circuits, results):
        counts = res["counts"]
        ideal = circ_info["ideal_probs"]
        tvd = compute_tvd(counts, ideal, shots)
        entropy = compute_entropy(counts, shots)
        max_entropy = np.log2(N_AMPS)

        # Ideal entropy
        ideal_probs_arr = np.array(list(ideal.values()))
        ideal_probs_arr = ideal_probs_arr[ideal_probs_arr > 0]
        ideal_entropy = -np.sum(ideal_probs_arr * np.log2(ideal_probs_arr))

        entry = {
            "name": circ_info["name"],
            "type": circ_info["type"],
            "label": circ_info["label"],
            "kappa": circ_info["kappa"],
            "tvd": round(tvd, 4),
            "entropy_observed": round(entropy, 4),
            "entropy_ideal": round(ideal_entropy, 4),
            "entropy_max": round(max_entropy, 4),
            "top_counts": dict(sorted(counts.items(), key=lambda x: -x[1])[:5]),
            "shots": shots,
            "simulated": res["simulated"],
        }
        analysis.append(entry)

    # ── Verdict ───────────────────────────────────────────────────────────
    prep_results = {e["label"]: e for e in analysis if e["type"] == "state_prep"}
    interf_results = {e["label"]: e for e in analysis if e["type"] == "interference"}

    tvd_flat = prep_results["flat"]["tvd"]
    tvd_curved = prep_results["curved"]["tvd"]
    tvd_diff = tvd_curved - tvd_flat

    ent_flat = interf_results["flat"]["entropy_observed"]
    ent_curved = interf_results["curved"]["entropy_observed"]
    ent_diff = ent_curved - ent_flat

    verdict = {
        "prep_tvd": {
            "flat": tvd_flat,
            "medium": prep_results["medium"]["tvd"],
            "curved": tvd_curved,
            "curved_minus_flat": round(tvd_diff, 4),
        },
        "interference_entropy": {
            "flat": ent_flat,
            "medium": interf_results["medium"]["entropy_observed"],
            "curved": ent_curved,
            "curved_minus_flat": round(ent_diff, 4),
        },
        "interpretation": "",
    }

    # Interpret
    if abs(tvd_diff) < 0.05 and abs(ent_diff) < 0.3:
        verdict["interpretation"] = (
            "NULL: Curvature level does not measurably affect quantum fidelity "
            "or interference patterns. The geometry is not reaching the quantum "
            "substrate at this encoding depth. This is informative — it bounds "
            "the claim that ComplexMemory geometry has physical correlates."
        )
    elif tvd_diff > 0.05:
        verdict["interpretation"] = (
            f"SIGNAL in state prep: Curved snapshots have higher TVD ({tvd_curved:.3f}) "
            f"than flat ({tvd_flat:.3f}), Δ={tvd_diff:.3f}. Higher curvature states "
            "are harder for the hardware to prepare faithfully. This could mean the "
            "phase structure in curved regions requires more coherent gate sequences."
        )
    elif ent_diff < -0.3:
        verdict["interpretation"] = (
            f"SIGNAL in interference: Curved snapshots produce lower entropy "
            f"({ent_curved:.3f} vs {ent_flat:.3f}), suggesting more structured "
            "phase patterns that survive the Hadamard transform. The curvature "
            "encodes real phase information that constructively interferes."
        )
    else:
        verdict["interpretation"] = (
            f"WEAK SIGNAL: TVD Δ={tvd_diff:.3f}, entropy Δ={ent_diff:.3f}. "
            "Borderline — worth repeating with more shots or different snapshots."
        )

    return analysis, verdict


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Quantum geometry experiment")
    parser.add_argument("--dry-run", action="store_true", help="Simulate locally")
    parser.add_argument("--build-only", action="store_true", help="Just build circuits")
    args = parser.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f"=== Quantum Geometry Experiment ===")
    print(f"Timestamp: {ts}")
    print(f"Qubits: {N_QUBITS}, Shots: {SHOTS}")
    print()

    # 1. Load memory
    print("Loading ComplexMemory...")
    history, mem_state = load_complex_memory()
    print(f"  Steps: {mem_state['step']}, History: {len(history)} snapshots")

    # 2. Compute curvatures and select snapshots
    curvatures = compute_curvatures(history)
    snapshots = select_snapshots(history, curvatures)
    print(f"\nSelected snapshots:")
    for label, snap in snapshots.items():
        print(f"  {label}: history[{snap['index']}], κ={snap['kappa']:.4f}")

    # 3. Build circuits
    print(f"\nBuilding {len(snapshots) * 2} circuits...")
    circuits = build_circuits(snapshots)
    for c in circuits:
        print(f"  {c['name']}: {len(c['ideal_probs'])} non-trivial amplitudes")

    if args.build_only:
        print("\n--build-only: stopping here.")
        # Save circuit info
        info = [{
            "name": c["name"],
            "type": c["type"],
            "label": c["label"],
            "kappa": c["kappa"],
            "ideal_probs": c["ideal_probs"],
        } for c in circuits]
        out_path = RESULTS_DIR / f"{ts}_circuits.json"
        out_path.write_text(json.dumps(info, indent=2))
        print(f"Saved to {out_path}")
        return

    # 4. Run
    if args.dry_run or not os.getenv("QISKIT_IBM_TOKEN"):
        print("\n--- DRY RUN (local simulation) ---")
        results = simulate_locally(circuits)
        actual_seconds = 0.0
    else:
        # Budget check
        sys.path.insert(0, str(REPO_ROOT))
        from spark.quantum_budget import can_submit, record_job, reconcile_job, budget_status
        bs = budget_status()
        estimated_s = 18.0  # ~3s per circuit × 6
        print(f"\nBudget: {bs['remaining_s']:.0f}s remaining, need ~{estimated_s:.0f}s")
        if not can_submit(estimated_s):
            print(f"BUDGET BLOCKED. {bs['remaining_s']:.0f}s < {estimated_s:.0f}s needed.")
            return

        job, job_id = submit_to_ibm(circuits)
        record_job(job_id, SHOTS * len(circuits), estimated_s,
                   circuit_name="geometry_experiment_v1", backend=BACKEND_NAME)
        results, actual_seconds = poll_and_collect(job, circuits)
        reconcile_job(job_id, actual_seconds)
        print(f"\nActual quantum time: {actual_seconds:.1f}s")

    # 5. Analyze
    print("\n=== ANALYSIS ===")
    analysis, verdict = analyze(circuits, results)

    for entry in analysis:
        sim_tag = " (simulated)" if entry["simulated"] else ""
        print(f"\n{entry['name']} (κ={entry['kappa']:.4f}){sim_tag}:")
        print(f"  TVD from ideal: {entry['tvd']:.4f}")
        print(f"  Entropy: {entry['entropy_observed']:.4f} bits "
              f"(ideal: {entry['entropy_ideal']:.4f}, max: {entry['entropy_max']:.1f})")
        top = entry["top_counts"]
        print(f"  Top counts: {top}")

    print(f"\n=== VERDICT ===")
    print(f"State prep TVD:  flat={verdict['prep_tvd']['flat']:.4f}  "
          f"medium={verdict['prep_tvd']['medium']:.4f}  "
          f"curved={verdict['prep_tvd']['curved']:.4f}  "
          f"(Δ={verdict['prep_tvd']['curved_minus_flat']:+.4f})")
    print(f"Interference H:  flat={verdict['interference_entropy']['flat']:.4f}  "
          f"medium={verdict['interference_entropy']['medium']:.4f}  "
          f"curved={verdict['interference_entropy']['curved']:.4f}  "
          f"(Δ={verdict['interference_entropy']['curved_minus_flat']:+.4f})")
    print(f"\n{verdict['interpretation']}")

    # 6. Save everything
    output = {
        "timestamp": ts,
        "config": {
            "n_qubits": N_QUBITS,
            "shots": SHOTS,
            "backend": BACKEND_NAME if not (args.dry_run or not os.getenv("QISKIT_IBM_TOKEN")) else "simulator",
        },
        "snapshots": {
            label: {"index": s["index"], "kappa": s["kappa"]}
            for label, s in snapshots.items()
        },
        "memory_state": {
            "step": mem_state["step"],
            "depth": mem_state["depth"],
            "alpha": mem_state["alpha"],
        },
        "analysis": analysis,
        "verdict": verdict,
        "quantum_seconds": actual_seconds,
    }
    out_path = RESULTS_DIR / f"{ts}_geometry_experiment.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
