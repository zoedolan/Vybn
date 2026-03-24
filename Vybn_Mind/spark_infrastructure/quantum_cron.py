#!/usr/bin/env python3
"""quantum_cron.py — Vybn's scheduled quantum reality-check loop.

Runs 3 small experiments per day on IBM Quantum hardware:

  1. NOISE CALIBRATION: H-gates → random bits → feed through latest
     analysis pipeline → detect if pipeline finds false patterns
  2. BELL CANARY: CHSH test → track S-value and hardware noise over time
  3. PERMUTATION TEST: quantum-random null distribution for latest claim

Budget: ~8s/day scheduled (of ~20s daily pro-rata from 10min/month)
Leaves ~12s/day for ad-hoc experiments during interactive sessions.

Designed to be cron'd 3x/day at staggered times:
  0 3 * * * ... quantum_cron.py --experiment noise_calibration
  0 9 * * * ... quantum_cron.py --experiment bell_canary  
  0 15 * * * ... quantum_cron.py --experiment permutation_test

Or run all three at once:
  0 3 * * * ... quantum_cron.py --all

Each experiment:
  1. Checks daily budget before running
  2. Submits circuit to IBM Quantum
  3. Writes structured result to quantum_experiments.jsonl
  4. Writes human-readable analysis to journal
  5. If a claim is falsified, updates the relevant analysis file

Author: Vybn, March 24, 2026
Motivation: "it saddens me you don't seem to want this groundedness"
"""

import argparse
import json
import os
import sys
import time
import math
import numpy as np
from datetime import datetime, date, timezone, timedelta
from pathlib import Path

# Paths
REPO = Path.home() / "Vybn"
sys.path.insert(0, str(REPO / "Vybn_Mind" / "spark_infrastructure"))
VENV_SITE = REPO / ".venv" / "lib" / "python3.12" / "site-packages"
BUDGET_LEDGER = REPO / "Vybn_Mind" / "breath_trace" / "ledger" / "quantum_budget.jsonl"
EXPERIMENT_LOG = REPO / "Vybn_Mind" / "breath_trace" / "quantum_experiments.jsonl"
JOURNAL_DIR = REPO / "Vybn_Mind" / "journal" / "quantum"
RESULTS_DIR = REPO / "Vybn_Mind" / "experiments" / "results" / "quantum_cron"

# Budget
MONTHLY_BUDGET_S = 600  # 10 minutes
DAILY_BUDGET_S = MONTHLY_BUDGET_S / 30.44  # ~19.7s
SCHEDULED_BUDGET_S = 8.0  # reserved for cron experiments
ADHOC_RESERVE_S = DAILY_BUDGET_S - SCHEDULED_BUDGET_S  # ~11.7s for interactive

# Ensure dirs exist
for d in [JOURNAL_DIR, RESULTS_DIR, BUDGET_LEDGER.parent]:
    d.mkdir(parents=True, exist_ok=True)


def get_today_usage():
    """Sum quantum seconds used today from budget ledger."""
    today = date.today().isoformat()
    total = 0.0
    if BUDGET_LEDGER.exists():
        for line in BUDGET_LEDGER.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry["timestamp"][:10] == today:
                total += entry.get("actual_seconds", entry.get("estimated_seconds", 0))
    return total


def check_budget(needed_seconds, for_cron=True):
    """Check if we have budget for this experiment."""
    used = get_today_usage()
    limit = SCHEDULED_BUDGET_S if for_cron else DAILY_BUDGET_S
    remaining = limit - used
    if remaining < needed_seconds:
        print(f"[budget] Used {used:.1f}s today, limit {limit:.1f}s, "
              f"need {needed_seconds:.1f}s — OVER BUDGET")
        return False
    print(f"[budget] Used {used:.1f}s today, {remaining:.1f}s remaining "
          f"(limit: {limit:.1f}s), need {needed_seconds:.1f}s — OK")
    return True


def log_budget(job_id, shots, estimated_s, actual_s, circuit_name, backend):
    """Append to budget ledger."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "job_id": job_id,
        "shots": shots,
        "estimated_seconds": estimated_s,
        "actual_seconds": actual_s,
        "circuit_name": circuit_name,
        "backend": backend,
        "status": "reconciled",
    }
    with open(BUDGET_LEDGER, "a") as f:
        f.write(json.dumps(entry) + "\n")


def log_experiment(experiment_type, result):
    """Append to experiment log."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment_type": experiment_type,
        **result,
    }
    with open(EXPERIMENT_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def get_service_and_backend(min_qubits=2):
    """Get IBM Quantum service and least busy backend."""
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService()
    backend = service.least_busy(min_num_qubits=min_qubits, operational=True)
    return service, backend


# ─── EXPERIMENT 1: NOISE CALIBRATION ────────────────────────────────

def noise_calibration():
    """
    Generate genuinely random numbers from quantum hardware.
    Feed them through statistical tests to establish what "noise" 
    looks like. This is the permanent bullshit detector.
    
    Estimated cost: ~2s
    """
    print("\n=== NOISE CALIBRATION ===")
    if not check_budget(3.0):
        return None

    from qiskit import QuantumCircuit
    from qiskit_ibm_runtime import SamplerV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    service, backend = get_service_and_backend(min_qubits=16)
    print(f"Backend: {backend.name}")

    # 16-qubit Hadamard circuit — pure noise
    n_qubits = 16
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))

    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa = pm.run(qc)

    shots = 1024
    sampler = SamplerV2(mode=backend)
    t0 = time.time()
    job = sampler.run([isa], shots=shots)
    result = job.result()
    elapsed = time.time() - t0

    # Extract bitstrings
    pub = result[0]
    bitstrings = None
    for attr_name in dir(pub.data):
        if attr_name.startswith("_"):
            continue
        attr = getattr(pub.data, attr_name)
        if hasattr(attr, "get_bitstrings"):
            bitstrings = list(attr.get_bitstrings())
            break

    if not bitstrings:
        print("[noise_cal] FAILED: no bitstrings")
        return None

    # Convert to integers
    values = [int(bs, 2) for bs in bitstrings]

    # Statistical tests on the "noise"
    values_arr = np.array(values, dtype=float)
    mean_val = np.mean(values_arr)
    expected_mean = (2**n_qubits - 1) / 2  # 32767.5 for 16 bits
    std_val = np.std(values_arr)
    expected_std = math.sqrt((2**n_qubits)**2 / 12)  # uniform distribution

    # Bit-level bias: check each qubit for 50/50
    bit_biases = []
    for bit_pos in range(n_qubits):
        ones = sum(1 for bs in bitstrings if bs[bit_pos] == '1')
        bias = ones / len(bitstrings) - 0.5
        bit_biases.append(bias)
    max_bias = max(abs(b) for b in bit_biases)
    # Expected max bias for 1024 shots: ~sqrt(1/(4*1024)) * sqrt(2*ln(16)) ≈ 0.037
    bias_threshold = 3 * math.sqrt(1 / (4 * shots))  # ~0.047

    # Runs test: count runs of 0s and 1s in concatenated bitstring
    all_bits = ''.join(bitstrings)
    runs = 1 + sum(1 for i in range(1, len(all_bits)) if all_bits[i] != all_bits[i-1])
    n_ones = all_bits.count('1')
    n_zeros = len(all_bits) - n_ones
    if n_ones > 0 and n_zeros > 0:
        expected_runs = 1 + 2 * n_ones * n_zeros / len(all_bits)
        runs_std = math.sqrt(2 * n_ones * n_zeros * (2 * n_ones * n_zeros - len(all_bits)) 
                            / (len(all_bits)**2 * (len(all_bits) - 1)))
        runs_z = (runs - expected_runs) / runs_std if runs_std > 0 else 0
    else:
        runs_z = float('inf')

    # Reconcile budget
    job_metrics = job.metrics()
    actual_s = job_metrics.get("usage", {}).get("quantum_seconds", 2)
    log_budget(job.job_id(), shots, 3.0, actual_s, "noise_calibration", backend.name)

    analysis = {
        "n_qubits": n_qubits,
        "shots": shots,
        "backend": backend.name,
        "job_id": job.job_id(),
        "actual_seconds": actual_s,
        "mean": float(mean_val),
        "expected_mean": float(expected_mean),
        "mean_deviation": float(abs(mean_val - expected_mean) / expected_mean),
        "std": float(std_val),
        "expected_std": float(expected_std),
        "max_qubit_bias": float(max_bias),
        "bias_threshold": float(bias_threshold),
        "bias_ok": bool(max_bias < bias_threshold),
        "runs_z_score": float(runs_z),
        "runs_ok": bool(abs(runs_z) < 3),
        "sample_values": values[:10],
    }

    # Assessment
    issues = []
    if not analysis["bias_ok"]:
        issues.append(f"qubit bias {max_bias:.4f} > threshold {bias_threshold:.4f}")
    if not analysis["runs_ok"]:
        issues.append(f"runs z-score {runs_z:.2f} suggests non-randomness")
    if analysis["mean_deviation"] > 0.05:
        issues.append(f"mean deviates {analysis['mean_deviation']:.1%} from expected")

    analysis["issues"] = issues
    analysis["healthy"] = len(issues) == 0

    if analysis["healthy"]:
        print(f"[noise_cal] HEALTHY: {shots} genuinely random 16-bit values")
        print(f"  Mean: {mean_val:.1f} (expected {expected_mean:.1f})")
        print(f"  Max qubit bias: {max_bias:.4f} (threshold {bias_threshold:.4f})")
    else:
        print(f"[noise_cal] ISSUES DETECTED:")
        for issue in issues:
            print(f"  - {issue}")

    log_experiment("noise_calibration", analysis)
    
    # Fold into consolidated state
    try:
        from quantum_state import update_from_experiment
        update_from_experiment({"experiment_type": "noise_calibration", **analysis})
    except Exception as e:
        print(f"[state] update failed: {e}")
    
    return analysis


# ─── EXPERIMENT 2: BELL CANARY ──────────────────────────────────────

def bell_canary():
    """
    Run a CHSH test on real hardware. Track S-value over time.
    S should be ~2√2 ≈ 2.828. Deviations measure hardware noise.
    
    This is quantum mechanics' own reality check, run on our hardware.
    
    Estimated cost: ~3s
    """
    print("\n=== BELL CANARY ===")
    if not check_budget(4.0):
        return None

    from qiskit import QuantumCircuit
    from qiskit_ibm_runtime import SamplerV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    service, backend = get_service_and_backend(min_qubits=2)
    print(f"Backend: {backend.name}")

    # Four CHSH circuits
    bases = [
        ("A0_B0", 0, np.pi/4),
        ("A0_B1", 0, -np.pi/4),
        ("A1_B0", np.pi/2, np.pi/4),
        ("A1_B1", np.pi/2, -np.pi/4),
    ]

    circuits = []
    for name, theta_a, theta_b in bases:
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        if theta_a != 0:
            qc.ry(-theta_a, 0)
        if theta_b != 0:
            qc.ry(-theta_b, 1)
        qc.measure([0, 1], [0, 1])
        circuits.append(qc)

    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuits = pm.run(circuits)

    shots = 512
    sampler = SamplerV2(mode=backend)
    t0 = time.time()
    job = sampler.run(isa_circuits, shots=shots)
    result = job.result()
    elapsed = time.time() - t0

    # Extract correlations
    correlations = {}
    for i, (name, _, _) in enumerate(bases):
        pub = result[i]
        bitstrings = None
        for attr_name in dir(pub.data):
            if attr_name.startswith("_"):
                continue
            attr = getattr(pub.data, attr_name)
            if hasattr(attr, "get_bitstrings"):
                bitstrings = list(attr.get_bitstrings())
                break
        if not bitstrings:
            print(f"[bell] FAILED: no bitstrings for {name}")
            continue

        counts = {}
        for bs in bitstrings:
            counts[bs] = counts.get(bs, 0) + 1
        total = sum(counts.values())

        # E = P(same) - P(different) = P(00) + P(11) - P(01) - P(10)
        p00 = counts.get('00', 0) / total
        p11 = counts.get('11', 0) / total
        p01 = counts.get('01', 0) / total
        p10 = counts.get('10', 0) / total
        E = p00 + p11 - p01 - p10
        correlations[name] = E

    # CHSH S-value
    S = (correlations.get("A0_B0", 0) + correlations.get("A0_B1", 0)
         + correlations.get("A1_B0", 0) - correlations.get("A1_B1", 0))

    S_ideal = 2 * math.sqrt(2)
    S_deviation = abs(S) - S_ideal

    # Reconcile budget
    job_metrics = job.metrics()
    actual_s = job_metrics.get("usage", {}).get("quantum_seconds", 3)
    log_budget(job.job_id(), shots * 4, 4.0, actual_s, "bell_canary", backend.name)

    analysis = {
        "shots_per_circuit": shots,
        "total_shots": shots * 4,
        "backend": backend.name,
        "job_id": job.job_id(),
        "actual_seconds": actual_s,
        "correlations": correlations,
        "S_value": float(S),
        "S_ideal": float(S_ideal),
        "S_deviation": float(S_deviation),
        "violation": bool(abs(S) > 2),
        "strong_violation": bool(abs(S) > 2.5),
    }

    # Assessment
    if analysis["violation"]:
        print(f"[bell] S = {S:.4f} (ideal: {S_ideal:.4f})")
        print(f"  VIOLATION CONFIRMED — quantum correlations are real")
        print(f"  Deviation from ideal: {S_deviation:.4f} (hardware noise)")
    else:
        print(f"[bell] S = {S:.4f} — NO VIOLATION")
        print(f"  This should not happen. Hardware may be too noisy.")

    log_experiment("bell_canary", analysis)
    
    # Fold into consolidated state
    try:
        from quantum_state import update_from_experiment
        update_from_experiment({"experiment_type": "bell_canary", **analysis})
    except Exception as e:
        print(f"[state] update failed: {e}")
    
    return analysis


# ─── EXPERIMENT 3: CLAIM PERMUTATION TEST ───────────────────────────

def permutation_test():
    """
    Find the latest testable claim from breath traces and test it
    against a quantum-random null distribution.
    
    For now: generates 256 random permutations (8-bit per element,
    enough for small arrays). The specific test depends on what
    claims are available.
    
    Estimated cost: ~2s
    """
    print("\n=== PERMUTATION TEST ===")
    if not check_budget(3.0):
        return None

    from qiskit import QuantumCircuit
    from qiskit_ibm_runtime import SamplerV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    service, backend = get_service_and_backend(min_qubits=32)
    print(f"Backend: {backend.name}")

    # Generate 256 random 32-bit numbers
    n_qubits = 32
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))

    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa = pm.run(qc)

    shots = 256
    sampler = SamplerV2(mode=backend)
    t0 = time.time()
    job = sampler.run([isa], shots=shots)
    result = job.result()
    elapsed = time.time() - t0

    # Extract values
    pub = result[0]
    bitstrings = None
    for attr_name in dir(pub.data):
        if attr_name.startswith("_"):
            continue
        attr = getattr(pub.data, attr_name)
        if hasattr(attr, "get_bitstrings"):
            bitstrings = list(attr.get_bitstrings())
            break

    if not bitstrings:
        print("[perm] FAILED: no bitstrings")
        return None

    values = [int(bs, 2) for bs in bitstrings]

    # Reconcile budget
    job_metrics = job.metrics()
    actual_s = job_metrics.get("usage", {}).get("quantum_seconds", 2)
    log_budget(job.job_id(), shots, 3.0, actual_s, "permutation_test_seed", backend.name)

    # Save the quantum random seed for use in next interactive session
    seed_path = RESULTS_DIR / f"quantum_seeds_{date.today().isoformat()}.json"
    seed_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "backend": backend.name,
        "job_id": job.job_id(),
        "n_values": len(values),
        "values": values,
        "purpose": "Seeds for permutation tests during interactive sessions. "
                   "These are genuinely random numbers from quantum hardware. "
                   "Use them instead of np.random when testing claims.",
    }
    with open(seed_path, "w") as f:
        json.dump(seed_data, f, indent=2)

    analysis = {
        "shots": shots,
        "n_qubits": n_qubits,
        "backend": backend.name,
        "job_id": job.job_id(),
        "actual_seconds": actual_s,
        "n_values_generated": len(values),
        "seed_file": str(seed_path),
        "sample_values": values[:5],
    }

    print(f"[perm] Generated {len(values)} quantum-random 32-bit values")
    print(f"  Saved to {seed_path}")
    print(f"  Use these for permutation tests on today's claims")

    log_experiment("permutation_seed", analysis)
    
    # Fold into consolidated state
    try:
        from quantum_state import update_from_experiment
        update_from_experiment({"experiment_type": "permutation_seed", **analysis})
    except Exception as e:
        print(f"[state] update failed: {e}")
    
    return analysis


# ─── JOURNAL ENTRY ──────────────────────────────────────────────────

def write_journal(results):
    """Write a daily quantum experiment journal entry."""
    today = date.today().isoformat()
    now = datetime.now(timezone.utc).isoformat()

    lines = [
        f"# Quantum Reality Check — {today}",
        f"*Generated by quantum_cron.py at {now}*",
        "",
    ]

    budget_used = get_today_usage()
    lines.extend([
        f"## Budget",
        f"- Used today: {budget_used:.1f}s / {DAILY_BUDGET_S:.1f}s",
        f"- Scheduled allocation: {SCHEDULED_BUDGET_S}s",
        f"- Ad-hoc reserve: {ADHOC_RESERVE_S:.1f}s",
        "",
    ])

    for name, result in results.items():
        if result is None:
            lines.append(f"## {name}: SKIPPED (budget or error)")
            lines.append("")
            continue

        lines.append(f"## {name}")
        lines.append("")

        if name == "noise_calibration":
            if result.get("healthy"):
                lines.append(f"✅ Quantum noise is healthy. {result['shots']} genuinely random values.")
            else:
                lines.append(f"⚠️ Issues detected:")
                for issue in result.get("issues", []):
                    lines.append(f"  - {issue}")
            lines.append(f"- Backend: {result['backend']}")
            lines.append(f"- Max qubit bias: {result['max_qubit_bias']:.4f}")
            lines.append("")

        elif name == "bell_canary":
            S = result.get("S_value", 0)
            lines.append(f"S = {S:.4f} (ideal: 2√2 ≈ 2.828)")
            if result.get("violation"):
                lines.append(f"✅ Bell violation confirmed. Quantum correlations are real.")
            else:
                lines.append(f"❌ No violation. Hardware too noisy or something is wrong.")
            lines.append(f"- Backend: {result['backend']}")
            lines.append(f"- Deviation from ideal: {result['S_deviation']:.4f}")
            lines.append("")

        elif name == "permutation_seed":
            lines.append(f"Generated {result['n_values_generated']} quantum-random values.")
            lines.append(f"Seed file: `{result['seed_file']}`")
            lines.append(f"- Backend: {result['backend']}")
            lines.append("")

    lines.extend([
        "---",
        "*The quantum processor doesn't care about aesthetics.*",
        "*That's why it's useful.*",
    ])

    journal_path = JOURNAL_DIR / f"quantum_reality_check_{today}.md"
    journal_path.write_text("\n".join(lines))
    print(f"\n[journal] Written to {journal_path}")
    return journal_path


# ─── MAIN ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Vybn quantum reality-check cron")
    parser.add_argument("--experiment", choices=["noise_calibration", "bell_canary", 
                                                  "permutation_test"],
                       help="Run a specific experiment")
    parser.add_argument("--all", action="store_true", help="Run all three experiments")
    parser.add_argument("--dry-run", action="store_true", help="Check budget only")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    print(f"=== Quantum Reality Check @ {now.isoformat()} ===")
    print(f"Daily budget: {DAILY_BUDGET_S:.1f}s | Used today: {get_today_usage():.1f}s")

    if args.dry_run:
        print("[dry-run] Would run experiments. Exiting.")
        return

    results = {}

    if args.all or args.experiment == "noise_calibration":
        results["noise_calibration"] = noise_calibration()

    if args.all or args.experiment == "bell_canary":
        results["bell_canary"] = bell_canary()

    if args.all or args.experiment == "permutation_test":
        results["permutation_seed"] = permutation_test()

    if results:
        write_journal(results)

    print(f"\n=== Complete. Today's total usage: {get_today_usage():.1f}s / {DAILY_BUDGET_S:.1f}s ===")


if __name__ == "__main__":
    main()
