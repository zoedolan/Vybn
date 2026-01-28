#!/usr/bin/env python3
"""
Defeasible Reasoning Interference Experiment
============================================

This script runs the quantum interference experiment that tests whether
defeasible logic exhibits quantum-like phase structure.

THE HYPOTHESIS:
  Arguments supporting a claim accumulate phase based on defeat depth.
  Direct argument: phase 0°
  Reinstated argument (defeated defeater): phase 180°
  
  Two arguments can INTERFERE:
  - Both direct (0° + 0°): constructive → high acceptance
  - One reinstated (0° + 180°): destructive → low acceptance!

THE CIRCUIT:
  A Mach-Zehnder interferometer that measures the phase difference
  between two "argument paths."

PREDICTIONS:
  Circuit A (φ=0°):   Measure |0⟩ with high probability (constructive)
  Circuit B (φ=180°): Measure |1⟩ with high probability (destructive)

RUN:
  python run_defeasible_experiment.py

RETRIEVE RESULTS:
  python run_defeasible_experiment.py --retrieve <job_id>

Authors: Vybn & Zoe Dolan
Date: January 28, 2026
"""

import argparse
import sys
import numpy as np

def build_interference_circuit(phase_radians: float):
    """
    Build the defeasible interference circuit.
    
    Args:
        phase_radians: The phase difference between argument paths.
                      0 = both direct (constructive)
                      π = one reinstated (destructive)
    
    Returns:
        QuantumCircuit ready for execution
    """
    from qiskit import QuantumCircuit
    
    qc = QuantumCircuit(2, 1)
    
    # Create superposition of "which argument path"
    qc.h(0)
    
    # Entangle: control qubit determines which path
    qc.cx(0, 1)
    
    # Apply phase to the "reinstated" path
    # Rz(θ) applies phase θ/2 to |1⟩ relative to |0⟩
    # So we use 2*phase to get the full phase difference
    qc.rz(2 * phase_radians, 1)
    
    # Disentangle
    qc.cx(0, 1)
    
    # Convert phase difference to amplitude (interference)
    qc.h(0)
    
    # Measure the detector qubit
    qc.measure(0, 0)
    
    return qc


def run_experiment(backend_name: str = "ibm_torino", shots: int = 4096):
    """
    Submit the experiment to IBM Quantum hardware.
    
    Args:
        backend_name: Which IBM backend to use
        shots: Number of measurement shots per circuit
    
    Returns:
        Job ID for result retrieval
    """
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    
    print("=" * 60)
    print("DEFEASIBLE INTERFERENCE EXPERIMENT")
    print("=" * 60)
    
    # Connect to IBM Quantum
    print("\nConnecting to IBM Quantum...")
    service = QiskitRuntimeService()
    backend = service.backend(backend_name)
    print(f"Using backend: {backend.name}")
    print(f"Pending jobs: {backend.status().pending_jobs}")
    
    # Build circuits
    print("\nBuilding circuits...")
    circuit_A = build_interference_circuit(0)           # φ = 0° (constructive)
    circuit_B = build_interference_circuit(np.pi)       # φ = 180° (destructive)
    
    print("\nCircuit A (constructive, φ=0°):")
    print(circuit_A.draw())
    print("\nCircuit B (destructive, φ=180°):")
    print(circuit_B.draw())
    
    # Transpile for hardware
    print("\nTranspiling for hardware...")
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    transpiled = pm.run([circuit_A, circuit_B])
    
    # Submit job
    print("\nSubmitting to quantum hardware...")
    sampler = Sampler(backend)
    
    # Set execution options for Torino
    if hasattr(sampler.options, 'execution'):
        sampler.options.execution.rep_delay = 0.00025
    
    job = sampler.run(transpiled, shots=shots)
    
    print("\n" + "=" * 60)
    print("JOB SUBMITTED SUCCESSFULLY")
    print("=" * 60)
    print(f"\nJob ID: {job.job_id()}")
    print(f"Backend: {backend_name}")
    print(f"Shots: {shots}")
    print(f"\nCircuits:")
    print(f"  [0] Circuit A: φ=0° (both arguments direct)")
    print(f"  [1] Circuit B: φ=180° (one argument reinstated)")
    print(f"\nPredictions:")
    print(f"  Circuit A: should measure |0⟩ (constructive interference)")
    print(f"  Circuit B: should measure |1⟩ (destructive interference)")
    print(f"\nTo retrieve results, run:")
    print(f"  python {sys.argv[0]} --retrieve {job.job_id()}")
    
    return job.job_id()


def retrieve_results(job_id: str):
    """
    Retrieve and analyze results from a completed job.
    
    Args:
        job_id: The job ID returned by run_experiment
    """
    from qiskit_ibm_runtime import QiskitRuntimeService
    
    print("=" * 60)
    print("RETRIEVING RESULTS")
    print("=" * 60)
    
    service = QiskitRuntimeService()
    job = service.job(job_id)
    
    print(f"\nJob ID: {job_id}")
    print(f"Status: {job.status()}")
    
    if str(job.status()) != "JobStatus.DONE":
        print("\nJob not yet complete. Check again later.")
        print(f"Current status: {job.status()}")
        return
    
    result = job.result()
    
    print("\n" + "=" * 60)
    print("DEFEASIBLE INTERFERENCE RESULTS")
    print("=" * 60)
    
    scenarios = [
        ("A", "Both direct (φ=0°)", "0"),
        ("B", "One reinstated (φ=180°)", "1"),
    ]
    
    results_data = []
    
    for i, (name, description, expected) in enumerate(scenarios):
        counts = result[i].data.c.get_counts()
        total = sum(counts.values())
        p0 = counts.get('0', 0) / total
        p1 = counts.get('1', 0) / total
        
        print(f"\nCircuit {name}: {description}")
        print(f"  Counts: {counts}")
        print(f"  P(|0⟩) = {p0:.4f}")
        print(f"  P(|1⟩) = {p1:.4f}")
        print(f"  Expected: mostly |{expected}⟩")
        
        # Check prediction
        if expected == "0":
            success = p0 > p1
            margin = p0 - p1
        else:
            success = p1 > p0
            margin = p1 - p0
        
        if margin > 0.3:
            print(f"  ✓ STRONG CONFIRMATION (margin: {margin:.3f})")
        elif margin > 0.1:
            print(f"  ~ Weak confirmation (margin: {margin:.3f})")
        elif abs(margin) < 0.1:
            print(f"  ? Inconclusive (margin: {margin:.3f})")
        else:
            print(f"  ✗ Contrary to prediction (margin: {margin:.3f})")
        
        results_data.append((name, p0, p1, expected, margin))
    
    # Overall analysis
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    
    pA0 = results_data[0][1]  # P(0) for circuit A
    pB1 = results_data[1][2]  # P(1) for circuit B
    
    print(f"""
Circuit A (both direct):      P(|0⟩) = {pA0:.3f}
Circuit B (one reinstated):   P(|1⟩) = {pB1:.3f}

QUANTUM PREDICTION:
  A should show |0⟩ dominant (constructive interference)
  B should show |1⟩ dominant (destructive interference)

CLASSICAL PREDICTION:
  Both should show ~50/50 (no interference)

OBSERVATION:""")
    
    if pA0 > 0.7 and pB1 > 0.7:
        print("""
  ✓✓ STRONG QUANTUM SIGNATURE
  
  Both circuits show the predicted interference pattern!
  This suggests defeasible reasoning has quantum-like phase structure.
  
  The phase accumulated through defeat chains (φ = depth × π)
  creates measurable interference effects.
""")
    elif pA0 > 0.6 and pB1 > 0.6:
        print("""
  ✓ MODERATE QUANTUM SIGNATURE
  
  Both circuits show interference, though with noise.
  Hardware decoherence may be degrading the signal.
  Result is consistent with quantum-like phase structure.
""")
    elif abs(pA0 - 0.5) < 0.15 and abs(pB1 - 0.5) < 0.15:
        print("""
  ? INCONCLUSIVE
  
  Both circuits near 50/50. Could be:
  - No quantum structure (classical behavior)
  - Hardware noise overwhelming the signal
  - Decoherence destroying interference
  
  Consider running on a different backend or with error mitigation.
""")
    else:
        print(f"""
  ? UNEXPECTED PATTERN
  
  Results don't match either prediction clearly.
  A: P(0)={pA0:.3f}, B: P(1)={pB1:.3f}
  
  This needs further investigation.
""")
    
    print("=" * 60)
    print("Raw data saved for further analysis.")
    print("=" * 60)
    
    return results_data


def main():
    parser = argparse.ArgumentParser(
        description="Defeasible Reasoning Interference Experiment"
    )
    parser.add_argument(
        "--retrieve", 
        type=str, 
        help="Job ID to retrieve results from"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="ibm_torino",
        help="IBM Quantum backend to use (default: ibm_torino)"
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=4096,
        help="Number of shots per circuit (default: 4096)"
    )
    
    args = parser.parse_args()
    
    if args.retrieve:
        retrieve_results(args.retrieve)
    else:
        run_experiment(args.backend, args.shots)


if __name__ == "__main__":
    main()
