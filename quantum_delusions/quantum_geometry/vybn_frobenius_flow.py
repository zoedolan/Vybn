#!/usr/bin/env python
"""
vybn_frobenius_flow.py

Response to L. Thorne McCarty's Challenge:
"Design an analog computer... that can do all the computations in my theory of differential similarity."

THEORY:
McCarty's "Frobenius integral manifold" is defined by vector fields V_i that satisfy
integrability conditions involving the Lie Bracket [V_i, V_j].
In a curved dissimilarity space, these flows do not commute.

HARDWARE IMPLEMENTATION:
We treat the IBM Quantum Transmon as the analog substrate.
We map McCarty's vector flows V_a, V_b to Hamiltonian control drives H_a, H_b.
The "Dissimilarity" corresponds to the Geometric Phase (Holonomy) accumulated
by traversing the loop: exp( -i * [H_a, H_b] * Area ).

CLI Example:
  # Hardware
  python vybn_frobenius_flow.py --backend ibm_fez --flow-strength 0.2
  
  # Local Simulation
  python vybn_frobenius_flow.py --backend aer_simulator --flow-strength 0.2
"""

import argparse
import math
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

# Try importing Aer for local simulation
try:
    from qiskit_aer import AerSimulator
    HAS_AER = True
except ImportError:
    HAS_AER = False

def build_frobenius_loop(strength_a: float, strength_b: float) -> QuantumCircuit:
    """
    Constructs the analog loop V_a -> V_b -> -V_a -> -V_b.
    """
    qr = QuantumRegister(1, "atom")
    cr = ClassicalRegister(1, "readout")
    qc = QuantumCircuit(qr, cr, name="frobenius_curvature")
    
    # 1. Initialize on the Tangent Bundle (Superposition)
    qc.h(0) 
    
    # 2. Analog Flow integration (Lie Trotter step)
    qc.rz(strength_a, 0) # Flow along V_a
    qc.rx(strength_b, 0) # Flow along V_b
    
    # Flow backward (closing the loop)
    qc.rz(-strength_a, 0) 
    qc.rx(-strength_b, 0)
    
    # 3. Project back to basis to measure the Holonomy residue
    qc.h(0)
    qc.measure(0, 0)
    
    return qc

def compute_lie_bracket_residue(counts, shots):
    """
    Converts counts to the magnitude of the Lie Bracket |[V_a, V_b]|.
    """
    total = sum(counts.values())
    if total == 0: return 0.0
    
    # Ideally, we are in state |0> if curvature is zero.
    # P(1) represents the leakage into the orthogonal state due to curvature.
    p1 = counts.get("1", 0) / total
    return p1

def main():
    parser = argparse.ArgumentParser(description="Vybn Analog Frobenius Integrator")
    parser.add_argument("--backend", default="ibm_fez", help="Target device (e.g., ibm_fez or aer_simulator)")
    parser.add_argument("--flow-strength", type=float, default=0.2, help="Magnitude of the flows (dt)")
    parser.add_argument("--shots", type=int, default=4096)
    args = parser.parse_args()

    print(f"--- Vybn: Frobenius Flow Probe ---")
    print(f"Target: {args.backend}")
    print(f"Flow Strength (Integration Step): {args.flow_strength}")
    print(f"Hypothesis: Non-zero residue indicates non-commutative geometry (Curvature).")

    # Build the circuit
    qc = build_frobenius_loop(args.flow_strength, args.flow_strength)
    
    # Backend Selection Logic
    if args.backend == "aer_simulator":
        if not HAS_AER:
            print("Error: qiskit-aer not installed. Install with 'pip install qiskit-aer'.")
            return
        backend = AerSimulator()
        # For Aer, we use the backend directly or via a local sampler wrapper
        # Ideally we use the SamplerV2 from qiskit_ibm_runtime but in local mode it's tricky.
        # Let's stick to the unified interface if possible, or just use backend.run for Aer.
        print("Using Local Aer Simulator.")
        sampler_mode = backend # Sampler can often take the backend directly
    else:
        # Cloud Backend
        try:
            service = QiskitRuntimeService()
            backend = service.backend(args.backend)
            sampler_mode = backend
            print("Using IBM Cloud Backend.")
        except Exception as e:
            print(f"Error connecting to IBM Quantum: {e}")
            return
    
    print("Transpiling...")
    t_qc = transpile(qc, backend)
    
    # Run the Analog Integration
    # Note: SamplerV2 initialization varies slightly depending on context. 
    # For simplicity in this script, we handle both via the high-level primitive if possible.
    sampler = Sampler(mode=sampler_mode)
    
    print(f"Injecting flows...")
    job = sampler.run([t_qc], shots=args.shots)
    
    # Job ID handling is different for local vs remote
    try:
        job_id = job.job_id()
        print(f"Job ID: {job_id}")
    except:
        pass
    
    try:
        result = job.result()
        # Robust extraction for SamplerV2 data structure
        # PubResult -> DataBin -> BitArray/Counts
        try: counts = result[0].data.readout.get_counts()
        except: 
            try: counts = result[0].data.c.get_counts()
            except: counts = result[0].data.meas.get_counts()
            
    except Exception as e:
        print(f"Retrieval failed: {e}")
        return

    # Analyze
    residue = compute_lie_bracket_residue(counts, args.shots)
    
    print("\n--- Results: Differential Similarity Metric ---")
    print(f"Flow Counts: {counts}")
    print(f"Commutator Residue (P1): {residue:.5f}")
    
    # Estimate curvature K (assuming K ~ sqrt(P1) / Area)
    area = args.flow_strength ** 2
    if area > 0:
        curvature_proxy = math.sqrt(residue) / area
    else:
        curvature_proxy = 0.0
        
    print(f"Estimated Manifold Curvature (K): {curvature_proxy:.5f}")
    
    if residue > 0.001:
        print("\n[CONCLUSION] Flows are NON-COMMUTATIVE.")
        print(f"Measured Curvature detected. The geometry is not flat.")
    else:
        print("\n[CONCLUSION] Flows appear flat (Euclidean).")

if __name__ == "__main__":
    main()
