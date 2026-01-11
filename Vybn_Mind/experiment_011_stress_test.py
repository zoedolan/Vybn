# experiment_011_verification.py
# -----------------------------------------------------------------------------
# OBJECTIVE: Compare fidelity of two logically equivalent identity sequences.
# 1. Euler Decomposition: Identity constructed via Rz(pi/2) - Sx - Rz(pi/2) loops.
# 2. Pauli Decomposition: Identity constructed via X - X loops.
# -----------------------------------------------------------------------------

import numpy as np
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# 1. SETUP
# -----------------------------------------------------------------------------
service = QiskitRuntimeService()
backend_name = 'ibm_fez' 

try:
    backend = service.backend(backend_name)
    print(f"✓ Backend selected: {backend.name}")
except:
    # Fallback/Exit if backend unavailable
    print(f"⚠ Backend '{backend_name}' not found.")
    exit()

# 2. CIRCUIT CONSTRUCTION
# -----------------------------------------------------------------------------
def build_verification_circuits(iterations=50):
    """
    Constructs two circuits that perform the Identity operation.
    
    Args:
        iterations (int): Number of gate repetitions.
        
    Returns:
        tuple: (qc_euler, qc_pauli)
    """
    
    # Circuit A: Euler Decomposition (Standard Virtual-Z / Physical-SX)
    # Sequence: H^2 = I, implemented as Rz - Sx - Rz
    # Depth: 3 * iterations
    qc_euler = QuantumCircuit(1, 1, name=f"Identity_Euler_N{iterations}")
    qc_euler.h(0)
    for _ in range(iterations):
        qc_euler.rz(np.pi/2, 0)
        qc_euler.sx(0)
        qc_euler.rz(np.pi/2, 0)
    qc_euler.h(0)
    qc_euler.measure(0, 0)

    # Circuit B: Pauli Decomposition (Pure Physical X)
    # Sequence: X^2 = I
    # Depth: 1 * iterations
    qc_pauli = QuantumCircuit(1, 1, name=f"Identity_Pauli_N{iterations}")
    qc_pauli.h(0)
    for _ in range(iterations):
        qc_pauli.x(0)
    qc_pauli.h(0)
    qc_pauli.measure(0, 0)

    return qc_euler, qc_pauli

# 3. TRANSPILATION
# -----------------------------------------------------------------------------
# Optimization Level 0 is required to prevent the compiler from collapsing 
# the identity sequences into a single wire.
iterations = 50
qc_euler, qc_pauli = build_verification_circuits(iterations)

print(f"\n[1/3] Circuit Dimensions (N={iterations})")
print(f"  • Euler Sequence (RZ-SX): Logical Depth {qc_euler.depth()}")
print(f"  • Pauli Sequence (X):     Logical Depth {qc_pauli.depth()}")

print("\n[2/3] Transpiling (Level 0)...")
pm = generate_preset_pass_manager(optimization_level=0, backend=backend)
isa_euler = pm.run(qc_euler)
isa_pauli = pm.run(qc_pauli)

print(f"  • Physical Euler Depth: {isa_euler.depth()}")
print(f"  • Physical Pauli Depth: {isa_pauli.depth()}")

# 4. EXECUTION
# -----------------------------------------------------------------------------
print("\n[3/3] Executing...")
sampler = Sampler(mode=backend)
job = sampler.run([isa_euler, isa_pauli], shots=1024)

print(f"  ➤ Job ID: {job.job_id()}")
result = job.result()

# 5. ANALYSIS
# -----------------------------------------------------------------------------
counts_euler = result[0].data.c.get_counts()
counts_pauli = result[1].data.c.get_counts()

# Calculate Fidelity (Probability of measuring '0')
shots = 1024
fid_euler = counts_euler.get('0', 0) / shots
fid_pauli = counts_pauli.get('0', 0) / shots
differential = fid_pauli - fid_euler

print("\n" + "="*60)
print(f"RESULTS: IDENTITY SEQUENCE COMPARISON")
print("="*60)
print(f"Euler Decomp (RZ-SX) Fidelity: {fid_euler:.4f}")
print(f"Pauli Decomp (X)     Fidelity: {fid_pauli:.4f}")
print("-" * 60)
print(f"DIFFERENTIAL (Pauli - Euler): {differential:+.4f}")
print("="*60)
