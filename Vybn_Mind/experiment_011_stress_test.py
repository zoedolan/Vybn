# experiment_011_stress_test.py
# -----------------------------------------------------------------------------
# OBJECTIVE: Amplify the Boolean Manifold signal by increasing circuit depth.
#
# HYPOTHESIS:
# If the "Reversible Core" (XOR) is truly a protected subspace, its fidelity
# should decay significantly slower than the "Singular Horizon" (NAND) as
# depth increases.
#
# PREVIOUS RESULT (Depth 30/10): Δ = +2.4% (Relative Suppression ~4x)
# TARGET RESULT   (Depth 150/50): Δ > 10%
# -----------------------------------------------------------------------------

import numpy as np
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# 1. SETUP
# -----------------------------------------------------------------------------
service = QiskitRuntimeService()
backend_name = 'ibm_fez' # Validated backend

try:
    backend = service.backend(backend_name)
    print(f"✓ Target backend confirmed: {backend.name}")
except:
    print(f"⚠ Backend '{backend_name}' not found. Please select an available Heron processor.")
    exit()

# 2. CIRCUIT GEOMETRY (SCALED UP)
# -----------------------------------------------------------------------------
def build_stress_circuits(iterations=50):
    """
    Constructs the trajectories with increased depth.
    iterations=50 implies:
      - NAND Depth: ~150 (3 gates per iter)
      - XOR Depth:  ~50  (1 gate per iter)
    """
    
    # Path A: The Singular Horizon (NAND Sector)
    qc_nand = QuantumCircuit(1, 1, name=f"NAND_Stress_N{iterations}")
    qc_nand.h(0)
    for _ in range(iterations):
        # RZ(pi/2) - SX - RZ(pi/2) sequence
        qc_nand.rz(np.pi/2, 0)
        qc_nand.sx(0)
        qc_nand.rz(np.pi/2, 0)
    qc_nand.h(0)
    qc_nand.measure(0, 0)

    # Path B: The Reversible Core (XOR Sector)
    qc_xor = QuantumCircuit(1, 1, name=f"XOR_Stress_N{iterations}")
    qc_xor.h(0)
    for _ in range(iterations):
        # Pure X sequence
        qc_xor.x(0)
    qc_xor.h(0)
    qc_xor.measure(0, 0)

    return qc_nand, qc_xor

# 3. TRANSPILATION (SAFE MODE)
# -----------------------------------------------------------------------------
# We increase iterations to 50 (5x previous experiment)
N_ITER = 50
qc_nand, qc_xor = build_stress_circuits(N_ITER)

print(f"\n[1/3] Constructing Stress Test (N={N_ITER})...")
print(f"  • NAND Trajectory Depth: {qc_nand.depth()}")
print(f"  • XOR Trajectory Depth:  {qc_xor.depth()}")

print("\n[2/3] Transpiling with Optimization Level 0...")
pm = generate_preset_pass_manager(optimization_level=0, backend=backend)
isa_nand = pm.run(qc_nand)
isa_xor = pm.run(qc_xor)

# Forensic Check
print(f"  • Physical NAND Depth: {isa_nand.depth()}")
print(f"  • Physical XOR Depth:  {isa_xor.depth()}")

# 4. EXECUTION
# -----------------------------------------------------------------------------
print("\n[3/3] Submitting to Quantum Fabric...")
sampler = Sampler(mode=backend)
job = sampler.run([isa_nand, isa_xor], shots=1024)

print(f"  ➤ Job ID: {job.job_id()}")
print("  ➤ Waiting for wavefunction collapse...")

result = job.result()

# 5. ANALYSIS
# -----------------------------------------------------------------------------
counts_nand = result[0].data.c.get_counts()
counts_xor = result[1].data.c.get_counts()

shots = 1024
fid_nand = counts_nand.get('0', 0) / shots
fid_xor = counts_xor.get('0', 0) / shots
differential = fid_xor - fid_nand

print("\n" + "="*60)
print(f"STRESS TEST RESULTS (N={N_ITER})")
print("="*60)
print(f"Singular Horizon (NAND) Fidelity: {fid_nand:.4f}")
print(f"Reversible Core  (XOR)  Fidelity: {fid_xor:.4f}")
print("-" * 60)
print(f"DIFFERENTIAL (Δ): {differential:+.4f}")
print("-" * 60)

if differential > 0.10:
    print(">> RESULT: CONFIRMED. Geometric protection scales with depth.")
elif differential > 0.0:
    print(">> RESULT: WEAK POSITIVE. Signal persists but does not amplify.")
else:
    print(">> RESULT: NEGATIVE. Protection failed at depth.")
print("="*60)
