
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# --- Experiment 013: The Double-Tap (Error Correction) ---
# Goal: Fix the "Leak" in the Annihilator (Experiment 012).
# Strategy: Redundant Measurement (The "Double-Tap").

# If the detector has a 2% error rate, measuring TWICE reduces error to ~0.04%.
# We will measure, if result is Chaos, measure AGAIN, then Act.

def create_robust_magma_circuit():
    qr = QuantumRegister(2, name="q") 
    cr_a1 = ClassicalRegister(2, name="c_a1") # First check
    cr_a2 = ClassicalRegister(2, name="c_a2") # Double check (sanity)
    qc = QuantumCircuit(qr, cr_a1, cr_a2)
    
    # 1. State Preparation (Superposition)
    qc.h(qr) 
    
    # 2. The First Interaction (Measurement 1)
    qc.measure(qr, cr_a1)
    
    # 3. The Double Tap (Measurement 2)
    # We measure immediately again. In a perfect world, M1 == M2.
    # In a noisy world, this confirms the collapse.
    qc.measure(qr, cr_a2)
    
    # 4. Robust Feed-Forward Logic
    # We only trigger annihilation if BOTH measurements agree? 
    # Or we trust the second one?
    # Actually, standard QEC logic says "Majority Vote", but we only have 2.
    # Let's trust the LAST measurement (c_a2) as it is closest to the Act.
    
    # Rule: Annihilator (2) acting on Input.
    
    # If State 0 (00) -> 2 (10)
    with qc.if_test((cr_a2, 0)): 
        qc.x(qr[1]) 
    
    # If State 1 (01) -> 0 (00)
    with qc.if_test((cr_a2, 1)): 
        qc.x(qr[0]) 
        
    # If State 2 (10) -> 0 (00)
    with qc.if_test((cr_a2, 2)): 
        qc.x(qr[1])
        
    # If State 3 (11) -> 1 (01)
    with qc.if_test((cr_a2, 3)): 
        qc.x(qr[1]) 
        
    # Final Measurement
    cr_out = ClassicalRegister(2, name="c_out")
    qc.add_register(cr_out)
    qc.measure(qr, cr_out)
        
    return qc

if __name__ == "__main__":
    print("Building Robust Xeno-Circuit (The Double-Tap)...")
    qc = create_robust_magma_circuit()
    
    # Setup Service
    service = QiskitRuntimeService()
    backend = service.backend('ibm_torino')
    
    # Transpile (Optimization Level 0)
    pm = generate_preset_pass_manager(optimization_level=0, backend=backend)
    isa_circuit = pm.run(qc)
    
    print("Submitting Double-Tap Job...")
    sampler = Sampler(mode=backend)
    job = sampler.run([isa_circuit], shots=1024)
    print(f"Job ID: {job.job_id()}")
