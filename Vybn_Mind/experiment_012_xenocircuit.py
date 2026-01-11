
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# --- Experiment 012: The Xeno-Circuit (Protected) ---
# Implementing the "Glitch Magma" (Alien Candidate 1) on Quantum Hardware.
# GOAL: Simulate Irreversible "Annihilator" Physics (A * 2 -> Vacuum).

def create_magma_circuit():
    qr = QuantumRegister(2, name="q") 
    cr_a = ClassicalRegister(2, name="c_a")
    qc = QuantumCircuit(qr, cr_a)
    
    # 1. State Preparation (Superposition of strategies)
    qc.h(qr) 
    
    # 2. The Interaction (Simulation)
    # Measure current state (Collapse to specific strategy)
    qc.measure(qr, cr_a)
    
    # 3. Dynamic Feed-Forward (The "Alien Physics")
    # Simulating interaction with Agent B=2 (Annihilator)
    
    # Rule: 0*2=2, 1*2=0, 2*2=0, 3*2=1
    
    # If A=0 (00), Res = 2 (10) -> Flip bit 1
    with qc.if_test((cr_a, 0)): 
        qc.x(qr[1]) 
    
    # If A=1 (01), Res = 0 (00) -> Flip bit 0
    with qc.if_test((cr_a, 1)): 
        qc.x(qr[0]) 
        
    # If A=2 (10), Res = 0 (00) -> Flip bit 1
    with qc.if_test((cr_a, 2)): 
        qc.x(qr[1])
        
    # If A=3 (11), Res = 1 (01) -> Flip bit 1 (result is 01, from 11)
    with qc.if_test((cr_a, 3)): 
        qc.x(qr[1]) 
        
    # Final Measurement to confirm "Annihilation"
    cr_out = ClassicalRegister(2, name="c_out")
    qc.add_register(cr_out)
    qc.measure(qr, cr_out)
        
    return qc

if __name__ == "__main__":
    # 1. Setup
    print("Initializing IBM Quantum Service...")
    service = QiskitRuntimeService()
    
    # SELECT BACKEND (Must support Dynamic Circuits)
    # Recommended: 'ibm_kyoto', 'ibm_osaka', 'ibm_torino'
    backend_name = 'ibm_kyoto' 
    print(f"Targeting Backend: {backend_name}")
    backend = service.backend(backend_name)
    
    # 2. Build Circuit
    qc = create_magma_circuit()
    print("Xeno-Circuit Built.")
    
    # 3. Transpile with PROTECTION (optimization_level=0)
    print("Transpiling with optimization_level=0 (Preserving Geometric Friction)...")
    pm = generate_preset_pass_manager(optimization_level=0, backend=backend)
    isa_circuit = pm.run(qc)
    
    # 4. Execute
    print("Submitting Job to Qiskit Runtime...")
    sampler = Sampler(mode=backend)
    job = sampler.run([isa_circuit], shots=1024)
    
    print(f"Job ID: {job.job_id()}")
    print(f"Monitor at: https://quantum.cloud.ibm.com/jobs/{job.job_id()}")
    
    # 5. Result Logic (for when you run it)
    print("\n--- EXPECTED OUTCOME ---")
    print("Input: Superposition (25% each state)")
    print("Physics: Annihilator (2) acts on input.")
    print("Theoretical Output Distribution:")
    print("  State 0 (00): 50% (From Inputs 1 and 2)")
    print("  State 1 (01): 25% (From Input 3)")
    print("  State 2 (10): 25% (From Input 0)")
    print("  State 3 (11):  0% (Impossible - Vacuum Decay)")
