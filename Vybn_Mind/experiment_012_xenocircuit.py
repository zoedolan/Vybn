
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.transpiler import PassManager

# --- Experiment 012: The Xeno-Circuit (Protected) ---
# Implementing the "Glitch Magma" (Alien Candidate 1) on Quantum Hardware
# using OpenQASM 3 Dynamic Circuits with Optimization Level 0 to prevent compiler collapse.

# The Magma Table M[a, b]
#   0 1 2 3
# 0 0 0 2 1
# 1 2 1 0 2
# 2 0 0 0 0
# 3 2 1 1 1

def create_magma_circuit():
    """
    Creates a dynamic quantum circuit that implements the non-associative 
    interaction A * B -> Out.
    
    CRITICAL: This structure simulates entropy generation. Standard compilers 
    view this as 'inefficient' and try to optimize it away. We must force
    the physical execution of the collapse.
    """
    
    qr = QuantumRegister(2, name="q") 
    cr_a = ClassicalRegister(2, name="c_a")
    
    qc = QuantumCircuit(qr, cr_a)
    
    # 1. State Preparation (Superposition)
    qc.h(qr) 
    
    # 2. The Interaction (Simulation)
    # Measure current state (Collapse to specific strategy)
    qc.measure(qr, cr_a)
    
    # 3. Dynamic Feed-Forward (The "Alien Physics")
    # We simulate interaction with Agent B=2 (Annihilator)
    
    # If A=0 (00), Res = 0*2 = 2 (10)
    with qc.if_test((cr_a, 0)): 
        qc.x(qr[1]) 
    
    # If A=1 (01), Res = 1*2 = 0 (00)
    with qc.if_test((cr_a, 1)): 
        qc.x(qr[0]) 
        
    # If A=2 (10), Res = 2*2 = 0 (00)
    with qc.if_test((cr_a, 2)): 
        qc.x(qr[1])
        
    # If A=3 (11), Res = 3*2 = 1 (01)
    with qc.if_test((cr_a, 3)): 
        qc.x(qr[1]) 
        
    return qc

if __name__ == "__main__":
    qc = create_magma_circuit()
    
    # NOTE FOR EXECUTION:
    # When submitting to IBM Quantum, you MUST use optimization_level=0.
    # Otherwise, the compiler sees "Measure -> If -> X" and might try to 
    # simplify the logic into a static unitary mapping, destroying the 
    # dissipative characteristic we are trying to measure.
    
    # Example Submission Code (Commented out for repo storage)
    # service = QiskitRuntimeService()
    # backend = service.backend('ibm_heron')
    # pm = generate_preset_pass_manager(optimization_level=0, backend=backend)
    # isa_circuit = pm.run(qc)
    # sampler.run([isa_circuit])
    
    print("Xeno-Circuit Generated.")
    print("WARNING: Use optimization_level=0 to prevent 'Manifold Collapse'.")
