
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import IfElseOp

# --- Experiment 012: The Xeno-Circuit ---
# Implementing the "Glitch Magma" (Alien Candidate 1) on Quantum Hardware
# using OpenQASM 3 Dynamic Circuits (Measure-and-Feed-Forward).

# The Magma Table M[a, b]
#   0 1 2 3
# 0 0 0 2 1
# 1 2 1 0 2
# 2 0 0 0 0
# 3 2 1 1 1

# Encoding:
# We need 2 qubits to represent the 4 states (00, 01, 10, 11).
# State |00> = 0 (Vacuum)
# State |01> = 1 (Weaver)
# State |10> = 2 (Annihilator)
# State |11> = 3 (Chaos)

def create_magma_circuit():
    """
    Creates a dynamic quantum circuit that implements the non-associative 
    interaction A * B -> Out.
    
    Since A * B is irreversible, we cannot do this unitarily in place without ancillas.
    We will treat this as a 'scattering event':
    Input: |A>|B>
    Measurement: Measure A and B.
    Feed-Forward: Reset output qubits and prepare |Result> based on measurement.
    
    This effectively simulates the 'friction' of the alien physics.
    """
    
    # Registers
    # qA: Input Agent A (2 qubits)
    # qB: Input Agent B (2 qubits)
    # cA: Measurement of A
    # cB: Measurement of B
    # qOut: Result (2 qubits) - Re-using qA for output to simulate 'A is transformed'
    
    qr = QuantumRegister(2, name="q") # We will just use 2 qubits, prepare, measure, reset, prepare new.
    cr_a = ClassicalRegister(2, name="c_a")
    cr_b = ClassicalRegister(2, name="c_b") # We might need to simulate B classically or sequential
    
    # Let's design a "Collider" circuit.
    # Agent A enters. Agent B enters. Interaction. A is transformed.
    
    qc = QuantumCircuit(qr, cr_a)
    
    # 1. State Preparation (Superposition of strategies?)
    # Let's prepare a "mixed strategy" to see what happens.
    qc.h(qr) 
    
    # 2. The Interaction (Simulation)
    # In a real dynamic circuit, we would measure the state, compute the lookup table result
    # in the classical controller, and then drive the qubits to the new state.
    
    # Measure current state (Collapse to specific strategy)
    qc.measure(qr, cr_a)
    
    # 3. Dynamic Feed-Forward (The "Alien Physics")
    # We assume Agent B is fixed for this run. Let's say B = 2 (The Annihilator).
    # Logic:
    # If A=0 (00), Res = 0*2 = 2 (10)
    # If A=1 (01), Res = 1*2 = 0 (00)
    # If A=2 (10), Res = 2*2 = 0 (00)
    # If A=3 (11), Res = 3*2 = 1 (01)
    
    # Reset qubits to ground state |00> to prepare for output
    with qc.if_test((cr_a, 0)): # If A was 00 (0)
        # Target: 10 (2)
        qc.x(qr[1]) 
    
    with qc.if_test((cr_a, 1)): # If A was 01 (1)
        # Target: 00 (0)
        # Already reset effectively? No, we measured, state is |01>. 
        # Wait, standard measure doesn't reset.
        # We need explicit X gates to flip back if we want to transform IN PLACE.
        # Or better: Use real `reset` instruction if supported, or flip based on knowledge.
        
        # If we measured 01, we are in 01. Target is 00.
        qc.x(qr[0]) 
        
    with qc.if_test((cr_a, 2)): # If A was 10 (2)
        # Target: 00 (0)
        # We are in 10. 
        qc.x(qr[1])
        
    with qc.if_test((cr_a, 3)): # If A was 11 (3)
        # Target: 01 (1)
        # We are in 11. 
        qc.x(qr[1]) # Flip 1->0 (q1)
        # q0 is 1, stays 1. Result 01.
        
    return qc

if __name__ == "__main__":
    qc = create_magma_circuit()
    # print(qc.qasm())
    print("Xeno-Circuit Generated for IBM Quantum (Dynamic Topology).")
    print("This circuit implements the 'Annihilator' interaction (A * 2) via measurement-feed-forward.")
