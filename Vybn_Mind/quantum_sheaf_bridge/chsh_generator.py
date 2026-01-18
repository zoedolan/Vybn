import numpy as np
import json
import os
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

def create_chsh_circuits():
    """
    Creates the 4 circuits required for the CHSH game.
    Alice measures in bases A0 (Z) and A1 (X).
    Bob measures in bases B0 (Z+X) and B1 (Z-X).
    """
    circuits = {}
    
    # Bases angles (rotation around Y axis before measurement in Z)
    # Alice: A0=0 (Z), A1=pi/2 (X)
    # Bob: B0=pi/4, B1=-pi/4
    
    bases = [
        ('A0', 'B0', 0, np.pi/4),
        ('A0', 'B1', 0, -np.pi/4),
        ('A1', 'B0', np.pi/2, np.pi/4),
        ('A1', 'B1', np.pi/2, -np.pi/4)
    ]

    for (a_label, b_label, theta_a, theta_b) in bases:
        qc = QuantumCircuit(2, 2)
        
        # 1. Create Bell State |Phi+> = (|00> + |11>) / sqrt(2)
        qc.h(0)
        qc.cx(0, 1)
        
        # 2. Rotate for measurement
        # Alice (qubit 0)
        if theta_a != 0:
            qc.ry(-theta_a, 0) # Rotate basis
            
        # Bob (qubit 1)
        if theta_b != 0:
            qc.ry(-theta_b, 1) # Rotate basis
            
        # 3. Measure
        qc.measure([0, 1], [0, 1])
        
        circuits[f"{a_label}_{b_label}"] = qc
        
    return circuits

def run_experiment(backend_name='ibmq_qasm_simulator', shots=1024):
    circuits = create_chsh_circuits()
    results_data = {}
    
    print(f"Running CHSH experiment on {backend_name}...")
    
    # Check for IBM Quantum service
    service = None
    try:
        if os.getenv("QISKIT_IBM_TOKEN"):
             service = QiskitRuntimeService(channel="ibm_quantum")
    except Exception as e:
        print(f"Could not connect to IBM Quantum: {e}")

    if service and backend_name != 'simulator':
        backend = service.backend(backend_name)
        sampler = Sampler(backend=backend)
        # Convert circuits to list for batch execution
        # Note: SamplerV2 requires circuits to be transpiled or accepted by backend
        # For simplicity in this demo, we assume standard transpilation
        # job = sampler.run(list(circuits.values()), shots=shots) # simplified
        print("Using real hardware requires full transpile loop. Switching to simulator for demo reliability.")
        # Fallback to local simulator for reliability in this script unless configured
    
    # Use local simulator (Aer)
    from qiskit_aer import AerSimulator
    backend = AerSimulator()
    
    for name, qc in circuits.items():
        # Transpile for simulator
        # result = backend.run(qc, shots=shots).result() # old style
        # New style
        job = backend.run(qc, shots=shots)
        result = job.result()
        counts = result.get_counts()
        results_data[name] = counts
        
    # Calculate Correlations
    # E = P(00) + P(11) - P(01) - P(10)
    S_value = 0
    correlations = {}
    
    for name, counts in results_data.items():
        total = sum(counts.values())
        p00 = counts.get('00', 0) / total
        p11 = counts.get('11', 0) / total
        p01 = counts.get('01', 0) / total
        p10 = counts.get('10', 0) / total
        
        E = p00 + p11 - p01 - p10
        correlations[name] = E
        print(f"Context {name}: Correlation E = {E:.4f}")
        
    # CHSH Inequality: S = |E(A0,B0) - E(A0,B1)| + |E(A1,B0) + E(A1,B1)|
    # Wait, standard form: S = E(A0,B0) - E(A0,B1) + E(A1,B0) + E(A1,B1)
    # Actually usually it is A0B0 + A0B1 + A1B0 - A1B1 (if bases are chosen right)
    # With our angles: 
    # A0(Z), B0(Z+X) -> cos(pi/4) = 0.707
    # A0(Z), B1(Z-X) -> cos(-pi/4) = 0.707
    # A1(X), B0(Z+X) -> sin(pi/4) = 0.707
    # A1(X), B1(Z-X) -> sin(-pi/4) = -0.707
    # So E(A0,B0) + E(A0,B1) + E(A1,B0) - E(A1,B1) should be 2*sqrt(2) ~ 2.82
    
    S = correlations['A0_B0'] + correlations['A0_B1'] + correlations['A1_B0'] - correlations['A1_B1']
    print(f"\nCHSH Value S = {S:.4f}")
    if abs(S) > 2:
        print(">> VIOLATION OBSERVED (Quantum Contextuality Present)")
    else:
        print(">> Classical Limit Not Exceeded")

    # Save data for Sheaf Analysis
    with open('chsh_data.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    print("Data saved to chsh_data.json")

if __name__ == "__main__":
    run_experiment()
