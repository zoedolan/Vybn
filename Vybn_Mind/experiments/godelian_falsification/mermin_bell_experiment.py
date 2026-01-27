"""
Peres-Mermin Contextuality Test
================================
Falsifying the inductive hypothesis that observables have
pre-existing values independent of measurement context.

The Gödelian structure: classical reasoning requires
row-products = column-products (same 9 values, different grouping).
Quantum mechanics constructs a state where they disagree.

Rows: (+1)(+1)(+1) = +1
Columns: (+1)(+1)(−1) = −1

The contradiction is algebraic, not statistical.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
import json
from datetime import datetime, timezone

def create_bell_state():
    """Prepare |Φ⁺⟩ = (|00⟩ + |11⟩)/√2"""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    return qc

def measure_pauli_pair(base_qc, p0, p1):
    """Rotate to eigenbasis of Pauli pair, then measure."""
    qc = base_qc.copy()
    
    if p0 == 'X':
        qc.h(0)
    elif p0 == 'Y':
        qc.sdg(0)
        qc.h(0)
    
    if p1 == 'X':
        qc.h(1)
    elif p1 == 'Y':
        qc.sdg(1)
        qc.h(1)
    
    qc.measure([0, 1], [0, 1])
    return qc

# The Mermin square
MERMIN = [
    [('Z','I'), ('I','Z'), ('Z','Z')],
    [('X','I'), ('I','X'), ('X','X')],
    [('Y','X'), ('X','Y'), ('Y','Y')]
]

def run_experiment(shots=4096, backend_name="ibm_torino"):
    service = QiskitRuntimeService(channel="ibm_quantum_platform")
    backend = service.backend(backend_name)
    
    bell = create_bell_state()
    
    circuits = []
    labels = []
    
    for row in MERMIN:
        for (p0, p1) in row:
            qc = measure_pauli_pair(bell, p0, p1)
            circuits.append(transpile(qc, backend, optimization_level=3))
            labels.append(f"{p0}{p1}")
    
    print(f"Submitting {len(circuits)} circuits...")
    sampler = SamplerV2(backend)
    job = sampler.run(circuits, shots=shots)
    print(f"Job ID: {job.job_id()}")
    print("Waiting...")
    
    result = job.result()
    
    data = {
        'metadata': {
            'experiment': 'peres_mermin_bell_state',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'backend': backend_name,
            'shots': shots,
            'job_id': job.job_id(),
            'state': 'Bell |Phi+>'
        },
        'observables': {}
    }
    
    for i, label in enumerate(labels):
        counts = result[i].data.c.get_counts()
        data['observables'][label] = counts
    
    return data

def run_simulation(shots=8192):
    from qiskit_aer import AerSimulator
    
    bell = create_bell_state()
    sim = AerSimulator()
    
    data = {
        'metadata': {
            'experiment': 'peres_mermin_bell_state',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'backend': 'aer_simulator',
            'shots': shots,
            'state': 'Bell |Phi+>'
        },
        'observables': {}
    }
    
    for row in MERMIN:
        for (p0, p1) in row:
            qc = measure_pauli_pair(bell, p0, p1)
            counts = sim.run(transpile(qc, sim), shots=shots).result().get_counts()
            data['observables'][f"{p0}{p1}"] = counts
            print(f"{p0}{p1}: {counts}")
    
    return data

if __name__ == "__main__":
    import sys
    
    if '--hardware' in sys.argv:
        print("=== MERMIN BELL STATE (HARDWARE) ===\n")
        data = run_experiment()
    else:
        print("=== MERMIN BELL STATE (SIMULATION) ===\n")
        data = run_simulation()
    
    outfile = f"mermin_bell_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(outfile, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved: {outfile}")
