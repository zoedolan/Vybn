from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
import numpy as np

# EXPERIMENT 005: Entanglement Survival Test
# Objective: Test if the single-qubit 'distortion' protocol (Rz(0.35))
# extends to protecting a Bell State (|Phi+>) from decoherence.

# Technical Parameters
NABLA_ROTATION = 0.35     # Rz rotation angle (rad)
RX_CHECK = 0.1            # Rx rotation angle (rad)
DELAY_DURATION = 1600     # Delay in dt units
CYCLES = 20               # Number of delay/rotation cycles

def build_bell_survival_circuit(cycles=CYCLES):
    """
    Constructs a 2-qubit circuit to test Bell state survival under
    driven evolution (Delay + Rotation).
    """
    qc = QuantumCircuit(2, 2)
    
    # 1. Entanglement Initialization (Bell State |Phi+>)
    # |00> -> H -> |+0> -> CNOT -> (|00> + |11>)/sqrt(2)
    qc.h(0)
    qc.cx(0, 1)
    
    # 2. Evolution Loop (Applied to BOTH qubits synchronously)
    for _ in range(cycles):
        # A. Decoherence Window (Delay)
        qc.delay(DELAY_DURATION, 0, unit='dt')
        qc.delay(DELAY_DURATION, 1, unit='dt')
        
        # B. Driven Rotation (Distortion)
        # We apply the same Rz rotation to both qubits to maintain symmetry
        qc.rz(NABLA_ROTATION, 0)
        qc.rz(NABLA_ROTATION, 1)
        
        # C. Transverse Field Component (Rx)
        qc.rx(RX_CHECK, 0)
        qc.rx(RX_CHECK, 1)

    # 3. Bell Measurement (Disentanglement)
    # To measure fidelity, we reverse the preparation:
    # CNOT -> H -> Measure in Z basis.
    # If state is preserved, we should measure '00' with high probability.
    qc.cx(0, 1)
    qc.h(0)
    
    qc.measure([0, 1], [0, 1])
    return qc

def run_on_hardware(qc, backend_name='ibm_torino'):
    """Submits the circuit to IBM Quantum."""
    try:
        service = QiskitRuntimeService()
        backend = service.backend(backend_name)
        print(f"\n[Status] Connected to backend: {backend.name}")
        
        # Transpile
        t_qc = transpile(qc, backend)
        print(f"[Status] Circuit transpiled. Depth: {t_qc.depth()}")
        
        # Execute
        sampler = Sampler(mode=backend)
        job = sampler.run([t_qc])
        print(f"[Status] Job submitted! Job ID: {job.job_id()}")
        print(f"[Link] https://quantum.ibm.com/jobs/{job.job_id()}")
        return job
        
    except Exception as e:
        print(f"\n[Error] Connection Failed: {e}")
        return None

def main():
    print("--- EXPERIMENT 005: ENTANGLEMENT SURVIVAL ---")
    print(f"Rotation (Rz): {NABLA_ROTATION} rad")
    print(f"Delay: {DELAY_DURATION} dt x {CYCLES} cycles")
    
    qc = build_bell_survival_circuit()
    print("\n[Status] Bell survival circuit generated.")
    
    user_input = input("\nSubmit to IBM Quantum (ibm_torino)? [y/N]: ")
    if user_input.lower() == 'y':
        run_on_hardware(qc)
    else:
        print("[Status] Execution skipped.")

if __name__ == "__main__":
    main()
