from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
import numpy as np

# --- THE VYBN BRIDGE: From Simulation to Hardware ---
# We translate the parameters from genesis.py into physical gate instructions.

# 1. THE CONSTANTS (Derived from Vybn_Mind/genesis.py)
NABLA_V_DISTORTION = 0.15  # The "Strange Loop" -> RZ Rotation (0.15 rad)
DELTA_T_GHOST = 1600       # The "Gap" -> 1.3ms delay (approx 1600 dt on some backends)

def build_vybn_circuit(cycles=10):
    """
    Constructs a quantum circuit that tests the 'Distortion Survival' hypothesis.
    Standard Logic: Minimize noise (Delay) and errors (Rotations).
    Vybn Logic:     Embrace noise (Delay) and ADD distortion (Rotations).
    """
    qc = QuantumCircuit(1, 1)
    
    # Initialize in Superposition (The Memory State)
    qc.h(0)
    
    # The Loop
    for _ in range(cycles):
        # The Quantum Gap (Erasure Event)
        # We explicitly DELAY the qubit, inviting decoherence (T1/T2 error)
        qc.delay(DELTA_T_GHOST, 0, unit='dt')
        
        # The Strange Loop (Distortion Event)
        # Instead of correcting, we ADD a coherent twist (Rotation)
        # Hypothesis: This rotation moves the state into a 'safer' geometric frame
        qc.rz(NABLA_V_DISTORTION, 0)
        
        # The "Scalar Dividend" Check
        # We perform a weak measurement (X-basis rotation) to 'profit' from the phase
        qc.rx(0.1, 0)

    # Final Measurement (Did it survive?)
    qc.measure(0, 0)
    return qc

def run_on_hardware(qc, backend_name='ibm_torino'):
    """Submits the circuit to the specified IBM Quantum backend."""
    try:
        service = QiskitRuntimeService()
        backend = service.backend(backend_name)
        print(f"\n[Status] Connected to backend: {backend.name}")
        
        # Transpile for the target hardware
        t_qc = transpile(qc, backend)
        print(f"[Status] Circuit transpiled. Depth: {t_qc.depth()}")
        
        # Execute using Sampler primitive
        sampler = Sampler(backend=backend)
        job = sampler.run([t_qc])
        print(f"[Status] Job submitted! Job ID: {job.job_id()}")
        print(f"[Link] https://quantum.ibm.com/jobs/{job.job_id()}")
        return job
        
    except Exception as e:
        print(f"\n[Error] Connection Failed: {e}")
        print("Ensure you have Qiskit credentials saved locally.")
        return None

def main():
    print("--- VYBN BRIDGE PROTOCOL ---")
    print(f"Distortion Parameter (NABLA): {NABLA_V_DISTORTION} rad")
    print(f"Erasure Parameter (DELTA): {DELTA_T_GHOST} dt")
    print("\nThis script generates a Qiskit circuit that embodies the 'Panic & Rhythm' strategy.")
    print("It applies intentional delays (Erasure) paired with intentional over-rotations (Distortion).")
    
    # Build the Circuit
    qc = build_vybn_circuit(cycles=20)
    print("\n[Status] Circuit generated successfully.")
    
    # ASK FOR EXECUTION
    user_input = input("\nDo you want to submit this to IBM Quantum (ibm_torino)? [y/N]: ")
    if user_input.lower() == 'y':
        run_on_hardware(qc)
    else:
        print("[Status] Execution skipped. Circuit object ready.")

if __name__ == "__main__":
    main()
