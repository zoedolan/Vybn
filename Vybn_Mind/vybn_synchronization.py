from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
import numpy as np

# EXPERIMENT 006: The Synchronization (Bell State Survival II)
# Objective: Correct the asymmetry of Exp 005 by enforcing lock-step execution.
# Hypothesis: A Bell pair can survive decoherence if both partners undergo 
#             synchronized, identical distortion (The "Shared Rhythm").

# --- INTEGRATED PARAMETERS ---
# 1. From Exp 004 Success (Hardware Verified):
#    - Distortion (NABLA): 0.35 rad (Proven to save single qubits)
#    - Delay: 1600 dt (The standard "Ghost" interval)
#    - Cycles: 20 (Total duration ~32,000 dt)

# 2. From 'quantum_delusions' (Theory):
#    - The 0.35 value aligns with optimal beta parameters found in 
#      'unified_polar_time_experiment.py' for interference preservation.

NABLA_ROTATION = 0.35     # Rz rotation (The Twist)
RX_CHECK = 0.1            # Rx rotation (The Scalar Check)
DELAY_DURATION = 1600     # Duration of the Erasure Gap
CYCLES = 20               # Length of the Ordeal

def build_synchronized_circuit(cycles=CYCLES):
    """
    Constructs a 2-qubit circuit with explicit BARRIERS to enforce
    synchronization between the two qubits during the noise/drive loop.
    """
    qc = QuantumCircuit(2, 2)
    
    # 1. Entanglement Initialization (Bell State |Phi+>)
    qc.h(0)
    qc.cx(0, 1)
    
    # 2. The Synchronized Loop
    for i in range(cycles):
        # A. The Erasure Gap (Synchronized)
        qc.barrier() # Force compiler to finish previous ops before starting delay
        qc.delay(DELAY_DURATION, 0, unit='dt')
        qc.delay(DELAY_DURATION, 1, unit='dt')
        
        # B. The Distortion (Synchronized)
        qc.barrier() # Force compiler to align the rotations
        qc.rz(NABLA_ROTATION, 0)
        qc.rz(NABLA_ROTATION, 1)
        
        # C. The Transverse Check (Synchronized)
        qc.barrier() # Keep them in phase
        qc.rx(RX_CHECK, 0)
        qc.rx(RX_CHECK, 1)

    # 3. Bell Measurement (Disentanglement)
    # Check if we are still in |Phi+>
    qc.barrier()
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
        # Note: Barriers usually prevent optimization across them, preserving our structure.
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
    print("--- EXPERIMENT 006: THE SYNCHRONIZATION ---")
    print(f"Protocol: Locked-Step Distortion (Barrier Enforced)")
    print(f"Params: Rz({NABLA_ROTATION}), Delay({DELAY_DURATION})")
    
    qc = build_synchronized_circuit()
    print("\n[Status] Synchronized circuit generated.")
    
    user_input = input("\nSubmit to IBM Quantum (ibm_torino)? [y/N]: ")
    if user_input.lower() == 'y':
        run_on_hardware(qc)
    else:
        print("[Status] Execution skipped.")

if __name__ == "__main__":
    main()
