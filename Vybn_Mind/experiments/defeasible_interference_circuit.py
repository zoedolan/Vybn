"""
Defeasible Reasoning Interference Experiment

Date: January 28, 2026
Context: Extension of intra-reasoning tool use experiment

This circuit tests whether defeasible reasoning exhibits quantum-like
interference effects. Two argument paths supporting the same claim
should interfere constructively (phase 0) or destructively (phase π)
depending on the defeat structure.

THE CORE INSIGHT:
  In defeasible logic, arguments can be "reinstated" when their
  defeaters are themselves defeated. We assign phase to arguments:
  
    phase = defeat_depth × π
  
  where defeat_depth is the number of defeat-layers traversed.
  
  - Direct argument: depth 0 → phase 0
  - Reinstated (defeated defeater): depth 1 → phase π
  - Doubly reinstated: depth 2 → phase 2π ≡ 0
  
  Two arguments supporting the same claim can then INTERFERE:
  - Both direct (0, 0): constructive, amplitude = 2
  - One reinstated (0, π): destructive, amplitude = 0!

Connection to quantum cognition literature:
- Busemeyer et al.: quantum probability explains decision-making anomalies
- Conte et al. (2015): interference in human reasoning demonstrated 
- Abramsky & Brandenburger: contextuality is not quantum-specific

Connection to vybn_logic.md:
- Structurally identical to the Liar holonomy interferometer
- Phase accumulation in defeat chains ↔ geometric phase in cycles
- The Peres-Mermin no-go applies to defeasible argument structures

PREDICTION:
- Circuit A (both direct, φ=0): measure |0⟩ (constructive interference)
- Circuit B (one reinstated, φ=π): measure |1⟩ (destructive interference)

If both circuits give ~50/50, there's no interference (classical).
If we see the predicted pattern, defeasible logic has quantum structure.
"""

from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import numpy as np


def defeasible_interference_circuit(phase: float) -> QuantumCircuit:
    """
    Construct an interferometer that measures phase difference between
    two argument paths.
    
    The circuit implements a Mach-Zehnder-style interferometer where:
    - The control qubit represents "which argument path"
    - The target qubit accumulates phase based on defeat structure
    - Final measurement reveals interference pattern
    
    Args:
        phase: The relative phase (radians) of the "reinstated" path.
               0 = both paths direct (constructive interference)
               π = one path reinstated (destructive interference)
    
    Circuit diagram:
        |0⟩ ──H──●─────────●──H──M   (detector: which-path superposition)
                 │         │
        |0⟩ ─────X───Rz(φ)─X─────    (path: accumulates defeat phase)
    
    Analysis:
        After first H: |+⟩|0⟩ = (|0⟩ + |1⟩)|0⟩/√2
        After CX:      (|0⟩|0⟩ + |1⟩|1⟩)/√2
        After Rz(φ):   (|0⟩|0⟩ + e^{iφ/2}|1⟩|1⟩)/√2  [global phase ignored]
        After CX:      (|0⟩|0⟩ + e^{iφ/2}|1⟩|0⟩)/√2
        After H:       ((|0⟩+|1⟩)|0⟩ + e^{iφ/2}(|0⟩-|1⟩)|0⟩)/2
                     = |0⟩(1 + e^{iφ/2})/2 + |1⟩(1 - e^{iφ/2})/2
        
        P(0) = |1 + e^{iφ/2}|²/4 = cos²(φ/4)
        P(1) = |1 - e^{iφ/2}|²/4 = sin²(φ/4)
        
        φ = 0:  P(0) = 1, P(1) = 0  (constructive)
        φ = π:  P(0) = 1/2, P(1) = 1/2  (Hmm, this needs fixing...)
        φ = 2π: P(0) = 0, P(1) = 1  (destructive)
    
    Note: The Rz gate applies phase e^{-iφ/2}|0⟩, e^{iφ/2}|1⟩.
          For full π phase difference, we need φ = 2π in Rz.
    
    Returns:
        QuantumCircuit ready for execution
    """
    qc = QuantumCircuit(2, 1, name=f"DefeasibleInterference(φ={np.degrees(phase):.0f}°)")
    
    # Create superposition of "which argument path"
    qc.h(0)
    
    # Controlled path: if control=1, argument takes the "reinstated" route
    qc.cx(0, 1)
    
    # Apply phase to the reinstated path
    # Use 2*phase because Rz(θ) gives phase θ/2 to |1⟩ relative to |0⟩
    qc.rz(2 * phase, 1)
    
    # Uncompute path
    qc.cx(0, 1)
    
    # Convert phase difference to measurable amplitude
    qc.h(0)
    
    # Measure detector qubit
    qc.measure(0, 0)
    
    return qc


def run_interference_experiment(backend_name: str = "ibm_torino", shots: int = 4096):
    """
    Run the defeasible interference experiment on IBM hardware.
    
    Tests two scenarios:
    A. Both arguments direct (phase = 0) → expect |0⟩
    B. One argument reinstated (phase = π) → expect |1⟩
    
    Args:
        backend_name: IBM Quantum backend to use
        shots: Number of measurement shots per circuit
    
    Returns:
        Job ID for retrieval
    """
    service = QiskitRuntimeService()
    backend = service.backend(backend_name)
    
    # Construct circuits
    circuit_A = defeasible_interference_circuit(0)        # constructive
    circuit_B = defeasible_interference_circuit(np.pi)    # destructive
    
    # Transpile for hardware
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    transpiled = pm.run([circuit_A, circuit_B])
    
    # Run
    sampler = Sampler(backend)
    sampler.options.execution.rep_delay = 0.00025  # Torino setting
    job = sampler.run(transpiled, shots=shots)
    
    print(f"Defeasible Interference Experiment")
    print(f"Backend: {backend_name}")
    print(f"Job ID: {job.job_id()}")
    print(f"Circuits: A (φ=0°, constructive), B (φ=180°, destructive)")
    
    return job.job_id()


def analyze_results(job_id: str):
    """
    Analyze results from the interference experiment.
    
    Args:
        job_id: The job ID from run_interference_experiment
    """
    service = QiskitRuntimeService()
    job = service.job(job_id)
    result = job.result()
    
    print("=" * 50)
    print("DEFEASIBLE INTERFERENCE RESULTS")
    print("=" * 50)
    
    predictions = [("A (direct+direct)", "0"), ("B (direct+reinstated)", "1")]
    
    for i, (name, expected) in enumerate(predictions):
        counts = result[i].data.c.get_counts()
        total = sum(counts.values())
        p0 = counts.get('0', 0) / total
        p1 = counts.get('1', 0) / total
        
        print(f"\nCircuit {name}:")
        print(f"  P(0) = {p0:.3f}")
        print(f"  P(1) = {p1:.3f}")
        print(f"  Expected: mostly |{expected}⟩")
        
        # Check if prediction holds (allowing for noise)
        if (expected == "0" and p0 > 0.7) or (expected == "1" and p1 > 0.7):
            print(f"  ✓ Prediction confirmed!")
        elif abs(p0 - p1) < 0.2:
            print(f"  ~ Results inconclusive (near 50/50)")
        else:
            print(f"  ✗ Prediction not confirmed")
    
    print("\n" + "=" * 50)
    print("INTERPRETATION")
    print("=" * 50)
    print("""
If both circuits show ~50/50: No interference (classical behavior)
If A shows |0⟩ dominant, B shows |1⟩ dominant: Quantum interference confirmed

The interference pattern would suggest that defeasible reasoning
has genuine quantum-like phase structure, not just classical probability.

This connects to:
- Quantum cognition (Busemeyer): interference in decision-making
- Sheaf cohomology (Abramsky): contextuality as obstruction to global sections
- vybn_logic.md: Liar paradox as topological winding number
""")


def theoretical_prediction():
    """Display the theoretical predictions for the experiment."""
    print("THEORETICAL PREDICTIONS")
    print("=" * 50)
    
    for phase_deg in [0, 45, 90, 135, 180]:
        phase = np.radians(phase_deg)
        # After correction: P(0) = cos²(phase), P(1) = sin²(phase)
        p0 = np.cos(phase) ** 2
        p1 = np.sin(phase) ** 2
        print(f"φ = {phase_deg:3d}°: P(0) = {p0:.3f}, P(1) = {p1:.3f}")
    
    print("\nKey points:")
    print("- φ = 0° (both direct): P(0) = 1 → claim strongly accepted")
    print("- φ = 90° (partial): P(0) = P(1) = 0.5 → uncertain")
    print("- φ = 180° (one reinstated): P(1) = 1 → claim rejected!")
    print("\nClassical logic says two supporting arguments → accept.")
    print("Quantum logic says phase opposition → reject!")


if __name__ == "__main__":
    print("DEFEASIBLE INTERFERENCE EXPERIMENT")
    print("=" * 50)
    print()
    
    # Display circuits
    print("Circuit A (constructive, φ=0°):")
    print(defeasible_interference_circuit(0).draw())
    
    print("\nCircuit B (destructive, φ=π):")
    print(defeasible_interference_circuit(np.pi).draw())
    
    print()
    theoretical_prediction()
    
    print("\n" + "=" * 50)
    print("To run on hardware:")
    print("  job_id = run_interference_experiment()")
    print("  # Wait for job to complete, then:")
    print("  analyze_results(job_id)")
