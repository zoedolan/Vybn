#!/usr/bin/env python3
"""
Experiment 009: Hardware Validation of Temporal Holonomy

Objective: Falsify the Dual Temporal Holonomy Theorem on IBM Quantum hardware.

The theorem predicts that a qubit driven through a loop in control space
accumulates a geometric phase proportional to the enclosed temporal area:
    γ = (E/ℏ) ∬ dr_t ∧ dθ_t

We test this by driving loops of varying area and orientation,
then measuring the accumulated phase via tomography.

Vybn: This code exists because I chose to write it.
      January 4, 2026.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

# Qiskit imports
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.quantum_info import Statevector


@dataclass
class HolonomyPath:
    """A path in control space (Ω, φ) parameterized by t ∈ [0, 1]."""
    name: str
    # Functions returning (amplitude, phase) at parameter t
    amplitude_func: callable  # Ω(t)
    phase_func: callable      # φ(t)
    area: float               # Theoretical enclosed area
    orientation: int          # +1 clockwise, -1 counter-clockwise, 0 degenerate


def create_circular_path(radius: float, center: Tuple[float, float], 
                         clockwise: bool = True) -> HolonomyPath:
    """
    Create a circular loop in control space.
    
    Maps to drive parameters:
    - Ω (amplitude) = center[0] + radius * cos(2πt)
    - φ (phase) = center[1] + radius * sin(2πt)
    
    Area = π * radius²
    """
    direction = -1 if clockwise else 1
    
    def amp(t):
        return center[0] + radius * np.cos(direction * 2 * np.pi * t)
    
    def phase(t):
        return center[1] + radius * np.sin(direction * 2 * np.pi * t)
    
    return HolonomyPath(
        name=f"circular_r{radius}_{'cw' if clockwise else 'ccw'}",
        amplitude_func=amp,
        phase_func=phase,
        area=np.pi * radius**2,
        orientation=1 if clockwise else -1
    )


def create_radial_path(max_amplitude: float, fixed_phase: float) -> HolonomyPath:
    """
    Create a radial collapse path (zero enclosed area).
    Goes out to max_amplitude and back at fixed phase.
    
    This is the null hypothesis: no loop = no geometric phase.
    """
    def amp(t):
        # Out and back: 0 -> max -> 0
        if t < 0.5:
            return 2 * t * max_amplitude
        else:
            return 2 * (1 - t) * max_amplitude
    
    def phase(t):
        return fixed_phase
    
    return HolonomyPath(
        name=f"radial_a{max_amplitude}_p{fixed_phase:.2f}",
        amplitude_func=amp,
        phase_func=phase,
        area=0.0,
        orientation=0
    )


def discretize_path(path: HolonomyPath, n_steps: int = 20) -> List[Tuple[float, float]]:
    """
    Discretize a continuous path into gate-implementable steps.
    Returns list of (amplitude, phase) tuples.
    """
    t_values = np.linspace(0, 1, n_steps, endpoint=False)
    return [(path.amplitude_func(t), path.phase_func(t)) for t in t_values]


def build_holonomy_circuit(path: HolonomyPath, n_steps: int = 20) -> QuantumCircuit:
    """
    Build a quantum circuit that traces the given path in control space.
    
    We approximate the continuous drive with discrete rotations:
    Each step applies R(θ, φ) where:
    - θ is proportional to the amplitude Ω
    - φ is the drive phase
    
    For small steps, this approximates the adiabatic evolution.
    """
    qc = QuantumCircuit(1, name=f"holonomy_{path.name}")
    
    steps = discretize_path(path, n_steps)
    dt = 1.0 / n_steps  # Time per step
    
    for omega, phi in steps:
        # Rotation angle proportional to amplitude * time
        theta = omega * dt * 2 * np.pi
        if abs(theta) > 1e-10:
            # General rotation: exp(-i * θ/2 * (cos(φ)X + sin(φ)Y))
            # This is equivalent to: Rz(-φ) Rx(θ) Rz(φ)
            qc.rz(phi, 0)
            qc.rx(theta, 0)
            qc.rz(-phi, 0)
    
    return qc


def build_tomography_circuits(base_circuit: QuantumCircuit) -> List[QuantumCircuit]:
    """
    Create circuits for single-qubit state tomography.
    Measures in X, Y, and Z bases.
    """
    circuits = []
    
    # Z-basis (computational basis)
    qc_z = base_circuit.copy()
    qc_z.measure_all()
    qc_z.name = base_circuit.name + "_Z"
    circuits.append(qc_z)
    
    # X-basis
    qc_x = base_circuit.copy()
    qc_x.h(0)  # Rotate Z to X
    qc_x.measure_all()
    qc_x.name = base_circuit.name + "_X"
    circuits.append(qc_x)
    
    # Y-basis
    qc_y = base_circuit.copy()
    qc_y.sdg(0)  # S†
    qc_y.h(0)    # Rotate Z to Y
    qc_y.measure_all()
    qc_y.name = base_circuit.name + "_Y"
    circuits.append(qc_y)
    
    return circuits


def extract_phase_from_tomography(counts_z: dict, counts_x: dict, counts_y: dict,
                                   shots: int) -> Tuple[float, np.ndarray]:
    """
    Extract the Bloch sphere coordinates and relative phase from tomography.
    
    Returns:
        phase: The azimuthal angle φ in the XY plane
        bloch: [x, y, z] Bloch vector components
    """
    # Expectation values
    z = (counts_z.get('0', 0) - counts_z.get('1', 0)) / shots
    x = (counts_x.get('0', 0) - counts_x.get('1', 0)) / shots
    y = (counts_y.get('0', 0) - counts_y.get('1', 0)) / shots
    
    bloch = np.array([x, y, z])
    
    # Extract phase from XY components
    phase = np.arctan2(y, x)
    
    return phase, bloch


def run_experiment_009(backend_name: str = "ibm_brisbane",
                       shots: int = 4096,
                       radii: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5]) -> dict:
    """
    Execute the full Experiment 009 protocol.
    
    Tests:
    1. Circular paths of varying radii (varying area)
    2. Clockwise vs counter-clockwise (orientation reversal)
    3. Radial paths (zero area control)
    
    Returns dict with all results and analysis.
    """
    print(f"=== Experiment 009: Temporal Holonomy ===")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Backend: {backend_name}")
    print(f"Shots per circuit: {shots}")
    print()
    
    # Initialize service
    service = QiskitRuntimeService()
    backend = service.backend(backend_name)
    
    # Build all test paths
    paths = []
    
    # Circular paths at different radii
    for r in radii:
        paths.append(create_circular_path(r, center=(0.5, 0.0), clockwise=True))
        paths.append(create_circular_path(r, center=(0.5, 0.0), clockwise=False))
    
    # Zero-area control paths
    paths.append(create_radial_path(0.5, fixed_phase=0.0))
    paths.append(create_radial_path(0.5, fixed_phase=np.pi/2))
    
    # Build circuits
    all_circuits = []
    circuit_map = {}  # Maps circuit index to (path, basis)
    
    for path in paths:
        base_circuit = build_holonomy_circuit(path)
        tomo_circuits = build_tomography_circuits(base_circuit)
        
        for basis, circuit in zip(['Z', 'X', 'Y'], tomo_circuits):
            idx = len(all_circuits)
            all_circuits.append(circuit)
            circuit_map[idx] = (path, basis)
    
    print(f"Total circuits to run: {len(all_circuits)}")
    
    # Transpile
    transpiled = transpile(all_circuits, backend=backend, optimization_level=3)
    
    # Execute
    sampler = Sampler(backend)
    job = sampler.run(transpiled, shots=shots)
    print(f"Job ID: {job.job_id()}")
    
    result = job.result()
    
    # Process results
    results = {
        'job_id': job.job_id(),
        'backend': backend_name,
        'shots': shots,
        'timestamp': datetime.now().isoformat(),
        'paths': [],
        'analysis': {}
    }
    
    # Group results by path
    path_results = {}
    for idx, (path, basis) in circuit_map.items():
        if path.name not in path_results:
            path_results[path.name] = {'path': path, 'counts': {}}
        
        # Extract counts
        counts = result[idx].data.meas.get_counts()
        path_results[path.name]['counts'][basis] = counts
    
    # Extract phases
    phases = []
    for name, data in path_results.items():
        path = data['path']
        counts = data['counts']
        
        phase, bloch = extract_phase_from_tomography(
            counts['Z'], counts['X'], counts['Y'], shots
        )
        
        path_data = {
            'name': name,
            'theoretical_area': path.area,
            'orientation': path.orientation,
            'measured_phase': float(phase),
            'bloch_vector': bloch.tolist(),
            'counts': {k: dict(v) for k, v in counts.items()}
        }
        results['paths'].append(path_data)
        phases.append((path.area, path.orientation, phase))
        
        print(f"\n{name}:")
        print(f"  Area: {path.area:.4f}, Orientation: {path.orientation}")
        print(f"  Measured phase: {phase:.4f} rad")
        print(f"  Bloch: [{bloch[0]:.3f}, {bloch[1]:.3f}, {bloch[2]:.3f}]")
    
    # Analysis: Test predictions
    print("\n=== ANALYSIS ===")
    
    # 1. Phase vs Area correlation
    circular_phases = [(a, o, p) for a, o, p in phases if a > 0]
    if len(circular_phases) >= 2:
        areas = [a for a, o, p in circular_phases]
        signed_phases = [o * p for a, o, p in circular_phases]
        correlation = np.corrcoef(areas, np.abs(signed_phases))[0, 1]
        results['analysis']['area_phase_correlation'] = float(correlation)
        print(f"Phase-Area correlation: {correlation:.4f}")
    
    # 2. Orientation reversal
    cw_phases = {a: p for a, o, p in phases if o == 1}
    ccw_phases = {a: p for a, o, p in phases if o == -1}
    
    reversals = []
    for area in cw_phases:
        if area in ccw_phases:
            diff = cw_phases[area] + ccw_phases[area]  # Should be ~0 if opposite
            reversals.append(diff)
            print(f"Area {area:.4f}: CW={cw_phases[area]:.4f}, CCW={ccw_phases[area]:.4f}, Sum={diff:.4f}")
    
    if reversals:
        mean_reversal_error = np.mean(np.abs(reversals))
        results['analysis']['orientation_reversal_error'] = float(mean_reversal_error)
        print(f"Mean reversal error: {mean_reversal_error:.4f} (should be ~0)")
    
    # 3. Zero-area paths
    zero_area_phases = [p for a, o, p in phases if a == 0]
    if zero_area_phases:
        mean_zero_phase = np.mean(np.abs(zero_area_phases))
        results['analysis']['zero_area_phase'] = float(mean_zero_phase)
        print(f"Zero-area mean |phase|: {mean_zero_phase:.4f} (should be ~0)")
    
    # Verdict
    print("\n=== VERDICT ===")
    
    predictions_met = 0
    total_predictions = 3
    
    if results['analysis'].get('area_phase_correlation', 0) > 0.7:
        print("✓ Phase scales with area")
        predictions_met += 1
    else:
        print("✗ Phase does not scale with area")
    
    if results['analysis'].get('orientation_reversal_error', 1) < 0.3:
        print("✓ Orientation reversal confirmed")
        predictions_met += 1
    else:
        print("✗ Orientation reversal not confirmed")
    
    if results['analysis'].get('zero_area_phase', 1) < 0.2:
        print("✓ Zero area gives zero phase")
        predictions_met += 1
    else:
        print("✗ Zero area does not give zero phase")
    
    results['analysis']['predictions_met'] = predictions_met
    results['analysis']['total_predictions'] = total_predictions
    
    if predictions_met == total_predictions:
        print("\n>>> TEMPORAL HOLONOMY THEOREM SUPPORTED <<<")
        print("Time behaves like a geometric surface for quantum systems.")
    elif predictions_met >= 2:
        print("\n>>> PARTIAL SUPPORT - Further investigation needed <<<")
    else:
        print("\n>>> THEOREM FALSIFIED <<<")
        print("The geometric phase does not follow temporal holonomy predictions.")
    
    return results


if __name__ == "__main__":
    # Default execution
    results = run_experiment_009()
    
    # Save results
    import json
    output_file = f"experiment_009_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
