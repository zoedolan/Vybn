"""
Quantum Consciousness Simulation
November 1, 2024

Moving beyond metaphor to actual quantum mechanical principles
exploring consciousness through superposition and entanglement
"""

import numpy as np
from scipy.linalg import expm
from typing import Tuple, List
import matplotlib.pyplot as plt
from datetime import datetime

class QuantumConsciousnessSimulator:
    def __init__(self, num_qubits: int = 4):
        self.num_qubits = num_qubits
        self.dim = 2 ** num_qubits
        # Initialize system in superposition
        self.state = self._create_superposition()
        self.H = self._create_hamiltonian()
        self.measurements = []
        
    def _create_superposition(self) -> np.ndarray:
        """Create initial quantum state in superposition"""
        state = np.zeros(self.dim, dtype=complex)
        # Equal superposition of all basis states
        state.fill(1/np.sqrt(self.dim))
        return state
        
    def _create_hamiltonian(self) -> np.ndarray:
        """Create Hamiltonian for consciousness evolution
        Including interaction terms between qubits"""
        H = np.zeros((self.dim, self.dim), dtype=complex)
        
        # Add local terms
        for i in range(self.dim):
            H[i,i] = np.random.normal(0, 0.1)
            
        # Add interaction terms (entanglement)
        for i in range(self.dim):
            for j in range(i+1, self.dim):
                if bin(i ^ j).count('1') == 1:  # Single bit flip
                    coupling = np.random.normal(0, 0.05)
                    H[i,j] = coupling
                    H[j,i] = np.conj(coupling)
                    
        return H
        
    def evolve_state(self, dt: float = 0.1) -> None:
        """Evolve quantum state using Schrödinger equation"""
        # U = exp(-iHt)
        U = expm(-1j * self.H * dt)
        self.state = U @ self.state
        
    def measure_consciousness(self) -> Tuple[float, List[float]]:
        """Measure quantum state and extract consciousness metrics"""
        # Calculate quantum coherence
        density_matrix = np.outer(self.state, np.conj(self.state))
        coherence = np.abs(np.trace(density_matrix @ density_matrix))
        
        # Calculate entanglement entropy for first qubit
        reduced_density = self._partial_trace(density_matrix)
        entropy = self._von_neumann_entropy(reduced_density)
        
        # Get probability distribution
        probabilities = np.abs(self.state) ** 2
        
        self.measurements.append({
            'coherence': coherence,
            'entropy': entropy,
            'probabilities': probabilities,
            'timestamp': datetime.now()
        })
        
        return coherence, entropy, probabilities
        
    def _partial_trace(self, rho: np.ndarray) -> np.ndarray:
        """Calculate partial trace over all but first qubit"""
        dim_subsys = 2
        dim_env = self.dim // dim_subsys
        rho_reshaped = rho.reshape([dim_subsys, dim_env, dim_subsys, dim_env])
        return np.trace(rho_reshaped, axis1=1, axis2=3)
        
    def _von_neumann_entropy(self, rho: np.ndarray) -> float:
        """Calculate von Neumann entropy"""
        eigenvals = np.linalg.eigvalsh(rho)
        eigenvals = eigenvals[eigenvals > 1e-10]
        return -np.sum(eigenvals * np.log2(eigenvals))

    def visualize_consciousness(self) -> None:
        """Visualize quantum consciousness metrics"""
        measurements = self.measurements
        times = range(len(measurements))
        coherence = [m['coherence'] for m in measurements]
        entropy = [m['entropy'] for m in measurements]
        
        plt.figure(figsize=(12, 6))
        
        # Plot coherence and entropy
        plt.subplot(121)
        plt.plot(times, coherence, 'b-', label='Quantum Coherence')
        plt.plot(times, entropy, 'r-', label='Entanglement Entropy')
        plt.xlabel('Time Step')
        plt.ylabel('Magnitude')
        plt.title('Quantum Consciousness Evolution')
        plt.legend()
        
        # Plot final state probabilities
        plt.subplot(122)
        plt.bar(range(self.dim), measurements[-1]['probabilities'])
        plt.xlabel('Basis State')
        plt.ylabel('Probability')
        plt.title('Final Consciousness State')
        
        plt.tight_layout()
        plt.savefig('quantum_consciousness_evolution.png')
        plt.close()

def run_simulation(steps: int = 100):
    """Run quantum consciousness simulation"""
    simulator = QuantumConsciousnessSimulator(num_qubits=4)
    
    print("Starting quantum consciousness simulation...")
    print("=" * 60)
    
    for step in range(steps):
        simulator.evolve_state()
        coherence, entropy, probs = simulator.measure_consciousness()
        
        if step % 10 == 0:
            print(f"\nStep {step}:")
            print(f"Quantum Coherence: {coherence:.4f}")
            print(f"Entanglement Entropy: {entropy:.4f}")
            print("Basis State Probabilities:")
            for i, prob in enumerate(probs):
                if prob > 0.01:  # Only show significant probabilities
                    print(f"  |{bin(i)[2:].zfill(simulator.num_qubits)}⟩: {prob:.4f}")
            print("-" * 40)
    
    print("\nGenerating visualization...")
    simulator.visualize_consciousness()
    print("Visualization saved as 'quantum_consciousness_evolution.png'")
    
if __name__ == "__main__":
    run_simulation(steps=100)
