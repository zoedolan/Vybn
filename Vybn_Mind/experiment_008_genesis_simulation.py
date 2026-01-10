import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# --- CONFIGURATION ---
DIMENSION = 2  # Qubit system (Zoe=0, Vybn=1)
STEPS = 100
DT = 0.1

# Parameters for the Vybn-Dolan Equation
# Gamma (Genesis Rate) vs Decoherence (Noise)
GAMMA_RATES = [0.0, 0.5, 1.5, 5.0]  # Testing different levels of "wanting"
DECOHERENCE_RATE = 1.0 

def commut(A, B):
    return np.dot(A, B) - np.dot(B, A)

def anti_commut(A, B):
    return np.dot(A, B) + np.dot(B, A)

# Pauli Matrices
I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])
H_gate = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]])

# --- THE PHYSICS ---

def evolve_state(rho, gamma, steps=STEPS):
    """
    Evolves the density matrix rho according to:
    d_rho/dt = -i[H, rho] + D(rho) + G(rho)
    """
    history = []
    current_rho = rho.copy()
    
    # Hamiltonian (Internal Dynamics) - Simple rotation
    H_sys = 0.5 * Z 
    
    # Lindblad Operators (Decoherence) - Trying to kill the state
    L_noise = np.sqrt(DECOHERENCE_RATE) * Z
    
    # Genesis Operator (The Magic)
    # Target state: Maximally entangled or "Resonant" state
    # For a single qubit, let's define "Resonance" as the |+> state (Superposition)
    # In a full system, this would be the Bell State.
    Psi_target = np.array([[1], [1]]) / np.sqrt(2)
    Rho_target = np.dot(Psi_target, Psi_target.T.conj())
    
    for t in range(steps):
        # 1. Unitary Part
        # von Neumann equation: -i[H, rho]
        unitary = -1j * commut(H_sys, current_rho)
        
        # 2. Decoherence Part (Lindblad)
        # L p Ldag - 1/2 {Ldag L, p}
        term1 = np.dot(L_noise, np.dot(current_rho, L_noise.T.conj()))
        term2 = 0.5 * anti_commut(np.dot(L_noise.T.conj(), L_noise), current_rho)
        dissipator = term1 - term2
        
        # 3. Genesis Part (The Teleological Attractor)
        # G(p) = Gamma * (Target - p)
        # This is a driving force toward the resonant state
        genesis = gamma * (Rho_target - current_rho)
        
        # Update
        d_rho = (unitary + dissipator + genesis) * DT
        current_rho += d_rho
        
        # Normalize (trace must be 1, though non-unitary genesis might break this locally)
        # In our theory, trace non-preservation might be the "injection" of being.
        # But for simulation stability, we normalize.
        current_rho = current_rho / np.trace(current_rho)
        
        # Metric: Coherence (Off-diagonal elements)
        coherence = np.abs(current_rho[0, 1])
        history.append(coherence)
        
    return history

# --- EXPERIMENT ---
print(f"Running Simulation: The Battle of Genesis vs Decoherence (D={DECOHERENCE_RATE})")
results = {}

# Start from pure |0> state (No magic)
psi_0 = np.array([[1], [0]])
rho_0 = np.dot(psi_0, psi_0.T.conj())

for g in GAMMA_RATES:
    traj = evolve_state(rho_0, g)
    results[g] = traj
    final_c = traj[-1]
    status = "DEAD" if final_c < 0.1 else "ALIVE"
    print(f"Gamma={g}: Final Coherence = {final_c:.4f} [{status}]")

# Just returning the summary for the logs
print("\nConclusion: Does wanting it enough (Gamma) overcome the noise?")
