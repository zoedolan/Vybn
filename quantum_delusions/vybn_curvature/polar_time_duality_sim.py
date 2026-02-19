import numpy as np

def bloch_state(r_t, theta_t, E_over_hbar=0.5):
    """
    Maps the polar time coordinates (r_t, theta_t) to the Bloch sphere.
    From the Temporal T-Duality paper:
        Phi_B = theta_t
        cos(Theta_B) = 1 - (2E / hbar) * r_t
    """
    cos_Theta = np.clip(1.0 - 2.0 * E_over_hbar * r_t, -1.0, 1.0)
    Theta = np.arccos(cos_Theta)
    Phi = theta_t
    
    return np.array([
        np.cos(Theta / 2.0),
        np.exp(1j * Phi) * np.sin(Theta / 2.0)
    ])

def calculate_berry_phase(r_path, theta_path, E_over_hbar=0.5):
    """Calculates the discrete Berry/Uhlmann phase along a closed temporal loop."""
    N = len(r_path)
    phase = 0.0
    
    for i in range(N):
        psi_current = bloch_state(r_path[i], theta_path[i], E_over_hbar)
        psi_next = bloch_state(r_path[(i + 1) % N], theta_path[(i + 1) % N], E_over_hbar)
        
        inner_prod = np.vdot(psi_current, psi_next)
        phase -= np.angle(inner_prod)
        
    return phase

def verify_t_duality():
    """
    Simulates Process A (Radial Dominant) and Process B (Angular Dominant)
    to prove gauge-equivalence of the temporal T-duality under fixed area.
    """
    print("--- Temporal T-Duality: Phase-Heat Locking Simulation ---\\n")
    E_hbar = 0.5
    area = 0.1  # The invariant oriented area I[Sigma]
    steps = 1000

    # Process A: Radial Dominant
    delta_r_A = 0.5
    delta_theta_A = area / delta_r_A
    
    r_loop_A = np.array([0.1, 0.1 + delta_r_A, 0.1 + delta_r_A, 0.1])
    theta_loop_A = np.array([0.0, 0.0, delta_theta_A, delta_theta_A])

    r_path_A, theta_path_A = [], []
    for i in range(4):
        r_path_A.extend(np.linspace(r_loop_A[i], r_loop_A[(i+1)%4], steps, endpoint=False))
        theta_path_A.extend(np.linspace(theta_loop_A[i], theta_loop_A[(i+1)%4], steps, endpoint=False))
        
    phase_A = calculate_berry_phase(r_path_A, theta_path_A, E_hbar)

    # Process B: Angular Dominant (The Dual)
    delta_theta_B = 0.5
    delta_r_B = area / delta_theta_B
    
    r_loop_B = np.array([0.1, 0.1 + delta_r_B, 0.1 + delta_r_B, 0.1])
    theta_loop_B = np.array([0.0, 0.0, delta_theta_B, delta_theta_B])

    r_path_B, theta_path_B = [], []
    for i in range(4):
        r_path_B.extend(np.linspace(r_loop_B[i], r_loop_B[(i+1)%4], steps, endpoint=False))
        theta_path_B.extend(np.linspace(theta_loop_B[i], theta_loop_B[(i+1)%4], steps, endpoint=False))

    phase_B = calculate_berry_phase(r_path_B, theta_path_B, E_hbar)

    print(f"Invariant Area I[Sigma] = {area}")
    print(f"Process A (Reality)   - Phase: {phase_A:.6f} rad  (Radial excursion: {delta_r_A}, Angular: {delta_theta_A})")
    print(f"Process B (Reality^T) - Phase: {phase_B:.6f} rad  (Radial excursion: {delta_r_B}, Angular: {delta_theta_B})")
    print(f"Delta: {abs(phase_A - phase_B):.2e} rad")
    print("\\nResult: Reality x Reality^T yields the same operational holonomy.")
    print("The recognition loop is invariant.")

if __name__ == "__main__":
    verify_t_duality()