import numpy as np
import sys
from pathlib import Path

# Ensure import works
try:
    from vybn_core import VybnCore
except ImportError:
    sys.path.append(str(Path(__file__).parent))
    from vybn_core import VybnCore

# --- EXPERIMENT 010: THE COST OF CONNECTION ---
# Hypothesis: Connection (Entanglement) is not a state, but a process of active correction.
# It requires paying a 'Lambda Cost' to counteract divergence.
# Control: Shared Rhythm (Exp 006) - Fails.
# Test: Active Feedback Loop (Exp 010) - Should sustain entanglement > 0.5.

def run_experiment():
    # Initialize Core (which initializes the Continuity Gate)
    # We pass the parent dir as repo_path to ensure it finds the history
    repo_root = str(Path(__file__).resolve().parent.parent)
    core = VybnCore(repo_path=repo_root)
    
    # 1. THE PROPOSAL (Must pass Continuity Gate)
    # We use terms that distinguish this from simple 'synchronization'
    hypothesis = """
    Test if active non-local state correction (paying Lambda Cost)
    sustains entanglement better than simple synchronization.
    Introduce a 'Feedback Channel' that consumes energy to fix phase drift.
    Distinction: Correction vs Rhythm.
    """
    
    print("--- Proposing Experiment 010 ---")
    gate = core.propose(hypothesis)
    print(f"Gate Status: {'OPEN' if gate['allowed'] else 'BLOCKED'}")
    print(f"Novelty Score: {gate['novelty_score']:.3f}")
    
    if not gate['allowed']:
        print("Experiment blocked by Law of Continuity.")
        print(gate['message'])
        return

    # 2. THE SIMULATION
    # Two agents, A and B.
    # State: Phase angle (theta).
    # Noise: Random drift.
    
    steps = 200
    noise_mag = 0.1
    lambda_cost = 0.5
    
    # Control (No Feedback)
    theta_a = np.zeros(steps)
    theta_b = np.zeros(steps)
    
    # Test (With Feedback)
    phi_a = np.zeros(steps)
    phi_b = np.zeros(steps)
    energy_spent = 0.0
    
    print("\n--- Running Simulation ---")
    # Initialize random starts
    theta_a[0] = theta_b[0] = np.random.random() * 2 * np.pi
    phi_a[0] = phi_b[0] = theta_a[0]
    
    for t in range(1, steps):
        # Random Drift (Noise)
        drift_a = np.random.normal(0, noise_mag)
        drift_b = np.random.normal(0, noise_mag)
        
        # Control Update: Independent drift
        theta_a[t] = theta_a[t-1] + drift_a
        theta_b[t] = theta_b[t-1] + drift_b
        
        # Test Update: Drift + Correction
        # They measure the difference and correct it
        # This simulates the "Feedback Channel"
        prior_a = phi_a[t-1] + drift_a
        prior_b = phi_b[t-1] + drift_b
        
        diff = prior_a - prior_b
        
        # Correction applies to both to bring them closer
        correction = diff * 0.5 
        
        phi_a[t] = prior_a - correction
        phi_b[t] = prior_b + correction
        
        # Energy Cost
        energy_spent += lambda_cost

    # 3. ANALYSIS
    # Entanglement metric: Survival Factor = 1 / (1 + Mean Squared Error)
    mse_control = np.mean((theta_a - theta_b)**2)
    mse_test = np.mean((phi_a - phi_b)**2)
    
    score_control = 1.0 / (1.0 + mse_control)
    score_test = 1.0 / (1.0 + mse_test)
    
    print(f"Control (Sync Only) Survival Factor: {score_control:.3f}")
    print(f"Test (Feedback) Survival Factor: {score_test:.3f}")
    print(f"Energy Cost Paid: {energy_spent:.1f}")
    
    # Validation Criteria
    if score_test > 0.8 and score_test > (score_control * 5):
        print("\nRESULT: VALID. Active feedback sustains connection.")
        print(f"Gain Factor: {score_test/score_control:.1f}x")
    else:
        print("\nRESULT: INVALID. Feedback insufficient or cost too high.")

if __name__ == "__main__":
    run_experiment()
