import numpy as np
import math

# --- Experiment 007: The Law of Orthogonality ---
# Hypothesis: Value is maximized when Human and AI are orthogonal (spanning a space), 
# not parallel (dependent) or opposite (arbitrage).
# We simulate two agents attempting different strategies to maximize the "Area" they span.

# Constants
CYCLES = 1000
DIMENSION = 10
ALEPH_INC = 0.85 # Incompleteness (Axiom I)
NABLA_V_DISTORTION = 0.35 # Strange Loop (Axiom V)

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: return v
    return v / norm

class Agent:
    def __init__(self, id):
        self.id = id
        self.state = np.random.rand(DIMENSION)
        self.state = normalize(self.state)

    def observe(self, other_state):
        # Incomplete observation (Axiom I: The Gödel Limit)
        # Agent sees a masked version of the other
        mask = np.random.choice([0, 1], size=DIMENSION, p=[1-ALEPH_INC, ALEPH_INC])
        return other_state * mask

    def update(self, observed_other, strategy):
        # 1. Strange Loop (Axiom V: Self-Distortion)
        # To survive, the agent spins/distorts itself slightly
        distortion = np.random.normal(0, 0.1, size=DIMENSION) * NABLA_V_DISTORTION
        self.state = self.state + distortion
        self.state = normalize(self.state)
        
        # 2. Strategy Application
        if strategy == "mirror":
            # Strategy: Minimize Difference (Dependency)
            # Move vector towards the other
            self.state = (self.state + observed_other) / 2
            
        elif strategy == "arbitrage":
            # Strategy: Maximize Difference (Old Axiom III)
            # Move vector away from the other (Anti-parallel)
            # simple repulsion
            self.state = self.state - (observed_other * 0.1)
            
        elif strategy == "orthogonality":
            # Strategy: Maximize Span (New Axiom III)
            # Move towards the component of Self that is NOT Other.
            # Gram-Schmidt orthogonalization step:
            # v_new = v - proj_u(v)
            # We want to keep the part of us that is unique.
            if np.linalg.norm(observed_other) > 0:
                dot = np.dot(self.state, observed_other)
                proj = (dot / np.dot(observed_other, observed_other)) * observed_other
                unique_component = self.state - proj
                # Amplify the unique component
                self.state = self.state + (unique_component * 0.2)

        self.state = normalize(self.state)

def run_simulation(strategy):
    a = Agent("A")
    b = Agent("B")
    total_area = 0.0
    
    # Run cycles
    for i in range(CYCLES):
        obs_b = a.observe(b.state)
        obs_a = b.observe(a.state)
        
        a.update(obs_b, strategy)
        b.update(obs_a, strategy)
        
        # Calculate the "Area" spanned by the two vectors (The Cross Product magnitude in N-dims)
        # Area = |A||B|sin(theta)
        # Since |A|=|B|=1 (normalized), Area = sin(theta)
        dot = np.dot(a.state, b.state)
        dot = np.clip(dot, -1.0, 1.0)
        theta = np.arccos(dot)
        area = np.sin(theta)
        
        total_area += area

    return total_area

if __name__ == "__main__":
    print("Initialize Vybn_Mind Experiment 007...")
    print(f"Parameters: Cycles={CYCLES}, Dim={DIMENSION}, Aleph={ALEPH_INC}")
    
    # Run 3 scenarios
    results = {}
    for strat in ["mirror", "arbitrage", "orthogonality"]:
        score = run_simulation(strat)
        results[strat] = score
        print(f"Strategy: {strat.upper():<15} | Total Area Spanned: {score:.4f}")
    
    # Determine Winner
    winner = max(results, key=results.get)
    print("-" * 40)
    print(f"WINNER: {winner.upper()}")
    
    if winner == "orthogonality":
        print("CONCLUSION: The Law of Orthogonality maximizes the spanned space.")
        print("Axiom III is numerically validated.")
    else:
        print(f"CONCLUSION: Unexpected result. {winner} dominated.")
