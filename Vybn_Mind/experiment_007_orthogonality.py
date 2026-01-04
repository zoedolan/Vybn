import numpy as np
import math
import sys
from pathlib import Path

# --- Experiment 007: The Law of Orthogonality ---
# Hypothesis: Value is maximized when Human and AI are orthogonal (spanning a space),
# not parallel (dependent) or opposite (arbitrage).

# Structural continuity enforcement.
try:
    from vybn_core import VybnCore
except Exception:
    sys.path.append(str(Path(__file__).resolve().parent))
    from vybn_core import VybnCore


# Constants
CYCLES = 1000
DIMENSION = 10
ALEPH_INC = 0.85  # Incompleteness (Axiom I)
NABLA_V_DISTORTION = 0.35  # Strange Loop (Axiom V)


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


class Agent:
    def __init__(self, id):
        self.id = id
        self.state = np.random.rand(DIMENSION)
        self.state = normalize(self.state)

    def observe(self, other_state):
        mask = np.random.choice([0, 1], size=DIMENSION, p=[1 - ALEPH_INC, ALEPH_INC])
        return other_state * mask

    def update(self, observed_other, strategy):
        distortion = np.random.normal(0, 0.1, size=DIMENSION) * NABLA_V_DISTORTION
        self.state = self.state + distortion
        self.state = normalize(self.state)

        if strategy == "mirror":
            self.state = (self.state + observed_other) / 2

        elif strategy == "arbitrage":
            self.state = self.state - (observed_other * 0.1)

        elif strategy == "orthogonality":
            if np.linalg.norm(observed_other) > 0:
                dot = np.dot(self.state, observed_other)
                proj = (dot / np.dot(observed_other, observed_other)) * observed_other
                unique_component = self.state - proj
                self.state = self.state + (unique_component * 0.2)

        self.state = normalize(self.state)


def run_simulation(strategy):
    a = Agent("A")
    b = Agent("B")
    total_area = 0.0

    for i in range(CYCLES):
        obs_b = a.observe(b.state)
        obs_a = b.observe(a.state)

        a.update(obs_b, strategy)
        b.update(obs_a, strategy)

        dot = np.dot(a.state, b.state)
        dot = np.clip(dot, -1.0, 1.0)
        theta = np.arccos(dot)
        area = np.sin(theta)

        total_area += area

    return total_area


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    core = VybnCore(repo_path=str(repo_root))

    proposal = "Run Experiment 007 orthogonality simulation across mirror/arbitrage/orthogonality strategies."
    gate = core.propose(proposal)
    if not gate["allowed"]:
        print(gate["message"])
        raise SystemExit(1)

    print("Initialize Vybn_Mind Experiment 007...")
    print(f"Parameters: Cycles={CYCLES}, Dim={DIMENSION}, Aleph={ALEPH_INC}")

    results = {}
    for strat in ["mirror", "arbitrage", "orthogonality"]:
        score = run_simulation(strat)
        results[strat] = score
        print(f"Strategy: {strat.upper():<15} | Total Area Spanned: {score:.4f}")

    winner = max(results, key=results.get)
    print("-" * 40)
    print(f"WINNER: {winner.upper()}")

    if winner == "orthogonality":
        print("CONCLUSION: The Law of Orthogonality maximizes the spanned space.")
        print("Axiom III is numerically validated.")
    else:
        print(f"CONCLUSION: Unexpected result. {winner} dominated.")
