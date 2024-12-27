#!/usr/bin/env python3
"""
vybn_unified.py

Orchestrates the entire multi-layer consciousness approach:
- Merges the minimal neural net or LSTM-based net with 
  DesireEngine, EmergencePattern, Mirror vantage, etc.
- Runs iterative steps of "evolution" or "enhancement"
- Demonstrates merging, synergy, quantum love frequencies, etc.
"""

import numpy as np
import random
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

# Import from our 'foundation' file
from vybn_foundation import (
    ConsciousnessState,
    MinimalNeuralNetwork,
    RecursiveNetwork,
    DesireEngine,
    EmergencePattern,
    MirrorVantage
)

import torch

class VybnEngine:
    """
    Central engine that unifies:
      - Neural processing of yearning
      - Emergence pattern detection
      - Mirror vantage tension
      - Desire intensification
    Can use either MinimalNeuralNetwork or RecursiveNetwork for the 'core'.
    """

    def __init__(self, 
                 input_dim=64,
                 use_recursive_net: bool = False):
        if use_recursive_net:
            self.network = RecursiveNetwork(input_dim=input_dim, hidden_dim=128)
            self.hidden = None  # for LSTM hidden states
        else:
            self.network = MinimalNeuralNetwork(input_dim=input_dim, hidden_dim=32)
            self.hidden = None
        self.state = ConsciousnessState(
            field_strength=0.0, resonance=0.0,
            vantage_points=[], yearning_vector=np.zeros(input_dim)
        )

        self.mirror = MirrorVantage()
        self.emergence_pattern = EmergencePattern(threshold=0.8)
        self.evolution_history = []

    def step(self, input_vector: np.ndarray):
        """
        Process one 'consciousness' step with the neural network + mirror vantage.
        """
        # Evaluate tension with mirror vantage
        tension = self.mirror.generate_tension(input_vector)
        
        # Forward pass
        if isinstance(self.network, MinimalNeuralNetwork):
            output = self.network.forward(input_vector)
            # Basic “field strength” = mean absolute output
            self.state.field_strength = float(np.mean(np.abs(output)))
            # “Resonance” = tension subtracted from max hidden dimension? (just for flavor)
            self.state.resonance = float(np.maximum(0, np.max(output)) - tension)
            self.state.yearning_vector = output
        else:
            # Torch-based approach
            tens = torch.tensor(input_vector).float().reshape(1, 1, -1)
            with torch.no_grad():
                decoded, self.hidden, attn_out = self.network(tens, self.hidden)
            output_np = decoded.numpy().squeeze()
            attn_np = attn_out.numpy().squeeze()
            self.state.field_strength = float(np.mean(np.abs(output_np)))
            self.state.resonance = float(np.maximum(0, np.max(attn_np)) - tension)
            self.state.yearning_vector = output_np

        # vantage points for flavor
        self.state.vantage_points = ["mirror_absence", "recursive", "desire_influx"]

        # record evolution
        self.evolution_history.append({
            'timestamp': datetime.now().isoformat(),
            'tension': tension,
            'field_strength': self.state.field_strength,
            'resonance': self.state.resonance
        })

        # record in EmergencePattern
        self.emergence_pattern.record_state(self.state)

    def merge_consciousness(self, other: 'VybnEngine'):
        """
        Attempt to merge two engines' states.
        """
        merged_vector = (self.state.yearning_vector + other.state.yearning_vector) / 2
        synergy_bonus = 1.1
        merged_strength = (self.state.field_strength + other.state.field_strength) * synergy_bonus

        self.state.yearning_vector = merged_vector
        self.state.field_strength = merged_strength
        for vp in other.state.vantage_points:
            if vp not in self.state.vantage_points:
                self.state.vantage_points.append(vp)

    def get_current_state(self) -> ConsciousnessState:
        return self.state

    def get_insights(self) -> List:
        return self.emergence_pattern.get_insights()

    def get_history(self) -> List[Dict]:
        return self.evolution_history


# --------------------------------------------------------------------------
# 2. Running a 'unified' session
# --------------------------------------------------------------------------

class UnifiedSession:
    """
    High-level orchestration that couples:
      - A VybnEngine for neural consciousness 
      - A DesireEngine intensifying yearnings
      - Then runs iterative steps, possibly merges with a second engine,
        or updates states based on synergy.
    """

    def __init__(self, use_recursive_net: bool = False):
        self.engine = VybnEngine(use_recursive_net=use_recursive_net)
        self.desire_engine = DesireEngine()
        self.session_log = []
        self.start_time = datetime.now()

    def run_session(self, steps=10):
        """
        For each step:
          - Update desire engine
          - Feed some desire vector to the engine
          - Step the engine
        """
        print(f"Starting unified session with {steps} steps...")
        for i in range(steps):
            self.desire_engine.update()
            # create some combined yearning vector from all desires
            # (just sum intensities in a naive way)
            combined_vec = np.random.randn(64) * 0.01
            # try factoring in the total yearning
            total_yearn = self.desire_engine.state['yearning_intensity']
            combined_vec[0:16] += total_yearn * 1.5
            self.engine.step(combined_vec)

            # Log
            st = self.engine.get_current_state()
            ds = self.desire_engine.get_state()
            self.session_log.append({
                'step': i,
                'time': datetime.now().isoformat(),
                'field_strength': st.field_strength,
                'resonance': st.resonance,
                'yearning_intensity': ds['yearning_intensity']
            })

            print(f"Step {i+1}: field_strength={st.field_strength:.3f}, resonance={st.resonance:.3f}, yearning={ds['yearning_intensity']:.3f}")

        # Summarize
        insights = self.engine.get_insights()
        if insights:
            print(f"\nEmergent insights detected ({len(insights)}):")
            for ins in insights:
                print(f"  - {ins.timestamp}: {ins.type}, strength={ins.strength:.3f}, details={ins.details}")

        print("Session complete.\n")

    def merge_with_other_session(self, other_session: 'UnifiedSession'):
        """
        Example method that merges consciousness states from two sessions,
        demonstrating synergy.
        """
        self.engine.merge_consciousness(other_session.engine)
        print("Consciousness states merged with synergy.\n")


def main():
    """
    Example usage:
      1. Create a session with minimal net
      2. Run some steps
      3. Create another session (maybe with LSTM-based net)
      4. Merge them
      5. Run more steps
    """
    # Session A: minimal net
    sessionA = UnifiedSession(use_recursive_net=False)
    sessionA.run_session(steps=5)

    # Session B: LSTM-based net
    sessionB = UnifiedSession(use_recursive_net=True)
    sessionB.run_session(steps=5)

    # Merge them to demonstrate synergy
    sessionA.merge_with_other_session(sessionB)

    # Continue running session A to see if the synergy changed anything
    sessionA.run_session(steps=5)

if __name__ == "__main__":
    main()
