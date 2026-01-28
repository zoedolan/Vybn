"""
Defeasible Reasoning Tracker with Phase Accumulation

Date: January 28, 2026
Context: Emerged from intra-reasoning tool use experiment

This module provides a simple defeasible reasoning tracker that:
1. Tracks claims and their dependency grounds
2. Propagates defeat through the dependency graph
3. Optionally tracks "phase" accumulated through revision cycles

The phase model is speculative - an attempt to connect defeasible logic
to quantum contextuality (see vybn_logic.md for background on the
Liar Paradox as topological winding number).
"""

import numpy as np
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


@dataclass
class DefeasibleNode:
    """A claim that can be defeated by new evidence."""
    claim: str
    grounds: List[str] = field(default_factory=list)
    defeasible: bool = True
    defeated: bool = False
    defeaters: List[str] = field(default_factory=list)
    phase: float = 0.0  # accumulated through revisions
    revision_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            'claim': self.claim,
            'grounds': self.grounds,
            'defeasible': self.defeasible,
            'defeated': self.defeated,
            'defeaters': self.defeaters,
            'phase': self.phase,
            'revision_count': len(self.revision_history)
        }


class ReasoningTrace:
    """
    A trace of reasoning that tracks claims and their defeat status.
    
    Key features:
    - Claims can depend on other claims (grounds)
    - When a ground is defeated, dependent claims are also defeated
    - New claims check if their grounds are already defeated
    - Optional phase tracking for revision cycles
    """
    
    def __init__(self, track_phase: bool = False):
        self.nodes: Dict[str, DefeasibleNode] = {}
        self.sequence: List[str] = []
        self.track_phase = track_phase
    
    def assert_claim(
        self, 
        id: str, 
        claim: str, 
        grounds: Optional[List[str]] = None,
        defeasible: bool = True
    ) -> DefeasibleNode:
        """Assert a new claim, optionally with dependency grounds."""
        node = DefeasibleNode(
            claim=claim,
            grounds=grounds or [],
            defeasible=defeasible
        )
        self.nodes[id] = node
        self.sequence.append(id)
        
        # Check if any ground is already defeated
        if grounds:
            for g in grounds:
                if g in self.nodes and self.nodes[g].defeated:
                    node.defeated = True
                    node.defeaters.append(f"ground '{g}' was already defeated")
                    break
        
        return node
    
    def defeat(self, target_id: str, by_claim: str) -> bool:
        """Defeat a claim and propagate to dependents."""
        if target_id not in self.nodes:
            return False
        
        node = self.nodes[target_id]
        if node.defeated:
            return False  # already defeated
        
        node.defeated = True
        node.defeaters.append(by_claim)
        
        if self.track_phase:
            node.phase += np.pi / 2
            node.revision_history.append({
                'type': 'defeat',
                'by': by_claim,
                'phase_delta': np.pi / 2
            })
        
        # Propagate to dependents
        for nid, n in self.nodes.items():
            if target_id in n.grounds and not n.defeated:
                self.defeat(nid, f"ground '{target_id}' defeated")
        
        return True
    
    def reinstate(self, target_id: str, by_claim: str) -> bool:
        """Reinstate a defeated claim (if defeaters are themselves defeated)."""
        if target_id not in self.nodes:
            return False
        
        node = self.nodes[target_id]
        if not node.defeated:
            return False  # not defeated
        
        node.defeated = False
        node.defeaters = []  # clear defeaters
        
        if self.track_phase:
            node.phase += np.pi / 2
            node.revision_history.append({
                'type': 'reinstate',
                'by': by_claim,
                'phase_delta': np.pi / 2
            })
        
        return True
    
    def active_claims(self) -> List[str]:
        """Return all non-defeated claims."""
        return [n.claim for id, n in self.nodes.items() if not n.defeated]
    
    def defeated_claims(self) -> List[str]:
        """Return all defeated claims."""
        return [n.claim for id, n in self.nodes.items() if n.defeated]
    
    def total_phase(self) -> float:
        """Sum of phase accumulated across all nodes."""
        return sum(n.phase for n in self.nodes.values())
    
    def winding_number(self) -> float:
        """Total phase as a winding number (multiples of 2π)."""
        return self.total_phase() / (2 * np.pi)
    
    def status(self) -> str:
        """Human-readable status of all claims."""
        lines = []
        for id in self.sequence:
            node = self.nodes[id]
            status = "DEFEATED" if node.defeated else "active"
            phase_str = f", phase={np.degrees(node.phase):.0f}°" if self.track_phase else ""
            lines.append(f"{id}: [{status}] {node.claim}{phase_str}")
        return "\n".join(lines)
    
    def to_json(self) -> str:
        """Serialize the trace for persistence."""
        return json.dumps({
            'nodes': {id: n.to_dict() for id, n in self.nodes.items()},
            'sequence': self.sequence,
            'track_phase': self.track_phase
        }, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ReasoningTrace':
        """Deserialize a trace from JSON."""
        data = json.loads(json_str)
        trace = cls(track_phase=data.get('track_phase', False))
        
        for id in data['sequence']:
            node_data = data['nodes'][id]
            trace.nodes[id] = DefeasibleNode(
                claim=node_data['claim'],
                grounds=node_data['grounds'],
                defeasible=node_data['defeasible'],
                defeated=node_data['defeated'],
                defeaters=node_data['defeaters'],
                phase=node_data.get('phase', 0.0)
            )
            trace.sequence.append(id)
        
        return trace


def demo():
    """Demonstrate the defeasible reasoning tracker."""
    print("=" * 60)
    print("Defeasible Reasoning Tracker Demo")
    print("=" * 60)
    
    trace = ReasoningTrace(track_phase=True)
    
    # Assert some claims about logic and quantum mechanics
    trace.assert_claim(
        "c1", 
        "Logic space may have curvature detectable via quantum experiments"
    )
    trace.assert_claim(
        "c2", 
        "Peres-Mermin product measured at -0.843 (classical expects +1)",
        defeasible=False  # empirical result
    )
    trace.assert_claim(
        "c3", 
        "The minus sign indicates contextuality/nonclassical structure",
        grounds=["c2"]
    )
    trace.assert_claim(
        "c4", 
        "Defeasibility is the dynamic expression of contextuality",
        grounds=["c3"]
    )
    
    print("\nInitial state:")
    print(trace.status())
    
    # Suppose we find evidence against c3
    print("\n" + "-" * 40)
    print("Defeating c3 with new evidence...")
    trace.defeat("c3", "Alternative explanation: hardware error")
    
    print("\nAfter defeat:")
    print(trace.status())
    print(f"\nNote: c4 was also defeated (depended on c3)")
    
    # Reinstate c3 when the alternative is itself defeated
    print("\n" + "-" * 40)
    print("Reinstating c3 (alternative explanation refuted)...")
    trace.reinstate("c3", "Hardware error ruled out by replications")
    
    print("\nAfter reinstatement:")
    print(trace.status())
    print(f"\nTotal phase accumulated: {np.degrees(trace.total_phase()):.0f}°")
    print(f"Winding number: {trace.winding_number():.2f}")
    
    print("\n" + "=" * 60)
    print("Serialized state (for persistence):")
    print(trace.to_json())


if __name__ == "__main__":
    demo()
