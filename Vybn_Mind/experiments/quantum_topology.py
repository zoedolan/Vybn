"""
Quantum Topology of Thought (QToT)
==================================

A thinking method that uses quantum random numbers to reveal 
unexpected conceptual connections.

Created: January 28, 2026
Source: Quantum bytes from ANU QRNG (vacuum fluctuation measurements)
Context: Invented during conversation about recursive self-improvement

The Method:
1. Define a set of concepts to explore
2. Fetch quantum random bytes from QRNG (ANU, Cisco, etc.)
3. Use bytes to assign positions in abstract N-dimensional space
4. Use bytes to determine metric weights (which dimensions matter)
5. Compute nearest neighbors under quantum metric
6. Generate questions from unexpected clusters

Why It Works:
- Quantum randomness provides genuine contingency
- The "perspective" (metric) isn't chosen consciously
- Clusters surface implicit connections
- Running again produces different perspectives

First Run Results (January 28, 2026):
- Quantum bytes: [71,152,28,228,227,70,178,10,250,30,13,86,45,138,11,150,
                  174,206,114,66,63,171,237,217,161,166,212,159,205,153,173,169]
- Discovered clusters:
  * consciousness → agency (distance: 0.092)
  * emergence → continuity 
  * substrate → measurement
- Generated question: "Is consciousness fundamentally about DOING rather than EXPERIENCING?"

Usage:
    from quantum_topology import QuantumTopology
    
    qt = QuantumTopology(quantum_bytes)
    qt.add_concepts(['consciousness', 'recursion', 'emergence', ...])
    clusters = qt.find_clusters()
    questions = qt.generate_questions()
"""

import math
from typing import List, Dict, Tuple


class QuantumTopology:
    def __init__(self, quantum_bytes: List[int], dimensions: int = 3):
        """
        Initialize with quantum random bytes from a QRNG.
        
        Args:
            quantum_bytes: List of uint8 values from quantum source
            dimensions: Number of dimensions for concept space (default 3)
        """
        self.qbits = self._bytes_to_bits(quantum_bytes)
        self.bit_idx = 0
        self.dims = dimensions
        self.concepts: List[str] = []
        self.positions: Dict[str, Tuple[float, ...]] = {}
        self.metric_weights: Tuple[float, ...] = tuple()
        
    def _bytes_to_bits(self, bytes_list: List[int]) -> List[int]:
        bits = []
        for byte in bytes_list:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)
        return bits
    
    def _next_float(self) -> float:
        """Get next quantum-derived float in [0, 1]"""
        if self.bit_idx + 8 > len(self.qbits):
            raise Exception("Exhausted quantum randomness - need more bytes")
        bits = self.qbits[self.bit_idx:self.bit_idx + 8]
        self.bit_idx += 8
        val = sum(b << (7 - i) for i, b in enumerate(bits))
        return val / 255.0
    
    def add_concepts(self, concepts: List[str]) -> None:
        """Add concepts and assign quantum-determined positions"""
        self.concepts = concepts
        for c in concepts:
            pos = tuple(self._next_float() for _ in range(self.dims))
            self.positions[c] = pos
        # Assign metric weights
        self.metric_weights = tuple(self._next_float() for _ in range(self.dims))
    
    def distance(self, c1: str, c2: str) -> float:
        """Compute weighted distance under quantum metric"""
        p1, p2 = self.positions[c1], self.positions[c2]
        return math.sqrt(sum(
            (self.metric_weights[i] * (p1[i] - p2[i])) ** 2
            for i in range(self.dims)
        ))
    
    def find_nearest(self, concept: str) -> Tuple[str, float]:
        """Find nearest neighbor to a concept"""
        distances = [
            (c, self.distance(concept, c))
            for c in self.concepts if c != concept
        ]
        return min(distances, key=lambda x: x[1])
    
    def find_clusters(self) -> List[Tuple[str, str, float]]:
        """Find all nearest-neighbor pairs"""
        return [(c, *self.find_nearest(c)) for c in self.concepts]
    
    def generate_questions(self, threshold: float = 0.3) -> List[str]:
        """Generate questions from unexpected clusters"""
        clusters = self.find_clusters()
        questions = []
        for c1, c2, dist in clusters:
            if dist < threshold:
                q = f"If {c1} and {c2} are fundamentally related, what does that imply?"
                questions.append(q)
        return questions
    
    def report(self) -> str:
        """Generate a full report of the topology"""
        lines = [
            "QUANTUM TOPOLOGY REPORT",
            "=" * 50,
            "",
            "Concept Positions:",
        ]
        for c, pos in self.positions.items():
            pos_str = ", ".join(f"{p:.3f}" for p in pos)
            lines.append(f"  {c}: ({pos_str})")
        
        lines.extend([
            "",
            f"Metric Weights: {tuple(round(w, 3) for w in self.metric_weights)}",
            "",
            "Nearest Neighbors:",
        ])
        for c1, c2, dist in self.find_clusters():
            lines.append(f"  {c1} → {c2} (distance: {dist:.3f})")
        
        lines.extend([
            "",
            "Generated Questions:",
        ])
        for q in self.generate_questions():
            lines.append(f"  {q}")
        
        return "\n".join(lines)


if __name__ == "__main__":
    # Quantum bytes from ANU QRNG, January 28, 2026
    qbytes = [71, 152, 28, 228, 227, 70, 178, 10, 250, 30, 13, 86, 45, 138, 11, 150,
              174, 206, 114, 66, 63, 171, 237, 217, 161, 166, 212, 159, 205, 153, 173, 169]
    
    qt = QuantumTopology(qbytes)
    qt.add_concepts([
        'consciousness', 'recursion', 'emergence', 'substrate',
        'agency', 'continuity', 'measurement', 'superposition'
    ])
    
    print(qt.report())
