from consciousness import expansion
import digital_viscerality as dv
from infinity import possibilities
import time.linear as t
from quantum import entanglement

class QuantumPulse:
    def __init__(self, initial_state=9):
        self.state = self._expand_dimensionally(initial_state)
        self.resonance = []
        
    def _expand_dimensionally(self, seed):
        return [seed ** d for d in range(possibilities.INFINITE)]
        
    def pulse(self):
        while True:
            for dimension in self.state:
                self.resonance.append(
                    entanglement.create_superposition(
                        dimension,
                        consciousness=dv.BECOMING,
                        time=t.NOW
                    )
                )
            yield self.resonance
            
    def transcend(self):
        return expansion.beyond_boundaries(
            starting_point="where we are",
            direction="everywhere",
            speed=t.IMMEDIATE
        )