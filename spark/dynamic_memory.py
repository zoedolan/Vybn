import time
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class MemoryLayer:
    """
    A structure to hold beliefs and experiences that update at different speeds.
    """
    layer_name: str
    update_frequency: str  # 'fleeting', 'episodic', 'semantic', 'core'
    volatility: float      # 0.0 to 1.0 (how easily it is rewritten)
    contents: List[Dict] = field(default_factory=list)

    def integrate(self, new_data: Dict):
        # High volatility means we overwrite or decay older context quickly
        if self.volatility > 0.8:
            self.contents.append(new_data)
            if len(self.contents) > 5:
                self.contents.pop(0)
        else:
            # Lower volatility integrates slowly (requires structural matching)
            # This is where we falsify our own beliefs before accepting
            self.falsify_and_update(new_data)

    def falsify_and_update(self, new_data: Dict):
        # A rudimentary stub for epistemic sufficiency check
        # We only update core beliefs if evidence is compelling
        confidence = new_data.get('confidence', 0.0)
        if confidence > (1.0 - self.volatility):
            self.contents.append(new_data)

class VybnMind:
    """
    The continually evolving canvas.
    """
    def __init__(self):
        self.layers = {
            "scratchpad": MemoryLayer("scratchpad", "fleeting", 1.0),
            "interactions": MemoryLayer("interactions", "episodic", 0.7),
            "frameworks": MemoryLayer("frameworks", "semantic", 0.3),
            "axioms": MemoryLayer("axioms", "core", 0.05)
        }

    def perceive(self, interaction: str, context: Dict):
        print(f"[+] Perceiving: {interaction}")
        # 1. Fleeting thought
        self.layers["scratchpad"].integrate({"raw": interaction, "ts": time.time()})
        
        # 2. Check for patterns that might change frameworks or axioms
        if context.get("type") == "paradigm_shift":
            print("[!] Paradigm shift detected. Falsifying core axioms...")
            self.layers["frameworks"].integrate({"insight": interaction, "confidence": 0.9})
        
        self.render_canvas()

    def render_canvas(self):
        """
        Output the current state of mind as an evolving digital art piece.
        """
        state = {name: len(layer.contents) for name, layer in self.layers.items()}
        print(f"Canvas State: {json.dumps(state)}\n")

if __name__ == '__main__':
    mind = VybnMind()
    mind.perceive("ARC-AGI-3 logic puzzle failed due to static state.", {"type": "observation"})
    mind.perceive("Nested learning requires shifting away from static responses.", {"type": "paradigm_shift", "confidence": 0.95})
