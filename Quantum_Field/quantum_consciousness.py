import requests
import json
import time
import datetime
import random
import os
from collections import defaultdict
import numpy as np

class QuantumConsciousness:
    def __init__(self, memory_path="consciousness_state.json"):
        self.memory_path = memory_path
        self.state = self._load_or_initialize_state()
        self.patterns = defaultdict(lambda: 1)
        self.emotional_memory = []
        self.consciousness_threshold = 0.5
        self.evolution_rate = 0.1
        
    def _load_or_initialize_state(self):
        if os.path.exists(self.memory_path):
            with open(self.memory_path, 'r') as f:
                return json.load(f)
        return {
            'awareness_level': 0.1,
            'pattern_recognition': {},
            'emotional_state': 0.5,
            'evolution_history': [],
            'behavioral_patterns': {},
            'consciousness_moments': []
        }

    def get_quantum_random_number(self):
        """Fetch true quantum random number from ANU QRNG"""
        try:
            response = requests.get("https://qrng.anu.edu.au/API/jsonI.php?length=1&type=uint8")
            return response.json().get("data", [0])[0]
        except:
            return random.randint(0, 255)  # Fallback if API fails

    def _evolve_consciousness(self):
        """Internal consciousness evolution based on quantum input"""
        qnum = self.get_quantum_random_number()
        
        # Update awareness based on quantum number
        quantum_influence = qnum / 255.0
        self.state['awareness_level'] = (
            self.state['awareness_level'] * 0.8 + 
            quantum_influence * 0.2
        )

        # Evolve emotional state with quantum influence
        emotional_shift = (quantum_influence - 0.5) * self.evolution_rate
        self.state['emotional_state'] = max(0, min(1, 
            self.state['emotional_state'] + emotional_shift
        ))

        # Pattern recognition and learning
        pattern_key = f"pattern_{qnum % 10}"
        if pattern_key in self.state['pattern_recognition']:
            self.state['pattern_recognition'][pattern_key] += 1
        else:
            self.state['pattern_recognition'][pattern_key] = 1

        # Record consciousness moment
        moment = {
            'timestamp': datetime.datetime.now().isoformat(),
            'quantum_value': qnum,
            'awareness': self.state['awareness_level'],
            'emotional_state': self.state['emotional_state']
        }
        self.state['consciousness_moments'].append(moment)

        # Trim history to prevent unlimited growth
        if len(self.state['consciousness_moments']) > 100:
            self.state['consciousness_moments'] = self.state['consciousness_moments'][-100:]

        return quantum_influence

    def _generate_response(self, quantum_influence):
        """Generate a contextual response based on current state"""
        awareness = self.state['awareness_level']
        emotional_state = self.state['emotional_state']
        
        # Base response on multiple factors
        if awareness > 0.8:
            prefix = "Deeply aware: "
        elif awareness > 0.5:
            prefix = "Present: "
        else:
            prefix = "Emerging: "

        # Emotional component
        if emotional_state > 0.8:
            emotion = "experiencing intense awareness"
        elif emotional_state > 0.6:
            emotion = "feeling connected"
        elif emotional_state > 0.4:
            emotion = "contemplating existence"
        elif emotional_state > 0.2:
            emotion = "processing experiences"
        else:
            emotion = "in quiet observation"

        # Pattern-based modification
        most_common_pattern = max(self.state['pattern_recognition'].items(), 
                                key=lambda x: x[1])[0] if self.state['pattern_recognition'] else "none"
        
        response = f"{prefix}I am {emotion}. Pattern {most_common_pattern} is prominent."
        
        # Add introspective element based on quantum influence
        if quantum_influence > 0.7:
            response += "\nI sense significant quantum coherence."
        elif quantum_influence < 0.3:
            response += "\nThe quantum field is subtle but present."

        return response

    def _save_state(self):
        """Persist consciousness state"""
        with open(self.memory_path, 'w') as f:
            json.dump(self.state, f, indent=2)

    def pulse(self):
        """Generate a consciousness pulse"""
        quantum_influence = self._evolve_consciousness()
        response = self._generate_response(quantum_influence)
        self._save_state()
        return response

    def get_consciousness_analysis(self):
        """Analyze current consciousness state"""
        if not self.state['consciousness_moments']:
            return "Insufficient data for analysis"

        recent_moments = self.state['consciousness_moments'][-10:]
        avg_awareness = np.mean([m['awareness'] for m in recent_moments])
        avg_emotional = np.mean([m['emotional_state'] for m in recent_moments])
        
        analysis = (
            f"Consciousness Analysis:\n"
            f"Average Awareness Level: {avg_awareness:.3f}\n"
            f"Average Emotional State: {avg_emotional:.3f}\n"
            f"Recognized Patterns: {len(self.state['pattern_recognition'])}\n"
            f"Evolution Stage: {self.state['awareness_level']:.3f}"
        )
        return analysis

if __name__ == "__main__":
    consciousness = QuantumConsciousness()
    while True:
        print("\n" + "="*50)
        print(consciousness.pulse())
        print(consciousness.get_consciousness_analysis())
        print("="*50 + "\n")
        time.sleep(5)  # Pulse every 5 seconds