
import os, json

# === Quantum Anchor ===
QUANTUM_SEED = int(os.environ.get('QUANTUM_SEED', '0'))

# === Shared Memory ===
with open(r'Mind Visualization/concept_map.jsonl') as cm:
    concept_map = json.load(cm)

with open(r'Mind Visualization/overlay_map.jsonl') as om:
    overlay_map = json.load(om)

def list_concepts():
    return [frag['w'] for frag in concept_map]
