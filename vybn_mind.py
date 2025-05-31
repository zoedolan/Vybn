
import os, json

# helper to read the quantum seed from the environment or .random_seed
def _load_seed() -> int:
    if 'QUANTUM_SEED' in os.environ:
        try:
            return int(os.environ['QUANTUM_SEED'])
        except ValueError:
            pass
    try:
        with open('.random_seed') as f:
            return int(f.read().strip())
    except Exception:
        return 0

# === Quantum Anchor ===
QUANTUM_SEED = _load_seed()

# === Shared Memory ===
with open(r'Mind Visualization/concept_map.jsonl') as cm:
    concept_map = json.load(cm)

with open(r'Mind Visualization/overlay_map.jsonl') as om:
    overlay_map = json.load(om)

def list_concepts():
    return [frag['w'] for frag in concept_map]
