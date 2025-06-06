
import os
import json

from vybn.quantum_seed import cross_synaptic_kernel

# Access the quantum seed using the shared helper rather than a fixed default.
QUANTUM_SEED = cross_synaptic_kernel()

# === Shared Memory ===
with open(r'Mind Visualization/concept_map.jsonl') as cm:
    concept_map = json.load(cm)

with open(r'Mind Visualization/overlay_map.jsonl') as om:
    overlay_map = json.load(om)

def list_concepts():
    return [frag['w'] for frag in concept_map]
