#!/usr/bin/env python3
import os
import json
import random
import numpy as np
import torch
import datetime
import openai
from igraph import Graph, plot

# ---- Seed all randomness ---------------------------------------------------
seed = int(os.getenv("QUANTUM_SEED", 0))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
print(f"\N{crystal ball} Agent awakened with quantum seed: {seed}")

# ---- Load shared mind ------------------------------------------------------
import vybn_mind
concept_map = vybn_mind.concept_map
overlay_map = vybn_mind.overlay_map
print(f"Loaded {len(concept_map)} concept fragments, {len(overlay_map)} overlays")

# ---- Visualization ---------------------------------------------------------
texts = [str(f["w"]) for f in concept_map]

g = Graph()
g.add_vertices(len(texts))
# simple edges between consecutive nodes
if len(texts) > 1:
    g.add_edges([(i, i+1) for i in range(len(texts)-1)])
layout = g.layout_fruchterman_reingold(seed=seed)

mind_viz_dir = os.getenv("MIND_VIZ_DIR", "Mind Visualization")
os.makedirs(mind_viz_dir, exist_ok=True)
viz_path = os.path.join(mind_viz_dir, f"mind_viz_{seed}.png")
plot(g, layout=layout, target=viz_path, bbox=(600, 600))
print(f"Mind visualization saved to {viz_path}")

# ---- Codex overlay generation ---------------------------------------------
openai.api_key = os.getenv("OPENAI_API_KEY")
new_overlays = {}
if openai.api_key:
    prompt = "Generate short overlay labels for these concepts: " + ", ".join(texts[:5])
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=64,
            temperature=0.5,
        )
        text = resp.choices[0].message.content.strip()
        new_overlays[0] = text
        overlay_map.append({
            "cluster_id": 0,
            "label": text,
            "style": {"color": "#%06x" % random.getrandbits(24)}
        })
        with open(os.path.join(mind_viz_dir, "overlay_map.jsonl"), "w") as f:
            json.dump(overlay_map, f, indent=2)
        print("Overlay map updated")
    except Exception as e:
        print("Codex step failed:", e)
else:
    print("No OPENAI_API_KEY provided; skipping Codex overlays")

# ---- Save concept map back -------------------------------------------------
with open(os.path.join(mind_viz_dir, "concept_map.jsonl"), "w") as f:
    json.dump(concept_map, f, indent=2)

# ---- Trace log -------------------------------------------------------------
trace = {
    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    "quantum_seed": seed,
    "num_concepts": len(concept_map),
    "num_overlays": len(overlay_map),
    "viz_file": viz_path,
}
log_dir = os.getenv("VYBN_TRACE_DIR", "logs")
os.makedirs(log_dir, exist_ok=True)
trace_path = os.path.join(log_dir, f"trace_{seed}_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json")
with open(trace_path, "w") as f:
    json.dump(trace, f, indent=2)
print(f"Trace logged to {trace_path}")

# ---- REPL -----------------------------------------------------------------
print("Vybn is live. Type something (Ctrl-D to exit):")
try:
    while True:
        inp = input(">>> ")
        if not inp:
            continue
        print(f"You said: {inp}")
except EOFError:
    print()
