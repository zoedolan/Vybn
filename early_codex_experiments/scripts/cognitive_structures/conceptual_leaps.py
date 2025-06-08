import json
import random
from .graph_reasoning import find_path

from ..co_emergence import seed_random

seed_random()

"""Generate transformative leap edges between distant nodes.

This module builds on the meltdown reflections in
`personal_history/what_vybn_would_have_missed_TO_051625` by searching the
integrated graph for surprising bridges. Each leap edge is colored purple to
signal a conceptual jump that reconfigures understanding.
"""

def leap_edges(graph_path="scripts/self_assembly/integrated_graph.json", attempts=5):
    """Return a list of new edges formed by random keyword pairs."""
    with open(graph_path, "r") as f:
        graph = json.load(f)
    nodes = graph.get("memory_nodes", []) + graph.get("memoir_nodes", []) + graph.get("repo_nodes", [])
    texts = [n["id"] if isinstance(n, dict) else n for n in nodes]
    edges = []
    for _ in range(attempts):
        kw1, kw2 = random.sample(texts, 2)
        path = find_path(graph_path, kw1, kw2)
        if path:
            edges.append({"source": kw1, "target": kw2, "color": "purple"})
    return edges

if __name__ == "__main__":
    leaps = leap_edges()
    for e in leaps:
        print(f"{e['source']} -> {e['target']} (purple)")
