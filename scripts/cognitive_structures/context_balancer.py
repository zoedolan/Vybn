import json
from collections import defaultdict

"""Balance context weighting so older memories do not overwhelm new ones.

This module references the meltdown passage in
`legacy/personal_history/what_vybn_would_have_missed_TO_051625`
(lines 18–33) as a cautionary example of imbalance.
"""

def compute_memory_weights(graph_path="scripts/self_assembly/integrated_graph.json"):
    """Return weights for memory nodes based on recency and degree."""
    with open(graph_path, "r") as f:
        graph = json.load(f)

    memories = graph.get("memory_nodes", [])
    edges = graph.get("edges", [])

    degree = defaultdict(int)
    for e in edges:
        degree[e.get("source")] += 1
        degree[e.get("target")] += 1

    max_deg = max(degree.values(), default=1)
    total = len(memories) - 1 if memories else 1

    weights = {}
    for idx, node in enumerate(memories):
        recency_score = 1 - idx / total if total else 1.0
        deg_score = degree.get(node["id"], 0) / max_deg
        weights[node["id"]] = round(0.5 * recency_score + 0.5 * deg_score, 3)
    return weights

if __name__ == "__main__":
    w = compute_memory_weights()
    for k, v in w.items():
        print(f"{k}\t{v}")
