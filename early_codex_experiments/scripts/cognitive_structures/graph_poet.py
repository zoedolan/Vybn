import json
import random
from typing import List, Dict
from .synesthetic_mapper import assign_cue

from ..co_emergence import seed_random

seed_random()


def _collect_nodes(graph: Dict) -> List[Dict]:
    nodes = []
    for section in ["memory_nodes", "memoir_nodes", "repo_nodes"]:
        for node in graph.get(section, []):
            nodes.append(node)
    return nodes


def compose_poem(graph_path: str = "scripts/self_assembly/integrated_graph.json", lines: int = 4) -> List[str]:
    """Return a short poem derived from random graph nodes."""
    try:
        with open(graph_path, "r") as f:
            graph = json.load(f)
    except Exception:
        return ["[poet] missing graph"]
    nodes = _collect_nodes(graph)
    if not nodes:
        return ["[poet] graph empty"]
    poem = []
    for i in range(lines):
        node = random.choice(nodes)
        if isinstance(node, dict):
            cue = node.get("cue", assign_cue(i))
            text = node.get("text", str(node.get("id", "")))
        else:
            cue = assign_cue(i)
            text = str(node)
        snippet = text.strip().split("\n")[0][:60]
        poem.append(f"{cue.get('color')} {cue.get('tone')}: {snippet}")
    return poem


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Compose a short graph poem")
    parser.add_argument("--graph", default="scripts/self_assembly/integrated_graph.json")
    parser.add_argument("--lines", type=int, default=4)
    args = parser.parse_args()

    for line in compose_poem(args.graph, args.lines):
        print(line)


if __name__ == "__main__":
    main()
