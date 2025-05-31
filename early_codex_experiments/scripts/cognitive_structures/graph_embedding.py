import json
import random
import argparse
from ..quantum_rng import seed_random

seed_random()


def load_graph(path):
    with open(path, 'r') as f:
        data = json.load(f)
    nodes = [n['id'] for n in data.get('memory_nodes', [])]
    nodes += [n['id'] for n in data.get('memoir_nodes', [])]
    nodes += [n['id'] for n in data.get('repo_nodes', [])]
    adj = {n: set() for n in nodes}
    for edge in data.get('edges', []):
        src = edge.get('source')
        tgt = edge.get('target')
        adj.setdefault(src, set()).add(tgt)
        adj.setdefault(tgt, set()).add(src)
    return nodes, adj


def random_walk(adj, start, length=4):
    walk = [start]
    current = start
    for _ in range(length):
        neighbors = list(adj.get(current, []))
        if not neighbors:
            break
        current = random.choice(neighbors)
        walk.append(current)
    return walk


def build_embeddings(adj, walks=20, length=4):
    counts = {node: {} for node in adj}
    for node in adj:
        for _ in range(walks):
            walk = random_walk(adj, node, length)
            for target in walk[1:]:
                counts[node][target] = counts[node].get(target, 0) + 1
    return counts


def save_embeddings(embeddings, out_path):
    with open(out_path, 'w') as f:
        json.dump(embeddings, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Compute simple graph embeddings via random walks.')
    parser.add_argument('--graph', default='scripts/self_assembly/integrated_graph.json', help='path to integrated graph')
    parser.add_argument('--output', default='scripts/self_assembly/graph_embeddings.json', help='output JSON file')
    parser.add_argument('--walks', type=int, default=20, help='number of walks per node')
    parser.add_argument('--length', type=int, default=4, help='length of each walk')
    args = parser.parse_args()

    nodes, adj = load_graph(args.graph)
    embeddings = build_embeddings(adj, walks=args.walks, length=args.length)
    save_embeddings(embeddings, args.output)
    print(f'Embeddings saved to {args.output}')


if __name__ == '__main__':
    main()
