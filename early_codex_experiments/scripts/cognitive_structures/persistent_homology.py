import json
import argparse
from collections import defaultdict, deque


def load_graph(path):
    with open(path, 'r') as f:
        data = json.load(f)
    nodes = [n['id'] for n in data.get('memory_nodes', [])]
    nodes += [n['id'] for n in data.get('memoir_nodes', [])]
    nodes += [n['id'] for n in data.get('repo_nodes', [])]
    adj = defaultdict(set)
    for edge in data.get('edges', []):
        src = edge.get('source')
        tgt = edge.get('target')
        adj[src].add(tgt)
        adj[tgt].add(src)
    return nodes, adj


def find_cycles(adj):
    cycles = []
    seen = set()
    for start in adj:
        if start in seen:
            continue
        stack = [(start, None, [])]
        parent = {start: None}
        while stack:
            node, pred, path = stack.pop()
            if node in path:
                cycle = path[path.index(node):] + [node]
                if len(cycle) > 2 and cycle not in cycles:
                    cycles.append(cycle)
                continue
            path = path + [node]
            for nbr in adj[node]:
                if nbr == pred:
                    continue
                if nbr not in parent:
                    parent[nbr] = node
                    stack.append((nbr, node, path))
                elif nbr in path:
                    cycle = path[path.index(nbr):] + [nbr]
                    if len(cycle) > 2 and cycle not in cycles:
                        cycles.append(cycle)
            seen.add(node)
    return cycles


def main():
    parser = argparse.ArgumentParser(description='Detect simple cycles in the integrated graph.')
    parser.add_argument('--graph', default='scripts/self_assembly/integrated_graph.json', help='path to integrated graph')
    parser.add_argument('--output', default='scripts/self_assembly/graph_cycles.json', help='output JSON file')
    args = parser.parse_args()

    nodes, adj = load_graph(args.graph)
    cycles = find_cycles(adj)
    with open(args.output, 'w') as f:
        json.dump({'cycles': cycles}, f, indent=2)
    print(f'Found {len(cycles)} cycles; results saved to {args.output}')


if __name__ == '__main__':
    main()
