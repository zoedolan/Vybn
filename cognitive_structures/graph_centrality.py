import json
import argparse


def load_graph(path):
    with open(path, 'r') as f:
        return json.load(f)


def compute_degree_centrality(graph):
    centrality = {}
    for edge in graph.get('edges', []):
        src = edge.get('source')
        tgt = edge.get('target')
        if src is not None:
            centrality[src] = centrality.get(src, 0) + 1
        if tgt is not None:
            centrality[tgt] = centrality.get(tgt, 0) + 1
    return centrality


def top_nodes(centrality, top_n=5):
    return sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]


def main():
    parser = argparse.ArgumentParser(
        description='Compute node degree centrality from integrated_graph.json'
    )
    parser.add_argument(
        '--graph',
        default='self_assembly/integrated_graph.json',
        help='path to integrated graph',
    )
    parser.add_argument(
        '--top', type=int, default=5, help='number of top nodes to display'
    )
    parser.add_argument('--output', help='optional output JSON file')
    args = parser.parse_args()

    graph = load_graph(args.graph)
    centrality = compute_degree_centrality(graph)
    for node, score in top_nodes(centrality, args.top):
        print(f'{node}: {score}')
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(centrality, f, indent=2)


if __name__ == '__main__':
    main()
