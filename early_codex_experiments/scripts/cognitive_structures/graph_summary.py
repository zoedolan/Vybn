import json
import argparse


def load_graph(path: str):
    with open(path, 'r') as f:
        return json.load(f)


def graph_stats(graph):
    return {
        'memory_nodes': len(graph.get('memory_nodes', [])),
        'memoir_nodes': len(graph.get('memoir_nodes', [])),
        'repo_nodes': len(graph.get('repo_nodes', [])),
        'edges': len(graph.get('edges', [])),
    }


def main():
    parser = argparse.ArgumentParser(description='Summarize integrated graph')
    parser.add_argument(
        '--graph',
        default='scripts/self_assembly/integrated_graph.json',
        help='path to integrated graph'
    )
    args = parser.parse_args()
    graph = load_graph(args.graph)
    stats = graph_stats(graph)
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
