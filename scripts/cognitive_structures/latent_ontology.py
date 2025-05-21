import json
import argparse
import re
from collections import defaultdict


def load_graph(path):
    with open(path, 'r') as f:
        return json.load(f)


def tokenize(text):
    return set(re.findall(r'[A-Za-z]+', text.lower()))


def build_latent_ontology(graph_path, threshold=0.2):
    graph = load_graph(graph_path)
    nodes = (
        graph.get('memory_nodes', [])
        + graph.get('memoir_nodes', [])
        + graph.get('repo_nodes', [])
    )
    tokens = {n['id']: tokenize(n.get('text', '')) for n in nodes if isinstance(n, dict)}
    clusters = []
    visited = set()
    for nid, toks in tokens.items():
        if nid in visited:
            continue
        cluster = [nid]
        visited.add(nid)
        for other_id, other_toks in tokens.items():
            if other_id in visited:
                continue
            if not toks or not other_toks:
                continue
            overlap = len(toks & other_toks) / len(toks | other_toks)
            if overlap >= threshold:
                cluster.append(other_id)
                visited.add(other_id)
        clusters.append(cluster)
    return clusters


def save_ontology(clusters, out_path):
    data = {f'cluster{i+1}': c for i, c in enumerate(clusters)}
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Infer latent ontology clusters from integrated graph.')
    parser.add_argument('--graph', default='scripts/self_assembly/integrated_graph.json', help='path to integrated graph')
    parser.add_argument('--output', default='scripts/self_assembly/latent_ontology.json', help='output JSON file')
    parser.add_argument('--threshold', type=float, default=0.2, help='Jaccard similarity threshold for clustering')
    args = parser.parse_args()
    clusters = build_latent_ontology(args.graph, threshold=args.threshold)
    save_ontology(clusters, args.output)
    print(f'Latent ontology saved to {args.output} with {len(clusters)} clusters.')
