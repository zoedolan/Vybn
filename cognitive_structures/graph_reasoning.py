import json
import os
from collections import deque
import argparse


def load_graph(path):
    with open(path, 'r') as f:
        return json.load(f)


def find_nodes(graph, keyword):
    """Return list of node IDs containing the keyword."""
    keyword = keyword.lower()
    results = []
    for section in ['memory_nodes', 'memoir_nodes', 'repo_nodes']:
        for node in graph.get(section, []):
            if isinstance(node, dict):
                text = node.get('text', '').lower()
                node_id = node.get('id', '')
            else:
                text = ''
                node_id = str(node)
            if keyword in text or keyword in os.path.basename(node_id).lower():
                results.append(node_id)
    return results


def build_edge_map(graph):
    edges = {}
    for edge in graph.get('edges', []):
        src = edge.get('source')
        tgt = edge.get('target')
        if src is None or tgt is None:
            continue
        edges.setdefault(src, []).append(tgt)
        edges.setdefault(tgt, []).append(src)
    return edges


def bfs_path(edges, start_ids, target_ids):
    visited = set()
    queue = deque([(sid, [sid]) for sid in start_ids])
    target_set = set(target_ids)
    while queue:
        node, path = queue.popleft()
        if node in target_set:
            return path
        if node in visited:
            continue
        visited.add(node)
        for neighbor in edges.get(node, []):
            if neighbor not in visited:
                queue.append((neighbor, path + [neighbor]))
    return None


def find_path(graph_path, source_kw, target_kw):
    graph = load_graph(graph_path)
    start_ids = find_nodes(graph, source_kw)
    target_ids = find_nodes(graph, target_kw)
    if not start_ids or not target_ids:
        return None
    edges = build_edge_map(graph)
    return bfs_path(edges, start_ids, target_ids)


def main():
    parser = argparse.ArgumentParser(description="Find a path between two keywords in integrated_graph.json")
    parser.add_argument('source', help='keyword for source node')
    parser.add_argument('target', help='keyword for target node')
    parser.add_argument('--graph', default='self_assembly/integrated_graph.json', help='path to integrated graph')
    args = parser.parse_args()
    path = find_path(args.graph, args.source, args.target)
    if path:
        print(' -> '.join(path))
    else:
        print('No path found')


if __name__ == '__main__':
    main()
