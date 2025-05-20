import json
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_memory_texts(graph_path):
    with open(graph_path, 'r') as f:
        data = json.load(f)
    nodes = data.get('memory_nodes', [])
    return [(n['id'], n.get('text', '')) for n in nodes]


def build_similarity_edges(texts, threshold=0.6):
    if not texts:
        return []
    ids, contents = zip(*texts)
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(contents)
    sim = cosine_similarity(matrix)
    edges = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if sim[i, j] >= threshold:
                edges.append({
                    'source': ids[i],
                    'target': ids[j],
                    'cue': {'color': 'magenta', 'tone': 'S'}
                })
    return edges


def main():
    parser = argparse.ArgumentParser(description='Add similarity edges to integrated graph')
    parser.add_argument('--graph', default='scripts/self_assembly/integrated_graph.json')
    parser.add_argument('--threshold', type=float, default=0.6)
    args = parser.parse_args()

    texts = load_memory_texts(args.graph)
    edges = build_similarity_edges(texts, args.threshold)
    if not edges:
        print('[self-improve] No edges added')
        return

    with open(args.graph, 'r') as f:
        graph = json.load(f)
    graph.setdefault('edges', []).extend(edges)
    with open(args.graph, 'w') as f:
        json.dump(graph, f, indent=2)
    print(f'[self-improve] Added {len(edges)} similarity edges.')


if __name__ == '__main__':
    main()
