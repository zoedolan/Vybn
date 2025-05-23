import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def load_memory_texts(graph_path):
    with open(graph_path, 'r') as f:
        data = json.load(f)
    memories = data.get('memory_nodes', [])
    return {node['id']: node['text'] for node in memories}


def compute_similarity(texts, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    ids = list(texts.keys())
    embeddings = [model.encode(texts[_id]) for _id in ids]
    sim_matrix = cosine_similarity(embeddings)
    return ids, sim_matrix


def save_matrix(ids, matrix, out_path):
    output = {}
    for i, src in enumerate(ids):
        output[src] = {}
        for j, tgt in enumerate(ids):
            output[src][tgt] = float(matrix[i][j])
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)


if __name__ == '__main__':
    texts = load_memory_texts('scripts/self_assembly/integrated_graph.json')
    ids, matrix = compute_similarity(texts)
    save_matrix(ids, matrix, 'scripts/self_assembly/memory_similarity_matrix.json')
    print('Similarity matrix saved to scripts/self_assembly/memory_similarity_matrix.json')
