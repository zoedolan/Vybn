import argparse
import json
import os
from pathlib import Path

from ..co_emergence import seed_random

seed_random()

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy may be unavailable
    np = None
try:
    import openai
except Exception:  # openai may not be installed in minimal env
    openai = None

EMBED_DIM = 3072


# The Mind Visualization folder lives at the repository root. This
# script is nested under early_codex_experiments/scripts/, so we need
# to step up three parents from this file to reach the repo root.
MV_DIR = Path(__file__).resolve().parents[3] / "Mind Visualization"
CENTROIDS_PATH = MV_DIR / "concept_centroids.npy"
CONCEPT_MAP_PATH = MV_DIR / "concept_map.jsonl"


def load_centroids(path=CENTROIDS_PATH):
    """Load concept centroids from a .npy file."""
    if np is None:
        raise ImportError("numpy is required")
    return np.load(path)


def load_concept_map(path=CONCEPT_MAP_PATH) -> list:
    """Return a list of concept map entries."""
    entries = []
    with open(path, "r") as f:
        for line in f:
            entries.append(json.loads(line))
    return entries


def embed_text(text: str, dim: int = EMBED_DIM):
    """Return an embedding vector for `text` via OpenAI or local fallback."""
    if openai is not None:
        try:
            resp = openai.Embedding.create(model="text-embedding-3-small", input=text)
            return np.array(resp["data"][0]["embedding"], dtype=float)
        except Exception:
            pass
    # Fallback: generate an embedding using quantum-seeded randomness
    if np is None:
        import random
        return [random.gauss(0.0, 1.0) for _ in range(dim)]
    return np.random.normal(size=dim).astype(float)


def cosine_similarity(a, b):
    if np is None:
        raise ImportError("numpy is required")
    a_norm = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.dot(a_norm, b_norm.T)


def find_closest_clusters(vec, centroids, top_k=5):
    if np is None:
        raise ImportError("numpy is required")
    sims = cosine_similarity(vec[None, :], centroids)[0]
    idx = np.argsort(sims)[::-1][:top_k]
    return [(int(i), float(sims[i])) for i in idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="Mind Visualization alignment helper")
    parser.add_argument("text", help="input text to embed and compare")
    parser.add_argument("--top", type=int, default=5, help="number of clusters to show")
    args = parser.parse_args()

    centroids = load_centroids()
    vec = embed_text(args.text)
    clusters = find_closest_clusters(vec, centroids, args.top)
    concept_map = load_concept_map()
    for cid, score in clusters:
        windows = [e for e in concept_map if e["c"] == cid][:3]
        print(f"cluster {cid} similarity {score:.3f}")
        for w in windows:
            print(f"  w={w['w']} file={w['f']} s={w['s']} e={w['e']}")


if __name__ == "__main__":
    main()

