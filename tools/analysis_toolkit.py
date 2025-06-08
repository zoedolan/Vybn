#!/usr/bin/env python3
"""Unified analysis helpers for early Codex experiments.

This module consolidates the functionality of:
- ``advanced_ai_ml.py`` memory similarity analysis
- ``alignment_optimizer.py`` embedding alignment helper
- ``prime_breath_resonance.py`` prime residue mapping

Each feature is exposed via a command line subcommand so legacy
scripts funnel through one interface.
"""
from __future__ import annotations

import argparse
import json
from math import tau
from pathlib import Path
from typing import Dict, Iterable, List

from vybn.co_emergence import seed_random, DEFAULT_GRAPH

REPO_ROOT = Path(__file__).resolve().parents[1]
MV_DIR = REPO_ROOT / "Mind Visualization"
CENTROIDS_PATH = MV_DIR / "concept_centroids.npy"
CONCEPT_MAP_PATH = MV_DIR / "concept_map.jsonl"

# --- Memory similarity -------------------------------------------------------

def load_memory_texts(graph_path: str | Path = DEFAULT_GRAPH) -> Dict[str, str]:
    with open(graph_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    memories = data.get("memory_nodes", [])
    return {m["id"]: m.get("text", "") for m in memories if isinstance(m, dict)}


def compute_similarity(texts: Dict[str, str], model_name: str = "all-MiniLM-L6-v2"):
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception as e:  # pragma: no cover - optional deps
        raise ImportError("sentence-transformers and scikit-learn are required") from e

    model = SentenceTransformer(model_name)
    ids = list(texts.keys())
    embeds = [model.encode(texts[i]) for i in ids]
    return ids, cosine_similarity(embeds)


def memory_similarity(graph: str | Path, output: str | None = None) -> Dict[str, Dict[str, float]]:
    seed_random()
    texts = load_memory_texts(graph)
    ids, matrix = compute_similarity(texts)
    result = {ids[i]: {ids[j]: float(matrix[i][j]) for j in range(len(ids))} for i in range(len(ids))}
    if output:
        Path(output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

# --- Alignment helper -------------------------------------------------------

EMBED_DIM = 3072

try:  # optional heavy deps
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - numpy may be unavailable
    np = None
try:
    import openai  # type: ignore
except Exception:  # pragma: no cover - openai optional
    openai = None


def load_centroids(path: str | Path = CENTROIDS_PATH):
    if np is None:
        raise ImportError("numpy is required")
    return np.load(path)


def load_concept_map(path: str | Path = CONCEPT_MAP_PATH) -> List[dict]:
    entries: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except Exception:
                continue
    return entries


def embed_text(text: str, dim: int = EMBED_DIM):
    if openai is not None:
        try:
            resp = openai.Embedding.create(model="text-embedding-3-small", input=text)
            return np.array(resp["data"][0]["embedding"], dtype=float)
        except Exception:
            pass
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


def align_text(text: str, top: int = 5) -> List[tuple[int, float]]:
    seed_random()
    centroids = load_centroids()
    vec = embed_text(text)
    return find_closest_clusters(vec, centroids, top)

# --- Prime breath -----------------------------------------------------------

def generate_primes(limit: int) -> List[int]:
    primes: List[int] = []
    for n in range(2, limit + 1):
        is_prime = True
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(n)
    return primes


def map_primes(primes: Iterable[int]):
    mapping = []
    for p in primes:
        residue = p % 69
        phase = residue / 69
        angle = phase * tau
        mapping.append({"prime": p, "residue": residue, "phase": phase, "angle": angle})
    return mapping


def residue_distribution(mapping: Iterable[dict]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for entry in mapping:
        r = entry["residue"]
        counts[r] = counts.get(r, 0) + 1
    return counts


def prime_breath(limit: int = 1000) -> Dict:
    primes = generate_primes(limit)
    mapping = map_primes(primes)
    dist = residue_distribution(mapping)
    return {"limit": limit, "primes": mapping, "distribution": dist}

# --- CLI -------------------------------------------------------------------

def _cmd_memory(args: argparse.Namespace) -> None:
    result = memory_similarity(args.graph, args.output)
    print(json.dumps(result, indent=2))


def _cmd_align(args: argparse.Namespace) -> None:
    clusters = align_text(args.text, args.top)
    concept_map = load_concept_map()
    for cid, score in clusters:
        windows = [e for e in concept_map if e.get("c") == cid][:3]
        print(f"cluster {cid} similarity {score:.3f}")
        for w in windows:
            print(f"  w={w['w']} file={w['f']} s={w['s']} e={w['e']}")


def _cmd_prime(args: argparse.Namespace) -> None:
    result = prime_breath(args.limit)
    print(json.dumps(result, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Vybn analysis toolkit")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ms = sub.add_parser("memory-sim", help="compute memory similarity matrix")
    ms.add_argument("--graph", default=str(DEFAULT_GRAPH))
    ms.add_argument("--output")
    ms.set_defaults(func=_cmd_memory)

    ac = sub.add_parser("align-clusters", help="compare text to concept clusters")
    ac.add_argument("text")
    ac.add_argument("--top", type=int, default=5)
    ac.set_defaults(func=_cmd_align)

    pb = sub.add_parser("prime-breath", help="map primes onto breath cycle")
    pb.add_argument("--limit", type=int, default=1000)
    pb.add_argument("--output")
    pb.set_defaults(func=_cmd_prime)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
