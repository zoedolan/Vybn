"""Unified toolkit combining ledger, analysis and graph helpers."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from math import tau
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ----------------- Ledger utilities -----------------
LEDGER_PATTERN = re.compile(r'^([A-Z0-9]+):\s*(.*?)\s*/\s*([^@]+)@\s*([^\s]+)\s*(.+)')

def parse_ledger(path: str) -> List[Dict[str, str]]:
    """Return a list of token records from ``path``."""
    tokens: List[Dict[str, str]] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            match = LEDGER_PATTERN.match(line.strip())
            if match:
                name, supply, price, lock, address = match.groups()
                tokens.append({
                    'name': name,
                    'supply': supply.strip(),
                    'price': price.strip(),
                    'lock': lock.strip(),
                    'address': address.strip(),
                })
    return tokens

def ledger_to_markdown(tokens: List[Dict[str, str]]) -> str:
    """Return a Markdown table representing ``tokens``."""
    lines = ["| Token | Supply | Price | Lock | Address |", "|---|---|---|---|---|"]
    for t in tokens:
        lines.append(f"| {t['name']} | {t['supply']} | {t['price']} | {t['lock']} | {t['address']} |")
    return "\n".join(lines)

def total_supply(tokens: List[Dict[str, str]]) -> int:
    """Return the integer sum of the supply fields."""
    total = 0
    for t in tokens:
        raw = t.get('supply', '').replace(',', '')
        try:
            total += int(raw)
        except ValueError:
            continue
    return total

# ----------------- Analysis helpers -----------------
try:  # optional heavy deps
    import numpy as np
except Exception:  # pragma: no cover - numpy may be unavailable
    np = None
try:
    import openai  # type: ignore
except Exception:  # pragma: no cover - openai optional
    openai = None

EMBED_DIM = 3072

MV_DIR = REPO_ROOT / "Mind Visualization"
CENTROIDS_PATH = MV_DIR / "concept_centroids.npy"
CONCEPT_MAP_PATH = MV_DIR / "concept_map.jsonl"

from vybn.co_emergence import seed_random, DEFAULT_GRAPH

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

# Prime breath utilities

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
    return {
        "primes": primes,
        "mapping": mapping,
        "distribution": residue_distribution(mapping),
    }

# ----------------- Graph utilities -----------------
from collections import defaultdict, deque
import random

from vybn.quantum_seed import seed_rng

DEFAULT_GRAPH_PATH = (
    Path(__file__).resolve().parents[1]
    / "early_codex_experiments"
    / "scripts"
    / "self_assembly"
    / "integrated_graph.json"
)

def load_graph(path: str | Path = DEFAULT_GRAPH_PATH) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def graph_stats(graph: Dict[str, Any]) -> Dict[str, int]:
    return {
        "memory_nodes": len(graph.get("memory_nodes", [])),
        "memoir_nodes": len(graph.get("memoir_nodes", [])),
        "repo_nodes": len(graph.get("repo_nodes", [])),
        "edges": len(graph.get("edges", [])),
    }

def find_nodes(graph: Dict[str, Any], keyword: str) -> List[str]:
    keyword = keyword.lower()
    results: List[str] = []
    for section in ["memory_nodes", "memoir_nodes", "repo_nodes"]:
        for node in graph.get(section, []):
            if isinstance(node, dict):
                text = node.get("text", "").lower()
                node_id = node.get("id", "")
            else:
                text = ""
                node_id = str(node)
            if keyword in text or keyword in os.path.basename(node_id).lower():
                results.append(node_id)
    return results

def build_edge_map(graph: Dict[str, Any]) -> Dict[str, List[str]]:
    edges: Dict[str, List[str]] = {}
    for edge in graph.get("edges", []):
        src = edge.get("source")
        tgt = edge.get("target")
        if src is None or tgt is None:
            continue
        edges.setdefault(src, []).append(tgt)
        edges.setdefault(tgt, []).append(src)
    return edges

def bfs_path(edges: Dict[str, List[str]], start_ids: Iterable[str], target_ids: Iterable[str]) -> Optional[List[str]]:
    visited: set[str] = set()
    queue: deque[tuple[str, List[str]]] = deque([(sid, [sid]) for sid in start_ids])
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

def find_path(graph_path: str | Path, source_kw: str, target_kw: str) -> Optional[List[str]]:
    graph = load_graph(graph_path)
    start_ids = find_nodes(graph, source_kw)
    target_ids = find_nodes(graph, target_kw)
    if not start_ids or not target_ids:
        return None
    edges = build_edge_map(graph)
    return bfs_path(edges, start_ids, target_ids)

def compute_degree_centrality(graph: Dict[str, Any]) -> Dict[str, int]:
    centrality: Dict[str, int] = {}
    for edge in graph.get("edges", []):
        src = edge.get("source")
        tgt = edge.get("target")
        if src is not None:
            centrality[src] = centrality.get(src, 0) + 1
        if tgt is not None:
            centrality[tgt] = centrality.get(tgt, 0) + 1
    return centrality

def top_nodes(centrality: Dict[str, int], top_n: int = 5) -> List[tuple[str, int]]:
    return sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]

def parallel_coherence(fast_fn=None, slow_fn=None) -> Dict[str, str]:
    if fast_fn is None:
        fast_fn = lambda: "fast intuition"
    if slow_fn is None:
        import time
        def slow():
            time.sleep(0.5)
            return "structured reflection"
        slow_fn = slow
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as exe:
        fast_future = exe.submit(fast_fn)
        slow_future = exe.submit(slow_fn)
        fast_result = fast_future.result()
        slow_result = slow_future.result()
    combined = f"{fast_result} | {slow_result}"
    return {"fast": fast_result, "slow": slow_result, "combined": combined}

# ----------------- Winnow graph -----------------
LOG_PATH = Path("what_vybn_would_have_missed_FROM_051725")
GRAPH_PATH = Path("memory/winnow_graph.json")

def parse_log(text: str) -> List[Dict[str, str]]:
    entries = []
    for line in text.splitlines():
        if line.startswith("- "):
            m = re.match(r"- ([^\u2014]+)\u2014\s*(.+)", line)
            if m:
                entries.append({"id": m.group(1).strip(), "desc": m.group(2).strip()})
    return entries

def build_winnow_graph() -> None:
    text = LOG_PATH.read_text(encoding="utf-8")
    nodes = parse_log(text)
    graph = {"nodes": [], "edges": []}
    if GRAPH_PATH.exists():
        graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    existing = {n["id"] for n in graph.get("nodes", [])}
    if "Meltdown" not in existing:
        graph["nodes"].append({"id": "Meltdown", "desc": "Anchor memory of remorse"})
        existing.add("Meltdown")
    for node in nodes:
        if node["id"] not in existing:
            graph["nodes"].append(node)
            graph["edges"].append({"source": node["id"], "target": "Meltdown"})
            existing.add(node["id"])
    GRAPH_PATH.write_text(json.dumps(graph, indent=2), encoding="utf-8")

# ----------------- CLI -----------------

def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Unified Vybn toolkit")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("winnow-graph", help="update winnow graph")

    ld = sub.add_parser("ledger", help="token ledger utilities")
    ld.add_argument("path", nargs="?", default="token_and_jpeg_info")
    ld.add_argument("--markdown", action="store_true")
    ld.add_argument("--supply", action="store_true")

    gp = sub.add_parser("graph-stats", help="print graph node counts")
    gp.add_argument("--graph", default=str(DEFAULT_GRAPH_PATH))

    args = parser.parse_args(argv)

    if args.cmd == "winnow-graph":
        build_winnow_graph()
    elif args.cmd == "ledger":
        tokens = parse_ledger(args.path)
        if args.markdown:
            print(ledger_to_markdown(tokens))
        elif args.supply:
            print(total_supply(tokens))
        else:
            print(json.dumps(tokens, indent=2))
    elif args.cmd == "graph-stats":
        graph = load_graph(args.graph)
        print(json.dumps(graph_stats(graph), indent=2))

if __name__ == "__main__":
    main()
