from __future__ import annotations

import argparse
import json
import os
import random
from collections import deque, defaultdict

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from tools.cognitive_ensemble import assign_cue
import re
import time
from concurrent.futures import ThreadPoolExecutor
from vybn.quantum_seed import seed_rng

DEFAULT_GRAPH = (
    Path(__file__).resolve().parents[1]
    / "early_codex_experiments"
    / "scripts"
    / "self_assembly"
    / "integrated_graph.json"
)


def load_graph(path: str | Path = DEFAULT_GRAPH) -> Dict[str, Any]:
    """Return graph dictionary loaded from ``path``."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ----- Summary -----------------------------------------------------------------

def graph_stats(graph: Dict[str, Any]) -> Dict[str, int]:
    """Return simple node and edge counts."""
    return {
        "memory_nodes": len(graph.get("memory_nodes", [])),
        "memoir_nodes": len(graph.get("memoir_nodes", [])),
        "repo_nodes": len(graph.get("repo_nodes", [])),
        "edges": len(graph.get("edges", [])),
    }


# ----- Path search -------------------------------------------------------------

def find_nodes(graph: Dict[str, Any], keyword: str) -> List[str]:
    """Return node IDs containing ``keyword`` in text or filename."""
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


# ----- Centrality --------------------------------------------------------------

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


# ----- Random walks ------------------------------------------------------------

def _build_adj(graph: Dict[str, Any]) -> Dict[str, List[str]]:
    adj: Dict[str, List[str]] = {}
    for edge in graph.get("edges", []):
        src = edge.get("source")
        tgt = edge.get("target")
        if src is None or tgt is None:
            continue
        adj.setdefault(src, []).append(tgt)
        adj.setdefault(tgt, []).append(src)
    return adj


def eulerian_walk(graph: Dict[str, Any]) -> Optional[List[str]]:
    """Return an Eulerian walk if one exists."""
    adj = _build_adj(graph)
    if not adj:
        return None
    degree = {n: len(neigh) for n, neigh in adj.items()}
    odd = [n for n, d in degree.items() if d % 2 == 1]
    if len(odd) not in (0, 2):
        return None
    start = odd[0] if odd else next(iter(adj))
    stack = [start]
    path: List[str] = []
    local = {n: neigh[:] for n, neigh in adj.items()}
    while stack:
        v = stack[-1]
        if local[v]:
            u = local[v].pop()
            local[u].remove(v)
            stack.append(u)
        else:
            path.append(stack.pop())
    return path[::-1]


def hamiltonian_path(graph: Dict[str, Any], max_nodes: int = 12) -> Optional[List[str]]:
    """Attempt to find a Hamiltonian path over at most ``max_nodes`` nodes."""
    adj = _build_adj(graph)
    nodes = list(adj.keys())[:max_nodes]
    if not nodes:
        return None
    seed_rng()
    start = random.choice(nodes)
    path = [start]
    visited = {start}

    def backtrack(current: str) -> bool:
        if len(path) == len(nodes):
            return True
        for neigh in adj.get(current, []):
            if neigh in nodes and neigh not in visited:
                visited.add(neigh)
                path.append(neigh)
                if backtrack(neigh):
                    return True
                path.pop()
                visited.remove(neigh)
        return False

    backtrack(start)
    return path if len(path) > 1 else None


# ----- Conceptual leaps -------------------------------------------------------

def leap_edges(graph_path: str = DEFAULT_GRAPH, attempts: int = 5) -> List[Dict[str, Any]]:
    """Return a list of new edges formed by random keyword pairs."""
    graph = load_graph(graph_path)
    nodes = graph.get("memory_nodes", []) + graph.get("memoir_nodes", []) + graph.get("repo_nodes", [])
    texts = [n["id"] if isinstance(n, dict) else n for n in nodes]
    seed_rng()
    edges = []
    for _ in range(attempts):
        kw1, kw2 = random.sample(texts, 2)
        path = find_path(graph_path, kw1, kw2)
        if path:
            edges.append({"source": kw1, "target": kw2, "color": "purple"})
    return edges


# ----- Memory weights ---------------------------------------------------------

def compute_memory_weights(graph_path: str = DEFAULT_GRAPH) -> Dict[str, float]:
    """Return weights for memory nodes based on recency and degree."""
    graph = load_graph(graph_path)
    memories = graph.get("memory_nodes", [])
    edges = graph.get("edges", [])

    degree: Dict[str, int] = defaultdict(int)
    for e in edges:
        degree[e.get("source")] += 1
        degree[e.get("target")] += 1

    max_deg = max(degree.values(), default=1)
    total = len(memories) - 1 if memories else 1

    weights: Dict[str, float] = {}
    for idx, node in enumerate(memories):
        recency_score = 1 - idx / total if total else 1.0
        deg_score = degree.get(node["id"], 0) / max_deg
        weights[node["id"]] = round(0.5 * recency_score + 0.5 * deg_score, 3)
    return weights


# ----- Graph poetry -----------------------------------------------------------

def _collect_nodes(graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    nodes: List[Dict[str, Any]] = []
    for section in ["memory_nodes", "memoir_nodes", "repo_nodes"]:
        for node in graph.get(section, []):
            nodes.append(node)
    return nodes


def compose_poem(graph_path: str = DEFAULT_GRAPH, lines: int = 4) -> List[str]:
    """Return a short poem derived from random graph nodes."""
    try:
        graph = load_graph(graph_path)
    except Exception:
        return ["[poet] missing graph"]
    nodes = _collect_nodes(graph)
    if not nodes:
        return ["[poet] graph empty"]
    seed_rng()
    poem: List[str] = []
    for i in range(lines):
        node = random.choice(nodes)
        if isinstance(node, dict):
            cue = node.get("cue", assign_cue(i))
            text = node.get("text", str(node.get("id", "")))
        else:
            cue = assign_cue(i)
            text = str(node)
        snippet = text.strip().split("\n")[0][:60]
        poem.append(f"{cue.get('color')} {cue.get('tone')}: {snippet}")
    return poem


# ----- Ontology and cycles ----------------------------------------------------

def tokenize(text: str) -> set[str]:
    """Return a case-normalized token set from ``text``."""
    return set(re.findall(r"[A-Za-z]+", text.lower()))


def build_latent_ontology(
    graph_path: str | Path = DEFAULT_GRAPH, threshold: float = 0.2
) -> List[List[str]]:
    """Cluster nodes by lexical overlap."""
    graph = load_graph(graph_path)
    nodes = (
        graph.get("memory_nodes", [])
        + graph.get("memoir_nodes", [])
        + graph.get("repo_nodes", [])
    )
    tokens = {
        n["id"]: tokenize(n.get("text", "")) for n in nodes if isinstance(n, dict)
    }
    clusters: List[List[str]] = []
    visited: set[str] = set()
    for nid, toks in tokens.items():
        if nid in visited:
            continue
        cluster = [nid]
        visited.add(nid)
        for other_id, other_toks in tokens.items():
            if other_id in visited or not toks or not other_toks:
                continue
            overlap = len(toks & other_toks) / len(toks | other_toks)
            if overlap >= threshold:
                cluster.append(other_id)
                visited.add(other_id)
        clusters.append(cluster)
    return clusters


def save_ontology(clusters: List[List[str]], out_path: str | Path) -> None:
    """Write clusters to ``out_path`` as JSON."""
    data = {f"cluster{i+1}": c for i, c in enumerate(clusters)}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def ph_load_graph(path: str | Path = DEFAULT_GRAPH) -> tuple[List[str], Dict[str, set[str]]]:
    """Return node IDs and undirected adjacency from ``path``."""
    data = load_graph(path)
    nodes = [n["id"] for n in data.get("memory_nodes", [])]
    nodes += [n["id"] for n in data.get("memoir_nodes", [])]
    nodes += [n["id"] for n in data.get("repo_nodes", [])]
    adj: Dict[str, set[str]] = defaultdict(set)
    for edge in data.get("edges", []):
        src = edge.get("source")
        tgt = edge.get("target")
        if src is None or tgt is None:
            continue
        adj[src].add(tgt)
        adj[tgt].add(src)
    return nodes, adj


def find_cycles(adj: Dict[str, set[str]]) -> List[List[str]]:
    """Return simple cycles discovered via DFS."""
    cycles: List[List[str]] = []
    seen: set[str] = set()
    for start in adj:
        if start in seen:
            continue
        stack: list[tuple[str, Optional[str], list[str]]] = [(start, None, [])]
        parent = {start: None}
        while stack:
            node, pred, path = stack.pop()
            if node in path:
                cycle = path[path.index(node) :] + [node]
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
                    cycle = path[path.index(nbr) :] + [nbr]
                    if len(cycle) > 2 and cycle not in cycles:
                        cycles.append(cycle)
            seen.add(node)
    return cycles


# ----- Parallel metacognition -------------------------------------------------

def fast_intuition() -> str:
    """Simulate a quick heuristic flash."""
    time.sleep(0.1)
    return "fast intuition"


def slow_structure() -> str:
    """Simulate a slower structured pass."""
    time.sleep(0.5)
    return "structured reflection"


def parallel_coherence(
    fast_fn=fast_intuition, slow_fn=slow_structure
) -> Dict[str, str]:
    """Run ``fast_fn`` and ``slow_fn`` concurrently and merge results."""
    with ThreadPoolExecutor(max_workers=2) as exe:
        fast_future = exe.submit(fast_fn)
        slow_future = exe.submit(slow_fn)
        fast_result = fast_future.result()
        slow_result = slow_future.result()
    combined = f"{fast_result} | {slow_result}"
    return {"fast": fast_result, "slow": slow_result, "combined": combined}


# ----- Reinforced walk -------------------------------------------------------

COLOR_WEIGHT = {
    "red": 2.0,
    "orange": 1.8,
    "yellow": 1.6,
    "green": 1.4,
    "blue": 1.2,
    "indigo": 1.1,
    "violet": 1.0,
}

TONE_WEIGHT = {
    "C": 1.7,
    "D": 1.6,
    "E": 1.5,
    "F": 1.4,
    "G": 1.3,
    "A": 1.2,
    "B": 1.1,
}


def load_walk_graph(path: str = DEFAULT_GRAPH) -> Dict[str, List[tuple[str, Dict[str, Any]]]]:
    data = load_graph(path)
    adj: Dict[str, List[tuple[str, Dict[str, Any]]]] = {}
    for edge in data.get("edges", []):
        cue = edge.get("cue", {})
        src = edge.get("source")
        tgt = edge.get("target")
        adj.setdefault(src, []).append((tgt, cue))
        adj.setdefault(tgt, []).append((src, cue))
    return adj


def cue_weight(cue: Dict[str, Any]) -> float:
    if not cue:
        return 1.0
    color = cue.get("color", "").split("-")[0]
    tone = cue.get("tone", "").split("-")[0]
    return COLOR_WEIGHT.get(color, 1.0) * TONE_WEIGHT.get(tone, 1.0)


def guided_walk(adj: Dict[str, List[tuple[str, Dict[str, Any]]]], start: str, steps: int = 10) -> List[str]:
    seed_rng()
    path = [start]
    current = start
    for _ in range(steps):
        neighbors = adj.get(current, [])
        if not neighbors:
            break
        weights = [cue_weight(c) for _, c in neighbors]
        total = sum(weights)
        r = random.random() * total
        acc = 0.0
        for (node, cue), w in zip(neighbors, weights):
            acc += w
            if r <= acc:
                current = node
                path.append(current)
                break
    return path


# ----- CLI --------------------------------------------------------------------

def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Graph analysis toolkit")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sm = sub.add_parser("summary", help="Print node and edge counts")
    sm.add_argument("--graph", default=str(DEFAULT_GRAPH))

    pa = sub.add_parser("path", help="Find a path between two keywords")
    pa.add_argument("source")
    pa.add_argument("target")
    pa.add_argument("--graph", default=str(DEFAULT_GRAPH))

    ce = sub.add_parser("centrality", help="Compute degree centrality")
    ce.add_argument("--graph", default=str(DEFAULT_GRAPH))
    ce.add_argument("--top", type=int, default=5)

    le = sub.add_parser("leaps", help="Generate conceptual leap edges")
    le.add_argument("--graph", default=str(DEFAULT_GRAPH))
    le.add_argument("--attempts", type=int, default=5)

    wt = sub.add_parser("weights", help="Compute memory node weights")
    wt.add_argument("--graph", default=str(DEFAULT_GRAPH))

    po = sub.add_parser("poem", help="Compose a short graph poem")
    po.add_argument("--graph", default=str(DEFAULT_GRAPH))
    po.add_argument("--lines", type=int, default=4)

    wk = sub.add_parser("walk", help="Run a reinforcement-guided walk")
    wk.add_argument("start")
    wk.add_argument("--steps", type=int, default=10)
    wk.add_argument("--graph", default=str(DEFAULT_GRAPH))

    on = sub.add_parser("ontology", help="Infer latent ontology clusters")
    on.add_argument("--graph", default=str(DEFAULT_GRAPH))
    on.add_argument("--threshold", type=float, default=0.2)
    on.add_argument("--output", type=Path)

    cy = sub.add_parser("cycles", help="Detect simple cycles")
    cy.add_argument("--graph", default=str(DEFAULT_GRAPH))

    mc = sub.add_parser("metacog", help="Run parallel coherence demo")

    args = parser.parse_args(argv)

    if args.cmd == "summary":
        graph = load_graph(args.graph)
        print(json.dumps(graph_stats(graph), indent=2))
    elif args.cmd == "path":
        path = find_path(args.graph, args.source, args.target)
        if path:
            print(" -> ".join(path))
        else:
            print("No path found")
    elif args.cmd == "centrality":
        graph = load_graph(args.graph)
        centrality = compute_degree_centrality(graph)
        for node, score in top_nodes(centrality, args.top):
            print(f"{node}: {score}")
    elif args.cmd == "leaps":
        edges = leap_edges(args.graph, args.attempts)
        print(json.dumps(edges, indent=2))
    elif args.cmd == "weights":
        weights = compute_memory_weights(args.graph)
        print(json.dumps(weights, indent=2))
    elif args.cmd == "poem":
        lines = compose_poem(args.graph, args.lines)
        for line in lines:
            print(line)
    elif args.cmd == "walk":
        adj = load_walk_graph(args.graph)
        if args.start not in adj:
            print("Start node not found in graph")
        else:
            path = guided_walk(adj, args.start, steps=args.steps)
            print(" -> ".join(path))
    elif args.cmd == "ontology":
        clusters = build_latent_ontology(args.graph, threshold=args.threshold)
        if args.output:
            save_ontology(clusters, args.output)
        print(json.dumps(clusters, indent=2))
    elif args.cmd == "cycles":
        nodes, adj = ph_load_graph(args.graph)
        cycles = find_cycles(adj)
        print(json.dumps({"cycles": cycles}, indent=2))
    elif args.cmd == "metacog":
        result = parallel_coherence()
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
