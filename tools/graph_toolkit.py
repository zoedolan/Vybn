from __future__ import annotations

import argparse
import json
import os
import random
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

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


if __name__ == "__main__":
    main()
