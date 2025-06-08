import json
import random

from ..co_emergence import seed_random

seed_random()


def _build_adj(graph):
    adj = {}
    for edge in graph.get('edges', []):
        src = edge.get('source')
        tgt = edge.get('target')
        if src is None or tgt is None:
            continue
        adj.setdefault(src, []).append(tgt)
        adj.setdefault(tgt, []).append(src)
    return adj


def eulerian_walk(graph):
    """Return a list of nodes forming an Eulerian walk if possible."""
    adj = _build_adj(graph)
    if not adj:
        return None
    degree = {n: len(neigh) for n, neigh in adj.items()}
    odd = [n for n, d in degree.items() if d % 2 == 1]
    if len(odd) not in (0, 2):
        return None
    start = odd[0] if odd else next(iter(adj))
    stack = [start]
    path = []
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


def hamiltonian_path(graph, max_nodes=12):
    """Attempt to find a Hamiltonian path over at most `max_nodes` nodes."""
    adj = _build_adj(graph)
    nodes = list(adj.keys())[:max_nodes]
    if not nodes:
        return None
    start = random.choice(nodes)
    path = [start]
    visited = {start}

    def backtrack(current):
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
