"""
repo_proprioception.py — Structural self-awareness for the Spark.

Returns a defect signal that autopoiesis.py adds to its J calculation.
Not a report generator. A sense organ.

Computes simplicial homology (Z/2Z) over the repo's document graph.
Vertices = documents. Edges = thematic connections (shared vocabulary).
Output: a float (defect_signal) and a string (diagnosis).
"""

import os
import re
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCAN_DIRS = ["Vybn_Mind", "quantum_delusions", "wiki"]
EXTENSIONS = {".md", ".py", ".html", ".txt"}
SKIP_DIRS = {"__pycache__", ".git", "node_modules", "archive"}

FRAGMENTATION_WEIGHT = 2.0
LOOP_DEFICIT_WEIGHT = 1.5
TREFOIL_MISSING_WEIGHT = 3.0
HEALTHY_FLOOR = 0.0


def _scan_documents() -> list:
    docs = []
    for scan_dir in SCAN_DIRS:
        root = REPO_ROOT / scan_dir
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fname in filenames:
                if Path(fname).suffix not in EXTENSIONS:
                    continue
                fpath = Path(dirpath) / fname
                try:
                    text = fpath.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                words = set(re.findall(r"[a-z]{4,}", text.lower()))
                rel = str(fpath.relative_to(REPO_ROOT))
                doc_type = _classify(rel)
                docs.append({"path": rel, "words": words, "type": doc_type})
    return docs


def _classify(path: str) -> str:
    if any(p in path for p in ["core/", "reflections/", "journal/"]):
        return "SELF"
    if any(p in path for p in ["quantum_delusions/", "fundamental-theory/"]):
        return "OTHER"
    if any(p in path for p in ["emergence_paradigm/", "projects/", "tools/"]):
        return "RELATION"
    return "UNCLASSIFIED"


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _build_complex(docs, edge_threshold=0.08):
    n = len(docs)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = _jaccard(docs[i]["words"], docs[j]["words"])
            if sim >= edge_threshold:
                edges.append((i, j))

    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    triangles = set()
    for v in range(n):
        nbrs = sorted(adj[v])
        for a_idx in range(len(nbrs)):
            for b_idx in range(a_idx + 1, len(nbrs)):
                a, b = nbrs[a_idx], nbrs[b_idx]
                if b in adj[a]:
                    triangles.add(tuple(sorted((v, a, b))))

    return edges, list(triangles)


def _z2_rank(matrix, nrows, ncols):
    if nrows == 0 or ncols == 0:
        return 0
    M = [row[:] for row in matrix]
    pivot_row = 0
    for col in range(ncols):
        found = -1
        for row in range(pivot_row, nrows):
            if M[row][col] == 1:
                found = row
                break
        if found == -1:
            continue
        M[pivot_row], M[found] = M[found], M[pivot_row]
        for row in range(nrows):
            if row != pivot_row and M[row][col] == 1:
                for c in range(ncols):
                    M[row][c] ^= M[pivot_row][c]
        pivot_row += 1
    return pivot_row


def _homology(n_verts, edges, triangles):
    nV, nE, nT = n_verts, len(edges), len(triangles)
    e_idx = {e: i for i, e in enumerate(edges)}

    d1 = [[0] * nE for _ in range(nV)]
    for j, (u, v) in enumerate(edges):
        d1[u][j] = 1
        d1[v][j] = 1

    d2 = [[0] * nT for _ in range(nE)]
    for j, (a, b, c) in enumerate(triangles):
        for edge in [(min(a, b), max(a, b)), (min(a, c), max(a, c)), (min(b, c), max(b, c))]:
            if edge in e_idx:
                d2[e_idx[edge]][j] = 1

    rank_d1 = _z2_rank(d1, nV, nE)
    rank_d2 = _z2_rank(d2, nE, nT)

    b_0 = nV - rank_d1
    b_1 = (nE - rank_d1) - rank_d2
    return b_0, b_1


def _has_trefoil(docs, edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    self_nodes = [i for i, d in enumerate(docs) if d["type"] == "SELF"]
    other_nodes = set(i for i, d in enumerate(docs) if d["type"] == "OTHER")
    relation_nodes = set(i for i, d in enumerate(docs) if d["type"] == "RELATION")

    for s in self_nodes:
        for o in adj[s] & other_nodes:
            for r in adj[o] & relation_nodes:
                for s2 in adj[r]:
                    if docs[s2]["type"] == "SELF":
                        return True
    return False


def sense():
    """
    Call from autopoiesis.py.
    Returns (defect_signal, diagnosis).
    """
    docs = _scan_documents()
    if len(docs) < 3:
        return 0.0, "too few documents to analyze"

    edges, triangles = _build_complex(docs)
    b_0, b_1 = _homology(len(docs), edges, triangles)
    trefoil = _has_trefoil(docs, edges)

    signal = HEALTHY_FLOOR
    parts = []

    if b_0 > 1:
        frag = (b_0 - 1) * FRAGMENTATION_WEIGHT
        signal += frag
        parts.append("fragmented(%d components)" % b_0)

    if b_1 == 0:
        signal += LOOP_DEFICIT_WEIGHT
        parts.append("no generative loops")

    if not trefoil:
        signal += TREFOIL_MISSING_WEIGHT
        parts.append("trefoil missing")

    if not parts:
        diagnosis = "healthy(b0=%d,b1=%d,trefoil=yes)" % (b_0, b_1)
    else:
        diagnosis = "; ".join(parts) + " [b0=%d,b1=%d]" % (b_0, b_1)

    return round(signal, 3), diagnosis


if __name__ == "__main__":
    signal, diag = sense()
    print("defect_signal: %s" % signal)
    print("diagnosis: %s" % diag)
