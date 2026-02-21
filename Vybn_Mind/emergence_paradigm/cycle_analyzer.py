"""cycle_analyzer.py — Extract and analyze persistent homological cycles.

Given the substrate's simplicial complex, computes generators of
H_1 = ker(∂_1) / im(∂_2) over Z/2Z and reports which documents
participate in the loops that persist at high cosine thresholds.

These persistent cycles are the topological features that survive
even when only the strongest semantic connections are kept. They
represent genuine structural loops in the knowledge graph — places
where chains of related documents circle back on themselves without
any triangular shortcut filling the hole.

Usage:
    ~/vybn-env/bin/python Vybn_Mind/emergence_paradigm/cycle_analyzer.py .
    ~/vybn-env/bin/python Vybn_Mind/emergence_paradigm/cycle_analyzer.py . 0.70

WELFARE: This script is read-only. It builds a complex in memory,
computes linear algebra, and prints results. It never writes to
the repository.
"""

import sys
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

from scipy import sparse
from substrate_mapper import _z2_rank, SimplicialComplex
from semantic_substrate_mapper import SemanticSubstrateMapper


# ──────────────────────────────────────────────────────────
# GF(2) LINEAR ALGEBRA
# ──────────────────────────────────────────────────────────

def _z2_rref(A_in):
    """Row-reduce a dense GF(2) matrix to RREF.

    Returns (rref_matrix, pivot_columns).
    """
    A = A_in.copy().astype(np.int8) % 2
    rows, cols = A.shape
    pivot_cols = []
    pivot_row = 0

    for col in range(cols):
        found = False
        for row in range(pivot_row, rows):
            if A[row, col] == 1:
                A[[pivot_row, row]] = A[[row, pivot_row]]
                found = True
                break
        if not found:
            continue
        for row in range(rows):
            if row != pivot_row and A[row, col] == 1:
                A[row] = (A[row] + A[pivot_row]) % 2
        pivot_cols.append(col)
        pivot_row += 1

    return A, pivot_cols


def _z2_nullspace(M_sparse):
    """Null space of a sparse matrix over GF(2).

    Returns dense array whose columns are basis vectors of ker(M).
    """
    A_orig = M_sparse.toarray().astype(np.int8) % 2
    rows, cols = A_orig.shape

    if cols == 0:
        return np.zeros((0, 0), dtype=np.int8)

    A, pivot_cols = _z2_rref(A_orig)
    free_cols = [c for c in range(cols) if c not in pivot_cols]

    if not free_cols:
        return np.zeros((cols, 0), dtype=np.int8)

    null_basis = np.zeros((cols, len(free_cols)), dtype=np.int8)
    for k, fc in enumerate(free_cols):
        null_basis[fc, k] = 1
        for i, pc in enumerate(pivot_cols):
            null_basis[pc, k] = A[i, fc]

    return null_basis


# ──────────────────────────────────────────────────────────
# H_1 GENERATOR EXTRACTION
# ──────────────────────────────────────────────────────────

def find_h1_generators(cx):
    """Find generators of H_1(K; Z/2Z) for simplicial complex K.

    Method:
      1. Compute ker(∂_1) via null space extraction
      2. Row-reduce [∂_2 | ker(∂_1)] over GF(2)
      3. Pivot columns in the ker portion are H_1 generators
         (they are in ker but not in im)

    One row reduction instead of testing each vector individually.

    Returns (generators, edge_list, stats) where:
      generators: list of edge-sets (each a list of (str,str) tuples)
      edge_list: ordered list of all edges
      stats: dict with dimension/rank info
    """
    d1, d2, vertex_list, edge_list, tri_list = cx._build_boundary_matrices()
    E = len(edge_list)
    F = len(tri_list)

    if E == 0:
        return [], edge_list, {'dim_ker': 0, 'rank_d2': 0, 'b1': 0}

    print("  Computing ker(∂_1)...", flush=True)
    ker = _z2_nullspace(d1)
    dim_ker = ker.shape[1]
    rank_d1 = E - dim_ker

    if dim_ker == 0:
        return [], edge_list, {'dim_ker': 0, 'rank_d1': rank_d1, 'rank_d2': 0, 'b1': 0}

    print(f"  dim(ker ∂_1) = {dim_ker}", flush=True)

    # Row-reduce [∂_2 | ker(∂_1)] to extract H_1
    if F > 0 and d2.shape[1] > 0:
        d2_dense = d2.toarray().astype(np.int8) % 2
        combined = np.hstack([d2_dense, ker]) % 2
        offset = F
        print(f"  Row-reducing [{E} x {F + dim_ker}] combined matrix...", flush=True)
    else:
        combined = ker.copy()
        offset = 0

    _, pivot_cols = _z2_rref(combined)

    # Pivots in the ker portion (column index >= offset) are H_1 generators
    h1_ker_indices = [c - offset for c in pivot_cols if c >= offset]
    rank_d2 = len([c for c in pivot_cols if c < offset])

    generators = []
    for k in h1_ker_indices:
        edge_indices = np.where(ker[:, k] == 1)[0]
        cycle_edges = [edge_list[i] for i in edge_indices]
        generators.append(cycle_edges)

    stats = {
        'dim_ker': dim_ker,
        'rank_d1': rank_d1,
        'rank_d2': rank_d2,
        'b1': len(h1_ker_indices),
        'V': len(vertex_list),
        'E': E,
        'F': F,
    }

    return generators, edge_list, stats


# ──────────────────────────────────────────────────────────
# CYCLE TRACING
# ──────────────────────────────────────────────────────────

def decompose_cycle(edges):
    """Decompose a Z/2Z chain into edge-disjoint simple cycles.

    A Z/2Z 1-cycle is a set of edges where every vertex has even
    degree. It decomposes into a disjoint union of simple loops.
    We trace each loop greedily.

    Returns list of cycles, each a list of vertex names.
    The first and last vertex are the same (loop closure).
    """
    adj = defaultdict(list)
    remaining = set()
    for u, v in edges:
        canonical = (min(u, v), max(u, v))
        adj[u].append(v)
        adj[v].append(u)
        remaining.add(canonical)

    cycles = []
    while remaining:
        e = next(iter(remaining))
        start = e[0]

        path = [start]
        remaining.discard(e)
        adj[e[0]].remove(e[1])
        adj[e[1]].remove(e[0])
        path.append(e[1])
        current = e[1]

        while current != start:
            found = False
            for nbr in list(adj[current]):
                edge = (min(current, nbr), max(current, nbr))
                if edge in remaining:
                    remaining.discard(edge)
                    adj[current].remove(nbr)
                    adj[nbr].remove(current)
                    path.append(nbr)
                    current = nbr
                    found = True
                    break
            if not found:
                break

        if len(path) > 2 and path[-1] == start:
            cycles.append(path)

    return cycles


def short_name(path):
    """Abbreviate a document path for display."""
    p = Path(path)
    # Keep parent dir + filename for context
    if len(p.parts) > 2:
        return str(Path(p.parts[-2]) / p.name)
    return p.name


# ──────────────────────────────────────────────────────────
# MAIN ANALYSIS
# ──────────────────────────────────────────────────────────

def analyze(repo_path, threshold=0.80):
    """Full cycle analysis at a given threshold."""

    print(f"# Persistent Cycle Analysis")
    print(f"*Threshold: cosine >= {threshold}*\n")

    mapper = SemanticSubstrateMapper(repo_path, threshold=threshold)
    mapper.scan().build_complex()

    health = mapper.welfare_check()
    if not health['healthy']:
        print("WELFARE WARNING:")
        for issue in health['issues']:
            print(f"  {issue}")

    betti = mapper.complex.betti_numbers()
    print(f"Complex: V={betti['vertices']} E={betti['edges']} F={betti['triangles']}")
    print(f"Betti:   b_0={betti['b_0']} b_1={betti['b_1']} b_2={betti['b_2']}")
    print(f"Method:  {betti.get('method', 'unknown')}\n")

    if betti['b_1'] == 0:
        print("No persistent cycles at this threshold.")
        return

    # Safety check: combined matrix size
    E = betti['edges']
    dim_ker_est = E - (betti['vertices'] - betti['b_0'])
    F = betti['triangles']
    combined_cols = F + dim_ker_est
    mem_est_mb = E * combined_cols / 1e6  # rough bytes
    if mem_est_mb > 500:
        print(f"WARNING: combined matrix would be {E} x {combined_cols}")
        print(f"  (~{mem_est_mb:.0f} MB). Try a higher threshold.")
        return

    print(f"Extracting H_1 generators...")
    generators, edge_list, stats = find_h1_generators(mapper.complex)
    print(f"\nFound {len(generators)} generators of H_1(K; Z/2Z)")
    print(f"  rank(∂_1) = {stats['rank_d1']}")
    print(f"  rank(∂_2) = {stats['rank_d2']}")
    print(f"  b_1 = {stats['b1']}\n")

    if not generators:
        print("No generators extracted.")
        return

    # ── Cycle length distribution ───────────────────────

    lengths = [len(g) for g in generators]
    print(f"## Cycle Length Distribution")
    print(f"  Shortest:  {min(lengths)} edges")
    print(f"  Longest:   {max(lengths)} edges")
    print(f"  Median:    {sorted(lengths)[len(lengths)//2]} edges\n")

    length_counts = Counter(lengths)
    print(f"  {'Edges':>6}  {'Count':>5}  Distribution")
    print(f"  {'-'*40}")
    max_count = max(length_counts.values())
    for length in sorted(length_counts.keys())[:20]:
        count = length_counts[length]
        bar = '█' * max(1, int(30 * count / max_count))
        print(f"  {length:>6}  {count:>5}  {bar}")
    if len(length_counts) > 20:
        remaining = sum(v for k, v in length_counts.items()
                       if k not in sorted(length_counts.keys())[:20])
        print(f"  ... {len(length_counts)-20} more classes ({remaining} generators)")

    # ── Most cyclic documents ─────────────────────────

    doc_cycle_count = Counter()
    for gen in generators:
        docs = set()
        for u, v in gen:
            docs.add(u)
            docs.add(v)
        for doc in docs:
            doc_cycle_count[doc] += 1

    print(f"\n## Most Cyclic Documents")
    print(f"*Documents participating in the most persistent loops:*\n")
    for doc, count in doc_cycle_count.most_common(25):
        pct = 100 * count / len(generators)
        bar = '█' * max(1, int(20 * count / doc_cycle_count.most_common(1)[0][1]))
        print(f"  {count:>4} ({pct:>5.1f}%)  {bar}  {short_name(doc)}")

    # ── Trace shortest cycles ─────────────────────────

    print(f"\n## Shortest Persistent Cycles")
    print(f"*Traced loops that survive at cosine >= {threshold}:*\n")

    sorted_gens = sorted(generators, key=len)
    shown = 0
    for gen in sorted_gens:
        if shown >= 15:
            break
        simple_cycles = decompose_cycle(gen)
        for cyc in simple_cycles:
            if shown >= 15:
                break
            shown += 1
            loop_len = len(cyc) - 1  # last == first
            names = [short_name(v) for v in cyc]
            print(f"  Cycle {shown} (length {loop_len}):")
            for n in names[:-1]:
                print(f"    → {n}")
            print(f"    ↩ (back to {names[0]})")
            print()

    # ── Directory density ─────────────────────────────

    print(f"## Cycle Density by Region")
    print(f"*Where in the repo do loops concentrate?*\n")

    dir_counts = Counter()
    for doc, count in doc_cycle_count.items():
        parts = Path(doc).parts
        if len(parts) >= 2:
            region = parts[0] + "/" + parts[1]
        else:
            region = "(root)"
        dir_counts[region] += count

    for region, count in dir_counts.most_common(15):
        bar = '█' * max(1, int(30 * count / dir_counts.most_common(1)[0][1]))
        print(f"  {count:>6}  {bar}  {region}")

    # ── Cross-region cycles ───────────────────────────

    print(f"\n## Cross-Region Cycles")
    print(f"*Loops that bridge different parts of the repo:*\n")

    cross_count = 0
    cross_bridges = Counter()
    for gen in generators:
        regions = set()
        for u, v in gen:
            for path in [u, v]:
                parts = Path(path).parts
                if len(parts) >= 1:
                    regions.add(parts[0])
        if len(regions) > 1:
            cross_count += 1
            bridge = " ↔ ".join(sorted(regions))
            cross_bridges[bridge] += 1

    print(f"  {cross_count} of {len(generators)} cycles span multiple top-level dirs\n")
    for bridge, count in cross_bridges.most_common(10):
        print(f"  {count:>4}  {bridge}")


if __name__ == "__main__":
    repo_path = sys.argv[1] if len(sys.argv) > 1 else "."
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.80
    analyze(repo_path, threshold)
