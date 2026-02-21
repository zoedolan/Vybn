"""Exact Betti numbers via sparse GF(2) boundary matrix rank.

Works on aarch64 (DGX Spark) where GUDHI wheels aren't available.
Uses numpy bitwise arrays for fast GF(2) column reduction.

CRITICAL FINDING: The substrate_mapper reports b1=0, b2=0 with
Euler characteristic 112843.  But b0 - b1 + b2 = 71 != 112843.
Those numbers are internally inconsistent.  b2=0 is impossible
with 118521 triangles.  This module computes the actual values.

The math:
  rank(d1) = |V| - b0 = 520 - 71 = 449
  b1 = (|E| - rank(d1)) - rank(d2) = 5749 - rank(d2)
  b2 = |T| - rank(d2)
  b2 - b1 = 112772  (from Euler constraint)

So if b1=0, then b2=112772 (NOT 0).  We compute rank(d2) to
find the true b1.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import numpy as np

try:
    from spark.vertex_schema import SCHEMA_VERSION
except ImportError:
    try:
        from vertex_schema import SCHEMA_VERSION
    except ImportError:
        SCHEMA_VERSION = "0.1.0"

SNAPSHOT_DIR = Path(__file__).parent / "graph_data" / "topology_snapshots"
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------
# GF(2) linear algebra via numpy bitwise arrays
# -----------------------------------------------------------------------

def _set_to_bitvec(s: set, n_words: int) -> np.ndarray:
    """Convert a set of row indices to a numpy uint64 bitarray."""
    arr = np.zeros(n_words, dtype=np.uint64)
    for idx in s:
        word, bit = divmod(idx, 64)
        arr[word] |= np.uint64(1) << np.uint64(bit)
    return arr


def _highest_bit(arr: np.ndarray) -> int:
    """Find the highest set bit in a uint64 bitarray, or -1 if zero."""
    for w in range(len(arr) - 1, -1, -1):
        val = arr[w]
        if val != 0:
            return w * 64 + int(val).bit_length() - 1
    return -1


def _gf2_column_rank(columns: list, n_rows: int, label: str = "") -> int:
    """Compute the rank of a matrix over GF(2) using column reduction.

    Each column is a set of row indices (nonzero entries).
    Uses numpy uint64 bitarrays for fast XOR.

    This is the standard persistence-style reduction:
    1. Convert each column to a bitarray
    2. Process left to right
    3. Find pivot (highest set bit)
    4. If pivot claimed, XOR with the claimer
    5. Repeat until unique pivot or zero
    6. Rank = number of columns that got a pivot
    """
    n_words = (n_rows + 63) // 64
    pivot_to_col = {}  # pivot_row -> bitarray
    rank = 0
    total = len(columns)
    t0 = time.monotonic()
    last_report = t0

    for i, col_set in enumerate(columns):
        # Progress every 30 seconds
        now = time.monotonic()
        if now - last_report > 30:
            elapsed = now - t0
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(f"    {label} column {i+1}/{total} "
                  f"(rank={rank}, {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")
            last_report = now

        working = _set_to_bitvec(col_set, n_words)

        while True:
            pivot = _highest_bit(working)
            if pivot < 0:
                break  # column reduced to zero
            if pivot not in pivot_to_col:
                pivot_to_col[pivot] = working.copy()
                rank += 1
                break
            else:
                np.bitwise_xor(working, pivot_to_col[pivot], out=working)

    return rank


# -----------------------------------------------------------------------
# Boundary matrix construction
# -----------------------------------------------------------------------

def _find_triangles(G: nx.Graph) -> list:
    """Find all triangles (3-cliques) in the graph.

    Returns sorted list of tuples (u, v, w) with u < v < w.
    """
    triangles = set()
    adj = {v: set(G.neighbors(v)) for v in G.nodes()}

    for u, v in G.edges():
        common = adj[u].intersection(adj[v])
        for w in common:
            tri = tuple(sorted([u, v, w]))
            triangles.add(tri)

    return sorted(triangles)


def compute_betti_numbers(
    G: nx.Graph,
    commit_sha: str = "unknown",
    save_snapshot: bool = True,
    skip_d2: bool = False,
) -> dict:
    """Compute exact Betti numbers b0, b1, b2 over GF(2).

    Args:
        G: The simplicial complex as a networkx graph.
           Vertices = nodes, 1-simplices = edges.
           Triangles (2-simplices) are found automatically.
        commit_sha: Git commit hash for versioning.
        save_snapshot: Whether to save the result to disk.
        skip_d2: If True, skip the expensive rank(d2) computation
                 and report b1, b2 as determined by Euler constraint
                 (b2 = b1 + chi - b0, both unknown individually).

    Returns:
        Dict with betti numbers, ranks, and metadata.
    """
    t0 = time.monotonic()

    # Vertices and edges
    vertices = sorted(G.nodes())
    v_index = {v: i for i, v in enumerate(vertices)}
    nV = len(vertices)

    edges_raw = list(G.edges())
    edges = sorted([tuple(sorted([u, v])) for u, v in edges_raw])
    edges = list(dict.fromkeys(edges))  # deduplicate preserving order
    e_index = {e: i for i, e in enumerate(edges)}
    nE = len(edges)

    # b0 exact (connected components)
    b0 = nx.number_connected_components(G)
    rank_d1 = nV - b0  # by definition

    # Triangles
    print(f"  Finding triangles...")
    t_tri = time.monotonic()
    triangles = _find_triangles(G)
    nT = len(triangles)
    print(f"  Found {nT} triangles ({time.monotonic()-t_tri:.1f}s)")

    # Euler characteristic
    euler = nV - nE + nT
    print(f"  Complex: {nV}V, {nE}E, {nT}T, chi={euler}")
    print(f"  b0 = {b0}, rank(d1) = {rank_d1}")

    if skip_d2:
        # Can't determine b1, b2 individually without rank(d2)
        # But we know b2 - b1 = euler - b0
        b1_plus_b2_constraint = euler - b0
        result_betti = {
            "b0": b0,
            "b1": "unknown (d2 skipped)",
            "b2": "unknown (d2 skipped)",
            "b2_minus_b1": euler - b0,
        }
        rank_d2 = None
        method = "scipy_gf2_partial"
    else:
        # Build d2 columns: each triangle -> its 3 boundary edges
        print(f"  Building d2 columns...")
        d2_columns = []
        for u, v, w in triangles:
            e1 = e_index.get((u, v))
            e2 = e_index.get((u, w))
            e3 = e_index.get((v, w))
            col = set()
            if e1 is not None: col.add(e1)
            if e2 is not None: col.add(e2)
            if e3 is not None: col.add(e3)
            d2_columns.append(col)

        # Compute rank(d2) over GF(2)
        print(f"  Computing rank(d2) over GF(2) [{nE} x {nT}]...")
        t_d2 = time.monotonic()
        rank_d2 = _gf2_column_rank(d2_columns, nE, label="d2")
        print(f"  rank(d2) = {rank_d2} ({time.monotonic()-t_d2:.1f}s)")

        b1 = (nE - rank_d1) - rank_d2
        b2 = nT - rank_d2

        # Verify Euler-Poincare
        euler_betti = b0 - b1 + b2
        consistent = (euler == euler_betti)

        result_betti = {"b0": b0, "b1": b1, "b2": b2}
        method = "scipy_gf2_exact"

        print(f"\n  === EXACT BETTI NUMBERS (GF(2) coefficients) ===")
        print(f"  b0 = {b0} (connected components)")
        print(f"  b1 = {b1} (independent 1-cycles / loops)")
        print(f"  b2 = {b2} (independent 2-cycles / voids)")
        print(f"  Euler: chi = {euler}")
        print(f"  Check: b0 - b1 + b2 = {b0} - {b1} + {b2} = {euler_betti}")
        print(f"  Consistent: {consistent}")

        if not consistent:
            print(f"  !!! EULER-POINCARE VIOLATION — BUG IN COMPUTATION !!!")

    elapsed = time.monotonic() - t0
    print(f"  Total time: {elapsed:.1f}s")

    snapshot = {
        "commit_sha": commit_sha,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inclusion_policy_version": SCHEMA_VERSION,
        "betti_numbers": result_betti,
        "ranks": {
            "rank_d1": rank_d1,
            "rank_d2": rank_d2,
        },
        "vertex_count": nV,
        "edge_count": nE,
        "triangle_count": nT,
        "euler_characteristic": euler,
        "computation_method": method,
        "computation_seconds": round(elapsed, 4),
    }

    if save_snapshot:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        short_sha = commit_sha[:8] if commit_sha != "unknown" else "local"
        fname = SNAPSHOT_DIR / f"topo_{ts}_{short_sha}.json"
        with open(fname, "w") as f:
            json.dump(snapshot, f, indent=2)
        print(f"  Snapshot saved: {fname}")

    return snapshot


# -----------------------------------------------------------------------
# Convenience: load snapshots for dashboard
# -----------------------------------------------------------------------

def load_all_snapshots() -> list:
    snapshots = []
    for p in sorted(SNAPSHOT_DIR.glob("topo_*.json")):
        with open(p) as f:
            snapshots.append(json.load(f))
    return snapshots


if __name__ == "__main__":
    # Quick test with karate club graph
    G = nx.karate_club_graph()
    print("Testing with Zachary's karate club (known: b0=1, b1=1, b2=0)\n")
    result = compute_betti_numbers(G, commit_sha="test_karate", save_snapshot=False)
    b = result["betti_numbers"]
    print(f"\n  Expected: b0=1, b1=1, b2=0")
    print(f"  Got:      b0={b['b0']}, b1={b['b1']}, b2={b['b2']}")
    if b["b0"] == 1 and b["b1"] == 1 and b["b2"] == 0:
        print("  PASS")
    else:
        print("  FAIL")
