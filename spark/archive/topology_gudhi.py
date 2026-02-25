"""Exact persistent homology via GUDHI.

Replaces the broken b1 approximation in topology.py with real computation.
The previous module used an Euler-characteristic heuristic that could be
off by thousands.  GUDHI computes simplicial homology in seconds for
graphs of this size.

Dependency: pip install gudhi
           (falls back to networkx-only approximation with loud warning)

Design principle: every call stores its result as a versioned snapshot
so the geometry dashboard can track deltas over time.  The snapshot
format is:
    {commit_sha, timestamp_utc, inclusion_policy_version,
     betti_numbers: {b0, b1, b2}, vertex_count, edge_count,
     simplex_count, computation_seconds}
"""

from __future__ import annotations

import json
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import networkx as nx

# ---------------------------------------------------------------------------
# Vertex schema (imported here so topology always respects the boundary)
# ---------------------------------------------------------------------------
try:
    from spark.vertex_schema import should_include, SCHEMA_VERSION
except ImportError:
    # Standalone execution
    from vertex_schema import should_include, SCHEMA_VERSION

# ---------------------------------------------------------------------------
# GUDHI import with graceful degradation
# ---------------------------------------------------------------------------
try:
    import gudhi
    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False
    import warnings
    warnings.warn(
        "GUDHI not installed. Betti numbers will use networkx approximation "
        "which is KNOWN TO BE INACCURATE. Run: pip install gudhi",
        RuntimeWarning,
        stacklevel=2,
    )

SNAPSHOT_DIR = Path(__file__).parent / "graph_data" / "topology_snapshots"
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


def _graph_to_simplex_tree(G: nx.Graph) -> "gudhi.SimplexTree":
    """Build a GUDHI SimplexTree from a networkx graph.

    Vertices become 0-simplices, edges become 1-simplices.
    Cliques of size 3 become 2-simplices (triangles), enabling
    b2 computation.  We cap at dimension 3 to keep it fast.
    """
    st = gudhi.SimplexTree()
    for node in G.nodes():
        st.insert([node], filtration=0.0)
    for u, v in G.edges():
        st.insert([u, v], filtration=0.0)
    # Add 2-simplices from triangles (cliques of size 3)
    for clique in nx.enumerate_all_cliques(G):
        if len(clique) == 3:
            st.insert(clique, filtration=0.0)
        elif len(clique) > 3:
            # Also add 3-simplices if present, then stop
            if len(clique) == 4:
                st.insert(clique, filtration=0.0)
            else:
                break
    return st


def _networkx_approximate_betti(G: nx.Graph) -> dict:
    """Fallback: rough approximation when GUDHI is unavailable.

    b0 = number of connected components (exact)
    b1 = edges - vertices + components  (Euler characteristic, INACCURATE)
    b2 = not computable without GUDHI
    """
    components = nx.number_connected_components(G)
    b0 = components
    b1_approx = G.number_of_edges() - G.number_of_nodes() + components
    return {
        "b0": b0,
        "b1": max(0, b1_approx),
        "b2": None,
        "method": "networkx_euler_approximation",
        "WARNING": "b1 is an upper bound via Euler characteristic, not exact homology",
    }


def compute_betti_numbers(
    G: nx.Graph,
    commit_sha: str = "unknown",
    save_snapshot: bool = True,
) -> dict:
    """Compute exact Betti numbers b0, b1, b2 for the repo knowledge graph.

    Returns a dict with betti numbers and metadata.
    Saves a timestamped snapshot to graph_data/topology_snapshots/.
    """
    t0 = time.monotonic()

    if GUDHI_AVAILABLE:
        st = _graph_to_simplex_tree(G)
        st.compute_persistence()
        betti = st.betti_numbers()
        result = {
            "b0": betti[0] if len(betti) > 0 else 0,
            "b1": betti[1] if len(betti) > 1 else 0,
            "b2": betti[2] if len(betti) > 2 else 0,
            "method": "gudhi_exact",
            "simplex_count": st.num_simplices(),
        }
    else:
        result = _networkx_approximate_betti(G)
        result["simplex_count"] = None

    elapsed = time.monotonic() - t0

    snapshot = {
        "commit_sha": commit_sha,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inclusion_policy_version": SCHEMA_VERSION,
        "betti_numbers": {
            "b0": result["b0"],
            "b1": result["b1"],
            "b2": result.get("b2"),
        },
        "vertex_count": G.number_of_nodes(),
        "edge_count": G.number_of_edges(),
        "simplex_count": result.get("simplex_count"),
        "computation_method": result["method"],
        "computation_seconds": round(elapsed, 4),
    }

    if save_snapshot:
        # Filename: timestamp + short hash for uniqueness
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        short_sha = commit_sha[:8] if commit_sha != "unknown" else "local"
        fname = SNAPSHOT_DIR / f"topo_{ts}_{short_sha}.json"
        with open(fname, "w") as f:
            json.dump(snapshot, f, indent=2)

    return snapshot


def load_all_snapshots() -> list[dict]:
    """Load all topology snapshots for longitudinal analysis."""
    snapshots = []
    for p in sorted(SNAPSHOT_DIR.glob("topo_*.json")):
        with open(p) as f:
            snapshots.append(json.load(f))
    return snapshots


def compute_deltas(snapshots: list[dict]) -> list[dict]:
    """Compute deltas between consecutive snapshots.

    This is the primary diagnostic signal: how did the topology
    change between commits?  Increasing b1 = new conceptual loops
    (generative).  Decreasing b1 = collapsed loops (degenerative).
    """
    deltas = []
    for i in range(1, len(snapshots)):
        prev, curr = snapshots[i - 1], snapshots[i]
        delta = {
            "from_commit": prev["commit_sha"],
            "to_commit": curr["commit_sha"],
            "from_ts": prev["timestamp_utc"],
            "to_ts": curr["timestamp_utc"],
            "delta_b0": curr["betti_numbers"]["b0"] - prev["betti_numbers"]["b0"],
            "delta_b1": curr["betti_numbers"]["b1"] - prev["betti_numbers"]["b1"],
            "delta_b2": (
                (curr["betti_numbers"]["b2"] - prev["betti_numbers"]["b2"])
                if curr["betti_numbers"]["b2"] is not None
                and prev["betti_numbers"]["b2"] is not None
                else None
            ),
            "delta_vertices": curr["vertex_count"] - prev["vertex_count"],
            "delta_edges": curr["edge_count"] - prev["edge_count"],
        }
        # Diagnostic interpretation
        if delta["delta_b1"] > 0:
            delta["interpretation"] = "GENERATIVE: new conceptual loops formed"
        elif delta["delta_b1"] < 0:
            delta["interpretation"] = "DEGENERATIVE: conceptual loops collapsed"
        else:
            delta["interpretation"] = "STABLE: loop structure unchanged"

        if delta["delta_b0"] < 0:
            delta["interpretation"] += " | CONSOLIDATING: components merging"
        elif delta["delta_b0"] > 0:
            delta["interpretation"] += " | FRAGMENTING: new disconnected clusters"

        deltas.append(delta)
    return deltas


if __name__ == "__main__":
    # Quick test with a small example
    G = nx.karate_club_graph()
    result = compute_betti_numbers(G, commit_sha="test_karate", save_snapshot=False)
    print(json.dumps(result, indent=2))
