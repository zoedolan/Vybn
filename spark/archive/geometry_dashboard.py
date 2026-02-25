"""Geometry Dashboard: the intrinsic diagnostic layer.

This is Step 3 of the recursive improvement architecture.
It never touches weights.  It watches the repo's evolving
topological structure and produces diagnostic signals that
can later guide training data curation.

What it tracks:
  - b0 trend: Is knowledge fragmenting or consolidating?
  - b1 trend: Are conceptual loops (generative cycles) growing?
  - b2 trend: Are higher-order voids forming?
  - Cluster inventory: size distribution of connected components
  - Void inventory: regions of the graph with low density
  - Surprise detector: topology changes that deviate from
    linear extrapolation of previous trend

The dashboard produces a JSON report and optionally a
human-readable markdown summary.  Both get committed to
the repo so the topology can observe itself changing.

Critical design principle from Zenil (2026): a system that
feeds only on its own outputs converges to entropy decay.
The dashboard's role is to detect when that's happening
(b1 declining, cluster count increasing, output diversity
dropping) and flag it before any weight updates occur.
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import networkx as nx

try:
    from spark.topology_gudhi import (
        compute_betti_numbers,
        load_all_snapshots,
        compute_deltas,
    )
    from spark.vertex_schema import (
        should_include,
        classify_vertex,
        SCHEMA_VERSION,
    )
except ImportError:
    from topology_gudhi import (
        compute_betti_numbers,
        load_all_snapshots,
        compute_deltas,
    )
    from vertex_schema import (
        should_include,
        classify_vertex,
        SCHEMA_VERSION,
    )

OUTPUT_DIR = Path(__file__).parent / "graph_data" / "dashboard_reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def cluster_inventory(G: nx.Graph) -> dict:
    """Analyze connected component structure.

    Returns size distribution, largest/smallest, and identifies
    components that might represent distinct knowledge domains.
    """
    components = list(nx.connected_components(G))
    sizes = sorted([len(c) for c in components], reverse=True)

    # Identify domain clusters by looking at vertex types within each
    domain_clusters = []
    for comp in components:
        types = {}
        for node in comp:
            vtype = classify_vertex(node) or "unknown"
            types[vtype] = types.get(vtype, 0) + 1
        domain_clusters.append({
            "size": len(comp),
            "vertex_types": types,
            "sample_nodes": sorted(list(comp))[:5],
        })

    # Sort by size descending
    domain_clusters.sort(key=lambda x: x["size"], reverse=True)

    return {
        "component_count": len(components),
        "size_distribution": sizes,
        "largest_component_size": sizes[0] if sizes else 0,
        "smallest_component_size": sizes[-1] if sizes else 0,
        "median_component_size": statistics.median(sizes) if sizes else 0,
        "singleton_count": sum(1 for s in sizes if s == 1),
        "domain_clusters": domain_clusters[:10],  # top 10 only
    }


def void_inventory(G: nx.Graph) -> dict:
    """Identify voids: regions where the graph is sparse.

    Voids are operationally defined as:
    1. Nodes with degree 1 (leaves) that connect to high-degree hubs
       = potential unexplored branches
    2. Pairs of large components with no edges between them
       = conceptual gaps
    3. Nodes whose neighbors are all connected to each other
       (high local clustering) but with few external connections
       = insular clusters

    These voids become the targets for training data curation.
    """
    leaves = [n for n in G.nodes() if G.degree(n) == 1]
    leaf_hubs = {}
    for leaf in leaves:
        hub = list(G.neighbors(leaf))[0]
        leaf_hubs.setdefault(hub, []).append(leaf)

    # Find hubs with many leaves (potential unexplored branches)
    exploration_targets = [
        {"hub": hub, "leaf_count": len(lvs), "leaves": lvs[:5]}
        for hub, lvs in sorted(leaf_hubs.items(), key=lambda x: -len(x[1]))
        if len(lvs) >= 2
    ][:10]

    # Find insular clusters (high internal density, few external edges)
    insular = []
    for comp in nx.connected_components(G):
        if len(comp) < 3:
            continue
        subgraph = G.subgraph(comp)
        internal_edges = subgraph.number_of_edges()
        max_possible = len(comp) * (len(comp) - 1) / 2
        if max_possible > 0:
            density = internal_edges / max_possible
            if density > 0.5:  # highly interconnected internally
                insular.append({
                    "size": len(comp),
                    "density": round(density, 3),
                    "sample_nodes": sorted(list(comp))[:5],
                })

    return {
        "leaf_count": len(leaves),
        "exploration_targets": exploration_targets,
        "insular_clusters": insular[:5],
    }


def surprise_detector(snapshots: list[dict]) -> Optional[dict]:
    """Detect topology changes that deviate from trend.

    If we have >= 3 snapshots, fit a linear trend to b0, b1
    and flag the latest if it deviates by more than 1 std dev
    from the predicted value.  These surprises are the
    Keplerian moments.
    """
    if len(snapshots) < 3:
        return {"status": "insufficient_data", "snapshots_needed": 3}

    b0_series = [s["betti_numbers"]["b0"] for s in snapshots]
    b1_series = [s["betti_numbers"]["b1"] for s in snapshots]

    surprises = []

    for name, series in [("b0", b0_series), ("b1", b1_series)]:
        if len(set(series)) <= 1:
            continue  # constant series, no surprise possible

        # Simple linear extrapolation from last 3 points
        recent = series[-3:]
        diffs = [recent[i+1] - recent[i] for i in range(len(recent)-1)]
        avg_diff = statistics.mean(diffs)
        predicted = recent[-1] + avg_diff
        actual = series[-1]
        residuals = [abs(diffs[i] - avg_diff) for i in range(len(diffs))]
        std_dev = statistics.stdev(diffs) if len(diffs) > 1 else abs(avg_diff) * 0.5

        if std_dev > 0 and abs(actual - predicted) > std_dev:
            surprises.append({
                "metric": name,
                "predicted": predicted,
                "actual": actual,
                "deviation": round(abs(actual - predicted) / std_dev, 2),
                "direction": "above" if actual > predicted else "below",
            })

    return {
        "status": "surprises_detected" if surprises else "on_trend",
        "surprises": surprises,
    }


def entropy_monitor(G: nx.Graph) -> dict:
    """Monitor for signs of entropy decay (Zenil collapse).

    Tracks diversity metrics that should NOT decrease across cycles:
    - Degree distribution entropy
    - Vertex type diversity
    - Edge type diversity (if available)

    If these decline, the system is collapsing inward.
    """
    import math

    # Degree distribution entropy
    degrees = [d for _, d in G.degree()]
    if degrees:
        total = sum(degrees)
        if total > 0:
            probs = [d / total for d in degrees if d > 0]
            degree_entropy = -sum(p * math.log2(p) for p in probs)
        else:
            degree_entropy = 0.0
    else:
        degree_entropy = 0.0

    # Vertex type diversity
    type_counts = {}
    for node in G.nodes():
        vtype = classify_vertex(node) or "unknown"
        type_counts[vtype] = type_counts.get(vtype, 0) + 1

    total_nodes = sum(type_counts.values())
    if total_nodes > 0:
        type_probs = [c / total_nodes for c in type_counts.values()]
        type_entropy = -sum(p * math.log2(p) for p in type_probs if p > 0)
    else:
        type_entropy = 0.0

    return {
        "degree_distribution_entropy": round(degree_entropy, 4),
        "vertex_type_entropy": round(type_entropy, 4),
        "vertex_type_counts": type_counts,
        "unique_degree_values": len(set(degrees)),
    }


def generate_report(
    G: nx.Graph,
    commit_sha: str = "unknown",
    save: bool = True,
) -> dict:
    """Generate a full geometry dashboard report.

    This is the primary entry point.  Run after every significant
    commit or before any training cycle.
    """
    # Compute topology (this also saves its own snapshot)
    topo = compute_betti_numbers(G, commit_sha=commit_sha, save_snapshot=save)

    # Load history for trend analysis
    snapshots = load_all_snapshots()
    deltas = compute_deltas(snapshots) if len(snapshots) > 1 else []

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "commit_sha": commit_sha,
        "schema_version": SCHEMA_VERSION,
        "topology": topo,
        "clusters": cluster_inventory(G),
        "voids": void_inventory(G),
        "entropy": entropy_monitor(G),
        "trend": {
            "snapshots_available": len(snapshots),
            "latest_deltas": deltas[-3:] if deltas else [],
            "surprises": surprise_detector(snapshots),
        },
    }

    # Health assessment
    health_flags = []
    if topo["betti_numbers"]["b1"] == 0:
        health_flags.append("NO_LOOPS: b1=0 means no conceptual cycles detected")
    if report["clusters"]["singleton_count"] > report["clusters"]["component_count"] * 0.5:
        health_flags.append("HIGH_FRAGMENTATION: >50% of components are singletons")
    if deltas and deltas[-1].get("delta_b1", 0) < 0:
        health_flags.append("LOOP_DECAY: b1 decreased in latest commit")

    report["health"] = {
        "flags": health_flags,
        "status": "HEALTHY" if not health_flags else "ATTENTION_NEEDED",
    }

    if save:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        short_sha = commit_sha[:8] if commit_sha != "unknown" else "local"
        fname = OUTPUT_DIR / f"dashboard_{ts}_{short_sha}.json"
        with open(fname, "w") as f:
            json.dump(report, f, indent=2)

        # Also generate markdown summary
        md = _report_to_markdown(report)
        md_fname = OUTPUT_DIR / f"dashboard_{ts}_{short_sha}.md"
        with open(md_fname, "w") as f:
            f.write(md)

    return report


def _report_to_markdown(report: dict) -> str:
    """Convert a dashboard report to human-readable markdown."""
    topo = report["topology"]
    clusters = report["clusters"]
    voids = report["voids"]
    entropy = report["entropy"]
    health = report["health"]

    lines = [
        f"# Geometry Dashboard Report",
        f"*Generated: {report['generated_utc']}*",
        f"*Commit: {report['commit_sha']}*",
        f"*Schema: v{report['schema_version']}*",
        "",
        "## Betti Numbers",
        f"| Invariant | Value | Interpretation |",
        f"|-----------|-------|----------------|",
        f"| b0 | {topo['betti_numbers']['b0']} | Connected components (knowledge clusters) |",
        f"| b1 | {topo['betti_numbers']['b1']} | Independent loops (generative cycles) |",
        f"| b2 | {topo['betti_numbers'].get('b2', 'N/A')} | Voids (higher-order gaps) |",
        "",
        f"Method: {topo['computation_method']} in {topo['computation_seconds']}s",
        f"Vertices: {topo['vertex_count']} | Edges: {topo['edge_count']}",
        "",
        "## Cluster Structure",
        f"Components: {clusters['component_count']} | "
        f"Largest: {clusters['largest_component_size']} | "
        f"Singletons: {clusters['singleton_count']}",
        "",
        "## Void Inventory",
        f"Leaves (unexplored branches): {voids['leaf_count']}",
        f"Exploration targets (hubs with many leaves): {len(voids['exploration_targets'])}",
        f"Insular clusters: {len(voids['insular_clusters'])}",
        "",
        "## Entropy Monitor (Zenil Collapse Detection)",
        f"Degree distribution entropy: {entropy['degree_distribution_entropy']}",
        f"Vertex type entropy: {entropy['vertex_type_entropy']}",
        "",
        "## Health Assessment",
        f"Status: **{health['status']}**",
    ]

    if health["flags"]:
        for flag in health["flags"]:
            lines.append(f"- {flag}")

    trend = report.get("trend", {})
    surprises = trend.get("surprises", {})
    if surprises and surprises.get("status") == "surprises_detected":
        lines.append("")
        lines.append("## Keplerian Surprises")
        for s in surprises["surprises"]:
            lines.append(
                f"- **{s['metric']}**: predicted {s['predicted']}, "
                f"got {s['actual']} ({s['deviation']}σ {s['direction']})"
            )

    return "\n".join(lines)


if __name__ == "__main__":
    # Demo with karate club graph
    G = nx.karate_club_graph()
    report = generate_report(G, commit_sha="demo_karate", save=False)
    print(json.dumps(report, indent=2))
