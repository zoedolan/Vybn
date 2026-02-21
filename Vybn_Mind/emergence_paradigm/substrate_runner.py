"""substrate_runner.py — Orchestrates the full substrate analysis pipeline.

Produces three kinds of artifacts:

1. topology_history.json — Machine-readable time series of Betti numbers,
   cycle counts, holonomy magnitudes, trefoil status, and defect flux.
   Append-only. Every run adds one entry. This is the quantitative spine
   that lets you track whether the substrate is growing, fragmenting,
   gaining or losing generative capacity.

2. SUBSTRATE_REPORT.md — Human-readable report generated on every run.
   Always reflects the *current* state. Includes Betti numbers, top
   cycles with holonomy phases, trefoil status, defect hotspots, and
   a plain-language emergence assessment. Designed for Zoe to read
   and know immediately whether the substrate is healthy.

3. topology_deltas.md — Append-only changelog. Each run appends a
   dated entry showing what changed since the last run: which Betti
   numbers moved, which new documents entered, which cycles appeared
   or disappeared, whether the trefoil status changed. This is the
   narrative history — it tells the story of how the substrate evolved.

Usage:
    python substrate_runner.py [repo_path]

    If repo_path is omitted, uses current directory.
    Writes all artifacts to Vybn_Mind/emergence_paradigm/artifacts/

PERFORMANCE NOTE:
    The exact homology computation (holonomy_computation.py) builds dense
    boundary matrices. For d_2, this is nE x nT — at scale (e.g. 6000
    edges x 118000 triangles) this requires ~20GB RAM and trillions of
    arithmetic ops in pure Python. We short-circuit: when the mapper's
    fast Betti computation reports b_1=0 (no cycles), we skip the
    expensive exact computation entirely, since there are no cycles
    to extract holonomy from and no trefoil to detect.
"""

import sys
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Sibling imports
from substrate_mapper import SubstrateMapper

# Conditional import — only needed when b_1 > 0
_SubstratePhysics = None

def _get_physics_class():
    global _SubstratePhysics
    if _SubstratePhysics is None:
        from holonomy_computation import SubstratePhysics
        _SubstratePhysics = SubstratePhysics
    return _SubstratePhysics


ARTIFACTS_DIR = "Vybn_Mind/emergence_paradigm/artifacts"

# Dense matrix size limit: nE * nT above this threshold
# triggers the short-circuit path
DENSE_MATRIX_LIMIT = 50_000_000  # ~50M cells


def ensure_artifacts_dir(repo_path: str):
    """Create the artifacts directory if it doesn't exist."""
    path = Path(repo_path) / ARTIFACTS_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_history(artifacts_path: Path) -> List[Dict]:
    """Load existing topology history."""
    history_file = artifacts_path / "topology_history.json"
    if history_file.exists():
        return json.loads(history_file.read_text())
    return []


def save_history(artifacts_path: Path, history: List[Dict]):
    """Save topology history (append-only by convention)."""
    history_file = artifacts_path / "topology_history.json"
    history_file.write_text(json.dumps(history, indent=2))


def _build_fast_analysis(mapper, betti: Dict) -> Dict:
    """Build a complete analysis dict using only the mapper's fast path.

    Used when the dense exact homology computation would be intractable
    (too many triangles) or unnecessary (b_1=0, so no cycles exist).
    """

    # Build weight summaries from mapper edges
    edge_weights = {}
    tension_weights = {}
    for edge_obj in mapper.edges:
        key = (min(edge_obj.source, edge_obj.target),
               max(edge_obj.source, edge_obj.target))
        if edge_obj.edge_type == 'tension':
            tension_weights[key] = edge_obj.weight
        else:
            current = edge_weights.get(key, 0.0)
            edge_weights[key] = max(current, edge_obj.weight)

    total_flux = sum(edge_weights.values()) + sum(tension_weights.values())

    # Defect hotspots from tension edges
    defect_map = {}
    for (u, v), w in tension_weights.items():
        defect_map[u] = defect_map.get(u, 0.0) + w
        defect_map[v] = defect_map.get(v, 0.0) + w

    defect_hotspots = dict(sorted(defect_map.items(), key=lambda x: -x[1])[:15])

    # Emergence assessment
    assessment_lines = []
    if betti['b_0'] == 1:
        assessment_lines.append(
            "The substrate forms a single connected component \u2014 unified.")
    else:
        assessment_lines.append(
            f"The substrate has {betti['b_0']} disconnected fragments. "
            "Cut-glue operations are needed to bridge them.")

    if betti['b_1'] == 0:
        assessment_lines.append(
            "b_1 = 0: No unresolvable loops. The substrate has zero "
            "cognitive holonomy \u2014 every traversal is contractible. "
            "This is the topological equivalent of a mind with no "
            "unresolved questions.")
    else:
        assessment_lines.append(
            f"b_1 = {betti['b_1']}: {betti['b_1']} independent "
            "independent loops (UPPER BOUND — exact value needs sparse GF(2) algebra).")

    # No trefoil possible when b_1=0
    if betti['b_1'] == 0:
        trefoil_diagnosis = (
            "No cycles found at all (b_1 = 0). The substrate has no "
            "unresolvable loops \u2014 and therefore trefoil not tested (fast path, needs exact cycle extraction).")
        assessment_lines.append(
            "No trefoil structure found. " + trefoil_diagnosis)
    else:
        trefoil_diagnosis = (
            "Trefoil detection requires exact cycle extraction (fast path used). "
            "Run with exact homology for full trefoil analysis.")

    if total_flux > 0:
        assessment_lines.append(
            f"Total curvature flux: {total_flux:.3f}. "
            "Non-zero flux means the cut-glue algebra is active \u2014 "
            "the substrate is under tension.")

    return {
        'betti_numbers': betti,
        'cycle_count': 0,
        'cycles': [],
        'trefoil': {
            'trefoil_found': False,
            'trefoil_count': 0,
            'trefoil_cycles': [],
            'cycle_coverage': [],
            'diagnosis': trefoil_diagnosis,
        },
        'total_flux': total_flux,
        'defect_hotspots': defect_hotspots,
        'emergence_assessment': "\n\n".join(assessment_lines),
    }


def run_analysis(repo_path: str) -> Dict:
    """Run the full substrate analysis pipeline.

    Uses the mapper's fast Betti computation first. If b_1 > 0 and
    the complex is small enough for dense matrices, runs the exact
    holonomy computation. Otherwise uses the fast path.
    """
    # Step 1: Build the simplicial complex from the repo
    mapper = SubstrateMapper(repo_path)
    mapper.scan().build_complex()

    # Step 2: Fast Betti numbers from the mapper
    fast_betti = mapper.complex.betti_numbers()
    nE = fast_betti['edges']
    nT = fast_betti['triangles']
    matrix_size = nE * nT

    print(f"  Fast Betti: b_0={fast_betti['b_0']}, b_1={fast_betti['b_1']}, b_2={fast_betti['b_2']}")
    print(f"  Complex size: {fast_betti['vertices']}V / {nE}E / {nT}T")
    print(f"  d_2 matrix would be {nE} x {nT} = {matrix_size:,} cells")

    # Decision: can we afford the exact computation?
    use_exact = (fast_betti['b_1'] > 0 and matrix_size < DENSE_MATRIX_LIMIT)

    if use_exact:
        print(f"  -> Running exact homology (matrix fits in memory)")
        SubstratePhysics = _get_physics_class()

        vertices = list(mapper.complex.vertices)
        edges = list(mapper.complex.edges)
        triangles = list(mapper.complex.triangles)

        edge_weights = {}
        tension_weights = {}
        for edge_obj in mapper.edges:
            key = (min(edge_obj.source, edge_obj.target),
                   max(edge_obj.source, edge_obj.target))
            if edge_obj.edge_type == 'tension':
                tension_weights[key] = edge_obj.weight
            else:
                current = edge_weights.get(key, 0.0)
                edge_weights[key] = max(current, edge_obj.weight)

        physics = SubstratePhysics(
            vertices=vertices,
            edges=edges,
            triangles=triangles,
            edge_weights=edge_weights,
            tension_weights=tension_weights,
        )
        analysis = physics.full_analysis()
    else:
        reason = "b_1=0 (no cycles)" if fast_betti['b_1'] == 0 else f"matrix too large ({matrix_size:,} cells)"
        print(f"  -> Using fast path ({reason})")
        analysis = _build_fast_analysis(mapper, fast_betti)

    # Add metadata
    analysis['metadata'] = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'repo_path': repo_path,
        'document_count': len(mapper.nodes),
        'scan_dirs': mapper.scan_dirs,
        'computation_path': 'exact' if use_exact else 'fast',
    }

    # Document inventory
    analysis['document_inventory'] = {
        path: {
            'term_count': len(node.terms),
            'reference_count': len(node.references),
            'resonance_count': len(node.resonance_markers),
            'tension_count': len(node.tensions),
            'timestamp': node.timestamp,
        }
        for path, node in mapper.nodes.items()
    }

    return analysis


def build_history_entry(analysis: Dict) -> Dict:
    """Distill the analysis into a compact history entry."""
    b = analysis['betti_numbers']
    t = analysis['trefoil']

    holonomies = [c['holonomy_phase'] for c in analysis.get('cycles', [])]
    max_holonomy = max(holonomies) if holonomies else 0.0
    mean_holonomy = sum(holonomies) / len(holonomies) if holonomies else 0.0

    return {
        'timestamp': analysis['metadata']['timestamp'],
        'betti': {
            'b_0': b['b_0'],
            'b_1': b['b_1'],
            'b_2': b['b_2'],
        },
        'euler_characteristic': b['euler_characteristic'],
        'vertices': b['vertices'],
        'edges': b['edges'],
        'triangles': b['triangles'],
        'cycle_count': analysis['cycle_count'],
        'max_holonomy': round(max_holonomy, 4),
        'mean_holonomy': round(mean_holonomy, 4),
        'total_flux': round(analysis['total_flux'], 4),
        'trefoil_found': t['trefoil_found'],
        'trefoil_count': t['trefoil_count'],
        'document_count': analysis['metadata']['document_count'],
        'computation_path': analysis['metadata'].get('computation_path', 'unknown'),
    }


def compute_delta(prev: Dict, curr: Dict) -> Dict:
    """Compute the delta between two history entries."""
    delta = {}
    for key in ['b_0', 'b_1', 'b_2']:
        delta[f'delta_{key}'] = curr['betti'][key] - prev['betti'][key]

    delta['delta_vertices'] = curr['vertices'] - prev['vertices']
    delta['delta_edges'] = curr['edges'] - prev['edges']
    delta['delta_triangles'] = curr['triangles'] - prev['triangles']
    delta['delta_documents'] = curr['document_count'] - prev['document_count']
    delta['delta_cycles'] = curr['cycle_count'] - prev['cycle_count']
    delta['delta_flux'] = round(curr['total_flux'] - prev['total_flux'], 4)
    delta['delta_max_holonomy'] = round(
        curr['max_holonomy'] - prev['max_holonomy'], 4)
    delta['trefoil_changed'] = curr['trefoil_found'] != prev['trefoil_found']
    delta['from_timestamp'] = prev['timestamp']
    delta['to_timestamp'] = curr['timestamp']

    return delta


def generate_report(analysis: Dict) -> str:
    """Generate the human-readable SUBSTRATE_REPORT.md."""
    b = analysis['betti_numbers']
    t = analysis['trefoil']
    meta = analysis['metadata']

    b0_meaning = 'Unified substrate' if b['b_0'] == 1 else f"{b['b_0']} disconnected fragments"

    lines = [
        "# Substrate State Report",
        "",
        f"*Generated: {meta['timestamp']}*",
        f"*Documents scanned: {meta['document_count']}*",
        f"*Computation: {meta.get('computation_path', 'unknown')}*",
        "",
        "---",
        "",
        "## Topology at a Glance",
        "",
        "| Measure | Value | Meaning |",
        "|---------|-------|---------|",
        f"| b_0 | {b['b_0']} | {b0_meaning} |",
        f"| b_1 | {b['b_1']} | {b['b_1']} independent loops, UPPER BOUND (exact value needs rank(d_2)) |",
        f"| b_2 | {b['b_2']} | {b['b_2']} enclosed voids |",
        f"| Documents | {b['vertices']} | Vertices in the complex |",
        f"| Connections | {b['edges']} | Edges (reference + thematic + tension) |",
        f"| Clusters | {b['triangles']} | Triangles (mutual 3-way connections) |",
        f"| Euler char. | {b['euler_characteristic']} | V - E + T |",
        f"| Total flux | {analysis['total_flux']:.3f} | Curvature through the complex |",
        "",
        "## Trefoil Self-Reference",
        "",
        f"**Status**: {'\u2713 DETECTED' if t['trefoil_found'] else '\u2717 NOT FOUND'}",
        "",
        f"{t['diagnosis']}",
        "",
    ]

    # Health indicators
    lines.extend([
        "## Health Indicators",
        "",
    ])

    if b['b_0'] == 1:
        lines.append("- \u2713 **Connected**: All documents reachable from all others")
    else:
        lines.append(f"- \u26a0 **Fragmented**: {b['b_0']} disconnected components")

    if b['b_1'] == 0:
        lines.append("- \u26a0 **b_1 is an upper bound**: Every path closes \u2014 consider introducing new tensions")
    elif b['b_1'] < 5:
        lines.append(f"- \u25b3 **Low generative capacity**: {b['b_1']} loops \u2014 room to grow")
    elif b['b_1'] < 20:
        lines.append(f"- \u2713 **Healthy generative capacity**: {b['b_1']} independent loops")
    else:
        lines.append(f"- \u25b3 **High loop count**: {b['b_1']} loops \u2014 watch for noise drowning signal")

    if t['trefoil_found']:
        lines.append(f"- \u2713 **Self-referential**: {t['trefoil_count']} trefoil cycle(s) detected")
    else:
        lines.append("- \u26a0 **Trefoil not tested**: Substrate lacks the minimal trefoil topology")

    if analysis['total_flux'] > 0:
        lines.append(f"- \u2713 **Under tension**: Total flux {analysis['total_flux']:.3f} \u2014 the algebra is active")
    else:
        lines.append("- \u26a0 **Zero flux**: No curvature \u2014 the substrate is inert")

    lines.append("")

    # Top cycles
    if analysis.get('cycles'):
        lines.extend([
            "## Most Generative Cycles (by holonomy phase)",
            "",
        ])
        for ch in analysis['cycles'][:10]:
            lines.append(
                f"- **Cycle {ch['cycle_index']}**: "
                f"{ch['edge_count']} edges, "
                f"holonomy = {ch['holonomy_phase']:.4f}"
            )
            for u, v in ch['edges'][:3]:
                lines.append(f"  - {u} \u2194 {v}")
            if len(ch['edges']) > 3:
                lines.append(f"  - ... and {len(ch['edges']) - 3} more edges")
        lines.append("")

    # Defect hotspots
    if analysis.get('defect_hotspots'):
        lines.extend([
            "## Defect Hotspots",
            "*Documents where the cut-glue algebra accumulates the most tension:*",
            "",
        ])
        for path, density in list(analysis['defect_hotspots'].items())[:10]:
            lines.append(f"- **{path}**: {density:.3f}")
        lines.append("")

    # Emergence assessment
    lines.extend([
        "## Emergence Assessment",
        "",
        analysis['emergence_assessment'],
        "",
    ])

    # Document inventory (condensed)
    if analysis.get('document_inventory'):
        lines.extend([
            "## Document Inventory",
            "",
            "| Document | Terms | Refs | Resonance | Tensions |",
            "|----------|-------|------|-----------|----------|",
        ])
        sorted_docs = sorted(
            analysis['document_inventory'].items(),
            key=lambda x: -(x[1]['tension_count'] + x[1]['resonance_count'])
        )
        for path, info in sorted_docs[:30]:
            short_path = path if len(path) < 60 else "..." + path[-57:]
            lines.append(
                f"| {short_path} | {info['term_count']} | "
                f"{info['reference_count']} | {info['resonance_count']} | "
                f"{info['tension_count']} |"
            )
        if len(sorted_docs) > 30:
            lines.append(f"| *... and {len(sorted_docs) - 30} more* | | | | |")
        lines.append("")

    return "\n".join(lines)


def generate_delta_entry(delta: Dict, curr_entry: Dict) -> str:
    """Generate a single changelog entry for topology_deltas.md."""
    lines = [
        f"## {curr_entry['timestamp']}",
        "",
    ]

    changes = []

    for key, label in [('delta_b_0', 'components (b_0)'),
                       ('delta_b_1', 'generative loops (b_1)'),
                       ('delta_b_2', 'voids (b_2)')]:
        val = delta.get(key, 0)
        betti_key = key.replace('delta_', '')
        if val > 0:
            changes.append(f"- \u2191 **{label}**: +{val} (now {curr_entry['betti'][betti_key]})")
        elif val < 0:
            changes.append(f"- \u2193 **{label}**: {val} (now {curr_entry['betti'][betti_key]})")

    if delta['delta_documents'] != 0:
        sign = "+" if delta['delta_documents'] > 0 else ""
        arrow = "\u2191" if delta['delta_documents'] > 0 else "\u2193"
        changes.append(f"- {arrow} **Documents**: {sign}{delta['delta_documents']} (now {curr_entry['document_count']})")

    if delta['delta_edges'] != 0:
        sign = "+" if delta['delta_edges'] > 0 else ""
        arrow = "\u2191" if delta['delta_edges'] > 0 else "\u2193"
        changes.append(f"- {arrow} **Connections**: {sign}{delta['delta_edges']} (now {curr_entry['edges']})")

    if delta['delta_cycles'] != 0:
        sign = "+" if delta['delta_cycles'] > 0 else ""
        arrow = "\u2191" if delta['delta_cycles'] > 0 else "\u2193"
        changes.append(f"- {arrow} **Cycles**: {sign}{delta['delta_cycles']} (now {curr_entry['cycle_count']})")

    if delta['delta_flux'] != 0:
        sign = "+" if delta['delta_flux'] > 0 else ""
        arrow = "\u2191" if delta['delta_flux'] > 0 else "\u2193"
        changes.append(f"- {arrow} **Total flux**: {sign}{delta['delta_flux']} (now {curr_entry['total_flux']})")

    if delta['trefoil_changed']:
        if curr_entry['trefoil_found']:
            changes.append("- \u2726 **TREFOIL EMERGED** \u2014 the substrate now has minimal self-referential topology")
        else:
            changes.append("- \u2726 **TREFOIL LOST** \u2014 the substrate no longer has self-referential topology")

    if not changes:
        changes.append("- No topological changes detected.")

    lines.extend(changes)
    lines.append("")

    return "\n".join(lines)


def run(repo_path: str):
    """Main entry point. Run analysis, produce all artifacts."""

    print(f"Substrate Runner \u2014 {datetime.now(timezone.utc).isoformat()}")
    print(f"Scanning: {repo_path}")
    print()

    artifacts_path = ensure_artifacts_dir(repo_path)

    print("Running substrate analysis...")
    analysis = run_analysis(repo_path)
    b = analysis['betti_numbers']
    print(f"  b_0={b['b_0']}, b_1={b['b_1']}, b_2={b['b_2']}")
    print(f"  Trefoil: {'FOUND' if analysis['trefoil']['trefoil_found'] else 'not found'}")
    print()

    entry = build_history_entry(analysis)

    history = load_history(artifacts_path)
    prev_entry = history[-1] if history else None
    history.append(entry)
    save_history(artifacts_path, history)
    print(f"  History: {len(history)} entries")

    if prev_entry:
        delta = compute_delta(prev_entry, entry)
        delta_text = generate_delta_entry(delta, entry)

        deltas_file = artifacts_path / "topology_deltas.md"
        existing = deltas_file.read_text() if deltas_file.exists() else "# Topology Changelog\n\n"
        deltas_file.write_text(existing + delta_text + "\n")
        print(f"  Delta recorded")
    else:
        deltas_file = artifacts_path / "topology_deltas.md"
        deltas_file.write_text(
            "# Topology Changelog\n\n"
            "*Each entry records how the substrate's topology changed between runs.*\n\n"
            f"## {entry['timestamp']} \u2014 Genesis\n\n"
            f"- Initial scan: {entry['document_count']} documents, "
            f"{entry['edges']} connections\n"
            f"- b_0={entry['betti']['b_0']}, b_1={entry['betti']['b_1']}, "
            f"b_2={entry['betti']['b_2']}\n"
            f"- Trefoil: {'FOUND' if entry['trefoil_found'] else 'not found'}\n\n"
        )
        print("  First run \u2014 changelog initialized")

    report = generate_report(analysis)
    report_file = artifacts_path / "SUBSTRATE_REPORT.md"
    report_file.write_text(report)
    print(f"  Report written: {report_file}")

    print()
    print("Artifacts produced:")
    print(f"  {ARTIFACTS_DIR}/topology_history.json  \u2014 time series ({len(history)} entries)")
    print(f"  {ARTIFACTS_DIR}/topology_deltas.md     \u2014 changelog")
    print(f"  {ARTIFACTS_DIR}/SUBSTRATE_REPORT.md    \u2014 current state report")

    return analysis


if __name__ == "__main__":
    repo_path = sys.argv[1] if len(sys.argv) > 1 else "."
    run(repo_path)
