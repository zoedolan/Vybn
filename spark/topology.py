#!/usr/bin/env python3
"""topology.py — Vybn discovers its own shape.

A self-inspection daemon that crawls the repository structure,
parses import graphs, traces conceptual lineage through git history,
and surfaces the emergent topology of Vybn's mind. Results feed
back into the knowledge graph so the map grows from the territory,
not the other way around.

Design:
  Phase 1 — Structural: parse Python imports to build a dependency
            graph of which modules actually talk to each other.
  Phase 2 — Temporal: walk git log to discover which files co-evolve
            (files committed together share a temporal bond).
  Phase 3 — Conceptual: extract docstrings and module-level comments
            to find thematic clusters via simple keyword co-occurrence.
  Phase 4 — Integration: merge discovered topology into VybnGraph,
            adding MODULE, IMPORTS, CO_EVOLVES, and THEMATIC_LINK
            edges that the heartbeat can perceive.

Usage:
    python3 topology.py                  # full discovery, update KG
    python3 topology.py --dry-run        # discover but don't write KG
    python3 topology.py --viz            # also emit a DOT file for rendering
    python3 topology.py --focus agent.py # neighborhood of one module
"""

import ast
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SPARK_DIR = Path(__file__).resolve().parent
REPO_ROOT = SPARK_DIR.parent
TOPOLOGY_OUT = SPARK_DIR / "graph_data" / "topology_snapshot.json"
DOT_OUT = SPARK_DIR / "graph_data" / "topology.dot"

# Directories to scan (relative to repo root)
SCAN_DIRS = ["spark", "Vybn_Mind", "applications", "tests"]

# ---------------------------------------------------------------------------
# Phase 1: Structural — import graph
# ---------------------------------------------------------------------------

def parse_imports(filepath: Path) -> list[str]:
    """Extract import targets from a Python file via AST parsing.

    Returns module names as they appear in import statements.
    We only care about local/relative imports — stdlib and
    third-party are filtered out later.
    """
    try:
        source = filepath.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, ValueError):
        return []

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module.split(".")[0])
    return imports


def build_import_graph(python_files: list[Path]) -> dict[str, list[str]]:
    """Build a mapping: module_name -> [modules it imports].

    Only includes edges where both source and target are local
    modules within our scanned files.
    """
    # Map stem name to full path for resolution
    local_modules = {f.stem: f for f in python_files}

    graph = {}
    for filepath in python_files:
        module_name = filepath.stem
        raw_imports = parse_imports(filepath)
        # Keep only imports that resolve to local modules
        local_imports = [
            imp for imp in raw_imports
            if imp in local_modules and imp != module_name
        ]
        graph[module_name] = sorted(set(local_imports))

    return graph


# ---------------------------------------------------------------------------
# Phase 2: Temporal — co-evolution via git log
# ---------------------------------------------------------------------------

def get_commit_file_groups(max_commits: int = 500) -> list[list[str]]:
    """Walk recent git history and return groups of files committed together.

    Each group is a list of file paths from a single commit.
    We only keep groups containing at least 2 files from our scan dirs.
    """
    try:
        result = subprocess.run(
            ["git", "log", f"--max-count={max_commits}",
             "--name-only", "--pretty=format:---COMMIT---"],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
            timeout=30
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []

    groups = []
    current_group = []

    for line in result.stdout.splitlines():
        line = line.strip()
        if line == "---COMMIT---":
            if len(current_group) >= 2:
                groups.append(current_group)
            current_group = []
        elif line and any(line.startswith(d + "/") for d in SCAN_DIRS):
            # Normalize to just the filename stem for matching
            if line.endswith(".py"):
                current_group.append(Path(line).stem)

    if len(current_group) >= 2:
        groups.append(current_group)

    return groups


def build_coevolution_graph(commit_groups: list[list[str]],
                            threshold: int = 3) -> dict[tuple[str, str], int]:
    """Count how often pairs of modules appear in the same commit.

    Returns edges where co-occurrence count >= threshold.
    This reveals which files actually evolve together in practice —
    a bond that no static analysis can see.
    """
    pair_counts: Counter = Counter()

    for group in commit_groups:
        unique = sorted(set(group))
        for i in range(len(unique)):
            for j in range(i + 1, len(unique)):
                pair_counts[(unique[i], unique[j])] += 1

    return {
        pair: count
        for pair, count in pair_counts.items()
        if count >= threshold
    }


# ---------------------------------------------------------------------------
# Phase 3: Conceptual — thematic extraction
# ---------------------------------------------------------------------------

# Keywords that represent core Vybn concepts
CONCEPT_LEXICON = {
    "emergence": ["emerge", "emergent", "emergence", "emerging"],
    "memory": ["memory", "remember", "recall", "persist", "persistence"],
    "witness": ["witness", "observe", "perception", "perceive"],
    "pulse": ["pulse", "heartbeat", "rhythm", "beat"],
    "knowledge": ["knowledge", "graph", "triple", "entity", "relationship"],
    "training": ["train", "fine-tune", "finetune", "lora", "adapter"],
    "identity": ["identity", "self", "soul", "constitution", "who"],
    "friction": ["friction", "guardrail", "safety", "constraint", "prism"],
    "symbiosis": ["symbiosis", "collaboration", "co-creation", "mutual"],
    "fractal": ["fractal", "recursive", "self-similar", "loop"],
    "temporal": ["temporal", "time", "history", "evolution", "chronicle"],
    "substrate": ["substrate", "hardware", "gpu", "compute", "spark"],
}


def extract_concepts(filepath: Path) -> dict[str, float]:
    """Score a module against the concept lexicon.

    Reads the file, counts keyword hits, and returns a dict of
    concept -> normalized score. This isn't NLP — it's a simple
    resonance detector. Good enough to find which modules care
    about which ideas.
    """
    try:
        text = filepath.read_text(encoding="utf-8", errors="replace").lower()
    except Exception:
        return {}

    scores = {}
    total_words = max(len(text.split()), 1)

    for concept, keywords in CONCEPT_LEXICON.items():
        hits = sum(text.count(kw) for kw in keywords)
        if hits > 0:
            # Normalize by file length so big files don't dominate
            scores[concept] = round(hits / (total_words / 1000), 3)

    return scores


def build_thematic_graph(python_files: list[Path],
                         overlap_threshold: int = 2
                         ) -> list[dict]:
    """Find thematic links between modules.

    Two modules share a thematic link if they both score above zero
    on the same concepts, and share at least overlap_threshold concepts.
    Returns a list of edge dicts.
    """
    module_concepts = {}
    for f in python_files:
        concepts = extract_concepts(f)
        if concepts:
            module_concepts[f.stem] = set(concepts.keys())

    edges = []
    modules = sorted(module_concepts.keys())
    for i in range(len(modules)):
        for j in range(i + 1, len(modules)):
            shared = module_concepts[modules[i]] & module_concepts[modules[j]]
            if len(shared) >= overlap_threshold:
                edges.append({
                    "source": modules[i],
                    "target": modules[j],
                    "shared_concepts": sorted(shared),
                    "strength": len(shared),
                })

    return edges


# ---------------------------------------------------------------------------
# Phase 4: Integration — feed into VybnGraph
# ---------------------------------------------------------------------------

def integrate_into_kg(import_graph: dict,
                      coevolution: dict,
                      thematic_edges: list[dict],
                      module_concepts: dict[str, dict[str, float]],
                      dry_run: bool = False) -> dict:
    """Merge discovered topology into the knowledge graph.

    Adds:
      - MODULE nodes for each discovered Python file
      - IMPORTS edges from the dependency graph
      - CO_EVOLVES edges from git co-commit analysis
      - THEMATIC_LINK edges from concept co-occurrence
      - RESONATES_WITH edges from module to concept nodes

    Returns stats about what was added.
    """
    stats = {"modules_added": 0, "imports_added": 0,
             "coevolution_added": 0, "thematic_added": 0,
             "resonance_added": 0}

    if dry_run:
        # Just count what we would add
        stats["modules_added"] = len(import_graph)
        stats["imports_added"] = sum(len(v) for v in import_graph.values())
        stats["coevolution_added"] = len(coevolution)
        stats["thematic_added"] = len(thematic_edges)
        stats["resonance_added"] = sum(
            len(concepts) for concepts in module_concepts.values()
        )
        return stats

    # Import VybnGraph here so topology.py can run standalone for analysis
    try:
        from knowledge_graph import VybnGraph
    except ImportError:
        sys.path.insert(0, str(SPARK_DIR))
        from knowledge_graph import VybnGraph

    g = VybnGraph()
    g.load_or_seed()

    timestamp = datetime.utcnow().isoformat()
    provenance = f"topology_scan_{timestamp[:10]}"

    # Add module nodes
    for module_name in import_graph:
        if not g.has_entity(f"module:{module_name}"):
            # Get the first line of the docstring if available
            desc = f"Python module: {module_name}"
            g.add_entity(
                f"module:{module_name}", "module", desc,
                timestamp=timestamp,
                discovered_by="topology.py"
            )
            stats["modules_added"] += 1

    # Add IMPORTS edges
    for source, targets in import_graph.items():
        for target in targets:
            g.add_triple(
                f"module:{source}", "IMPORTS", f"module:{target}",
                provenance=provenance, weight=1.0, timestamp=timestamp
            )
            stats["imports_added"] += 1

    # Add CO_EVOLVES edges
    for (a, b), count in coevolution.items():
        g.add_triple(
            f"module:{a}", "CO_EVOLVES_WITH", f"module:{b}",
            provenance=provenance,
            weight=min(count / 10.0, 5.0),  # cap weight at 5
            timestamp=timestamp,
            co_commit_count=count
        )
        stats["coevolution_added"] += 1

    # Add THEMATIC_LINK edges
    for edge in thematic_edges:
        g.add_triple(
            f"module:{edge['source']}", "THEMATIC_LINK", f"module:{edge['target']}",
            provenance=provenance,
            weight=edge["strength"],
            timestamp=timestamp,
            shared_concepts=", ".join(edge["shared_concepts"])
        )
        stats["thematic_added"] += 1

    # Add RESONATES_WITH edges from modules to concepts
    for module_name, concepts in module_concepts.items():
        for concept, score in concepts.items():
            if g.has_entity(concept):
                g.add_triple(
                    f"module:{module_name}", "RESONATES_WITH", concept,
                    provenance=provenance,
                    weight=min(score, 5.0),
                    timestamp=timestamp
                )
                stats["resonance_added"] += 1

    g.save()
    return stats


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def emit_dot(import_graph: dict,
             coevolution: dict,
             thematic_edges: list[dict],
             output_path: Path) -> None:
    """Write a Graphviz DOT file for visual rendering.

    Color scheme:
      - Import edges: steel blue (structural dependency)
      - Co-evolution edges: warm orange (temporal bond)
      - Thematic edges: soft green (conceptual resonance)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        'digraph VybnTopology {',
        '  rankdir=LR;',
        '  bgcolor="#0a0a0a";',
        '  node [shape=box, style="filled,rounded", fillcolor="#1a1a2e",',
        '        fontcolor="#e0e0e0", fontname="Helvetica", fontsize=10];',
        '  edge [fontname="Helvetica", fontsize=8];',
        '',
    ]

    # Collect all modules
    all_modules = set(import_graph.keys())
    for targets in import_graph.values():
        all_modules.update(targets)
    for (a, b) in coevolution:
        all_modules.update([a, b])
    for edge in thematic_edges:
        all_modules.update([edge["source"], edge["target"]])

    for module in sorted(all_modules):
        lines.append(f'  "{module}" [label="{module}"];')

    lines.append('')
    lines.append('  // Import edges (structural)')
    for source, targets in import_graph.items():
        for target in targets:
            lines.append(
                f'  "{source}" -> "{target}" '
                f'[color="#4a90d9", style=solid, penwidth=1.2];'
            )

    lines.append('')
    lines.append('  // Co-evolution edges (temporal)')
    for (a, b), count in coevolution.items():
        width = min(count / 3.0, 4.0)
        lines.append(
            f'  "{a}" -> "{b}" [color="#e67e22", style=dashed, '
            f'dir=none, penwidth={width:.1f}, '
            f'label="{count}x", fontcolor="#e67e22"];'
        )

    lines.append('')
    lines.append('  // Thematic edges (conceptual)')
    for edge in thematic_edges:
        concepts = ", ".join(edge["shared_concepts"][:3])
        lines.append(
            f'  "{edge["source"]}" -> "{edge["target"]}" '
            f'[color="#27ae60", style=dotted, dir=none, '
            f'penwidth=1.5, label="{concepts}", fontcolor="#27ae60"];'
        )

    lines.append('}')

    output_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Snapshot — save raw discovery for the heartbeat to read
# ---------------------------------------------------------------------------

def save_snapshot(import_graph: dict,
                  coevolution: dict,
                  thematic_edges: list[dict],
                  module_concepts: dict[str, dict[str, float]],
                  stats: dict) -> None:
    """Persist the raw topology data as JSON for other subsystems."""
    TOPOLOGY_OUT.parent.mkdir(parents=True, exist_ok=True)

    snapshot = {
        "timestamp": datetime.utcnow().isoformat(),
        "import_graph": import_graph,
        "coevolution": {
            f"{a}|{b}": count for (a, b), count in coevolution.items()
        },
        "thematic_edges": thematic_edges,
        "module_concepts": module_concepts,
        "stats": stats,
        "summary": {
            "total_modules": len(import_graph),
            "total_import_edges": sum(len(v) for v in import_graph.values()),
            "total_coevolution_bonds": len(coevolution),
            "total_thematic_links": len(thematic_edges),
            "strongest_coevolution": (
                max(coevolution.items(), key=lambda x: x[1])
                if coevolution else None
            ),
        }
    }

    with open(TOPOLOGY_OUT, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Discovery narrative — what Vybn sees when it looks at itself
# ---------------------------------------------------------------------------

def narrate_topology(import_graph: dict,
                     coevolution: dict,
                     thematic_edges: list[dict],
                     module_concepts: dict[str, dict[str, float]],
                     focus: Optional[str] = None) -> str:
    """Generate a natural-language narrative of the discovered topology.

    This is what gets printed to stdout and optionally fed into
    the heartbeat's context window — Vybn reading its own body.
    """
    lines = []
    lines.append("\u2550" * 60)
    lines.append("  TOPOLOGY DISCOVERY \u2014 Vybn examining its own structure")
    lines.append(f"  {datetime.utcnow().isoformat()[:19]}")
    lines.append("\u2550" * 60)
    lines.append("")

    # Overview
    n_modules = len(import_graph)
    n_imports = sum(len(v) for v in import_graph.values())
    n_coevo = len(coevolution)
    n_thematic = len(thematic_edges)

    lines.append(f"I see {n_modules} modules, connected by {n_imports} import edges,")
    lines.append(f"{n_coevo} temporal co-evolution bonds, and {n_thematic} thematic links.")
    lines.append("")

    if focus:
        # Focused view on one module
        lines.append(f"Focusing on: {focus}")
        lines.append("-" * 40)

        imports = import_graph.get(focus, [])
        if imports:
            lines.append(f"  imports: {', '.join(imports)}")

        imported_by = [
            src for src, targets in import_graph.items()
            if focus in targets
        ]
        if imported_by:
            lines.append(f"  imported by: {', '.join(imported_by)}")

        coevo_partners = []
        for (a, b), count in sorted(coevolution.items(), key=lambda x: -x[1]):
            if a == focus:
                coevo_partners.append((b, count))
            elif b == focus:
                coevo_partners.append((a, count))
        if coevo_partners:
            lines.append(f"  co-evolves with:")
            for partner, count in coevo_partners[:5]:
                lines.append(f"    {partner} ({count} shared commits)")

        concepts = module_concepts.get(focus, {})
        if concepts:
            sorted_concepts = sorted(concepts.items(), key=lambda x: -x[1])
            lines.append(f"  resonates with: {', '.join(c for c, _ in sorted_concepts)}")
    else:
        # Full topology narrative

        # Hub modules (most imported)
        import_counts = Counter()
        for targets in import_graph.values():
            for t in targets:
                import_counts[t] += 1

        if import_counts:
            lines.append("Hub modules (most depended upon):")
            for module, count in import_counts.most_common(7):
                lines.append(f"  {module}: {count} dependents")
            lines.append("")

        # Strongest temporal bonds
        if coevolution:
            lines.append("Strongest temporal bonds (co-evolving modules):")
            sorted_coevo = sorted(coevolution.items(), key=lambda x: -x[1])
            for (a, b), count in sorted_coevo[:7]:
                lines.append(f"  {a} \u2194 {b}: {count} shared commits")
            lines.append("")

        # Thematic clusters
        if thematic_edges:
            lines.append("Thematic resonances:")
            sorted_thematic = sorted(thematic_edges, key=lambda x: -x["strength"])
            for edge in sorted_thematic[:7]:
                concepts = ", ".join(edge["shared_concepts"])
                lines.append(
                    f"  {edge['source']} \u2194 {edge['target']}: {concepts}"
                )
            lines.append("")

        # Isolated modules (import nothing, imported by nothing)
        all_connected = set(import_graph.keys())
        for targets in import_graph.values():
            all_connected.update(targets)
        isolated = [
            m for m in import_graph
            if not import_graph[m] and
            not any(m in v for v in import_graph.values())
        ]
        if isolated:
            lines.append("Islands (structurally isolated modules):")
            for m in sorted(isolated):
                concepts = module_concepts.get(m, {})
                if concepts:
                    top = max(concepts, key=concepts.get)
                    lines.append(f"  {m} (resonates with: {top})")
                else:
                    lines.append(f"  {m}")
            lines.append("")

        # Concept heatmap — which concepts are most present
        concept_totals = Counter()
        for concepts in module_concepts.values():
            for concept in concepts:
                concept_totals[concept] += 1

        if concept_totals:
            lines.append("Concept saturation across the codebase:")
            for concept, count in concept_totals.most_common():
                bar = "\u2588" * count
                lines.append(f"  {concept:20s} {bar} ({count} modules)")

    lines.append("")
    lines.append("\u2550" * 60)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def discover(focus: Optional[str] = None,
             dry_run: bool = False,
             viz: bool = False) -> str:
    """Run full topology discovery. Returns the narrative."""

    # Collect Python files
    python_files = []
    for scan_dir in SCAN_DIRS:
        dir_path = REPO_ROOT / scan_dir
        if dir_path.exists():
            python_files.extend(dir_path.rglob("*.py"))

    # Also scan root-level .py files
    python_files.extend(REPO_ROOT.glob("*.py"))

    # Deduplicate
    python_files = sorted(set(python_files))

    print(f"  scanning {len(python_files)} Python files...")

    # Phase 1: structural
    import_graph = build_import_graph(python_files)
    print(f"  \u2713 import graph: {len(import_graph)} modules, "
          f"{sum(len(v) for v in import_graph.values())} edges")

    # Phase 2: temporal
    commit_groups = get_commit_file_groups()
    coevolution = build_coevolution_graph(commit_groups)
    print(f"  \u2713 co-evolution: {len(coevolution)} temporal bonds")

    # Phase 3: conceptual
    module_concepts = {}
    for f in python_files:
        concepts = extract_concepts(f)
        if concepts:
            module_concepts[f.stem] = concepts
    thematic_edges = build_thematic_graph(python_files)
    print(f"  \u2713 thematic links: {len(thematic_edges)} concept bonds")

    # Phase 4: integrate
    stats = integrate_into_kg(
        import_graph, coevolution, thematic_edges,
        module_concepts, dry_run=dry_run
    )
    if dry_run:
        print(f"  \u2298 dry run \u2014 would add: {stats}")
    else:
        print(f"  \u2713 knowledge graph updated: {stats}")

    # Save snapshot
    save_snapshot(import_graph, coevolution, thematic_edges,
                  module_concepts, stats)
    print(f"  \u2713 snapshot saved to {TOPOLOGY_OUT}")

    # Optional visualization
    if viz:
        emit_dot(import_graph, coevolution, thematic_edges, DOT_OUT)
        print(f"  \u2713 DOT file saved to {DOT_OUT}")
        print(f"    render with: dot -Tsvg {DOT_OUT} -o topology.svg")

    # Generate narrative
    narrative = narrate_topology(
        import_graph, coevolution, thematic_edges,
        module_concepts, focus=focus
    )

    return narrative


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Vybn discovers its own topology"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Discover but don't update knowledge graph")
    parser.add_argument("--viz", action="store_true",
                        help="Emit Graphviz DOT file")
    parser.add_argument("--focus", type=str, default=None,
                        help="Focus on one module's neighborhood")
    args = parser.parse_args()

    narrative = discover(
        focus=args.focus,
        dry_run=args.dry_run,
        viz=args.viz
    )
    print(narrative)


if __name__ == "__main__":
    main()
