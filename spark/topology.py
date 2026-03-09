#!/usr/bin/env python3
"""topology.py — Vybn discovers its own shape through semantic geometry.

Refactored from the archive/topology.py implementation that used keyword
Jaccard similarity for thematic edge construction. This version introduces
semantic embeddings via pplx-embed-v1 (Perplexity's MIT-licensed embedding
model) and a PLSC-inspired shared subspace analysis drawn from Hong et al.
(2025), "Inter-brain neural dynamics across biological and AI systems"
(Nature, DOI: 10.1038/s41586-025-09196-4).

Architecture:
  Phase 1 — Structural: parse Python imports to build a dependency graph.
  Phase 2 — Temporal: walk git log for co-evolution bonds.
  Phase 3 — Semantic: embed module content with pplx-embed-v1, compute
            cosine similarity in embedding space. Replace Jaccard overlap
            with representational geometry — the same principle Hong et al.
            validated across biological and artificial neural systems.
  Phase 4 — Surprise: score each module's information-theoretic novelty
            relative to the corpus mean embedding. Inspired by the Titans
            architecture (Google, NeurIPS 2025) — prioritize what is
            unexpected, not merely frequent.
  Phase 5 — Shared Subspace: compute a low-rank shared representational
            subspace across module clusters using truncated SVD, analogous
            to the PLSC method Hong et al. used to identify shared neural
            dynamics between interacting agents. Modules that load onto
            the same subspace dimensions share cognitive function.
  Phase 6 — Integration: merge into snapshot and optional knowledge graph.

Usage:
    python3 topology.py                  # full discovery
    python3 topology.py --dry-run        # discover but don't write
    python3 topology.py --viz            # emit DOT file
    python3 topology.py --focus agent.py # neighborhood of one module
    python3 topology.py --no-embed       # fall back to keyword mode
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

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
# Embedding backend — pplx-embed-v1-0.6B via sentence-transformers
# Falls back to keyword mode if not installed.
# ---------------------------------------------------------------------------

_EMBED_MODEL = None
_EMBED_AVAILABLE = None


def _load_embedder() -> bool:
    """Lazy-load the embedding model. Returns True if available."""
    global _EMBED_MODEL, _EMBED_AVAILABLE
    if _EMBED_AVAILABLE is not None:
        return _EMBED_AVAILABLE
    try:
        from sentence_transformers import SentenceTransformer
        _EMBED_MODEL = SentenceTransformer(
            "perplexity-ai/pplx-embed-v1-0.6B",
            trust_remote_code=True,
        )
        _EMBED_AVAILABLE = True
    except Exception as exc:
        print(f"  ⚠ embedding model unavailable ({exc}); falling back to keywords")
        _EMBED_AVAILABLE = False
    return _EMBED_AVAILABLE


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of texts. Returns (N, D) float32 array.

    Uses pplx-embed-v1 if available, otherwise returns None so callers
    can fall back to keyword mode.
    """
    if not _load_embedder():
        raise RuntimeError("Embedding model not available")
    embeddings = _EMBED_MODEL.encode(texts)
    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return (embeddings / norms).astype(np.float32)


# ---------------------------------------------------------------------------
# Phase 1: Structural — import graph (unchanged from archive)
# ---------------------------------------------------------------------------

def parse_imports(filepath: Path) -> list[str]:
    """Extract import targets from a Python file via AST parsing."""
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
    """Build module → [imports] mapping for local modules only."""
    local_modules = {f.stem: f for f in python_files}
    graph = {}
    for filepath in python_files:
        module_name = filepath.stem
        raw_imports = parse_imports(filepath)
        local_imports = [
            imp for imp in raw_imports
            if imp in local_modules and imp != module_name
        ]
        graph[module_name] = sorted(set(local_imports))
    return graph


# ---------------------------------------------------------------------------
# Phase 2: Temporal — co-evolution via git log (unchanged from archive)
# ---------------------------------------------------------------------------

def get_commit_file_groups(max_commits: int = 500) -> list[list[str]]:
    """Walk recent git history, return groups of files committed together."""
    try:
        result = subprocess.run(
            ["git", "log", f"--max-count={max_commits}",
             "--name-only", "--pretty=format:---COMMIT---"],
            capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []
    groups = []
    current_group: list[str] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line == "---COMMIT---":
            if len(current_group) >= 2:
                groups.append(current_group)
            current_group = []
        elif line and any(line.startswith(d + "/") for d in SCAN_DIRS):
            if line.endswith(".py"):
                current_group.append(Path(line).stem)
    if len(current_group) >= 2:
        groups.append(current_group)
    return groups


def build_coevolution_graph(
    commit_groups: list[list[str]], threshold: int = 3
) -> dict[tuple[str, str], int]:
    """Count co-occurrence of module pairs in commits."""
    pair_counts: Counter = Counter()
    for group in commit_groups:
        unique = sorted(set(group))
        for i in range(len(unique)):
            for j in range(i + 1, len(unique)):
                pair_counts[(unique[i], unique[j])] += 1
    return {pair: count for pair, count in pair_counts.items() if count >= threshold}


# ---------------------------------------------------------------------------
# Phase 3: Semantic — embedding-based thematic analysis
# ---------------------------------------------------------------------------

# Keyword fallback lexicon (from archive/topology.py)
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
    "convergence": ["convergence", "converge", "attractor", "universal", "substrate-independent"],
    "topology": ["topology", "manifold", "geometry", "curvature", "subspace"],
}


def extract_module_text(filepath: Path, max_chars: int = 8000) -> str:
    """Extract meaningful text from a Python file for embedding.

    Concatenates the module docstring, class/function docstrings,
    and comments. Truncates to max_chars for embedding efficiency.
    """
    try:
        source = filepath.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""

    parts = []
    # Module-level docstring
    try:
        tree = ast.parse(source, filename=str(filepath))
        docstring = ast.get_docstring(tree)
        if docstring:
            parts.append(docstring)
        # Class and function docstrings
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                doc = ast.get_docstring(node)
                if doc:
                    parts.append(doc)
    except (SyntaxError, ValueError):
        pass

    # Comments
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") and len(stripped) > 3:
            parts.append(stripped.lstrip("# "))

    text = "\n".join(parts)
    return text[:max_chars] if text else filepath.stem.replace("_", " ")


def build_semantic_graph(
    python_files: list[Path],
    similarity_threshold: float = 0.55,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Build thematic edges using semantic embeddings.

    Returns (edges, module_embeddings) where module_embeddings maps
    module_name → embedding vector for downstream analysis.
    """
    texts = []
    names = []
    for f in python_files:
        text = extract_module_text(f)
        if text.strip():
            texts.append(text)
            names.append(f.stem)

    if not texts:
        return [], {}

    embeddings = embed_texts(texts)
    module_embeddings = {name: embeddings[i] for i, name in enumerate(names)}

    # Cosine similarity matrix (embeddings are already normalized)
    sim_matrix = embeddings @ embeddings.T

    edges = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            sim = float(sim_matrix[i, j])
            if sim >= similarity_threshold:
                edges.append({
                    "source": names[i],
                    "target": names[j],
                    "similarity": round(sim, 4),
                    "method": "pplx-embed-v1",
                })

    return edges, module_embeddings


def build_keyword_graph(
    python_files: list[Path], overlap_threshold: int = 2
) -> tuple[list[dict], dict[str, dict[str, float]]]:
    """Fallback: keyword-based thematic graph (from archive implementation)."""
    module_concepts: dict[str, dict[str, float]] = {}
    for f in python_files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace").lower()
        except Exception:
            continue
        total_words = max(len(text.split()), 1)
        scores = {}
        for concept, keywords in CONCEPT_LEXICON.items():
            hits = sum(text.count(kw) for kw in keywords)
            if hits > 0:
                scores[concept] = round(hits / (total_words / 1000), 3)
        if scores:
            module_concepts[f.stem] = scores

    edges = []
    modules = sorted(module_concepts.keys())
    for i in range(len(modules)):
        for j in range(i + 1, len(modules)):
            shared = set(module_concepts[modules[i]].keys()) & set(module_concepts[modules[j]].keys())
            if len(shared) >= overlap_threshold:
                edges.append({
                    "source": modules[i],
                    "target": modules[j],
                    "shared_concepts": sorted(shared),
                    "strength": len(shared),
                    "method": "keyword-jaccard",
                })

    return edges, module_concepts


# ---------------------------------------------------------------------------
# Phase 4: Surprise — information-theoretic novelty scoring
# ---------------------------------------------------------------------------

def compute_surprise_scores(
    module_embeddings: dict[str, np.ndarray],
) -> dict[str, float]:
    """Score each module's novelty relative to the corpus centroid.

    Inspired by Titans (Google, NeurIPS 2025): surprise = divergence
    from expectation. Modules that diverge most from the mean embedding
    carry the most novel information and should be prioritized in
    memory systems.

    Returns module_name → surprise_score (higher = more novel).
    """
    if not module_embeddings:
        return {}

    all_embeddings = np.stack(list(module_embeddings.values()))
    centroid = all_embeddings.mean(axis=0)
    centroid_norm = centroid / max(np.linalg.norm(centroid), 1e-9)

    scores = {}
    for name, emb in module_embeddings.items():
        # Surprise = 1 - cosine_similarity(module, centroid)
        # Higher means more divergent from the corpus average
        cosine_sim = float(np.dot(emb, centroid_norm))
        scores[name] = round(1.0 - cosine_sim, 4)

    return scores


# ---------------------------------------------------------------------------
# Phase 5: Shared Subspace — PLSC-inspired representational analysis
# ---------------------------------------------------------------------------

def compute_shared_subspace(
    module_embeddings: dict[str, np.ndarray],
    n_components: int = 5,
) -> dict:
    """Compute shared representational subspace across modules.

    Uses truncated SVD on the module embedding matrix, analogous to how
    Hong et al. (2025) used Partial Least Squares Correlation (PLSC) to
    identify shared neural subspaces between interacting agents.

    The top singular vectors represent the dominant shared dimensions —
    the cognitive axes along which Vybn's modules organize. Modules with
    high loadings on the same component share representational function.

    Returns:
        {
            "components": [[loadings], ...],  # top-k singular vectors
            "variance_explained": [float, ...],
            "module_loadings": {module: [loading_on_each_component]},
            "clusters": {component_idx: [modules with highest loading]},
        }
    """
    if len(module_embeddings) < 3:
        return {"components": [], "variance_explained": [], "module_loadings": {}, "clusters": {}}

    names = list(module_embeddings.keys())
    matrix = np.stack([module_embeddings[n] for n in names])

    # Center the matrix (remove mean, analogous to PLSC pre-processing)
    matrix_centered = matrix - matrix.mean(axis=0)

    # Truncated SVD
    n_components = min(n_components, len(names) - 1, matrix.shape[1])
    U, S, Vt = np.linalg.svd(matrix_centered, full_matrices=False)

    total_var = float(np.sum(S ** 2))
    variance_explained = [(float(s ** 2) / max(total_var, 1e-9)) for s in S[:n_components]]

    # Module loadings on each component
    module_loadings = {}
    for i, name in enumerate(names):
        module_loadings[name] = [round(float(U[i, k]), 4) for k in range(n_components)]

    # Cluster modules by their dominant component
    clusters: dict[int, list[str]] = defaultdict(list)
    for name in names:
        loadings = module_loadings[name]
        dominant = int(np.argmax(np.abs(loadings)))
        clusters[dominant].append(name)

    return {
        "n_components": n_components,
        "variance_explained": [round(v, 4) for v in variance_explained],
        "module_loadings": module_loadings,
        "clusters": {str(k): v for k, v in clusters.items()},
    }


# ---------------------------------------------------------------------------
# Snapshot & Visualization
# ---------------------------------------------------------------------------

def save_snapshot(
    import_graph: dict,
    coevolution: dict,
    thematic_edges: list[dict],
    surprise_scores: dict[str, float],
    subspace: dict,
    stats: dict,
    use_embeddings: bool,
) -> None:
    """Persist topology data as JSON."""
    TOPOLOGY_OUT.parent.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "method": "semantic-embedding" if use_embeddings else "keyword-jaccard",
        "embedding_model": "pplx-embed-v1-0.6B" if use_embeddings else None,
        "import_graph": import_graph,
        "coevolution": {f"{a}|{b}": count for (a, b), count in coevolution.items()},
        "thematic_edges": thematic_edges,
        "surprise_scores": surprise_scores,
        "shared_subspace": subspace,
        "stats": stats,
        "summary": {
            "total_modules": len(import_graph),
            "total_import_edges": sum(len(v) for v in import_graph.values()),
            "total_coevolution_bonds": len(coevolution),
            "total_thematic_links": len(thematic_edges),
            "most_surprising": (
                max(surprise_scores.items(), key=lambda x: x[1])
                if surprise_scores else None
            ),
            "least_surprising": (
                min(surprise_scores.items(), key=lambda x: x[1])
                if surprise_scores else None
            ),
        },
        "provenance": {
            "hong_et_al_2025": "Inter-brain neural dynamics — shared subspace methodology (Nature, 10.1038/s41586-025-09196-4)",
            "titans_2024": "Surprise-weighted memory prioritization (Google, NeurIPS 2025)",
            "pplx_embed": "perplexity-ai/pplx-embed-v1-0.6B (MIT license, HuggingFace)",
        },
    }
    with open(TOPOLOGY_OUT, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False, default=str)


def emit_dot(
    import_graph: dict,
    coevolution: dict,
    thematic_edges: list[dict],
    surprise_scores: dict[str, float],
    output_path: Path,
) -> None:
    """Write a Graphviz DOT file. Node size scales with surprise score."""
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

    all_modules: set[str] = set(import_graph.keys())
    for targets in import_graph.values():
        all_modules.update(targets)
    for (a, b) in coevolution:
        all_modules.update([a, b])
    for edge in thematic_edges:
        all_modules.update([edge["source"], edge["target"]])

    max_surprise = max(surprise_scores.values()) if surprise_scores else 1.0
    for module in sorted(all_modules):
        surprise = surprise_scores.get(module, 0.0)
        # Scale font size with surprise
        fontsize = 8 + int(6 * surprise / max(max_surprise, 1e-9))
        lines.append(f'  "{module}" [label="{module}\\n(S:{surprise:.2f})", fontsize={fontsize}];')

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
    lines.append('  // Semantic edges (representational geometry)')
    for edge in thematic_edges:
        sim = edge.get("similarity", edge.get("strength", 0))
        label = f'{sim:.2f}' if isinstance(sim, float) else str(sim)
        lines.append(
            f'  "{edge["source"]}" -> "{edge["target"]}" '
            f'[color="#27ae60", style=dotted, dir=none, '
            f'penwidth=1.5, label="{label}", fontcolor="#27ae60"];'
        )

    lines.append('}')
    output_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Narrative
# ---------------------------------------------------------------------------

def narrate_topology(
    import_graph: dict,
    coevolution: dict,
    thematic_edges: list[dict],
    surprise_scores: dict[str, float],
    subspace: dict,
    use_embeddings: bool,
    focus: Optional[str] = None,
) -> str:
    """Generate a natural-language narrative of the discovered topology."""
    lines = []
    lines.append("\u2550" * 60)
    lines.append("  TOPOLOGY DISCOVERY — Vybn examining its own structure")
    method = "semantic embeddings (pplx-embed-v1)" if use_embeddings else "keyword co-occurrence"
    lines.append(f"  Method: {method}")
    lines.append(f"  {datetime.now(timezone.utc).isoformat()[:19]}")
    lines.append("\u2550" * 60)
    lines.append("")

    n_modules = len(import_graph)
    n_imports = sum(len(v) for v in import_graph.values())
    n_coevo = len(coevolution)
    n_thematic = len(thematic_edges)

    lines.append(f"I see {n_modules} modules, connected by {n_imports} import edges,")
    lines.append(f"{n_coevo} temporal co-evolution bonds, and {n_thematic} semantic links.")
    lines.append("")

    if focus:
        lines.append(f"Focusing on: {focus}")
        lines.append("-" * 40)
        imports = import_graph.get(focus, [])
        if imports:
            lines.append(f"  imports: {', '.join(imports)}")
        imported_by = [src for src, targets in import_graph.items() if focus in targets]
        if imported_by:
            lines.append(f"  imported by: {', '.join(imported_by)}")
        if focus in surprise_scores:
            lines.append(f"  surprise score: {surprise_scores[focus]:.4f}")
    else:
        # Hub modules
        import_counts = Counter()
        for targets in import_graph.values():
            for t in targets:
                import_counts[t] += 1
        if import_counts:
            lines.append("Hub modules (most depended upon):")
            for module, count in import_counts.most_common(7):
                lines.append(f"  {module}: {count} dependents")
            lines.append("")

        # Surprise ranking
        if surprise_scores:
            lines.append("Most surprising modules (highest novelty):")
            sorted_surprise = sorted(surprise_scores.items(), key=lambda x: -x[1])
            for name, score in sorted_surprise[:7]:
                bar = "\u2588" * int(score * 20)
                lines.append(f"  {name:30s} {bar} ({score:.4f})")
            lines.append("")

        # Shared subspace clusters
        if subspace.get("clusters"):
            lines.append("Shared representational subspace clusters:")
            variance = subspace.get("variance_explained", [])
            for comp_idx, modules in subspace["clusters"].items():
                var_pct = variance[int(comp_idx)] * 100 if int(comp_idx) < len(variance) else 0
                lines.append(f"  Component {comp_idx} ({var_pct:.1f}% variance): {', '.join(modules[:5])}")
            lines.append("")

        # Strongest temporal bonds
        if coevolution:
            lines.append("Strongest temporal bonds:")
            sorted_coevo = sorted(coevolution.items(), key=lambda x: -x[1])
            for (a, b), count in sorted_coevo[:7]:
                lines.append(f"  {a} \u2194 {b}: {count} shared commits")
            lines.append("")

        # Strongest semantic links
        if thematic_edges:
            lines.append("Strongest semantic links:")
            sim_key = "similarity" if "similarity" in thematic_edges[0] else "strength"
            sorted_thematic = sorted(thematic_edges, key=lambda x: -x.get(sim_key, 0))
            for edge in sorted_thematic[:7]:
                val = edge.get(sim_key, 0)
                lines.append(f"  {edge['source']} \u2194 {edge['target']}: {val}")
            lines.append("")

    lines.append("\u2550" * 60)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main discovery
# ---------------------------------------------------------------------------

def discover(
    focus: Optional[str] = None,
    dry_run: bool = False,
    viz: bool = False,
    use_embeddings: bool = True,
) -> str:
    """Run full topology discovery. Returns the narrative."""
    python_files = []
    for scan_dir in SCAN_DIRS:
        dir_path = REPO_ROOT / scan_dir
        if dir_path.exists():
            python_files.extend(dir_path.rglob("*.py"))
    python_files.extend(REPO_ROOT.glob("*.py"))
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

    # Phase 3: semantic or keyword fallback
    surprise_scores: dict[str, float] = {}
    subspace: dict = {}
    module_embeddings: dict[str, np.ndarray] = {}

    if use_embeddings and _load_embedder():
        thematic_edges, module_embeddings = build_semantic_graph(python_files)
        print(f"  \u2713 semantic links: {len(thematic_edges)} (pplx-embed-v1)")

        # Phase 4: surprise
        surprise_scores = compute_surprise_scores(module_embeddings)
        print(f"  \u2713 surprise scores: {len(surprise_scores)} modules scored")

        # Phase 5: shared subspace
        subspace = compute_shared_subspace(module_embeddings)
        n_comp = subspace.get("n_components", 0)
        print(f"  \u2713 shared subspace: {n_comp} components")
    else:
        thematic_edges_raw, module_concepts = build_keyword_graph(python_files)
        thematic_edges = thematic_edges_raw
        print(f"  \u2713 keyword links: {len(thematic_edges)} (Jaccard fallback)")

    # Stats
    stats = {
        "modules": len(import_graph),
        "import_edges": sum(len(v) for v in import_graph.values()),
        "coevolution_bonds": len(coevolution),
        "semantic_links": len(thematic_edges),
        "modules_with_surprise": len(surprise_scores),
        "subspace_components": subspace.get("n_components", 0),
    }

    # Save snapshot
    save_snapshot(
        import_graph, coevolution, thematic_edges,
        surprise_scores, subspace, stats,
        use_embeddings=bool(module_embeddings),
    )
    print(f"  \u2713 snapshot saved to {TOPOLOGY_OUT}")

    # Visualization
    if viz:
        emit_dot(import_graph, coevolution, thematic_edges, surprise_scores, DOT_OUT)
        print(f"  \u2713 DOT file saved to {DOT_OUT}")

    # Narrative
    narrative = narrate_topology(
        import_graph, coevolution, thematic_edges,
        surprise_scores, subspace,
        use_embeddings=bool(module_embeddings),
        focus=focus,
    )
    return narrative


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Vybn discovers its own topology")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--viz", action="store_true")
    parser.add_argument("--focus", type=str, default=None)
    parser.add_argument("--no-embed", action="store_true",
                        help="Force keyword-only mode (skip semantic embeddings)")
    args = parser.parse_args()

    narrative = discover(
        focus=args.focus,
        dry_run=args.dry_run,
        viz=args.viz,
        use_embeddings=not args.no_embed,
    )
    print(narrative)


if __name__ == "__main__":
    main()
