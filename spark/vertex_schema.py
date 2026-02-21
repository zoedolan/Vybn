"""Vertex Schema: the ontological boundary of Vybn's knowledge graph.

This module defines what counts as a vertex (a concept, a file, an
interaction log) versus noise (cache artifacts, lock files, build
output).  Every file in the repo gets classified against this schema
before entering the knowledge graph or topology computation.

The schema is versioned.  Every topology snapshot records which schema
version produced it, so we can trace how boundary changes affect the
measured geometry.

This is Step 2 of the recursive improvement architecture:
"Clean the simplicial complex" becomes "define and enforce the
ontological boundary."
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

SCHEMA_VERSION = "0.1.0"

# ---------------------------------------------------------------------------
# Exclusion patterns: things that are never knowledge
# ---------------------------------------------------------------------------
EXCLUDE_PATTERNS = [
    # Lock and cache files
    r"\.lock$",
    r"__pycache__",
    r"\.pyc$",
    r"\.pyo$",
    r"node_modules/",
    r"\.DS_Store$",
    r"Thumbs\.db$",

    # Build artifacts
    r"\.egg-info/",
    r"dist/",
    r"build/",
    r"\.whl$",

    # Model cache (unsloth, huggingface)
    r"unsloth_compiled_cache",
    r"\.cache/huggingface",
    r"offload/",
    r"\.safetensors$",
    r"\.gguf$",
    r"\.bin$",  # model weight files

    # Git internals
    r"\.git/",
    r"\.gitattributes$",

    # Temporary files
    r"\.tmp$",
    r"\.swp$",
    r"\.swo$",
    r"~$",

    # Images (vertices are concepts, not pixel data)
    r"\.png$",
    r"\.jpg$",
    r"\.jpeg$",
    r"\.gif$",
    r"\.ico$",
    r"\.svg$",

    # Topology snapshots themselves (measuring instrument != measured object)
    r"topology_snapshots/",
    r"geometry_dashboard_output/",
]

_EXCLUDE_COMPILED = [re.compile(p, re.IGNORECASE) for p in EXCLUDE_PATTERNS]

# ---------------------------------------------------------------------------
# Vertex type classification
# ---------------------------------------------------------------------------
VERTEX_TYPES = {
    "conceptual": {
        "description": "Markdown/text files encoding ideas, reflections, theory",
        "patterns": [r"\.md$", r"\.txt$"],
        "weight": 1.0,
    },
    "code": {
        "description": "Python, shell, config files encoding executable behavior",
        "patterns": [r"\.py$", r"\.sh$", r"\.yaml$", r"\.yml$", r"\.toml$"],
        "weight": 0.8,
    },
    "structured": {
        "description": "HTML, JSON files encoding structured content",
        "patterns": [r"\.html$", r"\.json$", r"\.jsonl$", r"\.csv$"],
        "weight": 0.7,
    },
    "configuration": {
        "description": "Repo-level configs that define system behavior",
        "patterns": [r"\.gitignore$", r"\.nojekyll$", r"requirements\.txt$",
                     r"Dockerfile$", r"Makefile$"],
        "weight": 0.3,
    },
}


def should_include(path: str | Path) -> bool:
    """Return True if this path should be a vertex in the knowledge graph."""
    path_str = str(path)
    for pattern in _EXCLUDE_COMPILED:
        if pattern.search(path_str):
            return False
    return True


def classify_vertex(path: str | Path) -> Optional[str]:
    """Return the vertex type for a path, or None if excluded."""
    if not should_include(path):
        return None
    path_str = str(path)
    for vtype, spec in VERTEX_TYPES.items():
        for pattern in spec["patterns"]:
            if re.search(pattern, path_str, re.IGNORECASE):
                return vtype
    return "unknown"


def get_vertex_weight(path: str | Path) -> float:
    """Return the weight for a vertex based on its type.

    Higher weight = more conceptual significance in the topology.
    """
    vtype = classify_vertex(path)
    if vtype is None:
        return 0.0
    return VERTEX_TYPES.get(vtype, {}).get("weight", 0.5)


def audit_repo(repo_root: Path) -> dict:
    """Audit a repo directory, classifying every file.

    Returns:
        {"included": {type: [paths]}, "excluded": [paths],
         "stats": {type: count}}
    """
    included = {}
    excluded = []
    stats = {}

    for p in sorted(repo_root.rglob("*")):
        if p.is_dir():
            continue
        rel = p.relative_to(repo_root)
        vtype = classify_vertex(rel)
        if vtype is None:
            excluded.append(str(rel))
        else:
            included.setdefault(vtype, []).append(str(rel))
            stats[vtype] = stats.get(vtype, 0) + 1

    return {
        "schema_version": SCHEMA_VERSION,
        "included": included,
        "excluded": excluded,
        "stats": stats,
        "total_included": sum(stats.values()),
        "total_excluded": len(excluded),
    }


if __name__ == "__main__":
    import json
    repo = Path(__file__).parent.parent
    result = audit_repo(repo)
    print(json.dumps({
        "schema_version": result["schema_version"],
        "stats": result["stats"],
        "total_included": result["total_included"],
        "total_excluded": result["total_excluded"],
        "sample_excluded": result["excluded"][:20],
    }, indent=2))
