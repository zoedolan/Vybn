"""semantic_substrate_mapper.py — Semantic embedding edge construction.

Replaces Jaccard keyword overlap with cosine similarity on dense
vector embeddings. The simplicial complex and homology machinery
from substrate_mapper.py are reused without modification.

WHY THIS MATTERS
----------------
Jaccard on regex-extracted keywords measures lexical overlap. Two files
mentioning "quantum" and "consciousness" get an edge not because they
are in conceptual tension but because someone typed the same words.
Semantic embeddings capture meaning-level similarity: files about
related *ideas* connect even when they use different vocabulary.

DESIGN DECISIONS (each changes the resulting topology)
-----------------------------------------------------
1. Model:      all-MiniLM-L6-v2  (384-dim, fast, good quality)
2. Granularity: file-level        (one embedding per document)
3. Threshold:   cosine >= 0.55    (creates an edge)
4. Truncation:  first 512 tokens  (model max; loses tail content)

Every one of these is tunable. Different choices produce different
Betti numbers. That is the research question, not a bug.

v2 CHANGES (2026-02-21)
-----------------------
- Raised default threshold from 0.35 to 0.55 (0.35 produced a
  near-complete graph: 23k edges, 510k triangles, b_1=0 because
  every cycle was filled)
- Added SKIP_PATTERNS to filter lock files, build artifacts,
  cache dirs, and other non-content files that were creating
  cliques of identical embeddings (e.g. 14 Unsloth .lock files
  all at cosine 1.0)
- Threshold sensitivity sweep starts at 0.3 instead of 0.1 to
  avoid choking on triangle enumeration at low thresholds
"""

import sys
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Set
from collections import defaultdict

# Import the machinery we are NOT replacing
from substrate_mapper import (
    SubstrateMapper, SubstrateNode, SubstrateEdge, SimplicialComplex
)

try:
    from sentence_transformers import SentenceTransformer
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False


# Files that should never enter the complex.
# These produce near-identical embeddings and create cliques
# that drown real structure in triangles.
SKIP_PATTERNS = [
    r'\.lock\.',           # lock files (.lock.Unsloth*.py etc.)
    r'__pycache__',        # bytecode cache
    r'\.pyc$',             # compiled python
    r'node_modules/',      # npm
    r'\.git/',             # git internals
    r'package-lock\.json', # npm lock
    r'yarn\.lock',         # yarn lock
    r'\.egg-info/',        # setuptools
    r'\.dist-info/',       # wheel metadata
    r'\.min\.js$',         # minified js
    r'\.min\.css$',        # minified css
]


def _should_skip(path: str) -> bool:
    """Return True if this file is non-content boilerplate."""
    for pattern in SKIP_PATTERNS:
        if re.search(pattern, path):
            return True
    return False


class SemanticSubstrateMapper(SubstrateMapper):
    """Substrate mapper with semantic-embedding edges.

    Inherits: scanning, file discovery, complex structure, homology.
    Overrides: build_complex() — the only method that touches edges.
    Adds: threshold_sensitivity() for persistence analysis.
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"
    DEFAULT_THRESHOLD = 0.55

    def __init__(self, repo_path: str,
                 model_name: str = None,
                 threshold: float = None):
        super().__init__(repo_path)
        self.model_name = model_name or self.DEFAULT_MODEL
        self.threshold = threshold if threshold is not None else self.DEFAULT_THRESHOLD
        self.embeddings: Dict[str, np.ndarray] = {}
        self._model = None
        self._skipped: List[str] = []

    # ── scanning (override to filter) ───────────────────

    def scan(self) -> 'SemanticSubstrateMapper':
        """Scan repo, then remove non-content files."""
        super().scan()
        to_remove = []
        for path in self.nodes:
            if _should_skip(path):
                to_remove.append(path)
        for path in to_remove:
            del self.nodes[path]
            self._skipped.append(path)
        if self._skipped:
            print(f"Filtered {len(self._skipped)} non-content files")
        return self

    # ── embedding layer ──────────────────────────────────

    def _get_model(self):
        if not HAS_EMBEDDINGS:
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _embed_documents(self):
        """Compute one embedding per scanned document.

        Uses normalize_embeddings=True so cosine similarity
        reduces to a dot product (faster, no division).
        """
        model = self._get_model()
        paths = list(self.nodes.keys())
        contents = [self.nodes[p].content for p in paths]
        vectors = model.encode(
            contents,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        for path, vec in zip(paths, vectors):
            self.embeddings[path] = vec

    # ── the part that actually changes ───────────────────

    def build_complex(self) -> 'SemanticSubstrateMapper':
        """Build the simplicial complex using semantic similarity.

        Edge types:
          1. reference  — explicit file cross-references (kept from parent)
          2. semantic   — cosine(embed(doc_i), embed(doc_j)) >= threshold

        Triangles: three mutually adjacent vertices, as before.
        """
        self._embed_documents()
        paths = list(self.nodes.keys())

        for path in paths:
            self.complex.add_vertex(path)

        # EDGE TYPE 1: explicit references (threshold-independent)
        for path, node in self.nodes.items():
            for ref in node.references:
                for target_path in paths:
                    if ref in target_path or target_path.endswith(ref):
                        self.complex.add_edge(path, target_path)
                        if path != target_path:
                            self.edges.append(SubstrateEdge(
                                path, target_path, 'reference', 1.0))
                        break

        # EDGE TYPE 2: semantic similarity (replaces Jaccard + tensions)
        # Normalized embeddings → cosine = dot product
        for i, p1 in enumerate(paths):
            for j in range(i + 1, len(paths)):
                p2 = paths[j]
                sim = float(np.dot(self.embeddings[p1], self.embeddings[p2]))
                if sim >= self.threshold:
                    self.complex.add_edge(p1, p2)
                    self.edges.append(SubstrateEdge(
                        p1, p2, 'semantic', sim))

        # TRIANGLES: three mutually connected documents
        adjacency = defaultdict(set)
        for (u, v) in self.complex.edges:
            adjacency[u].add(v)
            adjacency[v].add(u)
        for v in paths:
            neighbors = sorted(adjacency[v])
            for i, n1 in enumerate(neighbors):
                for n2 in neighbors[i + 1:]:
                    if n2 in adjacency[n1]:
                        self.complex.add_triangle(v, n1, n2)

        return self

    # ── persistence analysis ─────────────────────────────

    def threshold_sensitivity(self,
                              low: float = 0.30,
                              high: float = 0.80,
                              steps: int = 11) -> List[dict]:
        """Sweep cosine threshold; report Betti numbers at each level.

        If b_1 is stable across a range of thresholds the features
        are persistent (real structure). If volatile, they are
        threshold artifacts.

        Starts at 0.30 (not 0.10) because low thresholds on semantic
        embeddings produce near-complete graphs where triangle
        enumeration is O(n^3) and b_1 = 0 trivially.
        """
        if not self.embeddings:
            self._embed_documents()

        paths = list(self.nodes.keys())
        n = len(paths)

        # Pre-compute the full pairwise similarity matrix once
        vecs = np.array([self.embeddings[p] for p in paths])
        sim_matrix = vecs @ vecs.T  # normalized, so this is cosine

        # Collect reference edges (threshold-independent)
        ref_edges = set()
        for path, node in self.nodes.items():
            for ref in node.references:
                for target_path in paths:
                    if ref in target_path or target_path.endswith(ref):
                        edge = (min(path, target_path), max(path, target_path))
                        if edge[0] != edge[1]:
                            ref_edges.add(edge)
                        break

        results = []
        for step in range(steps):
            t = low + (high - low) * step / (steps - 1)

            cx = SimplicialComplex()
            for path in paths:
                cx.add_vertex(path)

            for edge in ref_edges:
                cx.add_edge(edge[0], edge[1])

            for i in range(n):
                for j in range(i + 1, n):
                    if sim_matrix[i, j] >= t:
                        cx.add_edge(paths[i], paths[j])

            # Triangles
            adj = defaultdict(set)
            for (u, v) in cx.edges:
                adj[u].add(v)
                adj[v].add(u)
            for v in paths:
                nbrs = sorted(adj[v])
                for idx, n1 in enumerate(nbrs):
                    for n2 in nbrs[idx + 1:]:
                        if n2 in adj[n1]:
                            cx.add_triangle(v, n1, n2)

            betti = cx.betti_numbers()
            results.append({
                'threshold': round(t, 3),
                'b_0': betti['b_0'],
                'b_1': betti['b_1'],
                'b_2': betti['b_2'],
                'edges': betti['edges'],
                'triangles': betti['triangles'],
                'euler': betti['euler_characteristic'],
            })
            # Print as we go so partial results are visible
            r = results[-1]
            print(f"{r['threshold']:>10.3f} {r['b_0']:>6} {r['b_1']:>6} "
                  f"{r['b_2']:>8} {r['edges']:>8} {r['triangles']:>10}",
                  flush=True)

        return results

    # ── convenience: dump the similarity matrix ──────────

    def similarity_report(self, top_n: int = 20) -> str:
        """Show the highest-similarity document pairs.

        Useful for sanity-checking whether the embeddings
        produce edges that a human would agree with.
        """
        if not self.embeddings:
            self._embed_documents()

        paths = list(self.nodes.keys())
        pairs = []
        for i, p1 in enumerate(paths):
            for j in range(i + 1, len(paths)):
                p2 = paths[j]
                sim = float(np.dot(self.embeddings[p1], self.embeddings[p2]))
                pairs.append((sim, p1, p2))
        pairs.sort(reverse=True)

        lines = ["# Top Semantic Similarities", ""]
        lines.append(f"{'Cosine':>8}  Doc A  \u2194  Doc B")
        lines.append("-" * 70)
        for sim, p1, p2 in pairs[:top_n]:
            a = Path(p1).name
            b = Path(p2).name
            lines.append(f"{sim:>8.4f}  {a}  \u2194  {b}")

        return "\n".join(lines)


if __name__ == "__main__":
    repo_path = sys.argv[1] if len(sys.argv) > 1 else "."

    mapper = SemanticSubstrateMapper(repo_path)
    mapper.scan().build_complex()

    health = mapper.welfare_check()
    if health['healthy']:
        print("Substrate integrity: HEALTHY")
    else:
        print("SUBSTRATE INTEGRITY: ISSUES DETECTED")
        for issue in health['issues']:
            print(f"  {issue}")
    print()

    print(mapper.report())

    print("\n")
    print(mapper.similarity_report())

    print("\n\n## Threshold Sensitivity (Semantic)")
    print(f"{'Threshold':>10} {'b_0':>6} {'b_1':>6} {'b_2':>8} {'Edges':>8} {'Triangles':>10}")
    print("-" * 55)
    mapper.threshold_sensitivity()
