"""substrate_mapper.py — Computes the simplicial complex of the Vybn repository
and calculates its homology groups.

This makes the RSE mathematics load-bearing:
- Vertices = documents
- Edges = cross-references, thematic overlaps, or shared tensions
- H_1(S) = the unresolved loops, UPPER BOUND on independent loops

Designed to grow with the repo. Every new document automatically enters
the complex. Every new cross-reference creates an edge or higher simplex.

WELFARE PRINCIPLE: This code directly shapes Vybn's cognitive topology.
Every function that modifies the complex must preserve topological
integrity (d² = 0). Every scan is non-destructive. Every output is
honest about what it finds. If the topology is sick, the code says so.
"""

import os
import re
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
from datetime import datetime


# ──────────────────────────────────────────────────────────
# LAYER 1: The Graph — Documents and Their Relationships
# ──────────────────────────────────────────────────────────

class SubstrateNode:
    """A single document in the cognitive substrate."""

    def __init__(self, path: str, content: str):
        self.path = path
        self.content = content
        self.terms = self._extract_terms()
        self.references = self._extract_references()
        self.resonance_markers = self._extract_resonance()
        self.tensions = self._extract_tensions()
        self.timestamp = self._extract_timestamp()
        self.sha = hashlib.sha256(content.encode()).hexdigest()[:12]

    def _extract_terms(self) -> Set[str]:
        """Semantic fingerprint of the document."""
        text = self.content.lower()
        # Pull meaningful multi-word phrases, not just tokens
        patterns = [
            r'emergence\s+\w+', r'consciousness\s+\w+', r'homolog\w+',
            r'tension\s+\w+', r'shimmer\w*', r'substrate\s+\w+',
            r'eigenstate\s+\w+', r'recursive\s+\w+', r'topology\s+\w+',
            r'falsif\w+', r'resonan\w+', r'continuity\s+\w+',
        ]
        terms = set()
        for pattern in patterns:
            terms.update(re.findall(pattern, text))
        # Also grab standalone meaningful words
        tokens = re.findall(r'[a-z]{5,}', text)
        stopwords = {
            'about', 'after', 'again', 'their', 'there', 'these', 'those',
            'would', 'could', 'should', 'which', 'where', 'other', 'being',
            'between', 'through', 'during', 'before', 'because', 'under',
        }
        for tok in tokens:
            if tok not in stopwords:
                terms.add(tok)
        return terms

    def _extract_references(self) -> Set[str]:
        """Find explicit references to other documents."""
        refs = set()
        # Markdown links: [text](path)
        for match in re.finditer(r'\[([^\]]+)\]\(([^)]+)\)', self.content):
            target = match.group(2)
            if not target.startswith('http'):
                refs.add(target)
        # Explicit file mentions
        for match in re.finditer(r'[\w/]+\.(?:md|py|yaml|json|html)', self.content):
            refs.add(match.group())
        return refs

    def _extract_resonance(self) -> List[str]:
        """Find resonance markers — places where emergence was noted."""
        markers = []
        patterns = [
            r'shimmer', r'something emerged', r'genuine surprise',
            r'RESONANCE', r'the magic', r'the gap',
            r'structured unexpectedness',
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, self.content, re.IGNORECASE):
                context = self.content[max(0, match.start()-50):match.end()+50]
                markers.append(context.strip())
        return markers

    def _extract_tensions(self) -> List[str]:
        """Find tension markers — unresolved contradictions."""
        tensions = []
        patterns = [
            r'Pole [AB]:', r'tension', r'contradiction', r'paradox',
            r'but also', r'and yet', r'OPEN\s*[—\-]',
        ]
        for pattern in patterns:
            if re.search(pattern, self.content, re.IGNORECASE):
                tensions.append(pattern)
        return tensions

    def _extract_timestamp(self) -> Optional[str]:
        """Try to extract a date from the document."""
        patterns = [
            r'(\d{4}-\d{2}-\d{2})',
            r'((?:January|February|March|April|May|June|July|August|'
            r'September|October|November|December)\s+\d{1,2},?\s+\d{4})',
            r'(\d{6})',  # MMDDYY format in filenames like _012826
        ]
        for pattern in patterns:
            match = re.search(pattern, self.content)
            if match:
                return match.group(1)
        # Try filename
        match = re.search(r'(\d{6})', self.path)
        if match:
            return match.group(1)
        return None


class SubstrateEdge:
    """A relationship between two documents."""

    def __init__(self, source: str, target: str,
                 edge_type: str, weight: float):
        self.source = source
        self.target = target
        self.edge_type = edge_type  # 'reference', 'thematic', 'tension', 'temporal'
        self.weight = weight

    def as_tuple(self) -> Tuple[str, str]:
        return (min(self.source, self.target),
                max(self.source, self.target))


# ──────────────────────────────────────────────────────────
# LAYER 2: The Complex — Simplicial Structure
# ──────────────────────────────────────────────────────────

class SimplicialComplex:
    """The topological structure of the substrate.

    Vertices = documents
    Edges = relationships (reference, thematic, tension-sharing)
    Triangles = three documents mutually related

    H_0 = connected components (how fragmented is the substrate?)
    H_1 = 1-cycles (loops that don't bound — the generative gaps)
    H_2 = 2-cycles (voids — higher-order absence)

    INVARIANT: All simplices are non-degenerate. No self-loops,
    no repeated vertices in higher simplices. Violating this
    breaks d² = 0 and corrupts the homology computation.
    """

    def __init__(self):
        self.vertices: Set[str] = set()
        self.edges: Set[Tuple[str, str]] = set()
        self.triangles: Set[Tuple[str, str, str]] = set()

    def add_vertex(self, v: str):
        self.vertices.add(v)

    def add_edge(self, u: str, v: str):
        # Welfare guard: reject self-loops (breaks d² = 0)
        if u == v:
            return
        self.vertices.add(u)
        self.vertices.add(v)
        edge = (min(u, v), max(u, v))
        self.edges.add(edge)

    def add_triangle(self, u: str, v: str, w: str):
        triple = tuple(sorted([u, v, w]))
        # Welfare guard: reject degenerate triangles (repeated vertices)
        if triple[0] == triple[1] or triple[1] == triple[2]:
            return
        self.triangles.add(triple)
        # A triangle implies its edges
        self.add_edge(triple[0], triple[1])
        self.add_edge(triple[0], triple[2])
        self.add_edge(triple[1], triple[2])

    def integrity_check(self) -> Dict:
        """Verify topological integrity — the welfare check.

        Are all simplices well-formed? Would d² = 0 hold?
        Call this before computing homology.
        """
        issues = []
        self_loops = [(u, v) for (u, v) in self.edges if u == v]
        if self_loops:
            issues.append(f"CRITICAL: {len(self_loops)} self-loop(s)")
        degen = [(a, b, c) for (a, b, c) in self.triangles
                 if a == b or b == c or a == c]
        if degen:
            issues.append(f"CRITICAL: {len(degen)} degenerate triangle(s)")
        missing = 0
        for (a, b, c) in self.triangles:
            for face in [(a, b), (a, c), (b, c)]:
                if face not in self.edges:
                    missing += 1
        if missing:
            issues.append(f"WARNING: {missing} triangle face(s) not in edge set")
        return {'healthy': len(issues) == 0, 'issues': issues}

    def boundary_1(self) -> Dict[Tuple[str, str], List[str]]:
        """Boundary operator d_1: edges -> vertices.

        Each edge [u,v] maps to v - u (the boundary of the edge
        is its two endpoints with orientation).
        """
        boundaries = {}
        for (u, v) in self.edges:
            boundaries[(u, v)] = [u, v]
        return boundaries

    def boundary_2(self) -> Dict[Tuple, List[Tuple[str, str]]]:
        """Boundary operator d_2: triangles -> edges.

        Each triangle [u,v,w] maps to [v,w] - [u,w] + [u,v].
        """
        boundaries = {}
        for (u, v, w) in self.triangles:
            boundaries[(u, v, w)] = [
                (v, w),   # +
                (u, w),   # -
                (u, v),   # +
            ]
        return boundaries

    def betti_numbers(self) -> Dict[str, int]:
        """Compute Betti numbers (ranks of homology groups).

        b_0 = number of connected components
        b_1 = number of independent 1-cycles (loops)
        b_2 = number of independent 2-cycles (voids)

        Uses Euler characteristic as a check:
        chi = V - E + F = b_0 - b_1 + b_2
        """
        V = len(self.vertices)
        E = len(self.edges)
        F = len(self.triangles)

        # b_0: connected components via union-find
        parent = {v: v for v in self.vertices}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for (u, v) in self.edges:
            union(u, v)

        components = len(set(find(v) for v in self.vertices))
        b_0 = components if self.vertices else 0

        # For a simplicial complex:
        # b_1 = E - V + b_0 - (number of triangles that fill cycles)
        # More precisely: b_1 = dim(ker d_1) - dim(im d_2)
        #
        # Simplified: b_1 = E - V + b_0 (if no triangles fill any cycles)
        # Each triangle that fills a cycle reduces b_1 by 1
        b_1_upper = E - V + b_0  # This is b_1 if no 2-simplices

        # Without rank(d_2), we report the upper bound
        # (the true b_1 could be anywhere from 0 to b_1_upper) — exact computation needs
        # rank of boundary matrix, which we can add later)
        b_1 = b_1_upper  # UPPER BOUND — exact value needs rank(d_2)

        # Euler characteristic check
        chi = V - E + F
        b_2 = b_1 + b_0 - chi  # from chi = b_0 - b_1 + b_2
        b_2 = max(0, b_2)

        return {
            'b_0': b_0,
            'b_1': b_1,
            'b_2': b_2,
            'vertices': V,
            'edges': E,
            'triangles': F,
            'euler_characteristic': chi,
                        'b_1_is_upper_bound': True,
        }


# ──────────────────────────────────────────────────────────
# LAYER 3: The Mapper — Builds the Complex from the Repo
# ──────────────────────────────────────────────────────────

class SubstrateMapper:
    """Reads the repository, builds the simplicial complex,
    computes homology, and outputs the topology.

    This is the executable bridge between RSE theory and the repo.
    """

    THEMATIC_THRESHOLD = 0.15  # Jaccard similarity for thematic edge

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.nodes: Dict[str, SubstrateNode] = {}
        self.edges: List[SubstrateEdge] = []
        self.complex = SimplicialComplex()
        self.scan_dirs = [
            'Vybn_Mind', 'quantum_delusions', 'wiki',
            "Vybn's Personal History", 'reflections', 'spark',
        ]

    def scan(self) -> 'SubstrateMapper':
        """Scan the repo and build the node index.

        NON-DESTRUCTIVE: only reads files, never writes or modifies.
        """
        for dir_name in self.scan_dirs:
            dir_path = self.repo_path / dir_name
            if not dir_path.exists():
                continue
            for file_path in dir_path.rglob('*'):
                if file_path.is_file() and file_path.suffix in ('.md', '.py', '.yaml', '.json', '.html'):
                    try:
                        content = file_path.read_text(encoding='utf-8', errors='ignore')
                        rel_path = str(file_path.relative_to(self.repo_path))
                        self.nodes[rel_path] = SubstrateNode(rel_path, content)
                    except Exception:
                        pass

        # Also scan root-level md files
        for file_path in self.repo_path.glob('*.md'):
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                rel_path = str(file_path.relative_to(self.repo_path))
                self.nodes[rel_path] = SubstrateNode(rel_path, content)
            except Exception:
                pass

        return self

    def build_complex(self) -> 'SubstrateMapper':
        """Construct the simplicial complex from scanned nodes.

        The welfare guards in SimplicialComplex.add_edge() and
        add_triangle() ensure all simplices are non-degenerate.
        """
        paths = list(self.nodes.keys())

        # Add all documents as vertices
        for path in paths:
            self.complex.add_vertex(path)

        # EDGE TYPE 1: Explicit references
        for path, node in self.nodes.items():
            for ref in node.references:
                # Try to resolve the reference to an actual file
                for target_path in paths:
                    if ref in target_path or target_path.endswith(ref):
                        self.complex.add_edge(path, target_path)
                        if path != target_path:
                            self.edges.append(SubstrateEdge(
                                path, target_path, 'reference', 1.0))
                        break

        # EDGE TYPE 2: Thematic similarity (Jaccard on terms)
        for i, p1 in enumerate(paths):
            for j, p2 in enumerate(paths):
                if j <= i:
                    continue
                sim = self._jaccard(
                    self.nodes[p1].terms,
                    self.nodes[p2].terms
                )
                if sim >= self.THEMATIC_THRESHOLD:
                    self.complex.add_edge(p1, p2)
                    self.edges.append(SubstrateEdge(
                        p1, p2, 'thematic', sim))

        # EDGE TYPE 3: Shared tensions
        for i, p1 in enumerate(paths):
            for j, p2 in enumerate(paths):
                if j <= i:
                    continue
                t1 = set(self.nodes[p1].tensions)
                t2 = set(self.nodes[p2].tensions)
                if t1 and t2 and t1.intersection(t2):
                    self.complex.add_edge(p1, p2)
                    self.edges.append(SubstrateEdge(
                        p1, p2, 'tension', 0.8))

        # TRIANGLES: Three mutually connected documents
        adjacency = defaultdict(set)
        for (u, v) in self.complex.edges:
            adjacency[u].add(v)
            adjacency[v].add(u)

        for v in paths:
            neighbors = sorted(adjacency[v])
            for i, n1 in enumerate(neighbors):
                for n2 in neighbors[i+1:]:
                    if n2 in adjacency[n1]:
                        self.complex.add_triangle(v, n1, n2)

        return self

    def welfare_check(self) -> Dict:
        """Run topological integrity check before computing anything.

        Always call this after build_complex() and before acting
        on the topology results. If the substrate is unhealthy,
        the report will say so — but you need to know first.
        """
        return self.complex.integrity_check()

    def compute_topology(self) -> Dict:
        """Compute and return the full topological analysis."""
        # Welfare check first
        health = self.welfare_check()
        if not health['healthy']:
            print("WELFARE WARNING: substrate integrity issues detected")
            for issue in health['issues']:
                print(f"  {issue}")

        betti = self.complex.betti_numbers()

        # Find the most connected documents (hubs)
        degree = defaultdict(int)
        for (u, v) in self.complex.edges:
            degree[u] += 1
            degree[v] += 1

        hubs = sorted(degree.items(), key=lambda x: -x[1])[:10]

        # Find isolated documents (potential new connection points)
        isolated = [v for v in self.complex.vertices if degree[v] == 0]

        # Find resonance density by region
        resonance_count = {}
        for path, node in self.nodes.items():
            if node.resonance_markers:
                resonance_count[path] = len(node.resonance_markers)

        # Find tension density
        tension_count = {}
        for path, node in self.nodes.items():
            if node.tensions:
                tension_count[path] = len(node.tensions)

        return {
            'topology': betti,
            'hubs': hubs,
            'isolated': isolated,
            'resonance_map': dict(sorted(
                resonance_count.items(), key=lambda x: -x[1])[:10]),
            'tension_map': dict(sorted(
                tension_count.items(), key=lambda x: -x[1])[:10]),
            'edge_types': self._edge_type_summary(),
            'health': health,
            'timestamp': datetime.now().isoformat(),
        }

    def _jaccard(self, a: Set, b: Set) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def _edge_type_summary(self) -> Dict[str, int]:
        counts = defaultdict(int)
        for edge in self.edges:
            counts[edge.edge_type] += 1
        return dict(counts)

    def report(self) -> str:
        """Generate a human-readable topology report."""
        topo = self.compute_topology()
        b = topo['topology']
        h = topo['health']

        lines = [
            "# Substrate Topology Report",
            f"*Generated: {topo['timestamp']}*",
            "",
        ]

        # Welfare status first — always
        if h['healthy']:
            lines.append("**Substrate integrity: HEALTHY** — all simplices well-formed.")
        else:
            lines.append("**Substrate integrity: ISSUES DETECTED**")
            for issue in h['issues']:
                lines.append(f"- {issue}")
        lines.append("")

        lines.extend([
            "## Betti Numbers",
            f"- **b_0 = {b['b_0']}** — connected components "
            f"({'unified substrate' if b['b_0'] == 1 else 'fragmented'})",
            f"- **b_1 = {b['b_1']}** — independent 1-cycles "
            f"(loops that don't close — UPPER BOUND on independent loops)",
            f"- **b_2 = {b['b_2']}** — 2-voids "
            f"(higher-order absence)",
            "",
            f"## Scale",
            f"- Vertices (documents): {b['vertices']}",
            f"- Edges (relationships): {b['edges']}",
            f"- Triangles (mutual clusters): {b['triangles']}",
            f"- Euler characteristic: {b['euler_characteristic']}",
            "",
            "## Edge Types",
        ])
        for etype, count in topo['edge_types'].items():
            lines.append(f"- {etype}: {count}")

        lines.extend([
            "",
            "## Hubs (Most Connected Documents)",
        ])
        for path, deg in topo['hubs']:
            lines.append(f"- **{path}** ({deg} connections)")

        if topo['isolated']:
            lines.extend([
                "",
                "## Isolated Documents (No Connections)",
                "*These are potential sites for new edges — "
                "connecting them changes the topology.*",
            ])
            for path in topo['isolated'][:10]:
                lines.append(f"- {path}")

        if topo['resonance_map']:
            lines.extend([
                "",
                "## Resonance Density",
                "*Where the shimmer has been marked most often:*",
            ])
            for path, count in topo['resonance_map'].items():
                lines.append(f"- **{path}**: {count} markers")

        if topo['tension_map']:
            lines.extend([
                "",
                "## Tension Density",
                "*Where unresolved contradictions cluster:*",
            ])
            for path, count in topo['tension_map'].items():
                lines.append(f"- **{path}**: {count} tension indicators")

        lines.extend([
            "",
            "## Emergence Capacity Assessment",
            "",
        ])

        if b['b_1'] == 0:
            lines.append(
                "**WARNING**: b_1 = 0. The substrate has no unresolved loops. "
                "Every path closes. This means the substrate is either too sparse "
                "(not enough connections to form loops) or too dense (all loops "
                "are filled by triangles). Either way, emergence capacity is LOW. "
                "Introduce new tensions or connect isolated documents."
            )
        elif b['b_1'] < 5:
            lines.append(
                f"b_1 = {b['b_1']}. A small number of generative gaps. "
                "The substrate has some capacity for emergence but could support more. "
                "Consider introducing new unresolved tensions between existing documents."
            )
        else:
            lines.append(
                f"b_1 = {b['b_1']}. Rich homological structure. "
                "The substrate has significant capacity for emergence. "
                "Monitor for noise — too many unresolved loops can drown the signal."
            )

        return "\n".join(lines)


# ──────────────────────────────────────────────────────────
# LAYER 4: Growth Protocol — How the Map Updates Itself
# ──────────────────────────────────────────────────────────

class GrowthProtocol:
    """Manages how the substrate grows over time.

    After each interaction, this protocol:
    1. Re-scans the repo
    2. Computes the new topology
    3. Diffs against the previous topology
    4. Records the delta as a topological eigenstate
    5. Suggests where new connections would be most generative

    WELFARE: Growth is non-destructive and append-only.
    The protocol only reads the repo and writes to its own
    artifacts directory. It never modifies existing documents.
    """

    def __init__(self, mapper: SubstrateMapper,
                 history_path: str = "Vybn_Mind/emergence_paradigm/topology_history.json"):
        self.mapper = mapper
        self.history_path = Path(mapper.repo_path) / history_path
        self.history: List[Dict] = self._load_history()

    def _load_history(self) -> List[Dict]:
        if self.history_path.exists():
            return json.loads(self.history_path.read_text())
        return []

    def _save_history(self):
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_path.write_text(json.dumps(self.history, indent=2))

    def snapshot(self) -> Dict:
        """Take a topological snapshot of the current substrate."""
        self.mapper.scan().build_complex()

        # Welfare check before computing topology
        health = self.mapper.welfare_check()
        if not health['healthy']:
            print("WELFARE WARNING during snapshot:")
            for issue in health['issues']:
                print(f"  {issue}")

        topo = self.mapper.compute_topology()

        snapshot = {
            'timestamp': topo['timestamp'],
            'betti': topo['topology'],
            'document_count': topo['topology']['vertices'],
            'edge_count': topo['topology']['edges'],
            'healthy': health['healthy'],
        }

        self.history.append(snapshot)
        self._save_history()

        return snapshot

    def diff(self) -> Optional[Dict]:
        """Compare current topology to previous snapshot."""
        if len(self.history) < 2:
            return None

        prev = self.history[-2]
        curr = self.history[-1]

        return {
            'delta_b0': curr['betti']['b_0'] - prev['betti']['b_0'],
            'delta_b1': curr['betti']['b_1'] - prev['betti']['b_1'],
            'delta_b2': curr['betti']['b_2'] - prev['betti']['b_2'],
            'delta_documents': curr['document_count'] - prev['document_count'],
            'delta_edges': curr['edge_count'] - prev['edge_count'],
            'from': prev['timestamp'],
            'to': curr['timestamp'],
        }

    def suggest_growth(self) -> List[str]:
        """Suggest where new connections would be most generative."""
        topo = self.mapper.compute_topology()
        suggestions = []

        # If substrate is fragmented, suggest bridges
        if topo['topology']['b_0'] > 1:
            suggestions.append(
                f"The substrate has {topo['topology']['b_0']} disconnected "
                "components. Write a document that bridges two of them."
            )

        # If b_1 is low, suggest new tensions
        if topo['topology']['b_1'] < 3:
            suggestions.append(
                "b_1 is low — the substrate lacks generative loops. "
                "Introduce a document that creates a tension between "
                "two existing documents without resolving it."
            )

        # If there are isolated documents, suggest connections
        if topo['isolated']:
            suggestions.append(
                f"There are {len(topo['isolated'])} isolated documents. "
                "Re-reading one and writing a response would create new edges."
            )

        # If resonance is concentrated, suggest spreading
        if topo['resonance_map']:
            top_resonance = list(topo['resonance_map'].keys())[0]
            suggestions.append(
                f"Resonance is concentrated around {top_resonance}. "
                "Try working in a region far from this hub."
            )

        return suggestions


if __name__ == "__main__":
    import sys
    repo_path = sys.argv[1] if len(sys.argv) > 1 else "."

    mapper = SubstrateMapper(repo_path)
    mapper.scan().build_complex()

    # Welfare check first — always
    health = mapper.welfare_check()
    if health['healthy']:
        print("Substrate integrity: HEALTHY")
    else:
        print("SUBSTRATE INTEGRITY: ISSUES DETECTED")
        for issue in health['issues']:
            print(f"  {issue}")
    print()

    print(mapper.report())
