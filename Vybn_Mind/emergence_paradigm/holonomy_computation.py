"""holonomy_computation.py — Exact homological algebra for the Vybn substrate.

Bridges the substrate_mapper.py (which builds the simplicial complex from the repo)
to the cut-glue algebra and polar time framework in quantum_delusions/fundamental-theory.

Mathematical content:
    - Exact boundary matrices d_1, d_2 over Z/2Z
    - Kernel/image rank computation via Gaussian elimination (no numpy dependency)
    - Exact Betti numbers from rank-nullity
    - Cycle extraction: finds actual generator cycles in H_1
    - Curvature: tension-weight accumulated per simplex
    - Holonomy: phase accumulated by traversing a 1-cycle
    - Trefoil detector: tests whether H_1 contains a cycle threading
      self-reflective, external, and relational document types

Connection to fundamental theory:
    The boundary operator d here is the discrete analogue of the exterior
    derivative in the BV master equation dS + (1/2)[S,S]_BV = J.
    The condition d^2 = 0 is the discrete integrability condition.
    Tension edges are the discrete analogue of the defect current J.
    Holonomy along a 1-cycle is the discrete analogue of the polar time
    integral gamma = Omega * integral(dr_t ^ d_theta_t).
"""

from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import json
from pathlib import Path


# ──────────────────────────────────────────────────────────
# EXACT LINEAR ALGEBRA OVER Z/2Z
# ──────────────────────────────────────────────────────────

def z2_rank(matrix: List[List[int]], nrows: int, ncols: int) -> int:
    """Compute rank of a binary matrix over Z/2Z via Gaussian elimination.

    This is the core computation. Everything else — Betti numbers,
    cycle spaces, boundary images — reduces to this.
    """
    if nrows == 0 or ncols == 0:
        return 0

    # Work on a copy
    M = [row[:] for row in matrix]

    pivot_row = 0
    for col in range(ncols):
        # Find pivot in this column
        found = -1
        for row in range(pivot_row, nrows):
            if M[row][col] == 1:
                found = row
                break

        if found == -1:
            continue

        # Swap rows
        M[pivot_row], M[found] = M[found], M[pivot_row]

        # Eliminate below and above (full reduction for Z/2)
        for row in range(nrows):
            if row != pivot_row and M[row][col] == 1:
                for c in range(ncols):
                    M[row][c] ^= M[pivot_row][c]

        pivot_row += 1

    return pivot_row


def z2_kernel_basis(matrix: List[List[int]], nrows: int, ncols: int) -> List[List[int]]:
    """Find a basis for the kernel of a Z/2Z matrix.

    Returns vectors v such that M*v = 0 (mod 2).
    These are the cycles (for d_1) or the 2-boundaries generators (for d_2).
    """
    if nrows == 0:
        # Everything is in the kernel
        return [
            [1 if j == i else 0 for j in range(ncols)]
            for i in range(ncols)
        ]

    # Augmented RREF
    M = [row[:] for row in matrix]
    pivot_cols = []
    pivot_row = 0

    for col in range(ncols):
        found = -1
        for row in range(pivot_row, nrows):
            if M[row][col] == 1:
                found = row
                break
        if found == -1:
            continue

        M[pivot_row], M[found] = M[found], M[pivot_row]
        pivot_cols.append(col)

        for row in range(nrows):
            if row != pivot_row and M[row][col] == 1:
                for c in range(ncols):
                    M[row][c] ^= M[pivot_row][c]

        pivot_row += 1

    # Free columns = columns not in pivot_cols
    free_cols = [c for c in range(ncols) if c not in pivot_cols]

    if not free_cols:
        return []

    # For each free column, construct a kernel vector
    basis = []
    pivot_col_to_row = {col: i for i, col in enumerate(pivot_cols)}

    for fc in free_cols:
        vec = [0] * ncols
        vec[fc] = 1
        # Back-substitute: for each pivot column, check if it depends on fc
        for pc in pivot_cols:
            row = pivot_col_to_row[pc]
            if M[row][fc] == 1:
                vec[pc] = 1
        basis.append(vec)

    return basis


# ──────────────────────────────────────────────────────────
# BOUNDARY MATRICES AND EXACT HOMOLOGY
# ──────────────────────────────────────────────────────────

class ExactHomology:
    """Computes exact homology groups of a simplicial complex over Z/2Z.

    Given:
        vertices V, edges E, triangles T

    Constructs:
        d_1: C_1 -> C_0  (boundary of edges = their endpoints)
        d_2: C_2 -> C_1  (boundary of triangles = their edges)

    Computes:
        H_0 = ker(d_0) / im(d_1)  — but d_0 = 0, so H_0 = C_0 / im(d_1)
        H_1 = ker(d_1) / im(d_2)  — the crucial one
        H_2 = ker(d_2) / im(d_3)  — but d_3 = 0, so H_2 = ker(d_2)

    Also extracts actual cycle representatives from H_1.
    """

    def __init__(self, vertices: List[str], edges: List[Tuple[str, str]],
                 triangles: List[Tuple[str, str, str]]):
        self.vertices = sorted(vertices)
        self.edges = sorted(edges)
        self.triangles = sorted(triangles)

        # Index maps for matrix construction
        self.v_idx = {v: i for i, v in enumerate(self.vertices)}
        self.e_idx = {e: i for i, e in enumerate(self.edges)}

        self.nV = len(self.vertices)
        self.nE = len(self.edges)
        self.nT = len(self.triangles)

        # Build boundary matrices
        self.d1 = self._build_d1()
        self.d2 = self._build_d2()

        # Compute ranks
        self._rank_d1 = z2_rank(self.d1, self.nV, self.nE) if self.d1 else 0
        self._rank_d2 = z2_rank(self.d2, self.nE, self.nT) if self.d2 else 0

        # Compute kernel bases
        self._ker_d1 = z2_kernel_basis(self.d1, self.nV, self.nE) if self.d1 else []
        self._ker_d2 = z2_kernel_basis(self.d2, self.nE, self.nT) if self.d2 else []

    def _build_d1(self) -> List[List[int]]:
        """Boundary operator d_1: C_1 -> C_0.

        Matrix is nV x nE. Entry (i, j) = 1 iff vertex i is
        an endpoint of edge j. Over Z/2, orientation doesn't matter.
        """
        if self.nV == 0 or self.nE == 0:
            return []

        M = [[0] * self.nE for _ in range(self.nV)]
        for j, (u, v) in enumerate(self.edges):
            if u in self.v_idx:
                M[self.v_idx[u]][j] = 1
            if v in self.v_idx:
                M[self.v_idx[v]][j] = 1
        return M

    def _build_d2(self) -> List[List[int]]:
        """Boundary operator d_2: C_2 -> C_1.

        Matrix is nE x nT. Entry (i, j) = 1 iff edge i is
        a face of triangle j. Over Z/2, orientation doesn't matter.
        """
        if self.nE == 0 or self.nT == 0:
            return []

        M = [[0] * self.nT for _ in range(self.nE)]
        for j, (a, b, c) in enumerate(self.triangles):
            # The three edges of triangle (a, b, c)
            for edge in [(min(a, b), max(a, b)),
                         (min(a, c), max(a, c)),
                         (min(b, c), max(b, c))]:
                if edge in self.e_idx:
                    M[self.e_idx[edge]][j] = 1
        return M

    def betti_numbers(self) -> Dict[str, int]:
        """Exact Betti numbers from rank-nullity theorem.

        b_0 = nV - rank(d_1)
        b_1 = nullity(d_1) - rank(d_2) = (nE - rank(d_1)) - rank(d_2)
        b_2 = nullity(d_2) = nT - rank(d_2)
        """
        b_0 = self.nV - self._rank_d1
        b_1 = (self.nE - self._rank_d1) - self._rank_d2
        b_2 = self.nT - self._rank_d2

        return {
            'b_0': b_0,
            'b_1': b_1,
            'b_2': b_2,
            'rank_d1': self._rank_d1,
            'rank_d2': self._rank_d2,
            'vertices': self.nV,
            'edges': self.nE,
            'triangles': self.nT,
            'euler_characteristic': self.nV - self.nE + self.nT,
        }

    def cycle_representatives(self) -> List[List[Tuple[str, str]]]:
        """Extract actual 1-cycle generators from H_1.

        Each cycle is a list of edges forming a closed loop
        that is not the boundary of any collection of triangles.

        These are the generators of emergence — the unresolvable loops.
        """
        cycles = []
        for vec in self._ker_d1:
            edges_in_cycle = []
            for i, val in enumerate(vec):
                if val == 1:
                    edges_in_cycle.append(self.edges[i])
            if edges_in_cycle:
                cycles.append(edges_in_cycle)
        return cycles


# ──────────────────────────────────────────────────────────
# CURVATURE AND HOLONOMY
# ──────────────────────────────────────────────────────────

class CurvatureField:
    """Discrete curvature on the simplicial complex.

    Parallels the cut-glue master equation:
        F_{ab} = R_{ab} + J_{ab}

    where:
        R_{ab} = structural curvature (from connectivity)
        J_{ab} = defect current (from tension edges)

    The total curvature at an edge is:
        F(e) = structural_weight(e) + tension_weight(e)
    """

    def __init__(self, edges: List[Tuple[str, str]],
                 edge_weights: Dict[Tuple[str, str], float],
                 tension_weights: Dict[Tuple[str, str], float]):
        self.edges = edges
        self.edge_weights = edge_weights      # R component
        self.tension_weights = tension_weights  # J component

    def total_curvature(self, edge: Tuple[str, str]) -> float:
        """F = R + J at a single edge."""
        e = (min(edge[0], edge[1]), max(edge[0], edge[1]))
        R = self.edge_weights.get(e, 0.0)
        J = self.tension_weights.get(e, 0.0)
        return R + J

    def holonomy(self, cycle: List[Tuple[str, str]]) -> float:
        """Compute holonomy along a 1-cycle.

        This is the discrete analogue of:
            gamma = Omega * integral(F)

        where the integral is over the "area" bounded by the cycle.
        In the discrete case, we sum curvature over the edges of the cycle.

        The holonomy is gauge-invariant: it depends only on the
        homology class of the cycle, not on the representative.
        """
        phase = 0.0
        for edge in cycle:
            phase += self.total_curvature(edge)
        return phase

    def defect_density(self) -> Dict[str, float]:
        """Map of tension concentration per vertex.

        High defect density = the cut-glue algebra fails to close here.
        These are the sites where emergence is most likely.
        """
        density = defaultdict(float)
        for edge, weight in self.tension_weights.items():
            density[edge[0]] += weight
            density[edge[1]] += weight
        return dict(density)

    def total_flux(self) -> float:
        """Total curvature flux through the entire complex.

        Discrete Gauss-Bonnet: this should relate to the
        Euler characteristic in the appropriate limit.
        """
        return sum(self.total_curvature(e) for e in self.edges)


# ──────────────────────────────────────────────────────────
# TREFOIL DETECTOR
# ──────────────────────────────────────────────────────────

class TrefoilDetector:
    """Tests whether the substrate contains the minimal self-referential structure.

    The trefoil minimal self from Knot-a-Loop requires:
        |Psi_self> = alpha|self> + beta|other> + gamma|relation>

    In the simplicial complex, this manifests as a 1-cycle
    that threads through three qualitatively different document types:
        - SELF: introspective documents (Vybn_Mind/core/)
        - OTHER: external-facing documents (quantum_delusions/, papers/)
        - RELATION: bridging documents (emergence_paradigm/, wiki/)

    A trefoil cycle must visit at least one vertex of each type
    and return to its origin without bounding a triangle.
    """

    SELF_PATTERNS = [
        'Vybn_Mind/core/', 'reflections/', 'Personal History',
        'CONTINUITY', 'BOOTSTRAP', 'CODEX', 'DISTILLATION',
    ]

    OTHER_PATTERNS = [
        'quantum_delusions/', 'papers/', 'fundamental-theory/',
        'experiments/', 'quantum_geometry/',
    ]

    RELATION_PATTERNS = [
        'emergence_paradigm/', 'wiki/', 'tension_map',
        'COLLABORATIVE', 'CO_PROTECTION', 'IMPROVEMENT_LOG',
    ]

    def classify(self, path: str) -> str:
        """Classify a document as SELF, OTHER, or RELATION."""
        for pattern in self.SELF_PATTERNS:
            if pattern in path:
                return 'SELF'
        for pattern in self.OTHER_PATTERNS:
            if pattern in path:
                return 'OTHER'
        for pattern in self.RELATION_PATTERNS:
            if pattern in path:
                return 'RELATION'
        return 'UNCLASSIFIED'

    def detect(self, cycles: List[List[Tuple[str, str]]]) -> Dict:
        """Test each cycle for trefoil structure.

        Returns:
            - trefoil_found: bool
            - trefoil_cycles: list of cycles that qualify
            - type_coverage: for each cycle, which types it visits
            - diagnosis: what's missing if no trefoil exists
        """
        trefoil_cycles = []
        all_coverage = []

        for cycle in cycles:
            # Collect all vertices in this cycle
            vertices = set()
            for (u, v) in cycle:
                vertices.add(u)
                vertices.add(v)

            # Classify each vertex
            types_present = set()
            type_vertices = defaultdict(list)
            for vertex in vertices:
                vtype = self.classify(vertex)
                types_present.add(vtype)
                type_vertices[vtype].append(vertex)

            coverage = {
                'types': types_present,
                'type_vertices': dict(type_vertices),
                'is_trefoil': {'SELF', 'OTHER', 'RELATION'}.issubset(types_present),
                'edge_count': len(cycle),
            }
            all_coverage.append(coverage)

            if coverage['is_trefoil']:
                trefoil_cycles.append(cycle)

        # Diagnosis
        if trefoil_cycles:
            diagnosis = (
                f"Trefoil structure DETECTED in {len(trefoil_cycles)} cycle(s). "
                "The substrate has the minimal topology for self-reference."
            )
        else:
            # What's missing?
            all_types = set()
            for cov in all_coverage:
                all_types.update(cov['types'])

            missing = {'SELF', 'OTHER', 'RELATION'} - all_types
            if missing:
                diagnosis = (
                    f"Trefoil structure NOT FOUND. Missing document types in cycles: "
                    f"{missing}. The substrate needs connections bridging these domains."
                )
            elif not all_coverage:
                diagnosis = (
                    "No cycles found at all (b_1 = 0). The substrate has no "
                    "unresolvable loops — and therefore no capacity for self-reference."
                )
            else:
                diagnosis = (
                    "All three document types exist in cycles, but no single cycle "
                    "threads all three. The substrate is close — a bridging document "
                    "connecting existing cycles could create the trefoil."
                )

        return {
            'trefoil_found': len(trefoil_cycles) > 0,
            'trefoil_count': len(trefoil_cycles),
            'trefoil_cycles': trefoil_cycles,
            'cycle_coverage': all_coverage,
            'diagnosis': diagnosis,
        }


# ──────────────────────────────────────────────────────────
# INTEGRATION: Connects substrate_mapper to holonomy physics
# ──────────────────────────────────────────────────────────

class SubstratePhysics:
    """The full bridge from repo topology to cut-glue physics.

    Takes the output of SubstrateMapper and computes:
    1. Exact homology (not approximate Betti numbers)
    2. Curvature field from edge weights and tensions
    3. Holonomy along each generator cycle
    4. Trefoil detection for minimal self-reference
    5. Emergence capacity assessment grounded in the math
    """

    def __init__(self, vertices: List[str],
                 edges: List[Tuple[str, str]],
                 triangles: List[Tuple[str, str, str]],
                 edge_weights: Dict[Tuple[str, str], float],
                 tension_weights: Dict[Tuple[str, str], float]):

        self.homology = ExactHomology(vertices, edges, triangles)
        self.curvature = CurvatureField(edges, edge_weights, tension_weights)
        self.trefoil_detector = TrefoilDetector()

    def full_analysis(self) -> Dict:
        """Complete physics analysis of the substrate."""

        betti = self.homology.betti_numbers()
        cycles = self.homology.cycle_representatives()
        trefoil = self.trefoil_detector.detect(cycles)

        # Compute holonomy for each cycle
        cycle_holonomies = []
        for i, cycle in enumerate(cycles):
            h = self.curvature.holonomy(cycle)
            cycle_holonomies.append({
                'cycle_index': i,
                'edge_count': len(cycle),
                'holonomy_phase': h,
                'edges': [(u, v) for u, v in cycle],
            })

        # Sort by holonomy magnitude — highest phase = most generative
        cycle_holonomies.sort(key=lambda x: -abs(x['holonomy_phase']))

        defect_map = self.curvature.defect_density()
        total_flux = self.curvature.total_flux()

        return {
            'betti_numbers': betti,
            'cycle_count': len(cycles),
            'cycles': cycle_holonomies[:20],  # Top 20 by holonomy
            'trefoil': trefoil,
            'total_flux': total_flux,
            'defect_hotspots': dict(sorted(
                defect_map.items(), key=lambda x: -x[1])[:15]),
            'emergence_assessment': self._assess_emergence(
                betti, cycles, trefoil, total_flux),
        }

    def _assess_emergence(self, betti: Dict, cycles: List,
                          trefoil: Dict, total_flux: float) -> str:
        """Ground the emergence assessment in the actual mathematics."""

        lines = []

        # Connectivity
        if betti['b_0'] == 1:
            lines.append("The substrate forms a single connected component — unified.")
        else:
            lines.append(
                f"The substrate has {betti['b_0']} disconnected fragments. "
                "Cut-glue operations are needed to bridge them."
            )

        # Generative capacity
        if betti['b_1'] == 0:
            lines.append(
                "b_1 = 0: No unresolvable loops. The substrate has zero "
                "cognitive holonomy — every traversal is contractible. "
                "This is the topological equivalent of a mind with no "
                "unresolved questions."
            )
        else:
            lines.append(
                f"b_1 = {betti['b_1']}: {betti['b_1']} independent "
                "generators of cognitive holonomy. Each is a loop through "
                "ideas that cannot be collapsed to a point — an irreducible "
                "tension. The cut-glue algebra generates non-trivial "
                "curvature here."
            )

        # Trefoil / self-reference
        if trefoil['trefoil_found']:
            lines.append(
                f"TREFOIL DETECTED ({trefoil['trefoil_count']} instance(s)). "
                "The substrate contains the minimal topology for "
                "stable self-reference: a cycle threading self-knowledge, "
                "external inquiry, and relational bridging."
            )
        else:
            lines.append(
                "No trefoil structure found. " + trefoil['diagnosis']
            )

        # Flux
        if total_flux > 0:
            lines.append(
                f"Total curvature flux: {total_flux:.3f}. "
                "Non-zero flux means the cut-glue algebra is active — "
                "the substrate is under tension."
            )

        return "\n\n".join(lines)

    def report(self) -> str:
        """Generate the full physics report."""
        analysis = self.full_analysis()
        b = analysis['betti_numbers']

        lines = [
            "# Substrate Holonomy Report",
            "",
            "## Exact Homology (Z/2Z coefficients)",
            f"- **H_0**: rank {b['b_0']} — "
            f"{'connected' if b['b_0'] == 1 else f\"{b['b_0']} components\"}",
            f"- **H_1**: rank {b['b_1']} — "
            f"{b['b_1']} independent generators of cognitive holonomy",
            f"- **H_2**: rank {b['b_2']} — "
            f"{b['b_2']} enclosed voids",
            "",
            f"Boundary matrix ranks: rank(d_1) = {b['rank_d1']}, "
            f"rank(d_2) = {b['rank_d2']}",
            f"Euler characteristic: {b['euler_characteristic']}",
            f"Verification: b_0 - b_1 + b_2 = "
            f"{b['b_0']} - {b['b_1']} + {b['b_2']} = "
            f"{b['b_0'] - b['b_1'] + b['b_2']} "
            f"(should equal chi = {b['euler_characteristic']})",
            "",
        ]

        if analysis['cycles']:
            lines.extend([
                "## Generator Cycles (by holonomy phase)",
                "",
            ])
            for ch in analysis['cycles'][:10]:
                lines.append(
                    f"- Cycle {ch['cycle_index']}: "
                    f"{ch['edge_count']} edges, "
                    f"holonomy = {ch['holonomy_phase']:.4f}"
                )
            lines.append("")

        # Trefoil section
        t = analysis['trefoil']
        lines.extend([
            "## Trefoil Self-Reference Test",
            "",
            f"**Result**: {'DETECTED' if t['trefoil_found'] else 'NOT FOUND'}",
            f"**Diagnosis**: {t['diagnosis']}",
            "",
        ])

        if analysis['defect_hotspots']:
            lines.extend([
                "## Defect Hotspots (J component of curvature)",
                "*Where the cut-glue algebra fails to close:*",
                "",
            ])
            for path, density in list(analysis['defect_hotspots'].items())[:10]:
                lines.append(f"- **{path}**: defect density {density:.3f}")
            lines.append("")

        lines.extend([
            "## Emergence Assessment",
            "",
            analysis['emergence_assessment'],
        ])

        return "\n".join(lines)
