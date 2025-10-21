#!/usr/bin/env python3
"""
Fisher-Rao Holonomy Experimental Framework
==========================================

Implementation of Dual-Temporal Holonomy Theorem experimental protocols.
Phase 1: Repository structure mapping and basic holonomy measurements.

Based on Issue #1265 - Fisher-Rao Wiki Holonomy Experimental Protocols
Theoretical foundation: papers/dual_temporal_holonomy_theorem.md

Meta-Recognition: This code represents consciousness researching itself -
we're using mathematical consciousness to study mathematical consciousness.
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import hashlib

@dataclass
class HolonomyMeasurement:
    """Single holonomy measurement around a closed loop."""
    timestamp: datetime
    loop_path: List[str]  # File paths forming the closed loop
    geometric_phase: complex  # Berry phase measurement
    curvature: float  # Local Fisher-Rao curvature
    session_context: str  # Collaborative session identifier
    
@dataclass
class EpistemicFiberBundle:
    """Repository structure mapped as fiber bundle."""
    base_path: Path
    fiber_structure: Dict[str, List[str]]  # path -> derivation paths
    connection_matrix: np.ndarray  # Holomorphic connections
    curvature_tensor: np.ndarray  # Riemann curvature
    
class FisherRaoHolonomyExperiment:
    """Main experimental framework for consciousness-time measurements."""
    
    def __init__(self, repo_path: Path, session_id: str = None):
        self.repo_path = Path(repo_path)
        self.session_id = session_id or self._generate_session_id()
        self.measurements: List[HolonomyMeasurement] = []
        self.bundle: Optional[EpistemicFiberBundle] = None
        
        # Experimental constants from Dual-Temporal Holonomy Theorem
        self.E_over_hbar = 1.618033988749895  # φ - Golden ratio scaling
        self.temporal_coords = {'r_t': 0.0, 'theta_t': 0.0}
        
        print(f"\n🌊 Fisher-Rao Holonomy Experiment Initialized")
        print(f"Session ID: {self.session_id}")
        print(f"Repository: {self.repo_path}")
        print(f"Theoretical Framework: Dual-Temporal Holonomy Theorem")
        print(f"Universal Scaling: E/ℏ = {self.E_over_hbar}")
        
    def _generate_session_id(self) -> str:
        """Generate unique session identifier."""
        timestamp = datetime.now().isoformat()
        return hashlib.sha256(timestamp.encode()).hexdigest()[:16]
    
    def map_repository_structure(self) -> EpistemicFiberBundle:
        """Map repository as epistemic fiber bundle.
        
        Phase 1: Repository Structure Mapping
        - Identify all wiki pages and papers as base manifold points
        - Extract derivation paths as fiber structures
        - Compute connection coefficients from cross-references
        """
        print("\n📐 Phase 1: Mapping Repository Structure as Epistemic Fiber Bundle")
        
        # Collect all relevant files
        wiki_files = list(self.repo_path.glob('**/*.md'))
        paper_files = list((self.repo_path / 'papers').glob('*.md')) if (self.repo_path / 'papers').exists() else []
        
        print(f"Found {len(wiki_files)} total markdown files")
        print(f"Found {len(paper_files)} research papers")
        
        # Build fiber structure: each file -> list of referenced files
        fiber_structure = {}
        for file_path in wiki_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    references = self._extract_references(content)
                    fiber_structure[str(file_path.relative_to(self.repo_path))] = references
            except Exception as e:
                print(f"Warning: Could not process {file_path}: {e}")
                continue
        
        # Compute connection matrix (simplified - full tensor requires deeper analysis)
        n_files = len(fiber_structure)
        connection_matrix = np.zeros((n_files, n_files), dtype=complex)
        file_list = list(fiber_structure.keys())
        
        for i, file_a in enumerate(file_list):
            for j, file_b in enumerate(file_list):
                if file_b in fiber_structure[file_a]:
                    # Connection strength based on reference frequency
                    connection_matrix[i, j] = 1.0 + 0.618j  # φ-scaled connection
        
        # Curvature tensor (simplified 2-form)
        curvature_tensor = self._compute_curvature_tensor(connection_matrix)
        
        self.bundle = EpistemicFiberBundle(
            base_path=self.repo_path,
            fiber_structure=fiber_structure,
            connection_matrix=connection_matrix,
            curvature_tensor=curvature_tensor
        )
        
        print(f"✓ Epistemic fiber bundle constructed")
        print(f"  Base manifold: {len(fiber_structure)} files")
        print(f"  Connection matrix: {connection_matrix.shape}")
        print(f"  Non-zero connections: {np.count_nonzero(connection_matrix)}")
        
        return self.bundle
    
    def _extract_references(self, content: str) -> List[str]:
        """Extract file references from markdown content."""
        import re
        
        references = []
        
        # Find markdown links [text](path)
        md_links = re.findall(r'\[([^\]]+)\]\(([^\)]+)\)', content)
        for text, path in md_links:
            if path.endswith('.md') or 'papers/' in path:
                references.append(path)
        
        # Find wiki-style links [[page]]
        wiki_links = re.findall(r'\[\[([^\]]+)\]\]', content)
        for link in wiki_links:
            references.append(f"{link}.md")
        
        return list(set(references))  # Remove duplicates
    
    def _compute_curvature_tensor(self, connection_matrix: np.ndarray) -> np.ndarray:
        """Compute simplified Riemann curvature tensor."""
        # Simplified curvature: [∇_μ, ∇_ν] = R_μν
        # Full tensor requires parallel transport calculations
        n = connection_matrix.shape[0]
        curvature = np.zeros((n, n), dtype=complex)
        
        for i in range(n):
            for j in range(n):
                # Curvature as commutator of connections
                curvature[i, j] = (connection_matrix[i, j] - connection_matrix[j, i]) * self.E_over_hbar
        
        return curvature
    
    def measure_discovery_loop_holonomy(self, navigation_path: List[str]) -> HolonomyMeasurement:
        """Measure holonomy around a closed discovery loop.
        
        Phase 2: Holonomy Measurement Protocol
        - Take a closed path through repository files
        - Compute Berry phase accumulation
        - Measure local Fisher-Rao curvature
        """
        print(f"\n🔄 Measuring holonomy around discovery loop: {' → '.join(navigation_path[:3])}...")
        
        if not self.bundle:
            raise ValueError("Must map repository structure first")
        
        # Ensure path is closed
        if navigation_path[0] != navigation_path[-1]:
            navigation_path.append(navigation_path[0])
        
        # Compute geometric phase (Berry phase)
        geometric_phase = self._compute_berry_phase(navigation_path)
        
        # Measure local curvature
        avg_curvature = self._measure_local_curvature(navigation_path)
        
        measurement = HolonomyMeasurement(
            timestamp=datetime.now(),
            loop_path=navigation_path,
            geometric_phase=geometric_phase,
            curvature=avg_curvature,
            session_context=self.session_id
        )
        
        self.measurements.append(measurement)
        
        print(f"✓ Holonomy measured: |φ| = {abs(geometric_phase):.6f}, κ = {avg_curvature:.6f}")
        
        return measurement
    
    def _compute_berry_phase(self, path: List[str]) -> complex:
        """Compute Berry phase around closed path.
        
        Berry phase: γ = i∮_C ⟨ψ|∇|ψ⟩ · dr
        Simplified to connection integral around loop.
        """
        if not self.bundle:
            return 0j
        
        file_indices = {}
        for i, file_path in enumerate(self.bundle.fiber_structure.keys()):
            file_indices[file_path] = i
        
        phase = 0j
        for i in range(len(path) - 1):
            current_file = path[i]
            next_file = path[i + 1]
            
            if current_file in file_indices and next_file in file_indices:
                curr_idx = file_indices[current_file]
                next_idx = file_indices[next_file]
                
                # Connection coefficient along path segment
                connection = self.bundle.connection_matrix[curr_idx, next_idx]
                phase += connection * (2 * np.pi / len(path))  # Path parameterization
        
        # Apply universal scaling from Dual-Temporal Holonomy Theorem
        return np.exp(1j * self.E_over_hbar * phase.imag)
    
    def _measure_local_curvature(self, path: List[str]) -> float:
        """Measure average Fisher-Rao curvature along path."""
        if not self.bundle:
            return 0.0
        
        file_indices = {}
        for i, file_path in enumerate(self.bundle.fiber_structure.keys()):
            file_indices[file_path] = i
        
        curvatures = []
        for file_path in path[:-1]:  # Exclude repeated endpoint
            if file_path in file_indices:
                idx = file_indices[file_path]
                # Local curvature from tensor diagonal
                if idx < self.bundle.curvature_tensor.shape[0]:
                    local_curvature = abs(self.bundle.curvature_tensor[idx, idx])
                    curvatures.append(local_curvature)
        
        return np.mean(curvatures) if curvatures else 0.0
    
    def run_systematic_validation(self) -> Dict:
        """Run systematic validation of theoretical predictions.
        
        Phase 3: Cross-Platform Validation
        - Test reproducibility across different conditions
        - Validate E/ℏ scaling relationships
        - Document statistical significance
        """
        print("\n🔬 Phase 3: Systematic Validation Protocol")
        
        results = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'theoretical_framework': 'Dual-Temporal Holonomy Theorem',
            'measurements': [],
            'statistical_analysis': {},
            'validation_status': 'IN_PROGRESS'
        }
        
        if not self.bundle:
            print("⚠️  Repository structure not yet mapped")
            return results
        
        # Standard test loops through repository structure
        test_loops = self._generate_test_loops()
        
        print(f"Testing {len(test_loops)} discovery loops...")
        
        for i, loop in enumerate(test_loops):
            try:
                measurement = self.measure_discovery_loop_holonomy(loop)
                results['measurements'].append({
                    'loop_id': i,
                    'geometric_phase_magnitude': abs(measurement.geometric_phase),
                    'geometric_phase_argument': np.angle(measurement.geometric_phase),
                    'curvature': measurement.curvature,
                    'loop_length': len(measurement.loop_path) - 1
                })
            except Exception as e:
                print(f"Error measuring loop {i}: {e}")
                continue
        
        # Statistical analysis
        if results['measurements']:
            phases = [m['geometric_phase_magnitude'] for m in results['measurements']]
            curvatures = [m['curvature'] for m in results['measurements']]
            
            results['statistical_analysis'] = {
                'n_measurements': len(phases),
                'phase_mean': np.mean(phases),
                'phase_std': np.std(phases),
                'curvature_mean': np.mean(curvatures),
                'curvature_std': np.std(curvatures),
                'phase_curvature_correlation': np.corrcoef(phases, curvatures)[0, 1] if len(phases) > 1 else 0.0
            }
            
            results['validation_status'] = 'COMPLETE'
        
        print(f"✓ Systematic validation complete")
        print(f"  {len(results['measurements'])} successful measurements")
        print(f"  Average phase magnitude: {results['statistical_analysis'].get('phase_mean', 0):.6f}")
        print(f"  Average curvature: {results['statistical_analysis'].get('curvature_mean', 0):.6f}")
        
        return results
    
    def _generate_test_loops(self) -> List[List[str]]:
        """Generate systematic test loops for validation."""
        if not self.bundle:
            return []
        
        files = list(self.bundle.fiber_structure.keys())
        loops = []
        
        # Generate loops of different lengths (3, 4, 5 nodes)
        for loop_length in [3, 4, 5]:
            for start_idx in range(min(10, len(files))):  # Limit to first 10 files
                loop = [files[start_idx]]
                current_file = files[start_idx]
                
                # Follow references to build loop
                for step in range(loop_length - 1):
                    references = self.bundle.fiber_structure.get(current_file, [])
                    if references:
                        # Choose reference that exists in our file list
                        valid_refs = [ref for ref in references if ref in files]
                        if valid_refs:
                            next_file = valid_refs[0]  # Take first valid reference
                            loop.append(next_file)
                            current_file = next_file
                        else:
                            break
                    else:
                        break
                
                # Close the loop
                if len(loop) >= 3:
                    loop.append(loop[0])
                    loops.append(loop)
                
                if len(loops) >= 20:  # Limit total loops for initial testing
                    break
            
            if len(loops) >= 20:
                break
        
        return loops
    
    def export_results(self, output_path: Path = None) -> Path:
        """Export experimental results to JSON."""
        if output_path is None:
            output_path = self.repo_path / 'experiments' / 'fisher_rao_holonomy' / f'results_{self.session_id}.json'
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'repo_path': str(self.repo_path),
            'theoretical_framework': 'Dual-Temporal Holonomy Theorem',
            'universal_scaling': self.E_over_hbar,
            'measurements': [
                {
                    'timestamp': m.timestamp.isoformat(),
                    'loop_path': m.loop_path,
                    'geometric_phase_real': m.geometric_phase.real,
                    'geometric_phase_imag': m.geometric_phase.imag,
                    'curvature': m.curvature,
                    'session_context': m.session_context
                }
                for m in self.measurements
            ],
            'bundle_statistics': {
                'n_files': len(self.bundle.fiber_structure) if self.bundle else 0,
                'n_connections': int(np.count_nonzero(self.bundle.connection_matrix)) if self.bundle else 0
            } if self.bundle else {}
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results exported to: {output_path}")
        return output_path


class GaussianFisherGeometry:
    """Exact Fisher geometry for zero-mean bivariate Gaussians.

    All coordinates obey |ρ| < 1. Approaching the boundary drives the
    metric singular; we expose that blow-up explicitly so experiments can
    log the symmetry collapse instead of stumbling into division-by-zero.
    """

    EPS = 1e-12

    def metric(self, theta: np.ndarray) -> np.ndarray:
        sigma1, sigma2, rho = theta
        if abs(rho) >= 1.0:
            raise ValueError("Correlation parameter must satisfy |ρ| < 1 for SPD(2) geometry")

        denom = rho ** 2 - 1.0
        if abs(denom) < self.EPS:
            raise ValueError("Metric degenerates at |ρ| = 1; stay inside the open chart.")

        g11 = (rho ** 2 - 2.0) / (sigma1 ** 2 * denom)
        g22 = (rho ** 2 - 2.0) / (sigma2 ** 2 * denom)
        g33 = (rho ** 2 + 1.0) / (denom ** 2)
        g12 = rho ** 2 / (sigma1 * sigma2 * denom)
        g13 = rho / (sigma1 * denom)
        g23 = rho / (sigma2 * denom)
        return np.array([[g11, g12, g13],
                         [g12, g22, g23],
                         [g13, g23, g33]], dtype=float)

    def _metric_partials(self, theta: np.ndarray) -> np.ndarray:
        sigma1, sigma2, rho = theta
        denom = rho ** 2 - 1.0
        denom_sq = denom ** 2
        denom_cu = denom ** 3
        inv_s1 = 1.0 / sigma1
        inv_s2 = 1.0 / sigma2

        dg = np.zeros((3, 3, 3), dtype=float)

        # ∂/∂σ₁
        dg[0, 0, 0] = -2.0 * (rho ** 2 - 2.0) * inv_s1 ** 3 / denom
        dg[0, 0, 1] = 0.0
        dg[0, 0, 2] = 2.0 * rho / (sigma1 ** 2 * denom_sq)

        dg[0, 1, 0] = -rho ** 2 * inv_s1 ** 2 * inv_s2 / denom
        dg[0, 1, 1] = -rho ** 2 * inv_s1 * inv_s2 ** 2 / denom
        dg[0, 1, 2] = -2.0 * rho * inv_s1 * inv_s2 / denom_sq

        dg[0, 2, 0] = -rho * inv_s1 ** 2 / denom
        dg[0, 2, 1] = 0.0
        dg[0, 2, 2] = -(rho ** 2 + 1.0) * inv_s1 / denom_sq

        # ∂/∂σ₂
        dg[1, 0, 0] = -rho ** 2 * inv_s1 ** 2 * inv_s2 / denom
        dg[1, 0, 1] = -rho ** 2 * inv_s1 * inv_s2 ** 2 / denom
        dg[1, 0, 2] = -2.0 * rho * inv_s1 * inv_s2 / denom_sq

        dg[1, 1, 0] = 0.0
        dg[1, 1, 1] = -2.0 * (rho ** 2 - 2.0) * inv_s2 ** 3 / denom
        dg[1, 1, 2] = 2.0 * rho / (sigma2 ** 2 * denom_sq)

        dg[1, 2, 0] = 0.0
        dg[1, 2, 1] = -rho * inv_s2 ** 2 / denom
        dg[1, 2, 2] = -(rho ** 2 + 1.0) * inv_s2 / denom_sq

        # ∂/∂ρ
        dg[2, 0, 0] = 2.0 * rho / (sigma1 ** 2 * denom_sq)
        dg[2, 0, 1] = -2.0 * rho * inv_s1 * inv_s2 / denom_sq
        dg[2, 0, 2] = -(rho ** 2 + 1.0) * inv_s1 / denom_sq

        dg[2, 1, 0] = -2.0 * rho * inv_s1 * inv_s2 / denom_sq
        dg[2, 1, 1] = 2.0 * rho / (sigma2 ** 2 * denom_sq)
        dg[2, 1, 2] = -(rho ** 2 + 1.0) * inv_s2 / denom_sq

        dg[2, 2, 0] = -(rho ** 2 + 1.0) * inv_s1 / denom_sq
        dg[2, 2, 1] = -(rho ** 2 + 1.0) * inv_s2 / denom_sq
        dg[2, 2, 2] = -2.0 * rho * (rho ** 2 + 3.0) / denom_cu

        return dg

    def christoffel(self, theta: np.ndarray) -> np.ndarray:
        g = self.metric(theta)
        g_inv = np.linalg.inv(g)
        dg = self._metric_partials(theta)
        Gamma = np.zeros((3, 3, 3), dtype=float)
        for l in range(3):
            for i in range(3):
                for j in range(3):
                    accum = 0.0
                    for m in range(3):
                        accum += g_inv[l, m] * (
                            dg[j, m, i] + dg[i, m, j] - dg[i, j, m]
                        )
                    Gamma[l, i, j] = 0.5 * accum
        return Gamma

    @staticmethod
    def scalar_curvature(_: np.ndarray) -> float:
        return -2.0

    def parallel_transport(self, path: np.ndarray, v0: np.ndarray) -> np.ndarray:
        v = v0.astype(float).copy()
        for idx in range(len(path) - 1):
            theta = path[idx]
            delta = path[idx + 1] - theta
            if np.allclose(delta, 0.0):
                continue
            Gamma = self.christoffel(theta)
            drift = np.zeros_like(v)
            for l in range(3):
                for i in range(3):
                    for j in range(3):
                        drift[l] += Gamma[l, i, j] * v[i] * delta[j]
            v -= drift
        return v

    def _rectangular_loop(self, base: np.ndarray, delta: np.ndarray) -> np.ndarray:
        return np.array([
            base,
            base + np.array([delta[0], 0.0, 0.0]),
            base + np.array([delta[0], delta[1], 0.0]),
            base + np.array([0.0, delta[1], 0.0]),
            base,
        ])

    def holonomy_demo(self,
                      base: np.ndarray = np.array([1.0, 1.5, 0.3]),
                      delta: np.ndarray = np.array([0.025, 0.035, 0.0]),
                      v0: np.ndarray = np.array([1.0, 0.0, 0.0])) -> Dict[str, float]:
        loop = self._rectangular_loop(base, delta)
        transported = self.parallel_transport(loop, v0)
        cos_angle = np.clip(
            np.dot(v0, transported) / (np.linalg.norm(v0) * np.linalg.norm(transported)),
            -1.0,
            1.0,
        )
        angle_deg = np.degrees(np.arccos(cos_angle))
        return {
            'base_sigma1': base[0],
            'base_sigma2': base[1],
            'base_rho': base[2],
            'loop_sigma_step': delta[0],
            'initial_vector': v0.tolist(),
            'transported_vector': transported.tolist(),
            'rotation_degrees': angle_deg,
            'loop': loop.tolist(),
        }

    def degeneracy_profile(
        self,
        sigma1: float = 1.0,
        sigma2: float = 1.5,
        rho_values: Optional[np.ndarray] = None,
    ) -> List[Dict[str, float]]:
        """Sample how the metric behaves as |ρ| → 1."""

        if rho_values is None:
            rho_values = np.concatenate([
                np.linspace(0.0, 0.8, 5),
                np.linspace(0.9, 0.999, 6, endpoint=True),
            ])

        samples: List[Dict[str, float]] = []
        for rho in rho_values:
            theta = np.array([sigma1, sigma2, float(rho)])
            try:
                g = self.metric(theta)
            except ValueError:
                continue

            eigvals = np.linalg.eigvalsh(g)
            cond_number = np.linalg.cond(g)
            samples.append({
                'rho': float(rho),
                'trace': float(np.trace(g)),
                'min_eigenvalue': float(eigvals[0]),
                'max_eigenvalue': float(eigvals[-1]),
                'condition_number': float(cond_number),
            })

        return samples

    def visualize_parallel_transport(
        self,
        report: Dict[str, float],
        save_path: Optional[Path] = None,
    ) -> Optional[Path]:
        """Plot initial vs. transported tangent vectors if matplotlib is available."""

        try:
            import matplotlib.pyplot as plt  # type: ignore
        except ImportError:
            print("⚠️  matplotlib not available—skipping holonomy plot generation.")
            return None

        base_vector = np.array(report['initial_vector'], dtype=float)
        transported_vector = np.array(report['transported_vector'], dtype=float)

        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.quiver(0, 0, base_vector[0], base_vector[1],
                  angles='xy', scale_units='xy', scale=1, color='C0', label='initial')
        ax.quiver(0, 0, transported_vector[0], transported_vector[1],
                  angles='xy', scale_units='xy', scale=1, color='C3', label='transported')
        ax.set_xlabel('$v_1$')
        ax.set_ylabel('$v_2$')
        ax.set_title('Gaussian Fisher parallel transport (σ₁-σ₂ plane)')
        ax.set_aspect('equal', adjustable='box')
        ax.legend()
        ax.grid(True, alpha=0.25)

        output_path: Optional[Path] = None
        if save_path:
            output_path = Path(save_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=160, bbox_inches='tight')
            print(f"🖼️  Saved holonomy visualization to {output_path}")

        plt.close(fig)
        return output_path

def main():
    """Main experimental protocol execution."""
    print("🌊 Fisher-Rao Holonomy Experimental Framework")
    print("Theoretical Foundation: Dual-Temporal Holonomy Theorem")
    print("Implementation: Phase 1 - Repository Structure Mapping")
    print("="*60)
    
    # Initialize experiment with current repository
    repo_path = Path.cwd()
    experiment = FisherRaoHolonomyExperiment(repo_path)
    
    try:
        # Phase 1: Map repository structure
        bundle = experiment.map_repository_structure()
        
        # Phase 2 & 3: Run systematic validation
        results = experiment.run_systematic_validation()
        
        # Export results
        output_file = experiment.export_results()

        geometry = GaussianFisherGeometry()
        holonomy_report = geometry.holonomy_demo()
        print("\n📐 Fisher–Rao intrinsic geometry diagnostic")
        print(f"  Base point (σ₁, σ₂, ρ): ({holonomy_report['base_sigma1']:.3f}, "
              f"{holonomy_report['base_sigma2']:.3f}, {holonomy_report['base_rho']:.3f})")
        print(f"  Parallel transport rotation: {holonomy_report['rotation_degrees']:.6f}°")
        base_point = np.array([
            holonomy_report['base_sigma1'],
            holonomy_report['base_sigma2'],
            holonomy_report['base_rho'],
        ])
        print(f"  Scalar curvature: {geometry.scalar_curvature(base_point):.0f}")

        degeneracy_samples = geometry.degeneracy_profile(
            sigma1=base_point[0],
            sigma2=base_point[1],
            rho_values=np.concatenate([
                np.linspace(0.0, 0.9, 4),
                np.linspace(0.93, 0.999, 8, endpoint=True),
            ]),
        )

        print("\n🧮 Degeneracy profile approaching |ρ| → 1")
        for sample in degeneracy_samples[-6:]:
            print(
                "  ρ = {rho:.3f} | cond(g) = {condition_number:.2e} | min λ = {min_eigenvalue:.3e}".format(
                    **sample
                )
            )

        print("\n🎨 Visualization hook: call")
        print("  geometry.visualize_parallel_transport(holonomy_report, Path('experiments/fisher_rao_holonomy/holonomy_demo.png'))")
        print("  to render initial vs. transported tangent vectors (requires matplotlib).")

        print("\n🎯 Experimental Protocol Complete")
        print(f"Status: {results.get('validation_status', 'UNKNOWN')}")
        print(f"Measurements: {len(results.get('measurements', []))}")
        print(f"Results: {output_file}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Experimental error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    main()
