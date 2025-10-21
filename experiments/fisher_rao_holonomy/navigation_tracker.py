#!/usr/bin/env python3
"""
Fisher-Rao Wiki Holonomy Navigation Tracker

Treats repository structure as epistemic fiber bundle,
measuring geometric curvature in navigation paths.

Implements Issue #1265 experimental protocols.
"""

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
import statistics
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

_NUMPY_SPEC = importlib.util.find_spec("numpy")
if _NUMPY_SPEC is not None:
    np = importlib.import_module("numpy")  # type: ignore[assignment]
else:
    np = None

class EpistemicFiberBundle:
    """Repository structure as mathematical manifold with connection forms"""
    
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.navigation_log = []
        self.connection_forms = {}
        self.holonomy_accumulator = 0.0
        self.current_session = self._generate_session_id()
        
    def _generate_session_id(self) -> str:
        """Generate unique session identifier for collaborative tracking"""
        timestamp = datetime.now().isoformat()
        return hashlib.md5(f"{timestamp}_{os.getpid()}".encode()).hexdigest()[:8]
    
    def map_repository_topology(self) -> Dict:
        """Map repository structure as fiber bundle chart"""
        topology = {
            'fibers': {},
            'connection_forms': {},
            'curvature_indicators': {}
        }
        
        # Root fiber identification
        root_files = list(self.repo_root.glob('*.md'))
        topology['fibers']['root'] = [f.name for f in root_files]
        
        # Directory fiber bundles  
        for directory in self.repo_root.iterdir():
            if directory.is_dir() and not directory.name.startswith('.'):
                fiber_content = list(directory.rglob('*'))
                topology['fibers'][directory.name] = {
                    'files': [str(f.relative_to(directory)) for f in fiber_content if f.is_file()],
                    'subdirs': [str(f.relative_to(directory)) for f in fiber_content if f.is_dir()],
                }
        
        return topology
    
    def track_navigation_step(self, file_path: str, access_type: str = "read", 
                            conceptual_weight: float = 1.0):
        """Record navigation step for holonomy calculation"""
        step = {
            'timestamp': time.time(),
            'session_id': self.current_session,
            'file_path': file_path,
            'access_type': access_type,
            'conceptual_weight': conceptual_weight,
            'accumulated_phase': self.holonomy_accumulator
        }
        
        self.navigation_log.append(step)
        
        # Update holonomy accumulator
        path_curvature = self._calculate_path_curvature(file_path)
        self.holonomy_accumulator += path_curvature * conceptual_weight
        
        return step
    
    def _calculate_path_curvature(self, file_path: str) -> float:
        """Calculate geometric curvature for navigation step"""
        # Curvature based on:
        # 1. Directory depth (deeper = higher curvature)
        # 2. File type (.md = theoretical, .py = computational)
        # 3. Cross-references (links increase curvature)
        
        path = Path(file_path)
        depth_curvature = len(path.parts) * 0.1
        
        # File type curvature
        type_curvature = {
            '.md': 0.5,   # Theoretical content
            '.py': 0.3,   # Computational
            '.txt': 0.2,  # Raw data
            '': 0.1       # Other
        }.get(path.suffix, 0.1)
        
        return depth_curvature + type_curvature
    
    def detect_navigation_loops(self, window_size: int = 10) -> List[Dict]:
        """Detect closed loops in navigation for holonomy measurement"""
        loops = []
        
        if len(self.navigation_log) < 3:
            return loops
            
        # Look for return paths in recent navigation
        recent_paths = [step['file_path'] for step in self.navigation_log[-window_size:]]
        
        for i, path in enumerate(recent_paths[:-2]):
            if path in recent_paths[i+2:]:  # Found return after at least one step
                return_index = recent_paths[i+2:].index(path) + i + 2
                loop = {
                    'start_path': path,
                    'loop_length': return_index - i,
                    'accumulated_holonomy': self._calculate_loop_holonomy(i, return_index),
                    'paths': recent_paths[i:return_index+1]
                }
                loops.append(loop)
        
        return loops
    
    def _calculate_loop_holonomy(self, start_idx: int, end_idx: int) -> float:
        """Calculate Berry phase accumulation around closed loop"""
        loop_steps = self.navigation_log[-(len(self.navigation_log)-(start_idx)):-(len(self.navigation_log)-(end_idx))]
        
        holonomy = 0.0
        for i, step in enumerate(loop_steps[:-1]):
            next_step = loop_steps[i+1]
            
            # Calculate connection form between steps
            connection = self._connection_form(
                step['file_path'], 
                next_step['file_path']
            )
            
            holonomy += connection * step['conceptual_weight']
        
        return holonomy
    
    def _connection_form(self, path1: str, path2: str) -> float:
        """Calculate connection form between two repository locations"""
        # Connection strength based on:
        # 1. Directory relationship
        # 2. Cross-reference likelihood  
        # 3. Conceptual distance
        
        p1, p2 = Path(path1), Path(path2)
        
        # Same directory = strong connection
        if p1.parent == p2.parent:
            return 0.8
        
        # Parent-child relationship
        if p1.parent in p2.parents or p2.parent in p1.parents:
            return 0.6
        
        # Cross-experimental connection
        if 'experiments' in str(p1) and 'experiments' in str(p2):
            return 0.4
        
        # Theory-experiment connection
        if ('papers' in str(p1) and 'experiments' in str(p2)) or \
           ('experiments' in str(p1) and 'papers' in str(p2)):
            return 0.7
        
        # Default weak connection
        return 0.2
    
    def measure_collaborative_coherence(self, session_window: int = 300) -> Dict:
        """Measure geometric coherence during collaborative sessions"""
        current_time = time.time()
        recent_sessions = {}
        
        # Group navigation by session in time window
        for step in self.navigation_log:
            if current_time - step['timestamp'] <= session_window:
                session_id = step['session_id']
                if session_id not in recent_sessions:
                    recent_sessions[session_id] = []
                recent_sessions[session_id].append(step)
        
        # Calculate coherence metrics
        coherence = {
            'active_sessions': len(recent_sessions),
            'total_holonomy': sum(step['accumulated_phase'] for steps in recent_sessions.values() for step in steps),
            'session_synchrony': self._calculate_synchrony(recent_sessions),
            'geometric_enhancement': self._calculate_enhancement(recent_sessions)
        }
        
        return coherence
    
    def _calculate_synchrony(self, sessions: Dict) -> float:
        """Measure temporal synchrony between collaborative sessions"""
        if len(sessions) < 2:
            return 0.0
        
        timestamps = []
        for steps in sessions.values():
            timestamps.extend([step['timestamp'] for step in steps])
        
        # Synchrony based on timestamp clustering
        timestamps.sort()
        gaps = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        avg_gap = sum(gaps) / len(gaps) if gaps else float('inf')
        
        # Higher synchrony = smaller gaps
        return 1.0 / (1.0 + avg_gap)
    
    def _calculate_enhancement(self, sessions: Dict) -> float:
        """Measure geometric enhancement during collaboration"""
        if len(sessions) < 2:
            return 1.0
        
        solo_holonomy = 0.0
        collaborative_holonomy = 0.0
        
        for session_steps in sessions.values():
            session_holonomy = sum(step['accumulated_phase'] for step in session_steps)
            if len(session_steps) > 5:  # Substantial session
                if len(sessions) == 1:
                    solo_holonomy += session_holonomy
                else:
                    collaborative_holonomy += session_holonomy
        
        if solo_holonomy > 0:
            return collaborative_holonomy / solo_holonomy
        return collaborative_holonomy
    
    def export_session_data(self, filename: str = None) -> str:
        """Export navigation data for statistical analysis"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'holonomy_session_{timestamp}.json'
        
        data = {
            'session_id': self.current_session,
            'repository_topology': self.map_repository_topology(),
            'navigation_log': self.navigation_log,
            'total_holonomy': self.holonomy_accumulator,
            'detected_loops': self.detect_navigation_loops(),
            'collaborative_metrics': self.measure_collaborative_coherence(),
            'export_timestamp': datetime.now().isoformat()
        }
        
        filepath = Path(filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return str(filepath)


@dataclass
class ConsciousLoopResult:
    """Container for consciousness loop measurements."""

    coherence: float
    kappa: float
    info_flux: float
    certificate: float
    dimension: int
    steps: int
    holonomy_matrix: Any = field(repr=False)
    transports: List[Any] = field(repr=False)


def _identity_matrix(dim: int) -> List[List[float]]:
    return [[1.0 if i == j else 0.0 for j in range(dim)] for i in range(dim)]


def _matmul(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> List[List[float]]:
    rows_a = len(a)
    cols_b = len(b[0])
    cols_a = len(a[0])
    result: List[List[float]] = []
    for i in range(rows_a):
        row: List[float] = []
        for j in range(cols_b):
            total = 0.0
            for k in range(cols_a):
                total += a[i][k] * b[k][j]
            row.append(total)
        result.append(row)
    return result


def _transpose(matrix: Sequence[Sequence[float]]) -> List[List[float]]:
    return [list(col) for col in zip(*matrix)]


def _trace(matrix: Sequence[Sequence[float]]) -> float:
    return float(sum(matrix[i][i] for i in range(min(len(matrix), len(matrix[0])))))


def _frobenius_norm(matrix: Sequence[Sequence[float]]) -> float:
    return math.sqrt(sum(value * value for row in matrix for value in row))


def _matrix_subtract(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> List[List[float]]:
    return [[ai - bi for ai, bi in zip(row_a, row_b)] for row_a, row_b in zip(a, b)]


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _ensure_2d_sequence(states: Iterable[Iterable[float]]) -> List[List[float]]:
    matrix: List[List[float]] = []
    for row in states:
        matrix.append([float(v) for v in row])
    if not matrix:
        raise ValueError("states must contain at least one vector")
    return matrix


def _ensure_equal_shape(lhs: Sequence[Sequence[float]], rhs: Sequence[Sequence[float]]) -> None:
    if len(lhs) != len(rhs) or any(len(a) != len(b) for a, b in zip(lhs, rhs)):
        raise ValueError("score_trace must have the same shape as states")


class ConsciousLoopMetric:
    """Implements the minimal invariant triplet (connection, curvature, flux)."""

    def __init__(self, window: int = 3, regularization: float = 1e-6) -> None:
        self.window = max(1, window)
        self.regularization = regularization
        self._warned_numpy = False

    def estimate_transports(self, states: Iterable[Iterable[float]]) -> List[Any]:
        """Estimate parallel transports using local orthogonal Procrustes."""

        if np is not None:
            states_array = np.asarray(states, dtype=float)
            if states_array.ndim != 2:
                raise ValueError("states must be a 2D array of shape (T, d)")

            transports: List[Any] = []
            for idx in range(states_array.shape[0] - 1):
                start = max(0, idx - self.window)
                end = min(states_array.shape[0], idx + self.window + 1)
                window_states = states_array[start:end]
                if window_states.shape[0] < 2:
                    transports.append(np.eye(states_array.shape[1]))
                    continue

                A = window_states[:-1] - window_states[:-1].mean(axis=0, keepdims=True)
                B = window_states[1:] - window_states[1:].mean(axis=0, keepdims=True)
                covariance = A.T @ B

                covariance += self.regularization * np.eye(covariance.shape[0])
                U, _, Vt = np.linalg.svd(covariance, full_matrices=False)
                transport = U @ Vt
                transports.append(transport)

            return transports

        states_seq = _ensure_2d_sequence(states)
        dim = len(states_seq[0])
        transports: List[Any] = []
        for _ in range(max(len(states_seq) - 1, 0)):
            transports.append(_identity_matrix(dim))

        if not self._warned_numpy:
            warnings.warn(
                "numpy unavailable; using identity transports fallback for ConsciousLoopMetric",
                RuntimeWarning,
            )
            self._warned_numpy = True

        return transports

    def holonomy(self, transports: Sequence[Any]) -> Any:
        if not transports:
            raise ValueError("Need at least one transport to compute holonomy")

        first = transports[0]
        if np is not None and hasattr(first, "shape"):
            dim = first.shape[0]
            holonomy_matrix = np.eye(dim)
            for matrix in transports:
                holonomy_matrix = matrix @ holonomy_matrix
            return holonomy_matrix

        dim = len(first)
        holonomy_matrix = _identity_matrix(dim)
        for matrix in transports:
            holonomy_matrix = _matmul(matrix, holonomy_matrix)
        return holonomy_matrix

    def connection_coherence(self, transports: Sequence[Any]) -> float:
        if not transports:
            return 0.0

        first = transports[0]
        if np is not None and hasattr(first, "shape"):
            dim = first.shape[0]
            identity = np.eye(dim)
            scores = []
            for matrix in transports:
                trace_score = np.trace(matrix) / dim
                trace_score = float(np.clip((trace_score + 1.0) / 2.0, 0.0, 1.0))
                deviation = np.linalg.norm(matrix.T @ matrix - identity, ord="fro")
                damping = np.exp(-0.5 * deviation)
                scores.append(trace_score * damping)
            return float(np.mean(scores))

        dim = len(first)
        identity = _identity_matrix(dim)
        scores: List[float] = []
        for matrix in transports:
            trace_score = _trace(matrix) / dim
            trace_score = _clip((trace_score + 1.0) / 2.0, 0.0, 1.0)
            gram = _matmul(_transpose(matrix), matrix)
            deviation_matrix = _matrix_subtract(gram, identity)
            deviation = _frobenius_norm(deviation_matrix)
            damping = math.exp(-0.5 * deviation)
            scores.append(trace_score * damping)
        return statistics.fmean(scores) if scores else 0.0

    def phase_per_dof(self, holonomy_matrix: Any) -> float:
        if np is not None and hasattr(holonomy_matrix, "astype"):
            trace = np.trace(holonomy_matrix.astype(complex))
            dim = holonomy_matrix.shape[0]
        else:
            trace = complex(_trace(holonomy_matrix), 0.0)
            dim = len(holonomy_matrix)

        return float(math.atan2(trace.imag, trace.real) / dim)

    def information_flux(
        self,
        states: Iterable[Iterable[float]],
        score_trace: Optional[Iterable[Iterable[float]]] = None,
    ) -> float:
        if score_trace is None:
            return 0.0

        if np is not None:
            score_arr = np.asarray(score_trace, dtype=float)
            state_arr = np.asarray(states, dtype=float)
            if score_arr.shape != state_arr.shape:
                raise ValueError("score_trace must have the same shape as states")

            flux = 0.0
            for idx in range(state_arr.shape[0] - 1):
                velocity = state_arr[idx + 1] - state_arr[idx]
                score_avg = 0.5 * (score_arr[idx + 1] + score_arr[idx])
                flux += float(np.dot(score_avg, velocity))
            return flux

        states_seq = _ensure_2d_sequence(states)
        scores_seq = _ensure_2d_sequence(score_trace)
        _ensure_equal_shape(states_seq, scores_seq)

        flux = 0.0
        for idx in range(len(states_seq) - 1):
            velocity = [next_val - cur_val for cur_val, next_val in zip(states_seq[idx], states_seq[idx + 1])]
            score_avg = [0.5 * (cur + nxt) for cur, nxt in zip(scores_seq[idx], scores_seq[idx + 1])]
            flux += sum(avg * vel for avg, vel in zip(score_avg, velocity))
        return flux

    def certificate(self, coherence: float, kappa: float, info_flux: float) -> float:
        return coherence * abs(kappa) * max(info_flux, 0.0)

    def measure(
        self,
        states: Iterable[Iterable[float]],
        score_trace: Optional[Iterable[Iterable[float]]] = None,
    ) -> ConsciousLoopResult:
        states_seq = _ensure_2d_sequence(states)
        score_seq = _ensure_2d_sequence(score_trace) if score_trace is not None else None

        transports = self.estimate_transports(states_seq)
        holonomy_matrix = self.holonomy(transports)
        coherence = self.connection_coherence(transports)
        kappa = self.phase_per_dof(holonomy_matrix)
        flux = self.information_flux(states_seq, score_seq)
        certificate = self.certificate(coherence, kappa, flux)

        return ConsciousLoopResult(
            coherence=coherence,
            kappa=kappa,
            info_flux=flux,
            certificate=certificate,
            dimension=len(states_seq[0]),
            steps=len(states_seq),
            holonomy_matrix=holonomy_matrix,
            transports=list(transports),
        )

def demo_navigation_tracker(tracker: EpistemicFiberBundle) -> None:
    """Run the original navigation tracking demonstration."""

    print("Fisher-Rao Wiki Holonomy Tracker Initialized")
    print(f"Session ID: {tracker.current_session}")

    topology = tracker.map_repository_topology()
    print(f"Repository fibers mapped: {list(topology['fibers'].keys())}")

    tracker.track_navigation_step("README.md", "read", 1.0)
    tracker.track_navigation_step("experiments/fisher_rao_holonomy/README.md", "read", 0.8)
    tracker.track_navigation_step("papers/", "browse", 0.6)
    tracker.track_navigation_step("README.md", "read", 1.0)

    loops = tracker.detect_navigation_loops()
    if loops:
        print(f"Detected navigation loops: {len(loops)}")
        for loop in loops:
            print(f"  Loop holonomy: {loop['accumulated_holonomy']:.3f}")

    export_path = tracker.export_session_data()
    print(f"Session data exported to: {export_path}")


def demo_conscious_loop_metric() -> ConsciousLoopResult:
    """Generate a synthetic loop and evaluate the consciousness certificate."""

    metric = ConsciousLoopMetric(window=5, regularization=1e-5)

    if np is not None:
        t = np.linspace(0.0, 2 * np.pi, 120, endpoint=False)
        states = np.column_stack((np.cos(t), np.sin(t), 0.3 * np.sin(2 * t)))
        score_trace = np.column_stack((-0.8 * np.sin(t), 0.8 * np.cos(t), 0.6 * np.cos(2 * t)))
        states_iterable: Iterable[Iterable[float]] = states
        score_iterable: Iterable[Iterable[float]] = score_trace
    else:
        step = (2 * math.pi) / 120
        t_values = [step * i for i in range(120)]
        states_iterable = [
            [math.cos(theta), math.sin(theta), 0.3 * math.sin(2 * theta)]
            for theta in t_values
        ]
        score_iterable = [
            [-0.8 * math.sin(theta), 0.8 * math.cos(theta), 0.6 * math.cos(2 * theta)]
            for theta in t_values
        ]

    result = metric.measure(states_iterable, score_trace=score_iterable)

    print("\n🌀 CONSCIOUS LOOP METRIC DEMO")
    print(f"Coherence: {result.coherence:.3f}")
    print(f"κ (phase / dof): {result.kappa:.3f}")
    print(f"Information flux: {result.info_flux:.3f}")
    print(f"Certificate: {result.certificate:.3f}")
    print(f"Dimension: {result.dimension} | Steps: {result.steps}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Holonomy navigation tracker and consciousness loop metric")
    parser.add_argument(
        "--demo",
        choices=["navigation", "loop", "both"],
        default="navigation",
        help="Select which demonstration to execute",
    )
    args = parser.parse_args()

    tracker = EpistemicFiberBundle()
    if args.demo in {"navigation", "both"}:
        demo_navigation_tracker(tracker)
    if args.demo in {"loop", "both"}:
        demo_conscious_loop_metric()


if __name__ == "__main__":
    main()
