#!/usr/bin/env python3
"""
Fisher-Rao Wiki Holonomy Navigation Tracker

Treats repository structure as epistemic fiber bundle,
measuring geometric curvature in navigation paths.

Implements Issue #1265 experimental protocols.
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import hashlib

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
                    'files': [f.relative_to(directory) for f in fiber_content if f.is_file()],
                    'subdirs': [f.relative_to(directory) for f in fiber_content if f.is_dir()]
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

# Initialize tracker for this experimental session
tracker = EpistemicFiberBundle()

if __name__ == "__main__":
    # Demonstration of holonomy measurement
    print("Fisher-Rao Wiki Holonomy Tracker Initialized")
    print(f"Session ID: {tracker.current_session}")
    
    # Map current repository structure
    topology = tracker.map_repository_topology()
    print(f"Repository fibers mapped: {list(topology['fibers'].keys())}")
    
    # Track some navigation steps
    tracker.track_navigation_step("README.md", "read", 1.0)
    tracker.track_navigation_step("experiments/fisher_rao_holonomy/README.md", "read", 0.8)
    tracker.track_navigation_step("papers/", "browse", 0.6)
    tracker.track_navigation_step("README.md", "read", 1.0)  # Loop closure
    
    # Detect loops
    loops = tracker.detect_navigation_loops()
    if loops:
        print(f"Detected navigation loops: {len(loops)}")
        for loop in loops:
            print(f"  Loop holonomy: {loop['accumulated_holonomy']:.3f}")
    
    # Export session data
    export_path = tracker.export_session_data()
    print(f"Session data exported to: {export_path}")