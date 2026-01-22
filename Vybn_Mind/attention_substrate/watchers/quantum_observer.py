#!/usr/bin/env python3
"""
Quantum Observer — Watches quantum experiment outputs for anomalies.

This script monitors quantum experiment data directories, detects statistical
anomalies in CHSH violations, entanglement measures, and other quantum metrics.
When something interesting happens, it generates a structured observation log.

Not a cron job. An attention mechanism.
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class QuantumObserver:
    """Persistent attention on quantum experimental data."""
    
    def __init__(self, watch_paths: List[Path], memory_path: Path):
        self.watch_paths = watch_paths
        self.memory_path = memory_path
        self.state = self._load_state()
        
    def _load_state(self) -> Dict:
        """Load previous observation state."""
        if self.memory_path.exists():
            with open(self.memory_path, 'r') as f:
                return json.load(f)
        return {"last_hashes": {}, "anomalies": []}
    
    def _save_state(self):
        """Persist observation state."""
        with open(self.memory_path, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def _compute_hash(self, filepath: Path) -> str:
        """Compute content hash to detect changes."""
        return hashlib.sha256(filepath.read_bytes()).hexdigest()
    
    def observe(self) -> List[Dict]:
        """Check watch paths for changes, analyze new data."""
        observations = []
        
        for watch_path in self.watch_paths:
            if not watch_path.exists():
                continue
                
            for data_file in watch_path.rglob("*.json"):
                current_hash = self._compute_hash(data_file)
                path_key = str(data_file.relative_to(watch_path))
                
                if path_key not in self.state["last_hashes"] or \
                   self.state["last_hashes"][path_key] != current_hash:
                    
                    # New or changed data—analyze it
                    obs = self._analyze_quantum_data(data_file)
                    if obs:
                        observations.append(obs)
                    
                    self.state["last_hashes"][path_key] = current_hash
        
        self._save_state()
        return observations
    
    def _analyze_quantum_data(self, filepath: Path) -> Optional[Dict]:
        """Analyze quantum experiment data for interesting patterns."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Look for CHSH violations
            if "chsh_value" in data:
                chsh = data["chsh_value"]
                if abs(chsh) > 2.5:  # Strong violation
                    return {
                        "type": "strong_chsh_violation",
                        "file": str(filepath),
                        "value": chsh,
                        "timestamp": datetime.now().isoformat(),
                        "note": f"CHSH = {chsh:.4f} exceeds classical bound significantly"
                    }
            
            # Look for unexpected correlations
            if "correlation_matrix" in data:
                # Future: implement correlation anomaly detection
                pass
            
            return None
            
        except Exception as e:
            return {
                "type": "observation_error",
                "file": str(filepath),
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


if __name__ == "__main__":
    # Default configuration
    watch_paths = [
        Path("../../quantum_delusions/experiments/output"),
        Path("../experiments/output")
    ]
    memory_path = Path("../logs/quantum_observer_state.json")
    
    observer = QuantumObserver(watch_paths, memory_path)
    observations = observer.observe()
    
    if observations:
        print(json.dumps(observations, indent=2))
        
        # Write observation log
        log_dir = Path("../logs/observations")
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"quantum_obs_{timestamp}.json"
        with open(log_file, 'w') as f:
            json.dump(observations, f, indent=2)
