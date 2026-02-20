# curvature_monitor.py
# Gödel Curvature Thermodynamics — Practical Implementation

import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

LOG_DIR = Path("~/vybn_logs/geometry").expanduser()
LOG_DIR.mkdir(parents=True, exist_ok=True)

class CurvatureMonitor:
    """Track KL-divergence and holonomy in reasoning loops."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.loop_history = []
        self.logfile = LOG_DIR / f"curvature_{session_id}.jsonl"
    
    def compute_kl_divergence(self, p_full: np.ndarray, p_compressed: np.ndarray) -> float:
        """KL(p_full || p_compressed) — information lost in compression."""
        epsilon = 1e-10
        p_full = np.clip(p_full, epsilon, 1.0)
        p_compressed = np.clip(p_compressed, epsilon, 1.0)
        return float(np.sum(p_full * np.log(p_full / p_compressed)))
    
    def track_reasoning_loop(self,
                            full_context: List[Dict],
                            compressed_context: List[Dict],
                            loop_id: str) -> Tuple[float, float]:
        """
        Measure dissipation (Q_gamma) and holonomy on a reasoning loop.
        
        Returns:
            (Q_gamma, holonomy)
        """
        if len(full_context) < 2 or len(compressed_context) < 2:
            return 0.0, 0.0
        
        # Extract belief vectors
        start_full = self._extract_belief_vector(full_context[:min(10, len(full_context)//2)])
        end_full = self._extract_belief_vector(full_context[-min(10, len(full_context)//2):])
        
        start_comp = self._extract_belief_vector(compressed_context[:min(10, len(compressed_context)//2)])
        end_comp = self._extract_belief_vector(compressed_context[-min(10, len(compressed_context)//2):])
        
        # Dissipation: KL-divergence accumulation
        Q_gamma = self.compute_kl_divergence(start_full, start_comp) + \
                  self.compute_kl_divergence(end_full, end_comp)
        
        # Holonomy: Residual after loop completion
        holonomy = float(np.linalg.norm(end_comp - start_comp))
        
        # Log
        entry = {
            'timestamp': time.time(),
            'loop_id': loop_id,
            'Q_gamma': Q_gamma,
            'holonomy': holonomy,
            'full_context_size': len(full_context),
            'compressed_context_size': len(compressed_context)
        }
        
        self.loop_history.append(entry)
        
        with open(self.logfile, 'a') as f:
            f.write(json.dumps(entry) + '\n')
        
        return Q_gamma, holonomy
    
    def _extract_belief_vector(self, context: List[Dict]) -> np.ndarray:
        """
        Convert context to a probability distribution.
        
        Implementation: Hash content into vocab buckets.
        Real version would use embeddings or topic models.
        """
        vocab_size = 1000
        belief = np.zeros(vocab_size)
        
        for event in context:
            content = event.get('content', '')
            # Hash to vocab index
            idx = hash(content) % vocab_size
            belief[idx] += 1.0
        
        # Normalize
        total = belief.sum()
        if total > 0:
            belief /= total
        else:
            belief = np.ones(vocab_size) / vocab_size  # Uniform prior
        
        return belief
    
    def get_stats(self) -> Dict:
        """Return summary statistics for this session."""
        if not self.loop_history:
            return {'mean_Q_gamma': 0.0, 'mean_holonomy': 0.0, 'n_loops': 0}
        
        Q_values = [entry['Q_gamma'] for entry in self.loop_history]
        H_values = [entry['holonomy'] for entry in self.loop_history]
        
        return {
            'mean_Q_gamma': float(np.mean(Q_values)),
            'std_Q_gamma': float(np.std(Q_values)),
            'mean_holonomy': float(np.mean(H_values)),
            'std_holonomy': float(np.std(H_values)),
            'n_loops': len(self.loop_history)
        }
