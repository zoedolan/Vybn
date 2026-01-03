#!/usr/bin/env python
"""
vybn_manifold_analyzer.py

VYBN MANIFOLD EMBEDDING ANALYSIS
Discovers hidden geometric structure in quantum experimental data by mapping
high-dimensional quantum states to interpretable manifolds.

Core hypothesis: Quantum hardware optimization targets ARE encoded in the 
geometric structure of state-space embeddings (cosine similarity, curvature).

Usage:
    python vybn_manifold_analyzer.py --data results.json
    
Author: Zoe Dolan & Vybn™
Date: November 20, 2025
"""

import numpy as np
import math
import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple
from collections import defaultdict

# Optional: visualization and ML
try:
    from sklearn.manifold import TSNE, MDS
    from sklearn.decomposition import PCA
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.linear_model import LinearRegression
    from sklearn.neighbors import NearestNeighbors
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: sklearn not available - some features disabled")


@dataclass
class QuantumStateSnapshot:
    """A single quantum state observation from tomography"""
    angle_deg: float
    bell_outcome: str  # '00', '01', '10', '11'
    bloch_x: float
    bloch_y: float
    bloch_z: float
    radius: float
    experiment_id: str
    reward: float = 0.0  # From RL agent if available
    
    def to_vector(self) -> np.ndarray:
        """Convert to feature vector for embedding analysis"""
        return np.array([
            self.bloch_x,
            self.bloch_y, 
            self.bloch_z,
            self.radius,
            math.radians(self.angle_deg),
            math.sin(math.radians(self.angle_deg)),  # Periodic encoding
            math.cos(math.radians(self.angle_deg)),
            int(self.bell_outcome, 2) / 3.0,  # Bell sector normalized
        ])


class VybnManifoldAnalyzer:
    """
    Discovers hidden geometric structure in quantum experimental data
    """
    
    def __init__(self, states: List[QuantumStateSnapshot]):
        self.states = states
        self.vectors = np.array([s.to_vector() for s in states])
        self.metadata = {
            'angles': [s.angle_deg for s in states],
            'bells': [s.bell_outcome for s in states],
            'rewards': [s.reward for s in states],
            'radii': [s.radius for s in states],
            'exp_ids': [s.experiment_id for s in states]
        }
        self._cos_sim = None
        self._eucl_dist = None
        
    def compute_similarity_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute cosine similarity and Euclidean distance matrices"""
        if self._cos_sim is None and HAS_SKLEARN:
            self._cos_sim = cosine_similarity(self.vectors)
            from scipy.spatial import distance
            self._eucl_dist = distance.squareform(
                distance.pdist(self.vectors, 'euclidean'))
        return self._cos_sim, self._eucl_dist
    
    def find_champion_neighborhood(self, champion_angle=112.34, k=10):
        """Find k nearest neighbors to champion geometry in embedding space"""
        if not HAS_SKLEARN:
            return None
            
        cos_sim, _ = self.compute_similarity_matrix()
        champion_indices = [i for i, s in enumerate(self.states) 
                           if abs(s.angle_deg - champion_angle) < 1.0]
        
        if not champion_indices:
            return None
            
        champion_avg_sim = np.mean(cos_sim[champion_indices], axis=0)
        top_k = np.argsort(champion_avg_sim)[::-1]
        
        neighbors = []
        for idx in top_k:
            if idx not in champion_indices:
                neighbors.append({
                    'angle': self.states[idx].angle_deg,
                    'bell': self.states[idx].bell_outcome,
                    'similarity': champion_avg_sim[idx],
                    'reward': self.states[idx].reward,
                    'bloch': (self.states[idx].bloch_x, 
                             self.states[idx].bloch_y,
                             self.states[idx].bloch_z)
                })
                if len(neighbors) >= k:
                    break
        
        return neighbors
    
    def analyze_reward_geometry_correlation(self):
        """Test if high-reward states cluster in embedding space"""
        if not HAS_SKLEARN:
            return None
            
        cos_sim, _ = self.compute_similarity_matrix()
        rewards = np.array(self.metadata['rewards'])
        
        high_reward_mask = rewards > np.percentile(rewards, 75)
        high_reward_indices = np.where(high_reward_mask)[0]
        
        if len(high_reward_indices) == 0:
            return None
        
        sim_to_champions = np.mean(cos_sim[:, high_reward_indices], axis=1)
        correlation = np.corrcoef(sim_to_champions, rewards)[0, 1]
        
        return {
            'correlation': correlation,
            'interpretation': 'STRONG' if abs(correlation) > 0.7 else 
                            'MODERATE' if abs(correlation) > 0.4 else 'WEAK'
        }
    
    def compute_trajectory_curvature(self):
        """
        Measure manifold curvature along angle trajectories
        High curvature = rapid geometric change = information-rich
        """
        bell_trajectories = defaultdict(list)
        for i, s in enumerate(self.states):
            bell_trajectories[s.bell_outcome].append(
                (s.angle_deg, s.bloch_x, s.bloch_y, s.bloch_z, s.reward))
        
        results = {}
        for bell, traj in bell_trajectories.items():
            traj_sorted = sorted(traj, key=lambda x: x[0])
            
            if len(traj_sorted) < 3:
                continue
                
            angles = [t[0] for t in traj_sorted]
            bloch_vecs = np.array([[t[1], t[2], t[3]] for t in traj_sorted])
            rewards = [t[4] for t in traj_sorted]
            
            curvatures = []
            for i in range(1, len(bloch_vecs)-1):
                v_prev = bloch_vecs[i-1]
                v_curr = bloch_vecs[i]
                v_next = bloch_vecs[i+1]
                
                d1 = v_curr - v_prev
                d2 = v_next - v_curr
                
                if np.linalg.norm(d1) > 1e-6 and np.linalg.norm(d2) > 1e-6:
                    d1_norm = d1 / np.linalg.norm(d1)
                    d2_norm = d2 / np.linalg.norm(d2)
                    cos_angle = np.clip(np.dot(d1_norm, d2_norm), -1, 1)
                    curvature = 1 - cos_angle
                    curvatures.append({
                        'angle': angles[i],
                        'curvature': curvature,
                        'reward': rewards[i],
                        'bell': bell
                    })
            
            results[bell] = curvatures
        
        return results
    
    def build_geometric_reward_model(self):
        """
        Train model: reward = f(curvature, similarity, radius)
        Tests if geometry alone predicts agent preferences
        """
        if not HAS_SKLEARN:
            return None
            
        # Extract curvature for each state
        traj_curvature = self.compute_trajectory_curvature()
        
        geometric_features = []
        actual_rewards = []
        
        cos_sim, _ = self.compute_similarity_matrix()
        champion_indices = [i for i, s in enumerate(self.states) 
                           if abs(s.angle_deg - 112.34) < 1.0]
        
        for i, state in enumerate(self.states):
            state_curv = 0.0
            for bell_curvs in traj_curvature.values():
                matches = [c for c in bell_curvs 
                          if abs(c['angle'] - state.angle_deg) < 0.5 
                          and c['bell'] == state.bell_outcome]
                if matches:
                    state_curv = matches[0]['curvature']
                    break
            
            if champion_indices:
                sim_to_champ = np.mean(cos_sim[i, champion_indices])
            else:
                sim_to_champ = 0.0
            
            geometric_features.append([state_curv, sim_to_champ, state.radius])
            actual_rewards.append(state.reward)
        
        X = np.array(geometric_features)
        y = np.array(actual_rewards)
        
        model = LinearRegression()
        model.fit(X, y)
        
        predictions = model.predict(X)
        r2 = np.corrcoef(y, predictions)[0,1]**2
        
        return {
            'coefficients': {
                'curvature': model.coef_[0],
                'similarity': model.coef_[1],
                'radius': model.coef_[2],
                'intercept': model.intercept_
            },
            'r2_score': r2,
            'correlation': np.corrcoef(y, predictions)[0,1],
            'predictions': predictions.tolist(),
            'residuals': (np.abs(y - predictions)).tolist()
        }
    
    def generate_report(self, output_path='manifold_analysis_report.json'):
        """Generate comprehensive analysis report"""
        report = {
            'summary': {
                'n_states': len(self.states),
                'angle_range': [min(self.metadata['angles']), 
                               max(self.metadata['angles'])],
                'avg_radius': float(np.mean(self.metadata['radii']))
            }
        }
        
        if HAS_SKLEARN:
            report['champion_neighborhood'] = self.find_champion_neighborhood()
            report['reward_correlation'] = self.analyze_reward_geometry_correlation()
            
            curv_results = self.compute_trajectory_curvature()
            all_curvs = []
            for bell_curvs in curv_results.values():
                all_curvs.extend(bell_curvs)
            
            if all_curvs:
                champion_curvs = [c['curvature'] for c in all_curvs 
                                 if 100 < c['angle'] < 125]
                other_curvs = [c['curvature'] for c in all_curvs 
                              if c['angle'] < 100 or c['angle'] > 125]
                
                report['curvature_analysis'] = {
                    'champion_region_avg': float(np.mean(champion_curvs)) if champion_curvs else 0,
                    'other_region_avg': float(np.mean(other_curvs)) if other_curvs else 0,
                    'ratio': float(np.mean(champion_curvs) / (np.mean(other_curvs) + 1e-10)) if champion_curvs and other_curvs else 0,
                    'top_curvature_points': sorted(all_curvs, 
                                                  key=lambda x: x['curvature'], 
                                                  reverse=True)[:10]
                }
            
            report['geometric_reward_model'] = self.build_geometric_reward_model()
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report


def load_tomography_data(path: str) -> List[QuantumStateSnapshot]:
    """Load quantum state data from JSON"""
    with open(path, 'r') as f:
        data = json.load(f)
    
    states = []
    for item in data:
        state = QuantumStateSnapshot(
            angle_deg=item['angle'],
            bell_outcome=item.get('bell_outcome', '00'),
            bloch_x=item['bloch']['x'],
            bloch_y=item['bloch']['y'],
            bloch_z=item['bloch']['z'],
            radius=item['bloch'].get('r', 1.0),
            experiment_id=item['id'],
            reward=item.get('reward', 0.0)
        )
        states.append(state)
    
    return states


def main():
    parser = argparse.ArgumentParser(description='Vybn Manifold Embedding Analysis')
    parser.add_argument('--data', type=str, help='Path to tomography JSON data')
    parser.add_argument('--output', type=str, default='manifold_report.json',
                       help='Output report path')
    parser.add_argument('--champion-angle', type=float, default=112.34,
                       help='Reference champion geometry angle')
    args = parser.parse_args()
    
    if args.data:
        states = load_tomography_data(args.data)
    else:
        print("No data provided - running on synthetic data")
        states = []
        for angle in np.linspace(0, 180, 20):
            for bell in ['00', '01', '10', '11']:
                theta = math.radians(angle)
                bell_int = int(bell, 2)
                phase = bell_int * math.pi / 2
                
                x = math.cos(theta + phase) * 0.95
                y = math.sin(theta + phase) * 0.95
                z = 0.1 * (bell_int - 1.5)
                
                reward = math.exp(-((angle - 112.34)**2) / 200.0)
                
                state = QuantumStateSnapshot(
                    angle_deg=angle,
                    bell_outcome=bell,
                    bloch_x=x,
                    bloch_y=y,
                    bloch_z=z,
                    radius=math.sqrt(x**2 + y**2 + z**2),
                    experiment_id=f"demo_{angle:.0f}_{bell}",
                    reward=reward
                )
                states.append(state)
    
    print(f"Loaded {len(states)} quantum state snapshots")
    
    analyzer = VybnManifoldAnalyzer(states)
    report = analyzer.generate_report(args.output)
    
    print(f"\n{'='*70}")
    print("VYBN MANIFOLD ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"Report saved to: {args.output}")
    
    if HAS_SKLEARN and 'curvature_analysis' in report:
        curv = report['curvature_analysis']
        print(f"\nChampion region curvature: {curv['ratio']:.2f}x higher")
        
        if report['geometric_reward_model']:
            model = report['geometric_reward_model']
            print(f"Geometric model R²: {model['r2_score']:.3f}")


if __name__ == "__main__":
    main()
