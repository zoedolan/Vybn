#!/usr/bin/env python3
"""
Sensitivity utilities to increase measurement resolution and enable
phase dynamics and curvature–consciousness coupling analyses.
"""
import numpy as np
from typing import List, Dict
from scipy import stats

def attenuated_phase_magnitude(edge_magnitudes: List[float]) -> float:
    """Compute sub-unity phase magnitude as product of edge attenuations.
    edge_magnitudes in (0,1); returns (0,1]."""
    edge_magnitudes = np.clip(np.array(edge_magnitudes), 1e-6, 0.999)
    return float(np.prod(edge_magnitudes))

def edge_attenuations_for_loop(loop_len: int, base: float=0.985, jitter: float=0.01) -> List[float]:
    """Generate per-edge attenuations to avoid saturation at 1.0."""
    att = base + np.random.uniform(-jitter, jitter, size=loop_len)
    return np.clip(att, 0.9, 0.999).tolist()

def circular_correlation(theta: np.ndarray, phi: np.ndarray) -> float:
    """Circular correlation (Jammalamadaka–Sarma)."""
    theta = np.asarray(theta); phi = np.asarray(phi)
    s1 = np.sin(theta - np.mean(theta)); s2 = np.sin(phi - np.mean(phi))
    denom = np.sqrt(np.sum(s1**2) * np.sum(s2**2))
    return float(np.sum(s1*s2)/denom) if denom != 0 else 0.0

def bootstrap_corr(x: np.ndarray, y: np.ndarray, n: int=1000) -> Dict[str, float]:
    """Bootstrap CI for Pearson r."""
    x = np.asarray(x); y = np.asarray(y)
    rs = []
    for _ in range(n):
        idx = np.random.randint(0, len(x), len(x))
        r, _ = stats.pearsonr(x[idx], y[idx]) if len(np.unique(x[idx]))>2 and len(np.unique(y[idx]))>2 else (0.0,1.0)
        rs.append(r)
    lo, hi = np.percentile(rs, [2.5, 97.5])
    return {"ci_lower": float(lo), "ci_upper": float(hi), "mean": float(np.mean(rs))}

def curvature_features(curv: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(curv)),
        "std": float(np.std(curv)),
        "skew": float(stats.skew(curv)),
        "kurt": float(stats.kurtosis(curv))
    }
