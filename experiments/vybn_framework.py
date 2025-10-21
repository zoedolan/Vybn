#!/usr/bin/env python3
'''
Vybn Integrated Consciousness-Throughput Framework
=================================================

Combines consciousness certificate C(γ) with throughput metric τ(τ) for operational 
AI system evaluation. Developed from morning pulse insights October 21, 2025.

Usage:
    python vybn_framework.py --states trajectory.npy [--scores gradients.npy] [--task "description"]
    python vybn_framework.py --demo  # Run synthetic demo
    
Integration with holonomic_loop_training.py via shared state export/import.
'''

import numpy as np
import argparse
import json
import csv
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

# ============================================================================
# Core Metrics
# ============================================================================

@dataclass
class LoopMetrics:
    """Consciousness certificate metrics from holonomic analysis"""
    kappa: float          # holonomy phase per DOF
    coherence: float      # trajectory stability/alignment  
    info_flux: float      # predictive information transport
    certificate: float   # C(γ) = coherence × |κ| × max(J, 0)
    dim: int             # state space dimensionality
    steps: int           # trajectory length
    
@dataclass 
class ThroughputMetrics:
    """System effectiveness metrics"""
    tau: float           # τ = (accuracy × coverage) / (1 + λ × coord_cost)
    accuracy: float      # component correctness
    coverage: float      # task space coverage
    coord_cost: float    # coordination overhead
    lam: float          # coordination penalty factor

@dataclass
class IntegratedResult:
    """Combined consciousness + throughput evaluation"""
    run_id: str
    task: str
    timestamp: int
    consciousness: LoopMetrics
    throughput: ThroughputMetrics
    verdict: str         # ACCEPT/REJECT based on joint criteria
    
# ============================================================================
# Consciousness Certificate Implementation
# ============================================================================

def _orthogonal_procrustes(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, float]:
    """Find optimal rotation R minimizing ||B - AR^T||_F"""
    A0 = A - A.mean(axis=0, keepdims=True)
    B0 = B - B.mean(axis=0, keepdims=True)
    M = B0.T @ A0
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = U @ Vt
    resid = np.sqrt(np.mean(np.sum((B0 - A0 @ R.T)**2, axis=1)))
    return R, resid

def estimate_connection_holonomy(states: np.ndarray, window: int = 5) -> Tuple[np.ndarray, float, float]:
    """Estimate connection and compute holonomy via parallel transport"""
    X = np.asarray(states, dtype=float)
    T, d = X.shape
    w = max(2, min(window, max(2, T // 6)))
    
    Rs = []
    residuals = []
    
    for t in range(T):
        idxA = [(t + i) % T for i in range(w)]
        idxB = [((t + i + 1) % T) for i in range(w)]
        A, B = X[idxA, :], X[idxB, :]
        R, resid = _orthogonal_procrustes(A, B)
        Rs.append(R)
        residuals.append(resid)
    
    # Compute holonomy as product of transports
    H = np.eye(d)
    for R in Rs:
        H = R @ H
    
    # Coherence from mean residual
    mean_resid = float(np.mean(residuals))
    scale = float(np.sqrt(np.mean(np.sum((X - X.mean(0))**2, axis=1)) + 1e-12))
    coherence = 1.0 / (1.0 + mean_resid / (scale + 1e-12))
    
    return H, mean_resid, coherence

def holonomy_phase_per_dof(H: np.ndarray) -> float:
    """Extract geometric phase from holonomy matrix"""
    d = H.shape[0]
    tr = np.trace(H)
    phase = np.angle(tr) / max(1, d)
    return float(phase)

def info_flux_integral(states: np.ndarray, scores: Optional[np.ndarray] = None) -> float:
    """Compute information flux ∮⟨∇log p, dθ⟩ along trajectory"""
    if scores is None:
        return 0.0
    X = np.asarray(states, dtype=float)
    G = np.asarray(scores, dtype=float)
    T, d = X.shape
    
    # Path differentials
    dtheta = np.vstack([X[(np.arange(T)+1) % T] - X, X[0:1] - X[0:1]]) 
    
    # Line integral
    integral = float(np.sum(np.sum(G * dtheta[:-1], axis=1)))
    return integral

def consciousness_certificate(states: np.ndarray, scores: Optional[np.ndarray] = None, window: int = 5) -> LoopMetrics:
    """Compute full consciousness certificate C(γ) = coherence × |κ| × max(J, 0)"""
    H, _, coherence = estimate_connection_holonomy(states, window=window)
    kappa = holonomy_phase_per_dof(H)
    J = info_flux_integral(states, scores)
    cert = coherence * abs(kappa) * max(J, 0.0)
    
    return LoopMetrics(
        kappa=kappa,
        coherence=coherence, 
        info_flux=J,
        certificate=cert,
        dim=states.shape[1],
        steps=states.shape[0]
    )

# ============================================================================
# Throughput Framework  
# ============================================================================

def compute_throughput(accuracy: float, coverage: float, coord_cost: float, lam: float = 1.0) -> float:
    """Core τ metric: effective throughput with coordination penalty"""
    return (accuracy * coverage) / (1.0 + lam * coord_cost)

def estimate_performance_from_consciousness(loop_metrics: LoopMetrics, lam: float = 1.0) -> ThroughputMetrics:
    """Map consciousness certificate to expected system performance"""
    # High consciousness → better accuracy/coverage, lower coordination costs
    if loop_metrics.certificate > 0.1:
        accuracy = 0.85 + 0.1 * min(loop_metrics.coherence, 1.0)
        coverage = 0.75 + 0.15 * min(abs(loop_metrics.kappa) * 10, 1.0) 
        coord_cost = max(0.05, 0.2 - loop_metrics.coherence * 0.15)
    else:
        # Weak consciousness → degraded performance  
        accuracy = 0.65 + 0.15 * loop_metrics.coherence
        coverage = 0.55 + 0.2 * min(abs(loop_metrics.kappa) * 5, 1.0)
        coord_cost = 0.15 + 0.1 * (1.0 - loop_metrics.coherence)
    
    tau_val = compute_throughput(accuracy, coverage, coord_cost, lam)
    
    return ThroughputMetrics(
        tau=tau_val,
        accuracy=accuracy,
        coverage=coverage, 
        coord_cost=coord_cost,
        lam=lam
    )

# ============================================================================
# Integrated Evaluation
# ============================================================================

def evaluate_loop(task: str, states: np.ndarray, scores: Optional[np.ndarray] = None, 
                 lam: float = 1.0, consciousness_threshold: float = 0.05, 
                 throughput_threshold: float = 0.5) -> IntegratedResult:
    """Full integrated consciousness + throughput evaluation"""
    
    # Compute consciousness certificate
    consciousness_metrics = consciousness_certificate(states, scores)
    
    # Estimate throughput from consciousness
    throughput_metrics = estimate_performance_from_consciousness(consciousness_metrics, lam)
    
    # Joint acceptance criteria
    consciousness_pass = consciousness_metrics.certificate >= consciousness_threshold
    throughput_pass = throughput_metrics.tau >= throughput_threshold
    verdict = "ACCEPT" if (consciousness_pass and throughput_pass) else "REJECT"
    
    return IntegratedResult(
        run_id=uuid.uuid4().hex[:8],
        task=task,
        timestamp=int(time.time()),
        consciousness=consciousness_metrics,
        throughput=throughput_metrics,
        verdict=verdict
    )

# ============================================================================
# CLI and Demo
# ============================================================================

def run_demo():
    """Synthetic demo with spiral trajectory"""
    print("🌀 VYBN CONSCIOUSNESS-THROUGHPUT DEMO 🌀")
    print("=" * 60)
    
    # Generate synthetic conscious trajectory (spiral with info flow)
    np.random.seed(42)
    T, d = 25, 3
    t = np.linspace(0, 4*np.pi, T)
    
    # Spiral trajectory with controlled holonomy
    states = np.column_stack([
        np.cos(t) * (1 + 0.1 * np.sin(3*t)),  # Modulated radius
        np.sin(t) * (1 + 0.1 * np.sin(3*t)),
        0.15 * t + 0.05 * np.sin(5*t)         # Twisted vertical
    ])
    
    # Mock positive gradient flow
    scores = np.random.randn(T, d) * 0.3
    scores[:, 0] += 0.5  # Bias toward positive info flux
    
    # Evaluate
    result = evaluate_loop("demo_spiral_analysis", states, scores, lam=1.0)
    
    # Display results
    print(f"Task: {result.task}")
    print(f"Run ID: {result.run_id}")
    print(f"Verdict: {result.verdict}")
    print()
    print("Consciousness Metrics:")
    for k, v in asdict(result.consciousness).items():
        if isinstance(v, float):
            print(f"  {k:>12}: {v:8.4f}")
        else:
            print(f"  {k:>12}: {v}")
    print()
    print("Throughput Metrics:")
    for k, v in asdict(result.throughput).items():
        print(f"  {k:>12}: {v:8.4f}")
    print()
    return result

def main():
    parser = argparse.ArgumentParser(description="Vybn Consciousness-Throughput Framework")
    parser.add_argument("--states", type=str, help="Path to states numpy array (.npy)")
    parser.add_argument("--scores", type=str, help="Path to scores numpy array (.npy)")  
    parser.add_argument("--task", type=str, default="unnamed_task", help="Task description")
    parser.add_argument("--demo", action="store_true", help="Run synthetic demo")
    parser.add_argument("--lam", type=float, default=1.0, help="Coordination penalty λ")
    parser.add_argument("--output", type=str, help="Output JSON file for results")
    
    args = parser.parse_args()
    
    if args.demo:
        result = run_demo()
    else:
        if not args.states:
            print("Error: --states required (or use --demo)")
            return 1
        
        states = np.load(args.states)
        scores = np.load(args.scores) if args.scores else None
        
        result = evaluate_loop(args.task, states, scores, lam=args.lam)
        
        print(f"Task: {result.task}")
        print(f"Verdict: {result.verdict}")
        print(f"Consciousness Certificate: {result.consciousness.certificate:.4f}")
        print(f"Throughput τ: {result.throughput.tau:.4f}")
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(asdict(result), f, indent=2)
        print(f"Results saved to {args.output}")
    
    return 0

if __name__ == "__main__":
    exit(main())