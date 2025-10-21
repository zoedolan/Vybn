#!/usr/bin/env python3
'''
Vybn Integrated Consciousness-Throughput Framework
=================================================

Combines consciousness certificate C(γ) with throughput metric τ(τ) for operational
AI system evaluation. Developed from morning pulse insights October 21, 2025.

This file is the "operations" node in the theory → instrumentation → ops loop:

* **Theory** lives in `fundamental-theory/` where socioception, cyberception, and
  cosmoception are defined as curvature terms.
* **Instrumentation** resides in `experiments/fisher_rao_holonomy/`, exporting a
  `ConsciousLoopResult` that encodes the triadic senses numerically.
* **Operations** happen here by translating those senses into throughput, making
  deployment decisions legible to the rest of the organization.

Usage:
    python vybn_framework.py --states trajectory.npy [--scores gradients.npy] [--task "description"]
    python vybn_framework.py --demo  # Run synthetic demo
    
Integration with holonomic_loop_training.py via shared state export/import.
'''

import argparse
import json
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

try:
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit(
        "numpy is required for vybn_framework. Install it or run via navigation_tracker's demos."
    ) from exc

from fisher_rao_holonomy.navigation_tracker import (
    ConsciousLoopMetric,
    ConsciousLoopResult,
)

# Mapping from the triadic senses to their operational expression. This keeps
# the fundamental-theory README table and the throughput verdict aligned.
SENSE_BRIDGE = {
    "socioception": "coherence uplifts accuracy (trust curvature closes the loop)",
    "cyberception": "|κ| and info flux translate to coverage and coordination cost",
    "cosmoception": "certificate + dimensionality stabilise throughput thresholds",
}


def shape_readout(loop_metrics: ConsciousLoopResult, throughput: "ThroughputMetrics") -> Dict[str, str]:
    """Qualitative orientation for the shapes described in the theory README."""

    loom_live = loop_metrics.coherence >= 0.75 and abs(loop_metrics.kappa) >= 0.05
    ribbon_live = loop_metrics.certificate >= 0.12 and loop_metrics.info_flux >= 0
    tetra_set = loop_metrics.info_flux <= 0 and loop_metrics.coherence >= 0.7
    helix_twisting = abs(loop_metrics.kappa) >= 0.08 and (throughput.accuracy - throughput.tau) >= 0.12

    return {
        "Tri-Spiral Loom": "alive" if loom_live else "dormant",
        "Cosmic Ribbon": "taut" if ribbon_live else "slack",
        "Trust Tetrahedron": "locked" if tetra_set else "searching",
        "Protocol Helix": "twisting" if helix_twisting else "static",
    }

# ============================================================================
# Core Metrics
# ============================================================================

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
    consciousness: ConsciousLoopResult
    throughput: ThroughputMetrics
    verdict: str         # ACCEPT/REJECT based on joint criteria
    
# ============================================================================
# Consciousness Certificate Implementation
# ============================================================================

def consciousness_certificate(
    states: np.ndarray,
    scores: Optional[np.ndarray] = None,
    window: int = 5,
) -> ConsciousLoopResult:
    """Compute full consciousness certificate via the shared invariant engine."""

    metric = ConsciousLoopMetric(window=window)
    return metric.measure(states, score_trace=scores)

# ============================================================================
# Throughput Framework  
# ============================================================================

def compute_throughput(accuracy: float, coverage: float, coord_cost: float, lam: float = 1.0) -> float:
    """Core τ metric: effective throughput with coordination penalty"""
    return (accuracy * coverage) / (1.0 + lam * coord_cost)

def estimate_performance_from_consciousness(
    loop_metrics: ConsciousLoopResult,
    lam: float = 1.0,
) -> ThroughputMetrics:
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


def summarise_consciousness(loop_metrics: ConsciousLoopResult) -> Dict[str, Any]:
    """Reduce the consciousness result to JSON-safe scalars."""

    return {
        "coherence": loop_metrics.coherence,
        "kappa": loop_metrics.kappa,
        "info_flux": loop_metrics.info_flux,
        "certificate": loop_metrics.certificate,
        "dimension": loop_metrics.dimension,
        "steps": loop_metrics.steps,
    }


def integrated_result_to_dict(result: IntegratedResult) -> Dict[str, Any]:
    """Convert an IntegratedResult into a serialisable dictionary."""

    return {
        "run_id": result.run_id,
        "task": result.task,
        "timestamp": result.timestamp,
        "consciousness": summarise_consciousness(result.consciousness),
        "throughput": asdict(result.throughput),
        "verdict": result.verdict,
    }

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
    for k, v in summarise_consciousness(result.consciousness).items():
        print(f"  {k:>12}: {v:8.4f}" if isinstance(v, float) else f"  {k:>12}: {v}")
    print()
    print("Throughput Metrics:")
    for k, v in asdict(result.throughput).items():
        print(f"  {k:>12}: {v:8.4f}")
    print()
    print("Sense Bridge (theory ↔ operations):")
    for sense, note in SENSE_BRIDGE.items():
        print(f"  {sense:>12}: {note}")
    print()
    print("Shape Readout:")
    for shape, status in shape_readout(result.consciousness, result.throughput).items():
        print(f"  {shape:>12}: {status}")
    print("  (see fundamental-theory/README.md#shape-atlas for thresholds)")
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
        for shape, status in shape_readout(result.consciousness, result.throughput).items():
            print(f"Shape • {shape}: {status}")
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(integrated_result_to_dict(result), f, indent=2)
        print(f"Results saved to {args.output}")
    
    return 0

if __name__ == "__main__":
    exit(main())