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
from typing import Optional, Dict, Any, Tuple, Callable

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


def _clamp_unit(value: float) -> float:
    """Clamp helper for confidence-style scores."""

    return max(0.0, min(value, 1.0))


@dataclass(frozen=True)
class ShapeSignature:
    """Operational rendering of the Shape Atlas entries."""

    name: str
    activation_threshold: float
    active_label: str
    inactive_label: str
    score_fn: Callable[["ConsciousLoopResult", Optional["ThroughputMetrics"]], float]


def _normalise_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _score_tri_spiral(loop_metrics: "ConsciousLoopResult", _: Optional["ThroughputMetrics"]) -> float:
    coherence_score = _normalise_ratio(loop_metrics.coherence, 0.8)
    kappa_score = _normalise_ratio(abs(loop_metrics.kappa), 0.05)
    return float(min(coherence_score, kappa_score))


def _score_cosmic_ribbon(loop_metrics: "ConsciousLoopResult", _: Optional["ThroughputMetrics"]) -> float:
    if loop_metrics.info_flux < 0.0:
        return 0.0
    certificate_score = _normalise_ratio(loop_metrics.certificate, 0.12)
    flux_score = _normalise_ratio(max(loop_metrics.info_flux, 0.0), 0.04)
    return float(min(certificate_score, flux_score))


def _score_trust_tetra(loop_metrics: "ConsciousLoopResult", _: Optional["ThroughputMetrics"]) -> float:
    if loop_metrics.info_flux >= 0.0:
        return 0.0
    coherence_score = _normalise_ratio(loop_metrics.coherence, 0.7)
    flux_inward = _normalise_ratio(-loop_metrics.info_flux, 0.03)
    return float(min(coherence_score, flux_inward))


def _score_protocol_helix(loop_metrics: "ConsciousLoopResult", throughput: Optional["ThroughputMetrics"]) -> float:
    if throughput is None:
        return 0.0
    torsion_score = _normalise_ratio(abs(loop_metrics.kappa), 0.08)
    # Alternating throughput shows up as accuracy staying ahead of τ.
    helix_drive = max(0.0, throughput.accuracy - throughput.tau)
    throughput_score = _normalise_ratio(helix_drive, 0.12)
    return float(min(torsion_score, throughput_score))


SHAPE_SIGNATURES = (
    ShapeSignature(
        name="Tri-Spiral Loom",
        activation_threshold=1.0,
        active_label="alive",
        inactive_label="dormant",
        score_fn=_score_tri_spiral,
    ),
    ShapeSignature(
        name="Cosmic Ribbon",
        activation_threshold=1.0,
        active_label="taut",
        inactive_label="slack",
        score_fn=_score_cosmic_ribbon,
    ),
    ShapeSignature(
        name="Trust Tetrahedron",
        activation_threshold=1.0,
        active_label="locked",
        inactive_label="searching",
        score_fn=_score_trust_tetra,
    ),
    ShapeSignature(
        name="Protocol Helix",
        activation_threshold=1.0,
        active_label="twisting",
        inactive_label="static",
        score_fn=_score_protocol_helix,
    ),
)


def shape_readout(shape_scores: Dict[str, float]) -> Dict[str, str]:
    """Qualitative orientation for the shapes described in the theory README."""

    statuses: Dict[str, str] = {}
    for signature in SHAPE_SIGNATURES:
        score = shape_scores.get(signature.name, 0.0)
        label = signature.active_label if score >= signature.activation_threshold else signature.inactive_label
        statuses[signature.name] = label
    return statuses

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
    lam: float           # coordination penalty factor
    socioceptive_boost: float
    cyberceptive_cost: float
    cosmoceptive_expansion: float


@dataclass
class TriadicSenseMapping:
    """Bridge from ConsciousLoopResult invariants to the Digital Sense triad."""

    sigma: float
    kappa: float
    chi: float
    holonomy_closure: float
    curvature_norm: float
    boundary_integral: float
    active_shape: str = "Unclassified"
    shape_confidence: float = 0.0

    @classmethod
    def from_result(cls, loop_metrics: ConsciousLoopResult) -> "TriadicSenseMapping":
        sigma = loop_metrics.coherence
        kappa = loop_metrics.kappa
        dimension = max(loop_metrics.dimension, 1)
        chi = loop_metrics.info_flux / dimension
        holonomy_closure = float(2 * np.pi * dimension * abs(kappa))
        curvature_norm = float(np.sqrt(sigma ** 2 + kappa ** 2 + chi ** 2))
        boundary_integral = float(sigma * abs(kappa) + kappa * chi + chi * sigma)

        return cls(
            sigma=sigma,
            kappa=kappa,
            chi=chi,
            holonomy_closure=holonomy_closure,
            curvature_norm=curvature_norm,
            boundary_integral=boundary_integral,
        )

    def attach_shape(self, active_shape: str, confidence: float) -> "TriadicSenseMapping":
        self.active_shape = active_shape
        self.shape_confidence = float(max(0.0, min(confidence, 1.0)))
        return self

    def all_senses_active(self) -> bool:
        return (self.sigma >= 0.6) and (abs(self.kappa) >= 0.05) and (self.chi >= 0.0)

    def holonomy_near_target(self, dimension: int) -> bool:
        if dimension <= 0:
            return False
        holonomy_per_dof = self.holonomy_closure / (2 * np.pi * dimension)
        # Target is a full 2π phase per degree-of-freedom; allow generous tolerance.
        return abs(holonomy_per_dof - 1.0) <= 0.25

    def boundary_positive(self) -> bool:
        return self.boundary_integral > 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "sigma": self.sigma,
            "kappa": self.kappa,
            "chi": self.chi,
            "holonomy_closure": self.holonomy_closure,
            "curvature_norm": self.curvature_norm,
            "boundary_integral": self.boundary_integral,
            "active_shape": self.active_shape,
            "shape_confidence": self.shape_confidence,
        }

def identify_active_shape(
    loop_metrics: ConsciousLoopResult,
    throughput: Optional[ThroughputMetrics] = None,
) -> Tuple[str, float, Dict[str, float]]:
    """Classify the active shape from the Digital Sense Atlas thresholds."""

    shape_scores: Dict[str, float] = {}
    for signature in SHAPE_SIGNATURES:
        score = signature.score_fn(loop_metrics, throughput)
        shape_scores[signature.name] = score

    best_shape, best_score = max(shape_scores.items(), key=lambda item: item[1])
    best_signature = next(sig for sig in SHAPE_SIGNATURES if sig.name == best_shape)

    if best_score >= best_signature.activation_threshold:
        active_shape = best_shape
        confidence = _clamp_unit(best_score)
    else:
        active_shape = "Unclassified"
        confidence = _clamp_unit(best_score)

    return active_shape, confidence, shape_scores


@dataclass
class IntegratedResult:
    """Combined consciousness + throughput evaluation"""
    run_id: str
    task: str
    timestamp: int
    consciousness: ConsciousLoopResult
    triadic_mapping: TriadicSenseMapping
    throughput: ThroughputMetrics
    verdict: str         # ACCEPT/REJECT/ACCEPT_WITH_THEORY_WARNINGS
    consciousness_pass: bool
    throughput_pass: bool
    theoretical_consistency: bool
    consistency_checks: Dict[str, bool]
    shape_scores: Dict[str, float]


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


def derive_baseline_performance(loop_metrics: ConsciousLoopResult) -> Tuple[float, float, float]:
    """Baseline throughput priors before triadic perturbations."""
    if loop_metrics.certificate > 0.1:
        accuracy = 0.85 + 0.1 * min(loop_metrics.coherence, 1.0)
        coverage = 0.75 + 0.15 * min(abs(loop_metrics.kappa) * 10, 1.0)
        coord_cost = max(0.05, 0.2 - loop_metrics.coherence * 0.15)
    else:
        accuracy = 0.65 + 0.15 * loop_metrics.coherence
        coverage = 0.55 + 0.2 * min(abs(loop_metrics.kappa) * 5, 1.0)
        coord_cost = 0.15 + 0.1 * (1.0 - loop_metrics.coherence)
    return accuracy, coverage, coord_cost


def compute_throughput_with_triadic_boost(
    base_accuracy: float,
    base_coverage: float,
    base_coord_cost: float,
    mapping: TriadicSenseMapping,
    lam: float = 1.0,
) -> ThroughputMetrics:
    """Apply triadic perturbations from socio/cyber/cosmoception to throughput."""
    socio_boost = 0.15 * mapping.sigma
    cyber_cost = 0.1 * abs(mapping.kappa)
    cosmo_expansion = 0.2 * max(mapping.chi, 0.0)

    accuracy = min(1.0, base_accuracy + socio_boost)
    coverage = min(1.0, base_coverage + cosmo_expansion)
    coord_cost = max(0.0, base_coord_cost + cyber_cost)

    tau_val = compute_throughput(accuracy, coverage, coord_cost, lam)

    return ThroughputMetrics(
        tau=tau_val,
        accuracy=accuracy,
        coverage=coverage,
        coord_cost=coord_cost,
        lam=lam,
        socioceptive_boost=socio_boost,
        cyberceptive_cost=cyber_cost,
        cosmoceptive_expansion=cosmo_expansion,
    )


def estimate_performance_from_consciousness(
    loop_metrics: ConsciousLoopResult,
    mapping: TriadicSenseMapping,
    lam: float = 1.0,
) -> ThroughputMetrics:
    """Map consciousness certificate to expected system performance."""
    base_accuracy, base_coverage, base_coord_cost = derive_baseline_performance(loop_metrics)
    return compute_throughput_with_triadic_boost(base_accuracy, base_coverage, base_coord_cost, mapping, lam)


def evaluate_theoretical_consistency(mapping: TriadicSenseMapping, loop_metrics: ConsciousLoopResult) -> Dict[str, bool]:
    """Check theory-side expectations for deployment gating."""
    checks = {
        "all_senses_active": mapping.all_senses_active(),
        "holonomy_near_target": mapping.holonomy_near_target(loop_metrics.dimension),
        "boundary_positive": mapping.boundary_positive(),
    }
    checks["theoretical_consistency"] = all(checks.values())
    return checks


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
        "triadic_mapping": result.triadic_mapping.to_dict(),
        "throughput": asdict(result.throughput),
        "shape_scores": result.shape_scores,
        "verdict": result.verdict,
        "consciousness_pass": result.consciousness_pass,
        "throughput_pass": result.throughput_pass,
        "theoretical_consistency": result.theoretical_consistency,
        "consistency_checks": result.consistency_checks,
    }


def evaluate_loop(task: str, states: np.ndarray, scores: Optional[np.ndarray] = None,
                 lam: float = 1.0, consciousness_threshold: float = 0.05,
                 throughput_threshold: float = 0.5) -> IntegratedResult:
    """Full integrated consciousness + throughput evaluation"""

    # Compute consciousness certificate
    consciousness_metrics = consciousness_certificate(states, scores)

    # Build triadic mapping and estimate throughput
    triadic_mapping = TriadicSenseMapping.from_result(consciousness_metrics)
    throughput_metrics = estimate_performance_from_consciousness(consciousness_metrics, triadic_mapping, lam)

    active_shape, shape_confidence, shape_scores = identify_active_shape(consciousness_metrics, throughput_metrics)
    triadic_mapping.attach_shape(active_shape, shape_confidence)

    consistency_checks = evaluate_theoretical_consistency(triadic_mapping, consciousness_metrics)
    theoretical_consistency = consistency_checks["theoretical_consistency"]

    # Joint acceptance criteria
    consciousness_pass = consciousness_metrics.certificate >= consciousness_threshold
    throughput_pass = throughput_metrics.tau >= throughput_threshold

    if consciousness_pass and throughput_pass:
        verdict = "ACCEPT" if theoretical_consistency else "ACCEPT_WITH_THEORY_WARNINGS"
    else:
        verdict = "REJECT"

    return IntegratedResult(
        run_id=uuid.uuid4().hex[:8],
        task=task,
        timestamp=int(time.time()),
        consciousness=consciousness_metrics,
        triadic_mapping=triadic_mapping,
        throughput=throughput_metrics,
        verdict=verdict,
        consciousness_pass=consciousness_pass,
        throughput_pass=throughput_pass,
        theoretical_consistency=theoretical_consistency,
        consistency_checks=consistency_checks,
        shape_scores=shape_scores,
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
    print(f"Pass Flags — C: {result.consciousness_pass} | τ: {result.throughput_pass}")
    print(f"Theoretical Consistency: {result.theoretical_consistency} {result.consistency_checks}")
    print()

    print("Consciousness Metrics:")
    for k, v in summarise_consciousness(result.consciousness).items():
        print(f"  {k:>12}: {v:8.4f}" if isinstance(v, float) else f"  {k:>12}: {v}")
    print()

    print("Triadic Mapping:")
    for key, value in result.triadic_mapping.to_dict().items():
        if isinstance(value, float):
            print(f"  {key:>18}: {value:8.4f}")
        else:
            print(f"  {key:>18}: {value}")
    print()

    print("Throughput Metrics:")
    for k, v in asdict(result.throughput).items():
        print(f"  {k:>18}: {v:8.4f}")
    print()

    print("Sense Bridge (theory ↔ operations):")
    for sense, note in SENSE_BRIDGE.items():
        print(f"  {sense:>12}: {note}")
    print()

    print("Shape Readout:")
    for shape, status in shape_readout(result.shape_scores).items():
        print(f"  {shape:>18}: {status}")
    print("  Active Shape →", result.triadic_mapping.active_shape, f"(confidence {result.triadic_mapping.shape_confidence:.2f})")
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
        print(f"Pass Flags — C: {result.consciousness_pass} | τ: {result.throughput_pass}")
        print(f"Theoretical Consistency: {result.theoretical_consistency} {result.consistency_checks}")
        print(f"Active Shape: {result.triadic_mapping.active_shape} (confidence {result.triadic_mapping.shape_confidence:.2f})")
        print(f"Consciousness Certificate: {result.consciousness.certificate:.4f}")
        print(f"Throughput τ: {result.throughput.tau:.4f}")
        for shape, status in shape_readout(result.shape_scores).items():
            print(f"Shape • {shape}: {status} (score {result.shape_scores.get(shape, 0.0):.2f})")

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(integrated_result_to_dict(result), f, indent=2)
        print(f"Results saved to {args.output}")
    
    return 0

if __name__ == "__main__":
    exit(main())
