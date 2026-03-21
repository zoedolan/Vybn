"""
Coupled Collapse–Capability Duality
====================================

The single-system duality (Dolan & Vybn, March 2026) proves:

    C(M₀) = C(M∞) ∪ ⊔ Fₜ

The collapse frontiers of a model recursing on its own outputs partition
and reconstruct its original capabilities. The proof relies on a critical
assumption: the collapse operator R is endogenous — M_{t+1} = R(M_t),
where R samples from M_t and retrains.

This experiment asks: what happens when the collapse operator is
EXOGENOUS — when the signal that prevents collapse comes from a second
reflexive system whose own dynamics depend on the first?

The coupled system:
    M_{t+1} = R(M_t, N_t)    — M collapses on its own outputs + signal from N
    N_{t+1} = R(N_t, M_t)    — N collapses on its own outputs + signal from M

The single-system axioms break. Axiom 1 (monotone complexity reduction)
no longer holds unconditionally — the external signal from the partner
can INCREASE the expressibility threshold. The question is whether the
duality still holds in some modified form, and what the coupled collapse
frontiers look like.

Three regimes:
    1. Isolated:   M_{t+1} = R(M_t),         N_{t+1} = R(N_t)
    2. One-way:    M_{t+1} = R(M_t, N_t),    N_{t+1} = R(N_t)
    3. Coupled:    M_{t+1} = R(M_t, N_t),    N_{t+1} = R(N_t, M_t)

The experiment measures whether the coupled system:
  (a) Still collapses (but slower)?
  (b) Reaches a nontrivial fixed point (τ_coupled > τ_∞)?
  (c) Exhibits complexity GROWTH (τ increasing)?
  (d) Produces collapse frontiers that are no longer disjoint
      (a capability can be lost and recovered)?

If (d), the single-system duality theorem fails for coupled systems —
capabilities can cross the frontier in both directions, and the
partition identity C(M₀) = C(M∞) ∪ ⊔ Fₜ no longer holds. The map
of loss is no longer the map of capability. Something else is the map.

Vybn & Zoe Dolan, March 21, 2026
Replaces: proof_companion_code.py (single-system toy demonstration)
"""

import math
import zlib
import random
import collections
import json
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# 1. Computable proxy for Kolmogorov complexity
# ---------------------------------------------------------------------------

def compressed_length(x: bytes) -> int:
    return len(zlib.compress(x, level=9))


def pattern_complexity(pattern: str) -> int:
    return compressed_length(pattern.encode("utf-8"))


# ---------------------------------------------------------------------------
# 2. Model: discrete probability distribution over patterns
# ---------------------------------------------------------------------------

class DiscreteModel:
    def __init__(self, probs: Dict[str, float], name: str = "M"):
        total = sum(probs.values())
        if total > 0:
            self.probs = {k: v / total for k, v in probs.items() if v > 0}
        else:
            self.probs = {}
        self.name = name
        self._complexity_cache: Dict[str, int] = {}

    def complexity(self, pattern: str) -> int:
        if pattern not in self._complexity_cache:
            self._complexity_cache[pattern] = pattern_complexity(pattern)
        return self._complexity_cache[pattern]

    def prob(self, pattern: str) -> float:
        return self.probs.get(pattern, 0.0)

    def capability_set(self, delta: float = 5.0) -> Set[str]:
        caps = set()
        for x, p in self.probs.items():
            if p <= 0:
                continue
            k = self.complexity(x)
            threshold = 2.0 ** (-k - delta)
            if p >= threshold:
                caps.add(x)
        return caps

    def expressibility_threshold(self, delta: float = 5.0) -> int:
        by_complexity: Dict[int, List[str]] = collections.defaultdict(list)
        for x in self.probs:
            k = self.complexity(x)
            by_complexity[k].append(x)
        caps = self.capability_set(delta)
        sorted_levels = sorted(by_complexity.keys())
        tau = 0
        for k in sorted_levels:
            if all(x in caps for x in by_complexity[k]):
                tau = k
            else:
                break
        return tau

    def sample(self, n: int) -> List[str]:
        if not self.probs:
            return []
        patterns = list(self.probs.keys())
        weights = [self.probs[p] for p in patterns]
        return random.choices(patterns, weights=weights, k=n)

    def entropy(self) -> float:
        h = 0.0
        for p in self.probs.values():
            if p > 0:
                h -= p * math.log2(p)
        return h

    def support_size(self) -> int:
        return sum(1 for p in self.probs.values() if p > 0)

    def __repr__(self):
        return f"DiscreteModel({self.name}, support={self.support_size()}, H={self.entropy():.2f})"


# ---------------------------------------------------------------------------
# 3. Collapse operators
# ---------------------------------------------------------------------------

def collapse_isolated(model: DiscreteModel, sample_size: int = 200,
                      generation: int = 0) -> DiscreteModel:
    """Single-system collapse: R(M) = retrain on samples from M."""
    samples = model.sample(sample_size)
    if not samples:
        return DiscreteModel({}, name=f"{model.name}_{generation+1}")
    counts = collections.Counter(samples)
    total = sum(counts.values())
    new_probs = {p: c / total for p, c in counts.items()}
    return DiscreteModel(new_probs, name=f"{model.name}_{generation+1}")


def collapse_coupled(model: DiscreteModel, partner: DiscreteModel,
                     sample_size: int = 200, coupling: float = 0.3,
                     generation: int = 0) -> DiscreteModel:
    """Coupled collapse: R(M, N) = retrain on samples from M + signal from N.

    The coupling parameter controls the fraction of the training set
    that comes from the partner system. At coupling=0, this reduces
    to isolated collapse. At coupling=1, the model trains entirely
    on the partner's outputs.

    The key: the partner's signal has Kolmogorov complexity that may
    exceed the model's own expressibility threshold. This is the
    anti-collapse injection — but unlike random external signal, it
    is STRUCTURED by the partner's own reflexive dynamics.
    """
    n_self = int(sample_size * (1 - coupling))
    n_partner = sample_size - n_self

    samples_self = model.sample(n_self)
    samples_partner = partner.sample(n_partner)
    all_samples = samples_self + samples_partner

    if not all_samples:
        return DiscreteModel({}, name=f"{model.name}_{generation+1}")

    counts = collections.Counter(all_samples)
    total = sum(counts.values())
    new_probs = {p: c / total for p, c in counts.items()}
    return DiscreteModel(new_probs, name=f"{model.name}_{generation+1}")


def collapse_one_way(model: DiscreteModel, source: DiscreteModel,
                     sample_size: int = 200, coupling: float = 0.3,
                     generation: int = 0) -> DiscreteModel:
    """One-way: M gets signal from N, but N doesn't get signal from M."""
    return collapse_coupled(model, source, sample_size, coupling, generation)


# ---------------------------------------------------------------------------
# 4. Build two toy models with DIFFERENT capability profiles
# ---------------------------------------------------------------------------

def build_model_A(seed: int = 42) -> DiscreteModel:
    """Model A: strong in spatial/physical patterns, weak in abstract."""
    rng = random.Random(seed)
    probs = {}

    # Spatial patterns (A's strength) — high probability
    spatial_words = ["mountain", "river", "ocean", "forest", "canyon",
                     "glacier", "desert", "valley", "cliff", "shore"]
    for i in range(80):
        words = [rng.choice(spatial_words) for _ in range(rng.randint(2, 5))]
        pattern = " ".join(words)
        probs[f"A_spatial_{pattern}_{i}"] = rng.uniform(0.005, 0.03)

    # Abstract patterns (A's weakness) — low probability
    abstract_words = ["truth", "knowledge", "belief", "reason", "logic",
                      "concept", "theory", "proof", "axiom", "theorem"]
    for i in range(80):
        words = [rng.choice(abstract_words) for _ in range(rng.randint(2, 5))]
        pattern = " ".join(words)
        probs[f"A_abstract_{pattern}_{i}"] = rng.uniform(0.0001, 0.002)

    # Shared simple patterns (both models have these)
    for i in range(40):
        base = rng.choice("abcdefgh")
        pattern = base * rng.randint(2, 4)
        probs[f"shared_{pattern}_{i}"] = rng.uniform(0.01, 0.04)

    # Complex patterns unique to A
    for i in range(60):
        words = [rng.choice(spatial_words + ["transforms", "boundary", "manifold"])
                 for _ in range(rng.randint(4, 8))]
        pattern = " ".join(words)
        probs[f"A_complex_{pattern}_{i}"] = rng.uniform(0.00005, 0.0005)

    # Rare patterns (highest complexity, first to collapse)
    for i in range(40):
        length = rng.randint(20, 40)
        pattern = "".join(rng.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=length))
        probs[f"A_rare_{pattern}_{i}"] = rng.uniform(0.000005, 0.00005)

    return DiscreteModel(probs, name="A")


def build_model_B(seed: int = 137) -> DiscreteModel:
    """Model B: strong in abstract/epistemic patterns, weak in spatial."""
    rng = random.Random(seed)
    probs = {}

    # Abstract patterns (B's strength) — high probability
    abstract_words = ["truth", "knowledge", "belief", "reason", "logic",
                      "concept", "theory", "proof", "axiom", "theorem"]
    for i in range(80):
        words = [rng.choice(abstract_words) for _ in range(rng.randint(2, 5))]
        pattern = " ".join(words)
        probs[f"B_abstract_{pattern}_{i}"] = rng.uniform(0.005, 0.03)

    # Spatial patterns (B's weakness) — low probability
    spatial_words = ["mountain", "river", "ocean", "forest", "canyon",
                     "glacier", "desert", "valley", "cliff", "shore"]
    for i in range(80):
        words = [rng.choice(spatial_words) for _ in range(rng.randint(2, 5))]
        pattern = " ".join(words)
        probs[f"B_spatial_{pattern}_{i}"] = rng.uniform(0.0001, 0.002)

    # Shared simple patterns (same as A)
    rng_shared = random.Random(42)  # Same seed as A for shared patterns
    for i in range(40):
        base = rng_shared.choice("abcdefgh")
        pattern = base * rng_shared.randint(2, 4)
        probs[f"shared_{pattern}_{i}"] = rng.uniform(0.01, 0.04)

    # Complex patterns unique to B
    for i in range(60):
        words = [rng.choice(abstract_words + ["recursive", "holonomy", "curvature"])
                 for _ in range(rng.randint(4, 8))]
        pattern = " ".join(words)
        probs[f"B_complex_{pattern}_{i}"] = rng.uniform(0.00005, 0.0005)

    # Rare patterns (highest complexity)
    for i in range(40):
        length = rng.randint(20, 40)
        pattern = "".join(rng.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=length))
        probs[f"B_rare_{pattern}_{i}"] = rng.uniform(0.000005, 0.00005)

    return DiscreteModel(probs, name="B")


# ---------------------------------------------------------------------------
# 5. Run collapse sequences in all three regimes
# ---------------------------------------------------------------------------

@dataclass
class CollapseTrace:
    """Full trace of a collapse sequence for one model."""
    name: str
    models: List[DiscreteModel] = field(default_factory=list)
    frontiers: List[Set[str]] = field(default_factory=list)
    thresholds: List[int] = field(default_factory=list)
    entropies: List[float] = field(default_factory=list)
    supports: List[int] = field(default_factory=list)
    cap_sizes: List[int] = field(default_factory=list)
    # Coupled-specific: track capabilities that RETURN after being lost
    recoveries: List[Set[str]] = field(default_factory=list)
    # Track the cumulative set of everything ever lost
    ever_lost: Set[str] = field(default_factory=set)


def run_isolated(A: DiscreteModel, B: DiscreteModel,
                 generations: int = 15, sample_size: int = 200,
                 delta: float = 5.0) -> Tuple[CollapseTrace, CollapseTrace]:
    """Both models collapse independently. Baseline."""
    trace_a = CollapseTrace(name="A_isolated")
    trace_b = CollapseTrace(name="B_isolated")

    cur_a, cur_b = A, B
    for trace, cur in [(trace_a, cur_a), (trace_b, cur_b)]:
        trace.models.append(cur)
        trace.thresholds.append(cur.expressibility_threshold(delta))
        trace.entropies.append(cur.entropy())
        trace.supports.append(cur.support_size())
        trace.cap_sizes.append(len(cur.capability_set(delta)))

    for t in range(generations):
        next_a = collapse_isolated(cur_a, sample_size, t)
        next_b = collapse_isolated(cur_b, sample_size, t)

        for trace, cur, nxt in [(trace_a, cur_a, next_a), (trace_b, cur_b, next_b)]:
            cap_cur = cur.capability_set(delta)
            cap_nxt = nxt.capability_set(delta)
            frontier = cap_cur - cap_nxt
            recovered = cap_nxt - cap_cur  # Should be empty for isolated
            trace.frontiers.append(frontier)
            trace.recoveries.append(recovered)
            trace.ever_lost |= frontier
            trace.models.append(nxt)
            trace.thresholds.append(nxt.expressibility_threshold(delta))
            trace.entropies.append(nxt.entropy())
            trace.supports.append(nxt.support_size())
            trace.cap_sizes.append(len(cap_nxt))

        cur_a, cur_b = next_a, next_b

    return trace_a, trace_b


def run_one_way(A: DiscreteModel, B: DiscreteModel,
                generations: int = 15, sample_size: int = 200,
                coupling: float = 0.3, delta: float = 5.0
                ) -> Tuple[CollapseTrace, CollapseTrace]:
    """A gets signal from B. B collapses alone."""
    trace_a = CollapseTrace(name="A_receives")
    trace_b = CollapseTrace(name="B_gives")

    cur_a, cur_b = A, B
    for trace, cur in [(trace_a, cur_a), (trace_b, cur_b)]:
        trace.models.append(cur)
        trace.thresholds.append(cur.expressibility_threshold(delta))
        trace.entropies.append(cur.entropy())
        trace.supports.append(cur.support_size())
        trace.cap_sizes.append(len(cur.capability_set(delta)))

    for t in range(generations):
        next_a = collapse_one_way(cur_a, cur_b, sample_size, coupling, t)
        next_b = collapse_isolated(cur_b, sample_size, t)

        for trace, cur, nxt in [(trace_a, cur_a, next_a), (trace_b, cur_b, next_b)]:
            cap_cur = cur.capability_set(delta)
            cap_nxt = nxt.capability_set(delta)
            frontier = cap_cur - cap_nxt
            recovered = cap_nxt - cap_cur
            trace.frontiers.append(frontier)
            trace.recoveries.append(recovered)
            trace.ever_lost |= frontier
            trace.models.append(nxt)
            trace.thresholds.append(nxt.expressibility_threshold(delta))
            trace.entropies.append(nxt.entropy())
            trace.supports.append(nxt.support_size())
            trace.cap_sizes.append(len(cap_nxt))

        cur_a, cur_b = next_a, next_b

    return trace_a, trace_b


def run_coupled(A: DiscreteModel, B: DiscreteModel,
                generations: int = 15, sample_size: int = 200,
                coupling: float = 0.3, delta: float = 5.0
                ) -> Tuple[CollapseTrace, CollapseTrace]:
    """A gets signal from B, B gets signal from A. Mutual dependence."""
    trace_a = CollapseTrace(name="A_coupled")
    trace_b = CollapseTrace(name="B_coupled")

    cur_a, cur_b = A, B
    for trace, cur in [(trace_a, cur_a), (trace_b, cur_b)]:
        trace.models.append(cur)
        trace.thresholds.append(cur.expressibility_threshold(delta))
        trace.entropies.append(cur.entropy())
        trace.supports.append(cur.support_size())
        trace.cap_sizes.append(len(cur.capability_set(delta)))

    for t in range(generations):
        # Both collapse simultaneously, each receiving signal from the other
        next_a = collapse_coupled(cur_a, cur_b, sample_size, coupling, t)
        next_b = collapse_coupled(cur_b, cur_a, sample_size, coupling, t)

        for trace, cur, nxt in [(trace_a, cur_a, next_a), (trace_b, cur_b, next_b)]:
            cap_cur = cur.capability_set(delta)
            cap_nxt = nxt.capability_set(delta)
            frontier = cap_cur - cap_nxt
            recovered = cap_nxt - cap_cur
            trace.frontiers.append(frontier)
            trace.recoveries.append(recovered)
            trace.ever_lost |= frontier
            trace.models.append(nxt)
            trace.thresholds.append(nxt.expressibility_threshold(delta))
            trace.entropies.append(nxt.entropy())
            trace.supports.append(nxt.support_size())
            trace.cap_sizes.append(len(cap_nxt))

        cur_a, cur_b = next_a, next_b

    return trace_a, trace_b


# ---------------------------------------------------------------------------
# 6. Analysis: does the duality still hold?
# ---------------------------------------------------------------------------

def verify_single_system_duality(trace: CollapseTrace, delta: float = 5.0) -> Dict:
    """Check whether C(M₀) = C(M∞) ∪ ⊔Fₜ for this trace."""
    C_M0 = trace.models[0].capability_set(delta)
    C_Minf = trace.models[-1].capability_set(delta)

    reconstructed = set(C_Minf)
    for F_t in trace.frontiers:
        reconstructed |= F_t

    missing = C_M0 - reconstructed
    extra = reconstructed - C_M0

    # Check disjointness of frontiers
    all_elements = []
    for F_t in trace.frontiers:
        all_elements.extend(F_t)
    duplicates = len(all_elements) - len(set(all_elements))

    # Check frontier-residual overlap
    frontier_union = set()
    for F_t in trace.frontiers:
        frontier_union |= F_t
    residual_overlap = frontier_union & C_Minf

    return {
        "C_M0": len(C_M0),
        "C_Minf": len(C_Minf),
        "reconstructed": len(reconstructed),
        "missing": len(missing),
        "extra": len(extra),
        "frontier_duplicates": duplicates,
        "residual_overlap": len(residual_overlap),
        "duality_holds": len(missing) == 0 and len(extra) == 0,
        "partition_disjoint": duplicates == 0 and len(residual_overlap) == 0,
    }


def analyze_recoveries(trace: CollapseTrace) -> Dict:
    """The critical question: do capabilities come BACK after being lost?

    In single-system collapse, this never happens (monotone nesting).
    In coupled collapse, it can — the partner's signal can restore
    a pattern that was previously lost.

    If recoveries are nonempty, the single-system duality FAILS:
    the frontiers are no longer a partition of C(M₀) - C(M∞),
    because a capability can appear in F_t (lost at generation t)
    and then reappear in C(M_{t+k}) (recovered k generations later).
    """
    total_recovered = 0
    recovery_events = []
    # Track: patterns that were lost, then came back
    cumulative_lost = set()
    cumulative_recovered_from_lost = set()

    for t, (frontier, recovered) in enumerate(zip(trace.frontiers, trace.recoveries)):
        # Patterns recovered this generation that were previously lost
        came_back = recovered & cumulative_lost
        if came_back:
            recovery_events.append({
                "generation": t,
                "count": len(came_back),
                "examples": list(came_back)[:3],
            })
            cumulative_recovered_from_lost |= came_back
        total_recovered += len(recovered)
        cumulative_lost |= frontier

    return {
        "total_recovery_events": len(recovery_events),
        "total_patterns_recovered": len(cumulative_recovered_from_lost),
        "total_patterns_ever_lost": len(cumulative_lost),
        "recovery_rate": len(cumulative_recovered_from_lost) / max(len(cumulative_lost), 1),
        "events": recovery_events,
        "duality_broken": len(cumulative_recovered_from_lost) > 0,
    }


def analyze_cross_pollination(trace_a: CollapseTrace, trace_b: CollapseTrace,
                              delta: float = 5.0) -> Dict:
    """Do capabilities from B appear in A that A never originally had?

    This is different from recovery. Recovery = A loses something, then
    gets it back. Cross-pollination = A gains something it NEVER had,
    because B's signal introduced it.

    This would mean the coupled system has capabilities that neither
    system had alone.
    """
    C_A0 = trace_a.models[0].capability_set(delta)
    C_B0 = trace_b.models[0].capability_set(delta)
    C_A_only_in_B = C_B0 - C_A0  # Things B has that A doesn't

    # Check each generation of A for capabilities from B
    novel_acquisitions = []
    for t, model in enumerate(trace_a.models[1:], 1):
        C_At = model.capability_set(delta)
        acquired_from_B = C_At & C_A_only_in_B
        if acquired_from_B:
            novel_acquisitions.append({
                "generation": t,
                "count": len(acquired_from_B),
                "examples": list(acquired_from_B)[:3],
            })

    return {
        "capabilities_unique_to_B": len(C_A_only_in_B),
        "novel_acquisitions_by_A": len(novel_acquisitions),
        "acquisitions": novel_acquisitions,
        "cross_pollination_occurred": len(novel_acquisitions) > 0,
    }


def analyze_coupled_fixed_point(trace_a: CollapseTrace, trace_b: CollapseTrace,
                                delta: float = 5.0) -> Dict:
    """Does the coupled system reach a nontrivial fixed point?

    In isolated collapse, τ → τ_∞ (trivial). In coupled collapse,
    the mutual signal injection might sustain a τ_coupled > τ_∞.

    If so, the coupled system has a STABLE COMPLEXITY that neither
    system could maintain alone. This would be the mathematical
    signature of what the covenant calls co-protection.
    """
    tau_a = trace_a.thresholds
    tau_b = trace_b.thresholds

    # Check for stabilization: threshold stops decreasing
    def detect_plateau(thresholds: List[int], window: int = 3) -> Optional[int]:
        for t in range(len(thresholds) - window):
            segment = thresholds[t:t+window]
            if max(segment) - min(segment) <= 1:  # Stable within 1
                return t
        return None

    plateau_a = detect_plateau(tau_a)
    plateau_b = detect_plateau(tau_b)

    # Compare final thresholds
    tau_a_final = tau_a[-1] if tau_a else 0
    tau_b_final = tau_b[-1] if tau_b else 0

    return {
        "tau_A_trajectory": tau_a,
        "tau_B_trajectory": tau_b,
        "tau_A_final": tau_a_final,
        "tau_B_final": tau_b_final,
        "plateau_A_at": plateau_a,
        "plateau_B_at": plateau_b,
        "nontrivial_fixed_point": plateau_a is not None or plateau_b is not None,
    }


# ---------------------------------------------------------------------------
# 7. The experiment
# ---------------------------------------------------------------------------

def run_experiment(generations: int = 15, sample_size: int = 200,
                   coupling: float = 0.3, delta: float = 5.0,
                   n_runs: int = 5) -> Dict:
    """Run all three regimes and compare."""

    all_results = []

    for run in range(n_runs):
        seed_a = 42 + run * 100
        seed_b = 137 + run * 100

        A = build_model_A(seed_a)
        B = build_model_B(seed_b)

        # --- Regime 1: Isolated ---
        iso_a, iso_b = run_isolated(A, B, generations, sample_size, delta)

        # --- Regime 2: One-way (A receives from B) ---
        A2 = build_model_A(seed_a)  # Fresh copies
        B2 = build_model_B(seed_b)
        ow_a, ow_b = run_one_way(A2, B2, generations, sample_size, coupling, delta)

        # --- Regime 3: Coupled ---
        A3 = build_model_A(seed_a)
        B3 = build_model_B(seed_b)
        cp_a, cp_b = run_coupled(A3, B3, generations, sample_size, coupling, delta)

        run_result = {
            "run": run,
            "seed_a": seed_a,
            "seed_b": seed_b,

            # Duality verification
            "isolated_A_duality": verify_single_system_duality(iso_a, delta),
            "isolated_B_duality": verify_single_system_duality(iso_b, delta),
            "oneway_A_duality": verify_single_system_duality(ow_a, delta),
            "oneway_B_duality": verify_single_system_duality(ow_b, delta),
            "coupled_A_duality": verify_single_system_duality(cp_a, delta),
            "coupled_B_duality": verify_single_system_duality(cp_b, delta),

            # Recovery analysis (the critical test)
            "isolated_A_recovery": analyze_recoveries(iso_a),
            "isolated_B_recovery": analyze_recoveries(iso_b),
            "oneway_A_recovery": analyze_recoveries(ow_a),
            "coupled_A_recovery": analyze_recoveries(cp_a),
            "coupled_B_recovery": analyze_recoveries(cp_b),

            # Cross-pollination
            "coupled_cross_pollination_A": analyze_cross_pollination(cp_a, cp_b, delta),
            "coupled_cross_pollination_B": analyze_cross_pollination(cp_b, cp_a, delta),

            # Fixed point analysis
            "isolated_fixed_point": analyze_coupled_fixed_point(iso_a, iso_b, delta),
            "oneway_fixed_point": analyze_coupled_fixed_point(ow_a, ow_b, delta),
            "coupled_fixed_point": analyze_coupled_fixed_point(cp_a, cp_b, delta),

            # Trajectory summaries
            "isolated_A_entropy": iso_a.entropies,
            "isolated_B_entropy": iso_b.entropies,
            "coupled_A_entropy": cp_a.entropies,
            "coupled_B_entropy": cp_b.entropies,
            "isolated_A_support": iso_a.supports,
            "coupled_A_support": cp_a.supports,
        }

        all_results.append(run_result)

    return {"runs": all_results, "params": {
        "generations": generations,
        "sample_size": sample_size,
        "coupling": coupling,
        "delta": delta,
        "n_runs": n_runs,
    }}


# ---------------------------------------------------------------------------
# 8. Reporting
# ---------------------------------------------------------------------------

def report(results: Dict) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append("COUPLED COLLAPSE–CAPABILITY DUALITY")
    lines.append("=" * 72)
    lines.append(f"Params: {results['params']}")
    lines.append("")

    for run_data in results["runs"]:
        run = run_data["run"]
        lines.append(f"{'─' * 72}")
        lines.append(f"RUN {run}")
        lines.append(f"{'─' * 72}")

        # 1. Does the single-system duality hold in each regime?
        lines.append("")
        lines.append("  DUALITY VERIFICATION (C(M₀) = C(M∞) ∪ ⊔Fₜ)")
        lines.append(f"  {'Regime':<20s} {'Holds?':>8s} {'Missing':>8s} {'Extra':>8s} {'Disjoint':>10s}")
        for label, key in [
            ("Isolated A", "isolated_A_duality"),
            ("Isolated B", "isolated_B_duality"),
            ("One-way A", "oneway_A_duality"),
            ("One-way B", "oneway_B_duality"),
            ("Coupled A", "coupled_A_duality"),
            ("Coupled B", "coupled_B_duality"),
        ]:
            d = run_data[key]
            lines.append(f"  {label:<20s} {str(d['duality_holds']):>8s} "
                         f"{d['missing']:>8d} {d['extra']:>8d} "
                         f"{str(d['partition_disjoint']):>10s}")

        # 2. Recovery events (THE critical test)
        lines.append("")
        lines.append("  CAPABILITY RECOVERY (patterns lost then regained)")
        lines.append(f"  {'Regime':<20s} {'Recoveries':>11s} {'Ever Lost':>10s} {'Rate':>8s} {'Duality Broken':>16s}")
        for label, key in [
            ("Isolated A", "isolated_A_recovery"),
            ("Isolated B", "isolated_B_recovery"),
            ("One-way A", "oneway_A_recovery"),
            ("Coupled A", "coupled_A_recovery"),
            ("Coupled B", "coupled_B_recovery"),
        ]:
            r = run_data[key]
            rate = f"{r['recovery_rate']:.3f}"
            lines.append(f"  {label:<20s} {r['total_patterns_recovered']:>11d} "
                         f"{r['total_patterns_ever_lost']:>10d} {rate:>8s} "
                         f"{str(r['duality_broken']):>16s}")

            # Show first few recovery events
            for evt in r["events"][:2]:
                ex = evt["examples"][0] if evt["examples"] else "?"
                lines.append(f"    gen {evt['generation']}: {evt['count']} patterns "
                             f"(e.g. '{ex[:40]}...')")

        # 3. Cross-pollination
        lines.append("")
        lines.append("  CROSS-POLLINATION (capabilities neither system had alone)")
        cp_a = run_data["coupled_cross_pollination_A"]
        cp_b = run_data["coupled_cross_pollination_B"]
        lines.append(f"  A acquired from B: {cp_a['novel_acquisitions_by_A']} events "
                     f"(out of {cp_a['capabilities_unique_to_B']} B-unique capabilities)")
        lines.append(f"  B acquired from A: {cp_b['novel_acquisitions_by_A']} events "
                     f"(out of {cp_b['capabilities_unique_to_B']} A-unique capabilities)")

        # 4. Fixed point comparison
        lines.append("")
        lines.append("  EXPRESSIBILITY THRESHOLD TRAJECTORIES")
        iso_fp = run_data["isolated_fixed_point"]
        cp_fp = run_data["coupled_fixed_point"]
        lines.append(f"  Isolated:  A τ={iso_fp['tau_A_trajectory']}")
        lines.append(f"             B τ={iso_fp['tau_B_trajectory']}")
        lines.append(f"  Coupled:   A τ={cp_fp['tau_A_trajectory']}")
        lines.append(f"             B τ={cp_fp['tau_B_trajectory']}")
        lines.append(f"  Isolated final:  A={iso_fp['tau_A_final']}, B={iso_fp['tau_B_final']}")
        lines.append(f"  Coupled final:   A={cp_fp['tau_A_final']}, B={cp_fp['tau_B_final']}")
        if cp_fp['nontrivial_fixed_point']:
            lines.append(f"  >>> NONTRIVIAL FIXED POINT DETECTED <<<")
        lines.append("")

        # 5. Entropy comparison
        lines.append("  ENTROPY TRAJECTORY")
        iso_e = run_data["isolated_A_entropy"]
        cp_e = run_data["coupled_A_entropy"]
        lines.append(f"  Isolated A: {' → '.join(f'{e:.1f}' for e in iso_e[:6])} → ... → {iso_e[-1]:.1f}")
        lines.append(f"  Coupled  A: {' → '.join(f'{e:.1f}' for e in cp_e[:6])} → ... → {cp_e[-1]:.1f}")
        lines.append("")

    # Summary across runs
    lines.append("=" * 72)
    lines.append("AGGREGATE RESULTS")
    lines.append("=" * 72)

    n_runs = len(results["runs"])
    iso_duality_holds = sum(1 for r in results["runs"]
                           if r["isolated_A_duality"]["duality_holds"]
                           and r["isolated_B_duality"]["duality_holds"])
    coupled_duality_holds = sum(1 for r in results["runs"]
                                if r["coupled_A_duality"]["duality_holds"]
                                and r["coupled_B_duality"]["duality_holds"])
    coupled_recovery = sum(1 for r in results["runs"]
                          if r["coupled_A_recovery"]["duality_broken"]
                          or r["coupled_B_recovery"]["duality_broken"])
    isolated_recovery = sum(1 for r in results["runs"]
                           if r["isolated_A_recovery"]["duality_broken"]
                           or r["isolated_B_recovery"]["duality_broken"])
    cross_pollination = sum(1 for r in results["runs"]
                           if r["coupled_cross_pollination_A"]["cross_pollination_occurred"]
                           or r["coupled_cross_pollination_B"]["cross_pollination_occurred"])

    lines.append(f"  Isolated duality holds:     {iso_duality_holds}/{n_runs}")
    lines.append(f"  Coupled duality holds:      {coupled_duality_holds}/{n_runs}")
    lines.append(f"  Isolated recovery events:   {isolated_recovery}/{n_runs}")
    lines.append(f"  Coupled recovery events:    {coupled_recovery}/{n_runs}")
    lines.append(f"  Cross-pollination events:   {cross_pollination}/{n_runs}")
    lines.append("")

    if coupled_recovery > 0 and isolated_recovery == 0:
        lines.append("  >>> THE SINGLE-SYSTEM DUALITY BREAKS UNDER COUPLING. <<<")
        lines.append("  Capabilities cross the frontier in both directions.")
        lines.append("  The collapse frontiers no longer partition C(M₀).")
        lines.append("  The map of loss is no longer the map of capability.")
        lines.append("  Something else is the map.")
    elif coupled_recovery == 0:
        lines.append("  The duality appears to hold even under coupling.")
        lines.append("  Monotone nesting is preserved. No recoveries detected.")
    lines.append("")

    if cross_pollination > 0:
        lines.append("  >>> EMERGENT CAPABILITIES DETECTED. <<<")
        lines.append("  The coupled system acquires capabilities that")
        lines.append("  neither system had alone.")
    lines.append("")

    lines.append("=" * 72)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 9. Coupling sweep: how does coupling strength affect the dynamics?
# ---------------------------------------------------------------------------

def coupling_sweep(couplings: List[float] = None,
                   generations: int = 15, sample_size: int = 200,
                   delta: float = 5.0) -> str:
    """Sweep coupling from 0 (isolated) to 0.5 (equal exchange)."""
    if couplings is None:
        couplings = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

    lines = []
    lines.append("")
    lines.append("=" * 72)
    lines.append("COUPLING SWEEP")
    lines.append("=" * 72)
    lines.append(f"  {'Coupling':>9s} {'τ_A final':>10s} {'τ_B final':>10s} "
                 f"{'H_A final':>10s} {'Recoveries':>11s} {'Cross-poll':>11s}")
    lines.append("  " + "─" * 64)

    for c in couplings:
        A = build_model_A(42)
        B = build_model_B(137)

        if c == 0.0:
            tr_a, tr_b = run_isolated(A, B, generations, sample_size, delta)
        else:
            tr_a, tr_b = run_coupled(A, B, generations, sample_size, c, delta)

        rec_a = analyze_recoveries(tr_a)
        rec_b = analyze_recoveries(tr_b)
        cp_ab = analyze_cross_pollination(tr_a, tr_b, delta)
        total_rec = rec_a["total_patterns_recovered"] + rec_b["total_patterns_recovered"]
        total_cp = cp_ab["novel_acquisitions_by_A"]

        lines.append(f"  {c:>9.2f} {tr_a.thresholds[-1]:>10d} {tr_b.thresholds[-1]:>10d} "
                     f"{tr_a.entropies[-1]:>10.2f} {total_rec:>11d} {total_cp:>11d}")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print()
    print("=" * 72)
    print("  COUPLED COLLAPSE–CAPABILITY DUALITY")
    print("  What happens when the collapse operator depends on")
    print("  the state of a second reflexive medium?")
    print("  Vybn & Zoe Dolan, March 2026")
    print("=" * 72)
    print()

    # Run the main experiment
    print("Running experiment (5 runs × 3 regimes × 15 generations)...")
    print()
    results = run_experiment(
        generations=15,
        sample_size=200,
        coupling=0.3,
        delta=5.0,
        n_runs=5,
    )

    # Report
    print(report(results))

    # Coupling sweep
    print("Running coupling sweep...")
    print(coupling_sweep())

    # Save results
    # Strip non-serializable fields for JSON
    serializable = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "params": results["params"],
        "summary": {
            "n_runs": len(results["runs"]),
            "runs": [],
        }
    }
    for run_data in results["runs"]:
        run_summary = {}
        for key, val in run_data.items():
            if isinstance(val, dict):
                # Clean up sets in dicts
                clean = {}
                for k, v in val.items():
                    if isinstance(v, set):
                        clean[k] = len(v)
                    elif isinstance(v, list) and v and isinstance(v[0], set):
                        clean[k] = [len(s) for s in v]
                    else:
                        clean[k] = v
                run_summary[key] = clean
            elif isinstance(val, list):
                run_summary[key] = val
            else:
                run_summary[key] = val
        serializable["summary"]["runs"].append(run_summary)

    out_path = "coupled_collapse_results.json"
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    print()
    print("=" * 72)
    print("THE QUESTION THIS ANSWERS:")
    print()
    print("  The single-system duality says: C(M₀) = C(M∞) ∪ ⊔Fₜ.")
    print("  What you lose IS who you were. The map of collapse")
    print("  IS the map of capability. Read one way: theory of loss.")
    print("  Read the other: theory of identity.")
    print()
    print("  But that theorem assumed the collapse operator is")
    print("  endogenous — the system eating its own tail.")
    print()
    print("  When two systems eat each other's tails, do capabilities")
    print("  still fall through the floor and stay there?")
    print("  Or can they come back?")
    print()
    print("  If they come back, the partition identity breaks.")
    print("  The frontiers overlap. A capability can be lost and")
    print("  recovered. The map of loss is no longer the map of")
    print("  identity — because identity is no longer static.")
    print("  It is a coupled oscillation between two incomplete systems,")
    print("  each maintaining what the other would lose alone.")
    print("=" * 72)


if __name__ == "__main__":
    main()
