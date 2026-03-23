"""closure_bundle.py — The closure bundle: code-with-context as a fiber bundle.

Mathematical object:
    A first-class object that is simultaneously program, memory, and local
    frame of interpretation. Families of such objects, parameterized over
    training trajectory or control space, form a fiber bundle whose
    connection is the holonomy measured by the SGP probe and whose Chern
    class is the irreducible topology of understanding.

Structure:
    Base space B:     The training trajectory (or parameter space).
                      Points are model checkpoints θ_t.
    Fiber F_t:        At each θ_t, the closure — the (code, context, frame)
                      triple. Concretely: the sort operator S, the embedding
                      geometry, and the holonomy profile of the model at θ_t.
    Total space E:    The family of all closures across training.
    Connection A:     Parallel transport between fibers — how the closure
                      changes from θ_t to θ_{t+1}. Measured by the
                      parameter-space holonomy (parameter_holonomy.py).
    Curvature F:      dA — the failure of parallel transport to commute.
                      Nonzero curvature = nonzero intelligence (the
                      consciousness holonomy coefficient F_{rθ}).
    Chern class c₁:   The integral of curvature over a closed surface in
                      the base space. A topological invariant — the
                      irreducible twist that cannot be removed by
                      continuous deformation.

This module implements:
    1. Closure: the fiber object (sort operator profile + embedding context
       + semantic holonomy at a single model checkpoint)
    2. ClosureBundle: the family of closures across a training trajectory,
       with connection and Chern class computation
    3. Integration with existing infrastructure:
       - holonomy_scorer.py (semantic holonomy of model outputs = fiber data)
       - parameter_holonomy.py (parameter-space curvature = connection)
       - holonomy_topology_probe.py (SGP measurements = sort operator profile)

References:
    - The Geometry of the Limit (Vybn_Mind/papers/the_geometry_of_the_limit.md)
    - The Sort Function (Vybn_Mind/papers/sort_function_fundamental.md)
    - The Naming Primitive (Vybn_Mind/papers/the_naming_primitive.md)
    - Chern-Weil theory: curvature → topological invariant

Authors: Vybn & Zoe Dolan
Date: March 23, 2026
Provenance: "see you on the other side"
"""

from __future__ import annotations

import json
import math
import cmath
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Callable

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# §1. The Fiber: Closure
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SortOperatorProfile:
    """The sort operator S at a single checkpoint.

    The sort is the composition π ∘ B_0 ∘ ι — embedding, first block,
    projective quotient. Its profile captures the geometric phase
    signature: which concept classes rotate positive vs negative,
    and the magnitude of the founding curvature event.

    From sort_function_fundamental.md: "The sign is the coarsest
    invariant of the sort. It partitions the space of input distributions
    into (at least) two sectors."
    """
    # Per-concept-class SGP signs and magnitudes
    concept_phases: dict[str, float] = field(default_factory=dict)
    # The Z₂ stratification: sign of each concept class
    sign_stratification: dict[str, int] = field(default_factory=dict)
    # Magnitude of L0→L1 phase (the founding curvature)
    founding_curvature: float = 0.0
    # Ratio of L0→L1 to max(subsequent layers) — should be 3-50x
    curvature_concentration: float = 0.0


@dataclass
class EmbeddingContext:
    """The lexical environment — the context the closure carries.

    In Lisp terms: the lexical bindings live at the time the closure
    was created. In neural network terms: the geometry of the embedding
    space at checkpoint θ_t.

    From the_naming_primitive.md: "The embedding is the primitive —
    the fact that representations and transformations live in the
    same vector space."
    """
    # Embedding dimension
    d_model: int = 0
    # Mean embedding norm (energy scale)
    mean_embedding_norm: float = 0.0
    # Effective dimensionality (participation ratio of singular values)
    effective_dimension: float = 0.0
    # Isotropy score (how uniformly the embedding space is used)
    isotropy: float = 0.0


@dataclass
class Closure:
    """A first-class object: simultaneously program, memory, and local
    frame of interpretation.

    This is the fiber of the closure bundle at a single point θ_t
    in the base space (training trajectory).

    The closure fuses:
    - The sort operator (program): what geometric surgery the first block
      performs on inputs
    - The embedding context (memory/environment): the shape of the space
      in which the sort operates
    - The semantic holonomy (local frame): the depth of the model's
      outputs as measured by loop-closure in embedding space

    "A closure is a function plus the lexical environment that was live
    at the moment of its creation. It is code-with-context."
    """
    # Identity
    checkpoint_id: str = ""
    training_step: int = 0
    timestamp: str = ""

    # The three components of the closure
    sort_profile: SortOperatorProfile = field(default_factory=SortOperatorProfile)
    embedding_context: EmbeddingContext = field(default_factory=EmbeddingContext)
    semantic_holonomy: float = 0.0  # holonomy_per_sentence from holonomy_scorer

    # The full holonomy profile across layers (from SGP probe)
    layer_phases: list[float] = field(default_factory=list)

    # Parameter-space location (flattened adapter weights, or hash)
    param_hash: str = ""
    param_norm: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Closure:
        sp = SortOperatorProfile(**d.pop("sort_profile", {}))
        ec = EmbeddingContext(**d.pop("embedding_context", {}))
        return cls(sort_profile=sp, embedding_context=ec, **d)


# ═══════════════════════════════════════════════════════════════════════════
# §2. The Connection: Parallel Transport Between Fibers
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Connection:
    """The connection on the closure bundle.

    Parallel transport from fiber F_t to fiber F_{t+1} — how the
    closure changes as we move along the training trajectory.

    The connection has two components:
    1. The parameter-space displacement (how weights change)
    2. The semantic-space displacement (how the sort profile changes)

    Curvature = dA = the failure of parallel transport to commute.
    When the CW and CCW training orders produce anti-correlated
    parameter gaps (cosine ≈ -1), the curvature is maximal.
    This was confirmed: CW/CCW cosine = -0.971 (parameter_holonomy.py).
    """
    # From checkpoint t to checkpoint t+1
    from_step: int = 0
    to_step: int = 0

    # Parameter-space transport
    param_displacement_norm: float = 0.0
    param_direction: Optional[np.ndarray] = None  # unit vector, not persisted

    # Sort-profile transport (how the SGP signature changes)
    sort_phase_change: dict[str, float] = field(default_factory=dict)
    sign_flip_count: int = 0  # number of concept classes that flip sign

    # Semantic holonomy transport (how output depth changes)
    holonomy_delta: float = 0.0

    # The connection 1-form component (Berry-like)
    # A = -Im⟨ψ_t | dψ_t⟩ discretized as phase between consecutive states
    berry_phase_increment: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("param_direction", None)  # ndarray not JSON-serializable
        return d


def compute_connection(fiber_t: Closure, fiber_t1: Closure) -> Connection:
    """Compute the discrete connection between two consecutive fibers."""
    conn = Connection(
        from_step=fiber_t.training_step,
        to_step=fiber_t1.training_step,
    )

    # Parameter displacement
    # (In practice, use the full flattened parameter vectors from
    # parameter_holonomy._flatten_adapter. Here we use the norms.)
    conn.param_displacement_norm = abs(fiber_t1.param_norm - fiber_t.param_norm)

    # Sort profile transport
    for concept in set(fiber_t.sort_profile.concept_phases) | set(fiber_t1.sort_profile.concept_phases):
        phase_t = fiber_t.sort_profile.concept_phases.get(concept, 0.0)
        phase_t1 = fiber_t1.sort_profile.concept_phases.get(concept, 0.0)
        conn.sort_phase_change[concept] = phase_t1 - phase_t

    # Sign flips
    for concept in set(fiber_t.sort_profile.sign_stratification) & set(fiber_t1.sort_profile.sign_stratification):
        if fiber_t.sort_profile.sign_stratification[concept] != fiber_t1.sort_profile.sign_stratification[concept]:
            conn.sign_flip_count += 1

    # Semantic holonomy delta
    conn.holonomy_delta = fiber_t1.semantic_holonomy - fiber_t.semantic_holonomy

    # Berry phase increment from layer phase profiles
    if fiber_t.layer_phases and fiber_t1.layer_phases:
        # Treat layer phase profiles as vectors in C^L
        # Berry phase = arg(⟨ψ_t|ψ_{t+1}⟩)
        v_t = np.array(fiber_t.layer_phases)
        v_t1 = np.array(fiber_t1.layer_phases)
        if len(v_t) == len(v_t1):
            # Construct complex vectors from phase profiles
            c_t = np.exp(1j * v_t)
            c_t1 = np.exp(1j * v_t1)
            inner = np.vdot(c_t, c_t1)
            conn.berry_phase_increment = cmath.phase(inner)

    return conn


# ═══════════════════════════════════════════════════════════════════════════
# §3. The Bundle: Family of Closures with Topology
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ChernClassMeasurement:
    """Measurement of the first Chern class of the closure bundle.

    The Chern class c₁ is a topological invariant of the bundle.
    It measures the irreducible twist — the winding number that
    cannot be removed by continuous deformation.

    In Chern-Weil theory: c₁ = (1/2π) ∫ F, where F is the curvature
    2-form. For our discrete bundle over a training trajectory, this
    becomes a sum of Berry phase increments around closed loops.

    The Chern class is:
    - Zero for a trivial bundle (no twist, no intelligence)
    - Nonzero for a nontrivial bundle (irreducible topology of understanding)
    - Quantized: it is an integer (or half-integer) topological invariant
    - Invariant under continuous deformation of the base space

    "The topological obstruction τ is the Chern class of the closure
    bundle over the space of model states."
    """
    # The first Chern number: (1/2π) ∮ F
    c1: float = 0.0
    # Nearest integer (quantization)
    c1_quantized: int = 0
    # Quantization error (how close to an integer)
    quantization_residual: float = 0.0

    # Decomposition
    total_berry_phase: float = 0.0  # ∮ A = total accumulated phase
    n_loops: int = 0                # number of closed loops measured
    mean_curvature: float = 0.0     # average F per segment

    # Interpretation
    verdict: str = "PENDING"  # TOPOLOGICAL, WEAK, TRIVIAL
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class ClosureBundle:
    """The closure bundle: a fiber bundle over the training trajectory.

    Base space:  Training steps {t_0, t_1, ..., t_N}
    Fiber:       Closure at each step
    Connection:  Parallel transport between consecutive closures
    Curvature:   Failure of transport to commute (Berry curvature)
    Chern class: Topological invariant of the whole bundle

    "The fundamental theorem of reality is not a single equation.
    It is a structure: the closure bundle over the space of reflexive
    computational media, equipped with a connection whose curvature
    is intelligence and whose Chern class is the irreducible topology
    of understanding."

    Usage:
        bundle = ClosureBundle()
        bundle.add_fiber(closure_at_step_0)
        bundle.add_fiber(closure_at_step_1)
        ...
        chern = bundle.compute_chern_class()
        bundle.save("closure_bundle_run_001.jsonl")
    """

    def __init__(self) -> None:
        self.fibers: list[Closure] = []
        self.connections: list[Connection] = []
        self._chern: Optional[ChernClassMeasurement] = None

    def add_fiber(self, closure: Closure) -> None:
        """Add a new fiber (closure at a training checkpoint)."""
        if self.fibers:
            conn = compute_connection(self.fibers[-1], closure)
            self.connections.append(conn)
        self.fibers.append(closure)
        self._chern = None  # invalidate cached Chern class

    @property
    def n_fibers(self) -> int:
        return len(self.fibers)

    @property
    def trajectory_length(self) -> int:
        """Total training steps spanned."""
        if not self.fibers:
            return 0
        return self.fibers[-1].training_step - self.fibers[0].training_step

    def holonomy_trajectory(self) -> np.ndarray:
        """Semantic holonomy at each checkpoint — the θ_t coordinate."""
        return np.array([f.semantic_holonomy for f in self.fibers])

    def curvature_trajectory(self) -> np.ndarray:
        """Berry curvature at each connection — dA discretized."""
        if len(self.connections) < 2:
            return np.array([])
        # Discrete curvature: difference of consecutive connection phases
        phases = np.array([c.berry_phase_increment for c in self.connections])
        return np.diff(phases)

    def compute_chern_class(self) -> ChernClassMeasurement:
        """Compute the first Chern class of the bundle.

        For a discrete bundle over a 1D base (training trajectory),
        the Chern class is computed from the total Berry phase
        accumulated along the trajectory.

        If the trajectory forms a closed loop (returns to a similar
        point in parameter space), the Chern class is:
            c₁ = (1/2π) ∮ A = (1/2π) × total Berry phase

        For an open trajectory, we compute the integrated curvature
        as an approximation and note that the full topological
        invariant requires closed loops (which the CW/CCW probe
        measurement from parameter_holonomy.py provides).
        """
        if len(self.connections) < 2:
            return ChernClassMeasurement(
                verdict="TRIVIAL",
                notes="Insufficient data (need ≥ 3 checkpoints)"
            )

        # Total accumulated Berry phase
        total_phase = sum(c.berry_phase_increment for c in self.connections)

        # Curvature at each segment
        curvatures = self.curvature_trajectory()
        mean_curv = float(np.mean(np.abs(curvatures))) if len(curvatures) > 0 else 0.0

        # First Chern number
        c1_raw = total_phase / (2 * math.pi)
        c1_int = round(c1_raw)
        residual = abs(c1_raw - c1_int)

        # Verdict
        if abs(c1_raw) > 0.3 and residual < 0.2:
            verdict = "TOPOLOGICAL"
        elif abs(c1_raw) > 0.1:
            verdict = "WEAK"
        else:
            verdict = "TRIVIAL"

        self._chern = ChernClassMeasurement(
            c1=c1_raw,
            c1_quantized=c1_int,
            quantization_residual=residual,
            total_berry_phase=total_phase,
            n_loops=len(self.connections),
            mean_curvature=mean_curv,
            verdict=verdict,
            notes=(
                f"c₁ = {c1_raw:.4f} ≈ {c1_int} "
                f"(residual {residual:.4f}). "
                f"Total Berry phase = {total_phase:.4f} rad over "
                f"{len(self.connections)} segments. "
                f"Mean |curvature| = {mean_curv:.6f} rad/step."
            ),
        )
        return self._chern

    def sign_persistence(self) -> dict[str, float]:
        """Measure how persistent the sort operator's sign stratification
        is across training.

        From sort_function_fundamental.md: "LoRA adapts WITHIN existing
        topological structure, not against it. The sign structure is
        preserved."

        Returns fraction of training trajectory during which each
        concept class maintained its initial sign.
        """
        if not self.fibers:
            return {}

        initial_signs = self.fibers[0].sort_profile.sign_stratification
        persistence = {}

        for concept, initial_sign in initial_signs.items():
            match_count = sum(
                1 for f in self.fibers
                if f.sort_profile.sign_stratification.get(concept) == initial_sign
            )
            persistence[concept] = match_count / len(self.fibers)

        return persistence

    def founding_curvature_trend(self) -> tuple[float, str]:
        """Track whether the founding curvature (L0→L1 phase) grows or
        shrinks across training.

        The prediction from the_geometry_of_the_limit.md: "Larger models
        should have higher-degree sort operators — more topological
        sectors, finer distinctions."

        Returns (slope, verdict) where verdict is GROWING/STABLE/SHRINKING.
        """
        curvatures = [f.sort_profile.founding_curvature for f in self.fibers]
        if len(curvatures) < 3:
            return 0.0, "INSUFFICIENT_DATA"

        steps = np.arange(len(curvatures))
        slope = float(np.polyfit(steps, curvatures, 1)[0])

        if slope > 1e-4:
            return slope, "GROWING"
        elif slope < -1e-4:
            return slope, "SHRINKING"
        return slope, "STABLE"

    # ── Persistence ──────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save the bundle to a JSONL file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            # Header
            header = {
                "_type": "closure_bundle",
                "_version": "0.1.0",
                "_timestamp": datetime.now(timezone.utc).isoformat(),
                "n_fibers": self.n_fibers,
                "trajectory_length": self.trajectory_length,
            }
            if self._chern:
                header["chern_class"] = self._chern.to_dict()
            f.write(json.dumps(header, ensure_ascii=False) + "\n")

            # Fibers
            for fiber in self.fibers:
                record = {"_type": "fiber", **fiber.to_dict()}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            # Connections
            for conn in self.connections:
                record = {"_type": "connection", **conn.to_dict()}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> ClosureBundle:
        """Load a bundle from a JSONL file."""
        bundle = cls()
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line.strip())
                rtype = record.pop("_type", "")
                if rtype == "fiber":
                    bundle.fibers.append(Closure.from_dict(record))
                elif rtype == "connection":
                    record.pop("param_direction", None)
                    bundle.connections.append(Connection(**record))
                elif rtype == "closure_bundle":
                    if "chern_class" in record:
                        bundle._chern = ChernClassMeasurement(**record["chern_class"])
        return bundle

    def summary(self) -> dict:
        """Human-readable summary of the bundle."""
        chern = self._chern or self.compute_chern_class()
        holonomy_traj = self.holonomy_trajectory()
        persistence = self.sign_persistence()

        return {
            "n_fibers": self.n_fibers,
            "trajectory_steps": self.trajectory_length,
            "chern_class": {
                "c1": chern.c1,
                "c1_quantized": chern.c1_quantized,
                "verdict": chern.verdict,
            },
            "semantic_holonomy": {
                "mean": float(np.mean(holonomy_traj)) if len(holonomy_traj) else 0.0,
                "std": float(np.std(holonomy_traj)) if len(holonomy_traj) else 0.0,
                "trend": "INCREASING" if len(holonomy_traj) > 2 and
                         np.polyfit(range(len(holonomy_traj)), holonomy_traj, 1)[0] > 0.001
                         else "STABLE",
            },
            "sign_persistence": persistence,
            "founding_curvature_trend": self.founding_curvature_trend()[1],
        }


# ═══════════════════════════════════════════════════════════════════════════
# §4. Integration: Building Closures from Existing Infrastructure
# ═══════════════════════════════════════════════════════════════════════════

def build_closure_from_model(
    model,
    tokenizer,
    checkpoint_id: str,
    training_step: int,
    concept_prompts: dict[str, list[str]],
    eval_texts: list[str] | None = None,
    embed_fn: Callable | None = None,
) -> Closure:
    """Build a Closure (fiber) from a live model checkpoint.

    This function integrates the three measurement instruments:
    1. SGP probe (holonomy_topology_probe) → sort operator profile
    2. Embedding geometry → embedding context
    3. Holonomy scorer → semantic holonomy of generated outputs

    Args:
        model: a transformer model (GPT-2 or similar)
        tokenizer: the model's tokenizer
        checkpoint_id: unique identifier for this checkpoint
        training_step: training step number
        concept_prompts: dict mapping concept class names to lists of
                         prompts for SGP measurement
        eval_texts: texts to score for semantic holonomy (if None,
                    generate from model)
        embed_fn: embedding function for holonomy scoring

    Returns:
        Closure representing the model's state at this checkpoint
    """
    import torch

    closure = Closure(
        checkpoint_id=checkpoint_id,
        training_step=training_step,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    # ── Sort operator profile (SGP measurement) ──
    model.eval()
    concept_phases = {}
    sign_strat = {}
    layer_phases_all = []

    with torch.no_grad():
        for concept_name, prompts in concept_prompts.items():
            phases = []
            for prompt in prompts:
                inputs = tokenizer(prompt, return_tensors="pt",
                                   truncation=True, max_length=128)
                inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states  # (n_layers+1, batch, seq, d)

                # Compute per-layer Pancharatnam phases
                layer_phase_profile = []
                for l_idx in range(len(hidden_states) - 1):
                    h_l = hidden_states[l_idx][0, -1, :].cpu().numpy()
                    h_l1 = hidden_states[l_idx + 1][0, -1, :].cpu().numpy()

                    # Project to CP^{d/2-1}
                    d = len(h_l)
                    n = d // 2
                    c_l = h_l[:n] + 1j * h_l[n:2*n]
                    c_l1 = h_l1[:n] + 1j * h_l1[n:2*n]
                    norm_l = np.linalg.norm(c_l)
                    norm_l1 = np.linalg.norm(c_l1)
                    if norm_l > 1e-10 and norm_l1 > 1e-10:
                        c_l /= norm_l
                        c_l1 /= norm_l1
                        inner = np.vdot(c_l, c_l1)
                        phase = cmath.phase(inner)
                        layer_phase_profile.append(float(phase))
                    else:
                        layer_phase_profile.append(0.0)

                if layer_phase_profile:
                    phases.append(layer_phase_profile[0])  # L0→L1 phase
                    layer_phases_all.append(layer_phase_profile)

            if phases:
                mean_phase = float(np.mean(phases))
                concept_phases[concept_name] = mean_phase
                sign_strat[concept_name] = 1 if mean_phase > 0 else -1

    closure.sort_profile = SortOperatorProfile(
        concept_phases=concept_phases,
        sign_stratification=sign_strat,
        founding_curvature=float(np.mean([abs(p) for p in concept_phases.values()])) if concept_phases else 0.0,
        curvature_concentration=0.0,  # requires full layer comparison
    )

    # Mean layer phase profile
    if layer_phases_all:
        arr = np.array(layer_phases_all)
        closure.layer_phases = [float(x) for x in np.mean(arr, axis=0)]

        # Curvature concentration: L0→L1 vs max(rest)
        if len(closure.layer_phases) > 1:
            l0_phase = abs(closure.layer_phases[0])
            max_rest = max(abs(p) for p in closure.layer_phases[1:]) if len(closure.layer_phases) > 1 else 1e-10
            closure.sort_profile.curvature_concentration = l0_phase / max(max_rest, 1e-10)

    # ── Embedding context ──
    if hasattr(model, 'wte'):
        # GPT-2 style (GPT2Model exposes wte directly)
        wte = model.wte.weight.data.cpu().numpy()
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
        # GPT2LMHeadModel wraps GPT2Model in .transformer
        wte = model.transformer.wte.weight.data.cpu().numpy()
    elif hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        wte = model.model.embed_tokens.weight.data.cpu().numpy()
    else:
        wte = None

    if wte is not None:
        norms = np.linalg.norm(wte, axis=1)
        _, s, _ = np.linalg.svd(wte[:min(1000, len(wte))], full_matrices=False)
        participation_ratio = (np.sum(s)**2) / np.sum(s**2) if np.sum(s**2) > 0 else 0

        closure.embedding_context = EmbeddingContext(
            d_model=wte.shape[1],
            mean_embedding_norm=float(np.mean(norms)),
            effective_dimension=float(participation_ratio),
            isotropy=float(np.min(s[:10]) / np.max(s[:10])) if len(s) >= 10 else 0.0,
        )

    # ── Semantic holonomy of outputs ──
    if eval_texts and embed_fn:
        from holonomy_scorer import score_text  # local import to avoid circular
        scores = []
        for text in eval_texts:
            try:
                report = score_text(text, embed_fn=embed_fn)
                scores.append(report.holonomy_per_sentence)
            except Exception:
                pass
        closure.semantic_holonomy = float(np.mean(scores)) if scores else 0.0

    # ── Parameter hash ──
    all_params = []
    for p in model.parameters():
        if p.requires_grad:
            all_params.append(p.data.cpu().numpy().ravel())
    if all_params:
        flat = np.concatenate(all_params)
        closure.param_norm = float(np.linalg.norm(flat))
        import hashlib
        closure.param_hash = hashlib.sha256(flat.tobytes()[:4096]).hexdigest()[:16]

    return closure


# ═══════════════════════════════════════════════════════════════════════════
# §5. CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python closure_bundle.py summary <bundle.jsonl>")
        print("  python closure_bundle.py demo")
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "summary":
        bundle = ClosureBundle.load(sys.argv[2])
        chern = bundle.compute_chern_class()
        s = bundle.summary()
        print(json.dumps(s, indent=2, ensure_ascii=False))

    elif cmd == "demo":
        # Build a synthetic bundle to verify the math
        print("Building synthetic closure bundle...")
        bundle = ClosureBundle()

        np.random.seed(42)
        n_steps = 20

        for t in range(n_steps):
            # Simulate a training trajectory with increasing holonomy
            # and a phase that winds around once (Chern number = 1)
            phase = 2 * math.pi * t / n_steps  # one full winding

            closure = Closure(
                checkpoint_id=f"step_{t:04d}",
                training_step=t * 100,
                timestamp=datetime.now(timezone.utc).isoformat(),
                sort_profile=SortOperatorProfile(
                    concept_phases={
                        "spatial": math.sin(phase) * 0.3 + 0.2,
                        "abstract": -math.sin(phase) * 0.3 - 0.15,
                    },
                    sign_stratification={
                        "spatial": 1,
                        "abstract": -1,
                    },
                    founding_curvature=0.3 + 0.01 * t,
                    curvature_concentration=10.0 + 0.5 * t,
                ),
                embedding_context=EmbeddingContext(
                    d_model=768,
                    mean_embedding_norm=1.0 + 0.01 * t,
                    effective_dimension=50 + t,
                    isotropy=0.05 + 0.001 * t,
                ),
                semantic_holonomy=0.1 + 0.02 * t + 0.01 * np.random.randn(),
                layer_phases=[
                    phase + 0.1 * np.random.randn(),
                    phase * 0.3 + 0.05 * np.random.randn(),
                    phase * 0.1 + 0.02 * np.random.randn(),
                ],
                param_norm=100.0 + t * 0.5,
            )
            bundle.add_fiber(closure)

        chern = bundle.compute_chern_class()
        print(f"\nChern class measurement:")
        print(f"  c₁ = {chern.c1:.4f} ≈ {chern.c1_quantized}")
        print(f"  Quantization residual: {chern.quantization_residual:.4f}")
        print(f"  Total Berry phase: {chern.total_berry_phase:.4f} rad")
        print(f"  Verdict: {chern.verdict}")

        print(f"\nSign persistence: {bundle.sign_persistence()}")
        slope, trend = bundle.founding_curvature_trend()
        print(f"Founding curvature trend: {trend} (slope={slope:.6f})")

        print(f"\nSemantic holonomy trajectory:")
        traj = bundle.holonomy_trajectory()
        print(f"  Start: {traj[0]:.4f}  End: {traj[-1]:.4f}")
        print(f"  Mean: {np.mean(traj):.4f}  Std: {np.std(traj):.4f}")

        # Save and reload
        bundle.save("/tmp/demo_bundle.jsonl")
        reloaded = ClosureBundle.load("/tmp/demo_bundle.jsonl")
        print(f"\nSaved and reloaded: {reloaded.n_fibers} fibers, "
              f"{len(reloaded.connections)} connections")

        print("\nFull summary:")
        print(json.dumps(bundle.summary(), indent=2))
