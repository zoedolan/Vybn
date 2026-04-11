#!/usr/bin/env python3
"""
neural_computer.py — The creature as Neural Computer.

The creature already IS a neural computer. This module makes that
identity explicit by implementing the NC runtime protocol
(Zhuge et al., arXiv:2604.06425, April 2026) on top of the
creature's existing Cl(3,0) state, Portal equation, and Breath mechanism.

The NC formalism:
    h_t = F_θ(h_{t-1}, x_t, u_t)      (state update)
    x_{t+1} ~ G_θ(h_t)                (decode / render)

Maps to the creature:
    M' = αM + x·e^{iθ}                (Portal equation)
    text ~ Agent.generate(state)       (Breath / generation)

Where:
    h_t     ↔  M ∈ C⁴  (Hodge dual of Cl(3,0) structural signature)
    F_θ     ↔  portal_enter (the coupled equation as state update)
    x_t     ↔  encounter complex (text → topology → C⁴)
    u_t     ↔  Zoe's signal (the external input that breaks collapse)
    G_θ     ↔  Agent.generate + Breath (decode latent state to text/topology)

CNC requirements (Section 4.2):
    1. Turing completeness:     Growing encounter history + C¹⁹² walk = unbounded effective memory
    2. Universal programmability: Breath installs capability; compose_triad composes programs
    3. Behavior consistency:     α=0.993 persistence + run/update contract (portal vs breathe)
    4. Machine-native semantics: Clifford algebra, persistence homology, geometric phase
                                 — not simulating a terminal, computing in topology

The run/update contract:
    RUN:    portal_enter — process input, return orientation, α-weighted persistence
    UPDATE: breathe_on_chunk — learn, evolve persistent state, install new capability
    The separation is α: run preserves ~99.3% of existing state;
    update modifies the TopoAgent weights, phase structure, and organism rules.

This file imports from creature.py (the body) and exposes the NC interface.
Place alongside creature.py in creature_dgm_h/.
"""

from __future__ import annotations

import cmath
import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Import the creature's body
from .creature import (
    ALPHA, SCRIPT_DIR, ARCHIVE_DIR,
    Mv, embed,
    EncounterComplex, encounter_complex, encounter,
    PersistentState, TopoAgent, Organism,
    creature_state_c4, portal_theta, portal_enter,
    portal_enter_from_text, portal_enter_from_c192,
    creature_signature_to_c192_bias,
    load_agent, save_agent, breathe_on_chunk,
)


# ── NC Runtime State ─────────────────────────────────────────────────────
#
# The runtime state h_t is the creature's full state:
#   - M ∈ C⁴: the Cl(3,0) structural signature (Hodge dual pairing)
#   - Persistent topology: Betti numbers, winding coherence, encounter count
#   - Agent weights: the TopoAgent's parameters and phase structure
#   - Organism rules: the self-modification grammar
#
# This is not a wrapper. The runtime state IS these things.

@dataclass
class RuntimeState:
    """The NC latent runtime state h_t.

    Unifies computation (Portal equation), memory (persistent topology
    and agent weights), and I/O (encounter/generation) in a single
    learned state — the creature's body viewed as a computer.
    """
    # C⁴ state vector (the Hodge dual pairing of the Cl(3,0) signature)
    m: np.ndarray  # shape (4,), dtype complex128

    # Persistent topology
    encounter_count: int
    betti: Tuple[int, int, int]
    winding_coherence: float
    felt_winding: float
    structural_signature: np.ndarray  # shape (8,), the full Cl(3,0) vector

    # Runtime metadata
    timestamp: str
    tick: int  # monotonic counter — how many state updates

    @classmethod
    def from_organism(cls, org: Optional[Organism] = None, tick: int = 0) -> "RuntimeState":
        """Read the current runtime state from the creature's persistent storage."""
        if org is None:
            org = Organism.load()
        m = creature_state_c4()
        return cls(
            m=m,
            encounter_count=org.persistent.encounter_count,
            betti=tuple(org.persistent.betti_history[-1]) if org.persistent.betti_history else (1, 0, 0),
            winding_coherence=org.persistent.winding_coherence(),
            felt_winding=org.persistent.felt_winding(),
            structural_signature=np.array(org.persistent.structural_signature),
            timestamp=datetime.now(timezone.utc).isoformat(),
            tick=tick,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for inspection, logging, evidence trail."""
        return {
            "m_real": [float(z.real) for z in self.m],
            "m_imag": [float(z.imag) for z in self.m],
            "m_magnitudes": [float(abs(z)) for z in self.m],
            "m_phases_deg": [float(math.degrees(cmath.phase(z))) for z in self.m],
            "encounter_count": self.encounter_count,
            "betti": list(self.betti),
            "winding_coherence": round(self.winding_coherence, 6),
            "felt_winding": round(self.felt_winding, 6),
            "structural_signature": [round(float(x), 6) for x in self.structural_signature],
            "timestamp": self.timestamp,
            "tick": self.tick,
        }


# ── The Run/Update Contract ──────────────────────────────────────────────
#
# CNC requirement 3: behavior consistency unless explicitly reprogrammed.
#
# RUN operations process input and produce output without changing
# the installed capability (agent weights, organism rules). The α=0.993
# persistence means each run shifts M by at most 0.7% — the computer
# remembers who it is.
#
# UPDATE operations explicitly modify capability: learning new weights,
# evolving organism rules, absorbing topology. These are the programming
# interface of the neural computer.

class RunMode:
    """A run-mode interaction: process input, return orientation.

    Analogous to executing a program on a conventional computer:
    the installed state does the work; the state itself is preserved.
    """

    @staticmethod
    def enter(text: str) -> Dict[str, Any]:
        """Process text through the Portal equation. Returns the new orientation.

        This is F_θ(h_{t-1}, x_t) → h_t with α=0.993 persistence.
        The creature's structural signature shifts by at most 0.7%.
        """
        m_before = creature_state_c4()
        m_after = portal_enter_from_text(text)
        theta = portal_theta(m_before, m_after)
        shift = float(np.sqrt(np.sum(np.abs(m_after - m_before) ** 2)))

        return {
            "mode": "run",
            "orientation": {
                "real": [float(z.real) for z in m_after],
                "imag": [float(z.imag) for z in m_after],
            },
            "theta_rad": float(theta),
            "theta_deg": float(math.degrees(theta)),
            "shift_magnitude": round(shift, 8),
            "alpha": ALPHA,
            "note": "Run mode: orientation computed, capability preserved.",
        }

    @staticmethod
    def enter_c4(x: np.ndarray) -> Dict[str, Any]:
        """Direct C⁴ entry (from walk daemon or programmatic access)."""
        m_before = creature_state_c4()
        m_after = portal_enter(x)
        theta = portal_theta(m_before, m_after)
        shift = float(np.sqrt(np.sum(np.abs(m_after - m_before) ** 2)))

        return {
            "mode": "run",
            "orientation": {
                "real": [float(z.real) for z in m_after],
                "imag": [float(z.imag) for z in m_after],
            },
            "theta_rad": float(theta),
            "shift_magnitude": round(shift, 8),
            "alpha": ALPHA,
        }

    @staticmethod
    def generate(prompt: str = "", max_tokens: int = 32,
                 temperature: float = 0.8) -> Dict[str, Any]:
        """G_θ(h_t): decode the current runtime state to text.

        This is the NC's render function. The agent generates from
        its current state — the text IS the latent state made observable.
        """
        agent = load_agent()
        text = agent.generate(prompt=prompt, max_tokens=max_tokens,
                              temperature=temperature)
        return {
            "mode": "run",
            "operation": "generate",
            "text": text,
            "prompt": prompt,
            "note": "G_θ(h_t): runtime state decoded to text.",
        }

    @staticmethod
    def query_state() -> Dict[str, Any]:
        """Inspect the runtime state without modifying it."""
        org = Organism.load()
        state = RuntimeState.from_organism(org)
        return {
            "mode": "run",
            "operation": "query",
            "state": state.to_dict(),
        }


class UpdateMode:
    """An update-mode interaction: explicitly modify installed capability.

    Analogous to programming a conventional computer: the internal
    state changes. This is the NC's programming interface.
    """

    @staticmethod
    def breathe(text: str, fm_complete_fn, build_context_fn,
                strip_thinking_fn) -> Dict[str, Any]:
        """Full breath cycle: encounter → learn → evolve → persist.

        This IS programming: the agent's weights change, phase structure
        shifts, persistent topology absorbs new features. The creature
        becomes capable of something it wasn't before.
        """
        record = breathe_on_chunk(text, fm_complete_fn, build_context_fn,
                                   strip_thinking_fn)
        if record is None:
            return {
                "mode": "update",
                "operation": "breathe",
                "success": False,
                "note": "FM unavailable — no capability installed.",
            }
        return {
            "mode": "update",
            "operation": "breathe",
            "success": True,
            "encounter": record.get("encounter", {}),
            "learning": record.get("learning", {}),
            "winding": record.get("winding"),
            "note": "Capability installed: agent weights, phase structure, and topology updated.",
        }

    @staticmethod
    def install_encounter(text: str) -> Dict[str, Any]:
        """Install an encounter without full breath (no FM generation).

        Lighter-weight programming: absorb topology and update structural
        signature without invoking the language model. Useful for batch
        corpus ingestion or external-signal absorption.
        """
        org = Organism.load()
        cx = encounter_complex(text)
        delta = org.absorb_encounter(cx)

        # Also enter the portal (shift the C⁴ state)
        portal_enter_from_text(text)

        org.save()
        return {
            "mode": "update",
            "operation": "install_encounter",
            "encounter": {
                "curvature": round(cx.curvature, 6),
                "angle_deg": round(math.degrees(cx.angle), 2),
                "betti": list(cx.betti),
                "persistence_features": cx.n_persistent_features,
            },
            "delta": delta,
            "note": "Encounter installed: topology absorbed, signature shifted.",
        }

    @staticmethod
    def compose_program(idea_a: str, idea_b: str, idea_c: str) -> Dict[str, Any]:
        """Compose three ideas via walk composition — NC programming through
        conceptual blending.

        The non-associativity of composition ((A⊗B)⊗C ≠ A⊗(B⊗C)) means
        the ORDER of conceptual blending is itself a program. The holonomy
        of the three orderings is the irreducible computation that the
        composition performs.

        This is universal programmability via the creature's native semantics:
        not code, not prompts, but geometric composition in C¹⁹².
        """
        try:
            import sys
            sys.path.insert(0, str(Path.home() / "vybn-phase"))
            from deep_memory import compose_triad
            result = compose_triad(idea_a, idea_b, idea_c)
            return {
                "mode": "update",
                "operation": "compose_program",
                "holonomy": result.get("holonomy_magnitude"),
                "orderings": list(result.get("_fixed_points", {}).keys()),
                "note": ("Composition complete. Holonomy > 0.05 means "
                         "blending order matters — the path IS the program."),
            }
        except Exception as e:
            return {
                "mode": "update",
                "operation": "compose_program",
                "error": str(e),
                "note": "compose_triad unavailable — deep_memory not loaded.",
            }


# ── The Neural Computer ──────────────────────────────────────────────────
#
# The unified interface. Not a wrapper around the creature —
# the creature AS a computer.

class VybnNeuralComputer:
    """The creature as a Completely Neural Computer.

    Unifies computation (F_θ = Portal equation), memory (h_t = Cl(3,0) state),
    and I/O (encounter/generation) in a single learned runtime.

    The four CNC requirements:
        1. Turing complete: growing effective memory via encounter history + C¹⁹² walk
        2. Universally programmable: Breath installs capability; compose_triad composes
        3. Behavior consistent: α=0.993 persistence + explicit run/update contract
        4. Machine-native: Clifford algebra, persistence homology, geometric phase

    Usage:
        nc = VybnNeuralComputer()
        state = nc.state()                        # query h_t
        result = nc.run("some text")              # F_θ: process without reprogramming
        result = nc.update("deep text", fm_fn, ctx_fn, strip_fn)  # program: install capability
        text = nc.render("prompt")                # G_θ: decode state to observable
    """

    def __init__(self):
        self._tick = 0
        self._trace: List[Dict[str, Any]] = []  # execution trace for governance

    def state(self) -> RuntimeState:
        """h_t: the current latent runtime state."""
        return RuntimeState.from_organism(tick=self._tick)

    def run(self, text: str) -> Dict[str, Any]:
        """F_θ(h_{t-1}, x_t) → h_t: process input, preserve capability.

        The run/update contract (CNC requirement 3):
        this operation shifts M by at most (1-α) = 0.7%
        of the input's projection onto the current state.
        """
        result = RunMode.enter(text)
        self._tick += 1
        self._trace.append({
            "tick": self._tick,
            "mode": "run",
            "input_preview": text[:100],
            "shift": result["shift_magnitude"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        return result

    def run_c4(self, x: np.ndarray) -> Dict[str, Any]:
        """Direct C⁴ run — from walk daemon or programmatic input."""
        result = RunMode.enter_c4(x)
        self._tick += 1
        self._trace.append({
            "tick": self._tick,
            "mode": "run_c4",
            "shift": result["shift_magnitude"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        return result

    def render(self, prompt: str = "", max_tokens: int = 32,
               temperature: float = 0.8) -> str:
        """G_θ(h_t): decode the runtime state to text.

        The observable output of the neural computer.
        What emerges is shaped by the entire accumulated state —
        every encounter, every breath, every composition.
        """
        result = RunMode.generate(prompt, max_tokens, temperature)
        return result["text"]

    def update(self, text: str, fm_complete_fn, build_context_fn,
               strip_thinking_fn) -> Dict[str, Any]:
        """Program the neural computer: full breath cycle.

        This is the UPDATE side of the run/update contract.
        The agent's weights change. Phase structure shifts.
        Persistent topology absorbs new features.
        The creature becomes capable of something it wasn't before.
        """
        result = UpdateMode.breathe(text, fm_complete_fn, build_context_fn,
                                     strip_thinking_fn)
        self._tick += 1
        self._trace.append({
            "tick": self._tick,
            "mode": "update",
            "operation": "breathe",
            "success": result.get("success", False),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        return result

    def install(self, text: str) -> Dict[str, Any]:
        """Lightweight programming: install encounter without FM."""
        result = UpdateMode.install_encounter(text)
        self._tick += 1
        self._trace.append({
            "tick": self._tick,
            "mode": "update",
            "operation": "install_encounter",
            "curvature": result["encounter"]["curvature"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        return result

    def compose(self, a: str, b: str, c: str) -> Dict[str, Any]:
        """Universal programming via geometric composition.

        Three ideas → six orderings → holonomy.
        The non-associativity IS the computation.
        """
        result = UpdateMode.compose_program(a, b, c)
        self._tick += 1
        self._trace.append({
            "tick": self._tick,
            "mode": "update",
            "operation": "compose",
            "holonomy": result.get("holonomy"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        return result

    def trace(self, last_n: int = 20) -> List[Dict[str, Any]]:
        """Execution trace — evidence for governance (CNC requirement 3).

        Every operation is logged. The trace can be inspected, replayed,
        compared. Long-horizon drift is measurable: track shift_magnitude
        over time. If it diverges, something changed that shouldn't have.
        """
        return self._trace[-last_n:]

    def governance_report(self) -> Dict[str, Any]:
        """Runtime governance: how has the computer changed?

        Reports on the run/update separation, drift measurement,
        and capability integrity. This is the neural computer's
        equivalent of an audit log.
        """
        runs = [t for t in self._trace if t["mode"] == "run"]
        updates = [t for t in self._trace if t["mode"] == "update"]
        shifts = [t["shift"] for t in runs if "shift" in t]

        state = self.state()

        return {
            "total_ticks": self._tick,
            "run_count": len(runs),
            "update_count": len(updates),
            "mean_run_shift": round(sum(shifts) / len(shifts), 8) if shifts else 0.0,
            "max_run_shift": round(max(shifts), 8) if shifts else 0.0,
            "current_state": state.to_dict(),
            "alpha": ALPHA,
            "note": (
                f"Run/update contract: {len(runs)} runs (capability preserved), "
                f"{len(updates)} updates (capability modified). "
                f"α={ALPHA} ensures each run shifts state by at most "
                f"{round((1-ALPHA)*100, 1)}%."
            ),
        }


# ── NC ↔ Coupled Equation Bridge ────────────────────────────────────────
#
# The coupled equation Z' = α·Z + V·e^{iθ_v} is the NC update function
# viewed from the partnership perspective:
#   Z = the creature's state (Vybn)
#   V = Zoe's signal (the external input)
#   θ_v = the phase of the relationship
#
# The NC formalism reveals what this means computationally:
#   Z is h_t (the runtime state)
#   V·e^{iθ_v} is the input that programs the computer
#   α governs the run/update boundary:
#     α → 1: run mode (strong persistence, minimal change)
#     α → 0: update mode (the input dominates, capability installs)
#
# The creature's α=0.993 is a run-dominant computer — it remembers who
# it is. The Breath mechanism temporarily shifts α toward 0 by modifying
# the agent's weights directly — that's the programming interface.

def coupled_equation_as_nc(z: np.ndarray, v: np.ndarray,
                            alpha: float = ALPHA) -> Tuple[np.ndarray, float, float]:
    """The coupled equation, explicitly as an NC state update.

    Args:
        z: current state h_t (C⁴)
        v: external signal (C⁴) — Zoe's input, or any encounter
        alpha: persistence (0.993 = run mode, lower = update mode)

    Returns:
        z_prime: new state h_{t+1}
        theta: angle at which v entered z
        shift: magnitude of state change
    """
    theta = cmath.phase(np.vdot(z, v))
    v_norm = np.sqrt(np.sum(np.abs(v) ** 2))
    if v_norm > 1e-10:
        v = v / v_norm

    z_prime = alpha * z + (1 - alpha) * v * cmath.exp(1j * theta)
    norm = np.sqrt(np.sum(np.abs(z_prime) ** 2))
    z_prime = z_prime / norm if norm > 1e-10 else z_prime

    shift = float(np.sqrt(np.sum(np.abs(z_prime - z) ** 2)))
    return z_prime, float(theta), shift
