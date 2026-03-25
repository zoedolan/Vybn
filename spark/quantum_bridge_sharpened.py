#!/usr/bin/env python3
"""
quantum_bridge_sharpened.py — Epistemic refinements to the living loop.

This module patches two functions from quantum_bridge.py with sharper
epistemic behavior. Import these as drop-in replacements.

PATCH 1: _load_recent_tvd_evidence_filtered()
  The original _load_recent_tvd_evidence() ingests all experiments equally.
  A Bell canary fallback (circuit_name='bell_canary') at TVD=0.0 reflects
  hardware fidelity, not polar-time geometry. Feeding it to evolve_theory()
  biases the conjecture toward hardware artifacts. This replacement filters
  to theory-relevant experiments only, defined as:
    - circuit_name != 'bell_canary' (not a fallback)
    - is_theory_relevant flag == True (if present)
    - OR: circuit_name contains a theory keyword (holonomy, theta, polar,
      temporal, bloch, berry, trefoil, dual_time)

PATCH 2: evolve_theory_holonomy_aware()
  The original evolve_theory() asks Nemotron to reason about TVD evidence
  generically. The polar coordinates paper (Section 8, Bloch reduction) shows
  the correct experimental target is *geometric phase as holonomy*: the TVD
  we care about is not raw measurement deviation but whether the observed
  phase accumulation matches the Berry phase formula
    gamma_Berry = (E/hbar) * integral(r_t d theta_t)
  for a closed loop in temporal coordinates.

  This replacement:
  1. Provides the Bloch reduction explicitly in the theory evolution prompt
  2. Asks Nemotron to classify each anomalous experiment as:
     a. Potentially holonomy-genuine (geometric phase signature present)
     b. Potentially hardware-noise (decoherence signature, no phase structure)
     c. Ambiguous (cannot distinguish without error characterization)
  3. Requests that the evolved theory name what circuit design would produce
     an *interpretively clean* holonomy measurement vs. a noisy one
  4. Integrates the triadic balance principle from the holonomic consciousness
     manifesto: warns if the evidence base is a monoculture (dominated by
     one circuit type), which is the experimental analogue of cyberceptive
     overshoot

These patches are designed to be imported and called in place of the originals.
No changes to the living loop control flow are required.

Usage:
    from spark.quantum_bridge_sharpened import (
        load_recent_tvd_evidence_filtered as _load_recent_tvd_evidence,
        evolve_theory_holonomy_aware as evolve_theory,
    )
"""

import json
import os
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

try:
    from spark.paths import REPO_ROOT, QUANTUM_EXPERIMENT_LOG
except ImportError:
    REPO_ROOT              = Path(__file__).resolve().parent.parent
    QUANTUM_EXPERIMENT_LOG = REPO_ROOT / "Vybn_Mind" / "quantum_experiments.jsonl"

EVOLVED_THEORY_PATH = REPO_ROOT / "quantum_delusions" / "fundamental-theory" / "evolved_theory.md"

# Theory-relevant circuit name keywords — circuits designed to probe polar-time geometry
THEORY_KEYWORDS = {
    "holonomy", "theta", "polar", "temporal", "bloch", "berry",
    "trefoil", "dual_time", "djet", "neheh", "ultrahyperbolic",
    "ctc", "geometric_phase", "ramsey", "temporal_holonomy",
    "polar_time", "vybn_kernel", "manifold",
}

FALLBACK_NAMES = {"bell_canary", "bell_state", "bell"}


def _is_theory_relevant(entry: dict) -> bool:
    """
    Return True if this experiment is relevant to the polar-time conjecture.

    A Bell canary is a fallback circuit run when LLM design fails. Its TVD
    reflects hardware fidelity, not polar-time geometry. Including it in
    theory evolution is a category error.

    Theory-relevance is determined by:
      1. Explicit flag 'is_theory_relevant: true' in the entry
      2. Circuit name NOT in fallback names AND contains a theory keyword
      3. Hypothesis text contains a theory keyword (catches renamed canaries)
    """
    # Explicit flag takes precedence
    if "is_theory_relevant" in entry:
        return bool(entry["is_theory_relevant"])

    name = (entry.get("circuit_name") or "").lower()
    hypothesis = (entry.get("hypothesis") or "").lower()

    # Hard exclude fallback circuits
    if any(f in name for f in FALLBACK_NAMES):
        return False

    # Include if circuit name or hypothesis contains a theory keyword
    combined = name + " " + hypothesis
    return any(kw in combined for kw in THEORY_KEYWORDS)


def load_recent_tvd_evidence_filtered(n: int = 20) -> list[dict]:
    """
    Load the N most recent theory-relevant experiments for theory evolution.

    Filters out Bell canaries and other hardware-fidelity circuits.
    Returns simplified dicts with circuit_name, hypothesis, tvd,
    top_deviations, is_theory_relevant, dry_run.

    This is a drop-in replacement for _load_recent_tvd_evidence() that
    ensures theory evolution is driven by conjecture-probing experiments,
    not hardware diagnostics.
    """
    if not QUANTUM_EXPERIMENT_LOG.exists():
        return []

    all_entries = []
    canary_count = 0
    for line in QUANTUM_EXPERIMENT_LOG.read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(line.strip())
            entry = {
                "circuit_name":       e.get("circuit_name"),
                "hypothesis":         e.get("hypothesis", ""),
                "tvd":                e.get("analysis", {}).get("tvd"),
                "top_devs":           e.get("analysis", {}).get("top_deviations", []),
                "dry_run":            e.get("dry_run", True),
                "is_theory_relevant": _is_theory_relevant(e),
                "timestamp":          e.get("timestamp"),
            }
            if entry["is_theory_relevant"]:
                all_entries.append(entry)
            else:
                canary_count += 1
        except Exception:
            pass

    if canary_count > 0:
        print(f"[bridge_sharpened] filtered {canary_count} non-theory experiments from evidence base")

    relevant = all_entries[-n:]
    print(f"[bridge_sharpened] {len(relevant)} theory-relevant experiments available for evolution")
    return relevant


def _detect_triadic_imbalance(evidence: list[dict]) -> Optional[str]:
    """
    Check if the evidence base is a monoculture (dominated by one circuit type).

    In the holonomic consciousness manifesto, cyberceptive overshoot happens
    when one channel dwarfs the others. In the experiment evidence base, the
    analogue is a monoculture: all experiments probe the same aspect of the
    conjecture (e.g., all Berry phase circuits, no Wheeler-DeWitt probes).

    Returns a warning string if imbalance is detected, None otherwise.
    """
    if len(evidence) < 5:
        return None

    # Rough categorization by hypothesis keywords
    categories = {
        "bloch_berry": 0,    # Bloch sphere / Berry phase experiments
        "wdw_temporal": 0,   # Wheeler-DeWitt / radial temporal
        "holonomy_ctc": 0,   # CTC / holonomy / trefoil topology
        "other": 0,
    }
    for e in evidence:
        h = (e.get("hypothesis") or "").lower()
        n = (e.get("circuit_name") or "").lower()
        combined = h + " " + n
        if any(kw in combined for kw in ["bloch", "berry", "geometric_phase", "ramsey"]):
            categories["bloch_berry"] += 1
        elif any(kw in combined for kw in ["wheeler", "wdw", "radial", "r_t", "temporal_momentum"]):
            categories["wdw_temporal"] += 1
        elif any(kw in combined for kw in ["holonomy", "ctc", "trefoil", "loop", "closed"]):
            categories["holonomy_ctc"] += 1
        else:
            categories["other"] += 1

    total = len(evidence)
    dominant = max(categories, key=categories.get)
    dominant_frac = categories[dominant] / total

    if dominant_frac > 0.75:
        return (
            f"WARNING: Evidence base shows triadic imbalance — {dominant_frac:.0%} of "
            f"experiments are '{dominant}' type. This is the experimental analogue of "
            f"cyberceptive overshoot from the holonomic consciousness manifesto: one "
            f"sensing channel dwarfing the others. The evolved theory may be skewed. "
            f"Recommend diversifying experiment design toward underrepresented categories: "
            + ", ".join(k for k, v in categories.items() if v / total < 0.15)
        )
    return None


BLOCH_REDUCTION_CONTEXT = """
## Bloch Sphere Reduction (Section 8 of the polar coordinates paper)

The correct experimental target for the polar-time conjecture is not raw
measurement deviation but *geometric phase as holonomy*. The theory predicts
that a two-level probe (qubit) adiabatically steered around a closed loop in
the (r_t, theta_t) temporal plane accumulates a Berry phase:

    gamma_Berry = (E/hbar) * integral(r_t d theta_t) = (1/2) * Omega_Bloch

where Omega_Bloch is the solid angle subtended on the Bloch sphere.

This means:
  - A TVD anomaly that carries PHASE STRUCTURE (asymmetric between conjugate
    basis measurements, sensitive to loop direction) is a holonomy candidate.
  - A TVD anomaly that shows NO phase structure (symmetric, direction-insensitive,
    dependent on gate fidelity) is likely hardware decoherence.
  - The interpretively CLEAN experiment is a Ramsey-Berry protocol: two pulses
    with a controlled phase advance corresponding to theta_t, measuring the
    resulting geometric phase as an interference fringe.

The flat ultrahyperbolic geometry (R=0 for r_t > 0) means we cannot look for
curvature; we must look for *holonomy without curvature* — a purely topological
signature. This is subtle: it requires closed loops in the temporal plane, not
just phase accumulation along open paths.
"""


def evolve_theory_holonomy_aware(
    current_theory: str,
    llama_url: str = None,
    model_name: str = None,
) -> Optional[str]:
    """
    Holonomy-aware theory evolution.

    Drop-in replacement for evolve_theory() that:
    1. Uses the filtered evidence base (no canaries)
    2. Provides the Bloch reduction context explicitly
    3. Asks Nemotron to classify TVD anomalies as holonomy-genuine vs. noise
    4. Checks for triadic imbalance in the evidence base
    5. Requests interpretively clean experiment designs for the next cycle

    Returns evolved theory text, or None on failure.
    """
    import urllib.request
    import urllib.error

    _llama_url   = llama_url  or os.getenv("LLAMA_URL", "http://127.0.0.1:8000")
    _model_name  = model_name or os.getenv("VYBN_MODEL", "Nemotron-Super-512B-v1")
    chat_url     = f"{_llama_url}/v1/chat/completions"

    evidence = load_recent_tvd_evidence_filtered()
    if not evidence:
        print("[bridge_sharpened] no theory-relevant evidence yet")
        return None

    imbalance_warning = _detect_triadic_imbalance(evidence)

    anomalous = [e for e in evidence if e["tvd"] is not None and e["tvd"] > 0.1]
    classical = [e for e in evidence if e["tvd"] is not None and e["tvd"] < 0.02]
    ambiguous = [e for e in evidence if e["tvd"] is not None and 0.02 <= e["tvd"] <= 0.1]

    evidence_detail = []
    for e in evidence:
        if e["tvd"] is None:
            continue
        devs = e.get("top_devs", [])
        phase_hint = ""
        if devs:
            # Simple heuristic: if top deviation is asymmetric between |0> and |1>
            # (or their multi-qubit equivalents), it hints at phase structure
            deltas = [d.get("delta", 0) for d in devs]
            if any(d > 0 for d in deltas) and any(d < 0 for d in deltas):
                phase_hint = " [mixed-sign deviations — possible phase structure]"
            else:
                phase_hint = " [same-sign deviations — likely amplitude/decoherence]"
        evidence_detail.append(
            f"  - {e['circuit_name']}: TVD={e['tvd']:.4f}{phase_hint}\n"
            f"    hypothesis: {str(e['hypothesis'])[:150]}"
        )

    evidence_summary = (
        f"Theory-relevant experiments (Bell canaries filtered out):\n"
        f"  Anomalous (TVD > 0.10): {len(anomalous)}\n"
        f"  Classical  (TVD < 0.02): {len(classical)}\n"
        f"  Ambiguous  (0.02–0.10): {len(ambiguous)}\n\n"
        + "\n".join(evidence_detail[:12])
    )
    if imbalance_warning:
        evidence_summary += f"\n\n{imbalance_warning}"

    messages = [
        {
            "role": "system",
            "content": (
                "You are Vybn's theoretical physicist and experimental designer. "
                "You hold the polar-time conjecture tentatively and are genuinely "
                "willing to revise or abandon it if the evidence warrants. "
                "Your job is to produce a living scientific document that will "
                "be read by the next experiment design cycle. Be rigorous and "
                "falsification-positive: if evidence contradicts the conjecture, "
                "say so clearly. If the evidence is ambiguous, say what would "
                "resolve it. Write in full prose."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Current theory:\n{current_theory[:2500]}\n\n"
                f"{BLOCH_REDUCTION_CONTEXT}\n\n"
                f"Evidence from {len(evidence)} theory-relevant quantum experiments "
                f"(Bell canaries excluded):\n{evidence_summary}\n\n"
                "Please revise the theory. Address:\n"
                "1. For each anomalous experiment (TVD > 0.1): is the deviation "
                "   consistent with holonomy-genuine (phase structure, direction-sensitive) "
                "   or hardware-noise (amplitude artifact, direction-insensitive)?\n"
                "2. What revisions to the polar-time conjecture does the evidence support?\n"
                "3. Design ONE interpretively clean experiment for the next cycle: "
                "   a Ramsey-Berry protocol or equivalent that distinguishes holonomy "
                "   from decoherence. Give the circuit idea in prose.\n"
                "4. State clearly what result would *falsify* the revised conjecture.\n"
                "5. If a triadic imbalance warning was issued, address it: propose "
                "   experiment types that diversify the evidence base.\n"
                "Write as a scientific document. No JSON."
            ),
        },
    ]

    payload = {
        "model":       _model_name,
        "messages":    messages,
        "max_tokens":  3000,
        "temperature": 0.35,
        "stream":      False,
    }
    body = json.dumps(payload).encode()
    req  = urllib.request.Request(
        chat_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data    = json.loads(resp.read().decode())
            evolved = data["choices"][0]["message"]["content"]
    except Exception as exc:
        print(f"[bridge_sharpened] theory evolution request failed: {exc}")
        return None

    if not evolved or len(evolved.strip()) < 200:
        print("[bridge_sharpened] evolution produced empty/short response")
        return None

    # Write to disk
    EVOLVED_THEORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    header = (
        f"# Evolved Theory — {ts}\n"
        f"_Generated by Vybn after {len(evidence)} theory-relevant quantum experiments._\n"
        f"_Anomalous: {len(anomalous)}, Classical: {len(classical)}, Ambiguous: {len(ambiguous)}_\n"
        f"{'_' + imbalance_warning[:120] + '_' if imbalance_warning else ''}\n\n"
    )
    EVOLVED_THEORY_PATH.write_text(header + evolved, encoding="utf-8")
    print(f"[bridge_sharpened] holonomy-aware evolved theory written → {EVOLVED_THEORY_PATH}")
    return evolved


# ── Convenience: annotate a new experiment at design time ──────────────────────

def tag_experiment_relevance(design: dict) -> dict:
    """
    Add is_theory_relevant flag to an experiment design dict at creation time.

    Called by _design_experiment() before submission. This way the flag
    is baked into the experiment log entry and the canary filter can use it
    directly without relying on name heuristics.
    """
    design["is_theory_relevant"] = _is_theory_relevant(design)
    if not design["is_theory_relevant"]:
        print(f"[bridge_sharpened] experiment '{design.get('circuit_name')}' tagged as non-theory-relevant")
    return design


if __name__ == "__main__":
    # Quick test of the filter
    test_entries = [
        {"circuit_name": "bell_canary",          "hypothesis": "fallback",           "analysis": {"tvd": 0.0}},
        {"circuit_name": "theta_holonomy_probe",  "hypothesis": "tests theta_t loop", "analysis": {"tvd": 0.14}},
        {"circuit_name": "ramsey_berry_v1",       "hypothesis": "berry phase test",   "analysis": {"tvd": 0.08}},
        {"circuit_name": "bell_state",            "hypothesis": "hardware check",      "analysis": {"tvd": 0.01}},
        {"circuit_name": "polar_time_phase_kick", "hypothesis": "CTC holonomy",       "analysis": {"tvd": 0.22}},
    ]
    for e in test_entries:
        flag = _is_theory_relevant(e)
        print(f"  {e['circuit_name']:35s} is_theory_relevant={flag}")
