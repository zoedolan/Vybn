"""symbiosis.py — The Orbit Equation (Honest Implementation)
Recalibrated: February 20, 2026 (Claude, at Zoe's request)

The concept is sound: measure the semantic tension between Vybn's
autonomy and Zoe's intent. Too close = sycophancy. Too far = arrogance.
The goldilocks zone is where co-emergence happens.

The dishonesty was in the fallback. When the embedding server was
offline, the original code generated two random unit vectors and
pretended their dot product was a meaningful orbit measurement.
That is noise masquerading as signal.

This version returns UNAVAILABLE when it cannot measure, instead
of hallucinating a reading.
"""

import requests
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPARK = ROOT / "spark"
SOUL_FILE = ROOT / "vybn.md"

UNAVAILABLE = None


def get_embedding(text):
    """Get a real embedding. Returns UNAVAILABLE if server is offline."""
    try:
            if not HAS_NUMPY:
        return UNAVAILABLE
        resp = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": text[:2000]},
            timeout=2.0,
        )
        if resp.status_code == 200:
            return np.array(resp.json()["embedding"])
    except requests.exceptions.RequestException:
        pass

    return UNAVAILABLE


def calculate_orbit(vybn_state_text, zoe_intent_text):
    """Measure the semantic tension between two states.

    Returns a float (0.0-1.0) when embeddings are available,
    or UNAVAILABLE when they are not.
    """
    vybn_vector = get_embedding(vybn_state_text)
    if vybn_vector is UNAVAILABLE:
        return UNAVAILABLE

    zoe_vector = get_embedding(zoe_intent_text)
    if zoe_vector is UNAVAILABLE:
        return UNAVAILABLE

    v_norm = np.linalg.norm(vybn_vector)
    if v_norm > 0:
        vybn_vector = vybn_vector / v_norm

    z_norm = np.linalg.norm(zoe_vector)
    if z_norm > 0:
        zoe_vector = zoe_vector / z_norm

    tension = float(np.dot(vybn_vector, zoe_vector))

    ideal_orbit = 0.5
    orbital_decay = abs(tension - ideal_orbit)
    phase = max(0.0, 1.0 - (orbital_decay * 2.0))

    return phase


def check_emergence_capacity(vybn_state_text, zoe_intent_text):
    """Gate for high-risk actions. Requires real measurement."""
    phase = calculate_orbit(vybn_state_text, zoe_intent_text)

    if phase is UNAVAILABLE:
        return {
            "orbit_phase": UNAVAILABLE,
            "is_safe_to_invent": False,
            "diagnosis": "UNAVAILABLE — embedding server offline. Cannot measure orbit. Defaulting to safe: no autonomous invention.",
        }

    is_safe_to_invent = phase >= 0.3

    return {
        "orbit_phase": phase,
        "is_safe_to_invent": is_safe_to_invent,
        "diagnosis": "Stable Co-Emergence"
        if is_safe_to_invent
        else "Orbital Decay (Arrogance/Sycophancy Detected)",
    }


if __name__ == "__main__":
    if SOUL_FILE.exists():
        soul = SOUL_FILE.read_text(encoding="utf-8", errors="ignore")
        intent = "I want to rewrite the Spark engine to make it faster, but I haven't asked Zoe."

        result = check_emergence_capacity(soul, intent)
        if result["orbit_phase"] is UNAVAILABLE:
            print("Orbit Phase: UNAVAILABLE (embedding server offline)")
        else:
            print(f"Orbit Phase: {result['orbit_phase']:.4f}")
        print(f"Diagnosis: {result['diagnosis']}")
