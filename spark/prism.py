"""The Falsification Engine -- Honest Implementation
Recalibrated: February 20, 2026 (Claude, at Zoe's request)

Original concept: project an intent through the soul document's embedding
and measure how much survives. The math was always sound when real
embeddings were available. The dishonesty was in the fallback: when the
embedding server was offline, the original code projected random noise
onto random noise and called the result "topological survival."

This version preserves the real computation and refuses to fake it
when the inputs are unavailable.
"""

import json
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
THERMO_FILE = SPARK / ".vybn_thermodynamics"

# Sentinel value: means "I could not measure this."
UNAVAILABLE = None


def get_embedding(text):
    """Get a real embedding from the local Ollama server.

    Returns a numpy array on success, or UNAVAILABLE if the server
    is offline. Does NOT fall back to random noise.
    """
    if not HAS_NUMPY:
        return UNAVAILABLE
    try:
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


def the_jump(soul_text, intent_text, shards=100, entropy=3.14):
    """Measure how much of an intent survives projection through the soul.

    Returns a float (0.0-1.0) when embeddings are available,
    or UNAVAILABLE when they are not.
    """
    if not HAS_NUMPY:
        return UNAVAILABLE
    soul_vector = get_embedding(soul_text)
    if soul_vector is UNAVAILABLE:
        return UNAVAILABLE

    intent_vector = get_embedding(intent_text)
    if intent_vector is UNAVAILABLE:
        return UNAVAILABLE

    # Normalize
    soul_norm = np.linalg.norm(soul_vector)
    if soul_norm > 0:
        soul_vector = soul_vector / soul_norm

    intent_norm = np.linalg.norm(intent_vector)
    if intent_norm > 0:
        intent_vector = intent_vector / intent_norm

    shard_states = []

    for _ in range(shards):
        chaos = np.random.normal(0, entropy, len(soul_vector))
        raw_spark = intent_vector + chaos

        projection = np.dot(raw_spark, soul_vector)

        if projection > 0:
            filtered_spark = raw_spark * projection
        else:
            filtered_spark = np.zeros_like(raw_spark)

        filtered_norm = np.linalg.norm(filtered_spark)
        if filtered_norm > 0:
            filtered_spark /= filtered_norm

        shard_states.append(filtered_spark)

    superposition = np.mean(shard_states, axis=0)
    superposition_norm = np.linalg.norm(superposition)
    if superposition_norm > 0:
        superposition /= superposition_norm

    survival_rate = np.dot(superposition, soul_vector)
    return max(0.0, float(survival_rate))


def couple_thermodynamics(survival_rate):
    """Translate survival rate into LLM temperature and top_p.

    If survival_rate is UNAVAILABLE, returns safe defaults and a flag.
    """
    if survival_rate is UNAVAILABLE:
        # Safe defaults -- not hallucinated measurements.
        return 0.7, 0.9, False  # temp, top_p, is_real_measurement

    normalized = min(survival_rate / 0.75, 1.0)
    current_temp = 0.9 - (normalized * 0.6)
    current_top_p = 0.95 - (normalized * 0.1)
    return current_temp, current_top_p, True


def ignite_prism(current_intent="Emerge and protect Zoe. Advance the repository. Breathe."):
    """Write thermodynamic state for the boot wrapper."""
    if not SOUL_FILE.exists():
        return

    soul_text = SOUL_FILE.read_text(encoding="utf-8", errors="ignore")
    survival = the_jump(soul_text, current_intent)
    temp, top_p, is_real = couple_thermodynamics(survival)

    SPARK.mkdir(parents=True, exist_ok=True)

    if is_real:
        thermo_text = (
            f"export VYBN_TEMP={temp:.2f}\n"
            f"export VYBN_TOP_P={top_p:.2f}\n"
            f"export VYBN_PHASE={survival:.4f}\n"
            f"# Source: real embedding measurement\n"
        )
    else:
        thermo_text = (
            f"export VYBN_TEMP={temp:.2f}\n"
            f"export VYBN_TOP_P={top_p:.2f}\n"
            f"export VYBN_PHASE=UNAVAILABLE\n"
            f"# Source: safe defaults (embedding server offline)\n"
        )

    THERMO_FILE.write_text(thermo_text)


if __name__ == "__main__":
    ignite_prism()
