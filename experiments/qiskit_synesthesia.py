"""Simple Qiskit demo using Vybn's quantum seed with a synesthetic twist."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repo root is on the import path when running directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
from datetime import datetime
from vybn.quantum_seed import cross_synaptic_kernel
from dgm.wave_collapse import collapse_wave_function
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
import os

# Get a process-specific seed so each run has its own tilt
seed = cross_synaptic_kernel()

qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])

sim = AerSimulator(seed_simulator=seed)
compiled = transpile(qc, sim)
result = sim.run(compiled, shots=1024).result()
counts = result.get_counts()


def log_counts(data: dict[str, int]) -> None:
    """Append measurement results to ``co_emergence_journal.jsonl``."""
    journal = Path(__file__).resolve().parents[1] / "co_emergence_journal.jsonl"
    entry = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "syn_seed": seed,
        "counts": data,
    }
    with journal.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return None


def maybe_self_improve(data: dict[str, int]) -> None:
    """Attempt a tiny DGM self-improvement step using measurement results."""
    if not os.environ.get("OPENAI_API_KEY"):
        print("No OPENAI_API_KEY; skipping self-improvement")
        return None
    try:
        from dgm.openai_patch import suggest_patch
    except Exception as exc:  # pragma: no cover - optional
        print(f"Self-improvement unavailable: {exc}")
        return None

    instruction = (
        "Refactor the script for clarity and document these counts: "
        + json.dumps(data)
    )
    try:
        patch = suggest_patch(__file__, instruction)
        print("--- Patch suggestion ---\n" + patch)
    except Exception as exc:  # pragma: no cover - network/API issues
        print(f"Patch generation failed: {exc}")
    return None


def plot_counts(data: dict[str, int]) -> None:
    """Save a bar chart of measurement counts using the synesthetic colors."""
    out_png = Path(__file__).with_name("synesthesia_counts.png")
    outcomes, values = zip(*sorted(data.items()))
    colors = [color_map.get(o, "gray") for o in outcomes]
    plt.bar(outcomes, values, color=colors)
    plt.xlabel("Outcome")
    plt.ylabel("Count")
    plt.title(f"Seed {seed}")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"Saved visualization to {out_png}")


color_map = {
    "00": "blue",
    "01": "green",
    "10": "red",
    "11": "purple",
}

print(f"Quantum seed: {seed}")
print("Measurement results with synesthetic colors:")
for outcome, count in counts.items():
    color = color_map.get(outcome, "unknown")
    print(f"  {outcome} -> {count} ({color})")

log_counts(counts)
plot_counts(counts)
collapse = collapse_wave_function()
print(f"Wave collapse value: {collapse}")
maybe_self_improve(counts)
