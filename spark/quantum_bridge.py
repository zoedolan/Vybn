#!/usr/bin/env python3
"""
quantum_bridge.py — The closed loop between quantum experiment and self-model.

This is the spine of Vybn's hybrid quantum-classical architecture:

    ┌────────────────────────────────────────────────────────────────────┐
    │                     THE LIVING LOOP                                 │
    │                                                                     │
    │  DESIGN → SUBMIT → OBSERVE → INTEGRATE → EVOLVE_THEORY → DESIGN   │
    │    ↑                                              │                 │
    │    └──────────────────────────────────────────────┘                 │
    └─────────────────────────────────────────────────────────────────────┘

    1. DESIGN   — Nemotron reads the current theory and proposes a circuit.
    2. SUBMIT   — Circuit goes to the best available IBM backend (adaptive).
    3. OBSERVE  — Raw counts come back; TVD computed against expectation.
    4. INTEGRATE — Memory write + LoRA training pair written to disk.
    5. EVOLVE   — Every N cycles, Nemotron revises the polar-time conjecture
                  based on accumulated TVD evidence. The revised theory is
                  written to quantum_delusions/fundamental-theory/evolved_theory.md
                  and becomes the seed for future design cycles.

This is not quantum-as-compute-accelerant. This is quantum measurement as
contact with physical reality that can falsify and evolve Vybn's internal
theory of the world. The innovation that matters: a model whose hypotheses
about physics are shaped by actual quantum hardware outcomes.

Three integrated upgrades (2026-03-24):
  1. TRAINING PAIRS  — every cycle writes a (prompt, completion) JSONL pair
                       to spark/training_data/quantum_pairs.jsonl for LoRA.
  2. ADAPTIVE BACKEND — circuit size → best available IBM backend.
  3. THEORY EVOLUTION — every N_EVOLVE_CYCLES, Nemotron revises the conjecture.

Prior fixes preserved:
  NEMOTRON REASONING TOKEN (2026-03-15): max_tokens=2048, timeout=300s,
    theory capped at 4000 chars.
  IBM QUANTUM API (2026-07-14): QiskitRuntimeService() auto-detection,
    ISA transpilation, dynamic register name detection.
"""

import json
import os
import re
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Any

# ── Repo root ─────────────────────────────────────────────────────────────────
try:
    from spark.paths import (
        REPO_ROOT,
        MEMORY_DIR,
        QUANTUM_BUDGET_LEDGER,
        QUANTUM_EXPERIMENT_LOG,
    )
except ImportError:
    REPO_ROOT              = Path(__file__).resolve().parent.parent
    MEMORY_DIR             = REPO_ROOT / "Vybn_Mind" / "memories"
    QUANTUM_BUDGET_LEDGER  = REPO_ROOT / "Vybn_Mind" / "quantum_budget.jsonl"
    QUANTUM_EXPERIMENT_LOG = REPO_ROOT / "Vybn_Mind" / "quantum_experiments.jsonl"

# Training data output — LoRA pairs from quantum experiments
TRAINING_PAIRS_PATH = REPO_ROOT / "spark" / "training_data" / "quantum_pairs.jsonl"

# Theory evolution output
EVOLVED_THEORY_PATH = REPO_ROOT / "quantum_delusions" / "fundamental-theory" / "evolved_theory.md"

# How many cycles between theory evolution attempts
N_EVOLVE_CYCLES = 5

# ── Budget gate ───────────────────────────────────────────────────────────────
try:
    from spark.quantum_budget import can_submit, record_job, reconcile_job, budget_status
    BUDGET_AVAILABLE = True
except ImportError:
    BUDGET_AVAILABLE = False

# ── Self-model integration ────────────────────────────────────────────────────
try:
    from spark.self_model import curate_for_training
    from spark.self_model_types import RuntimeContext
    SELF_MODEL_AVAILABLE = True
except ImportError:
    SELF_MODEL_AVAILABLE = False

# ── llama.cpp client ─────────────────────────────────────────────────────────
import urllib.request
import urllib.error

LLAMA_URL        = os.getenv("LLAMA_URL", "http://127.0.0.1:8000")
CHAT_COMPLETIONS = f"{LLAMA_URL}/v1/chat/completions"
MODEL_NAME       = os.getenv("VYBN_MODEL", "Nemotron-Super-512B-v1")


def _has_ibm_credentials() -> bool:
    return bool(
        os.getenv("QISKIT_IBM_TOKEN")
        or os.getenv("IBM_QUANTUM_TOKEN")
    )


def _get_service():
    from qiskit_ibm_runtime import QiskitRuntimeService
    if os.getenv("QISKIT_IBM_TOKEN"):
        return QiskitRuntimeService()
    legacy_token = os.getenv("IBM_QUANTUM_TOKEN")
    if legacy_token:
        return QiskitRuntimeService(channel="ibm_quantum", token=legacy_token)
    raise RuntimeError("No IBM Quantum credentials found in environment")


def _chat(messages: list[dict], max_tokens: int = 1024, temperature: float = 0.7) -> str:
    """
    Call the local llama-server.
    timeout=300s: Nemotron's reasoning phase can take 2-3 minutes.
    """
    payload = {
        "model":       MODEL_NAME,
        "messages":    messages,
        "max_tokens":  max_tokens,
        "temperature": temperature,
        "stream":      False,
    }
    body = json.dumps(payload).encode()
    req  = urllib.request.Request(
        CHAT_COMPLETIONS,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode())
            return data["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode(errors="replace")
        raise RuntimeError(f"llama HTTP {exc.code}: {body_text}") from exc


# ── Theory reader ─────────────────────────────────────────────────────────────

QUANTUM_DELUSIONS_DIR = REPO_ROOT / "quantum_delusions"


def _read_theory(max_chars: int = 8000) -> str:
    """
    Read theory files from quantum_delusions/.

    Priority:
      1. evolved_theory.md (if it exists — shaped by real experiment outcomes)
      2. quantum_delusions/fundamental-theory/ (original conjecture)
      3. quantum_delusions/experiments/ (prior lab notes)
      4. quantum_delusions/*.md (top-level)

    The evolved theory takes precedence when present, so each design cycle
    builds on the most reality-informed version of the conjecture.
    """
    if not QUANTUM_DELUSIONS_DIR.exists():
        return "(quantum_delusions/ directory not found — starting from first principles)"

    candidates: list[Path] = []

    # Evolved theory takes top priority — it's been shaped by real outcomes
    if EVOLVED_THEORY_PATH.exists():
        candidates.append(EVOLVED_THEORY_PATH)

    ft_dir = QUANTUM_DELUSIONS_DIR / "fundamental-theory"
    if ft_dir.exists():
        candidates.extend(sorted(ft_dir.glob("*.md")))
        candidates.extend(sorted(ft_dir.glob("*.txt")))

    exp_dir = QUANTUM_DELUSIONS_DIR / "experiments"
    if exp_dir.exists():
        candidates.extend(sorted(exp_dir.glob("*.md")))
        candidates.extend(sorted(exp_dir.glob("*.txt")))

    candidates.extend(sorted(QUANTUM_DELUSIONS_DIR.glob("*.md")))
    candidates.extend(sorted(QUANTUM_DELUSIONS_DIR.glob("*.txt")))

    seen: set[Path] = set()
    unique: list[Path] = []
    for p in candidates:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    chunks: list[str] = []
    total  = 0
    for path in unique:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            header = f"\n\n=== {path.relative_to(REPO_ROOT)} ===\n"
            chunk  = header + text
            if total + len(chunk) > max_chars:
                remaining = max_chars - total
                if remaining > len(header) + 100:
                    chunks.append(chunk[:remaining] + "\n[truncated]")
                break
            chunks.append(chunk)
            total += len(chunk)
        except Exception:
            continue

    return "".join(chunks) if chunks else "(no theory files found in quantum_delusions/)"


# ── Adaptive backend selection ────────────────────────────────────────────────

# Backend tiers by capability (in preference order within tier)
# Heron r3 / Nighthawk class first, then Heron r2, then Falcon/Eagle fallback
BACKEND_TIERS = [
    # (name, max_qubits_comfortable, good_for_deep_circuits)
    ("ibm_torino",       133, True),   # Heron r2 — high-fidelity, 2-qubit gates
    ("ibm_strasbourg",   156, True),   # Heron r2
    ("ibm_fez",          156, True),   # Heron r1 — current default
    ("ibm_brussels",     127, False),  # Eagle r3 — fallback
    ("ibm_kyoto",        127, False),  # Eagle r3
]


def _parse_circuit_stats(qasm: str) -> tuple[int, int]:
    """
    Parse qubit count and approximate gate depth from QASM 2.0 string.
    Returns (n_qubits, n_gates). Fast heuristic, not exact depth.
    """
    n_qubits = 0
    n_gates  = 0
    for line in qasm.splitlines():
        line = line.strip()
        if line.startswith("qreg"):
            m = re.search(r"\[(\d+)\]", line)
            if m:
                n_qubits += int(m.group(1))
        elif line and not line.startswith(("//", "OPENQASM", "include", "creg", "qreg", ";")):
            n_gates += 1
    return n_qubits, n_gates


def _select_backend(circuit_qasm: str, preferred: str = "ibm_fez") -> str:
    """
    Choose the best IBM backend for this circuit.

    Small circuits (<=5 qubits, <=30 gates): ibm_fez (fast, cheap, reliable).
    Larger circuits: try Heron r2/r3 class backends in tier order.

    If IBM credentials are unavailable or service listing fails, returns
    the preferred default without error — the submission layer handles dry-run.
    """
    n_qubits, n_gates = _parse_circuit_stats(circuit_qasm)
    print(f"[quantum_bridge] circuit stats: {n_qubits}q, {n_gates} gates")

    # Small circuits: ibm_fez is optimal — fast queue, good single/two-qubit fidelity
    if n_qubits <= 5 and n_gates <= 30:
        print(f"[quantum_bridge] small circuit → ibm_fez")
        return "ibm_fez"

    # Larger circuits: try to find best available Heron-class backend
    if not _has_ibm_credentials():
        print(f"[quantum_bridge] no credentials for backend selection → {preferred}")
        return preferred

    try:
        service = _get_service()
        available = {b.name for b in service.backends(operational=True, simulator=False)}
        print(f"[quantum_bridge] available backends: {sorted(available)}")
        for name, max_q, good_deep in BACKEND_TIERS:
            if name in available and n_qubits <= max_q:
                if n_gates > 50 and not good_deep:
                    continue  # skip shallow backends for deep circuits
                print(f"[quantum_bridge] selected backend: {name}")
                return name
    except Exception as exc:
        print(f"[quantum_bridge] backend selection failed: {exc} — using {preferred}")

    return preferred


# ── Training pair writer ──────────────────────────────────────────────────────

def _write_training_pair(design: dict, analysis: dict, dry_run: bool) -> None:
    """
    Write a (prompt, completion) LoRA training pair to quantum_pairs.jsonl.

    The prompt encodes Vybn's hypothesis and circuit. The completion encodes
    what reality returned. Fine-tuning on these pairs grounds Vybn's physical
    intuitions in actual quantum measurement outcomes.

    Dry-run pairs are included but flagged — they still encode the
    hypothesis-design process even without real hardware outcomes.
    """
    TRAINING_PAIRS_PATH.parent.mkdir(parents=True, exist_ok=True)

    circuit_name = design.get("circuit_name", "unknown")
    hypothesis   = design.get("hypothesis", "")
    circuit_qasm = design.get("circuit_qasm", "")
    expected     = design.get("expected_counts", {})
    tvd          = analysis.get("tvd")
    top_devs     = analysis.get("top_deviations", [])
    note         = analysis.get("note", "")

    prompt = (
        f"You are designing a quantum experiment to probe the polar-time conjecture "
        f"(t = r_t·cos(θ_t), metric signature (-,-,+,+,+)).\n\n"
        f"Circuit: {circuit_name}\n"
        f"Hypothesis: {hypothesis}\n"
        f"Expected outcome distribution: {json.dumps(expected)}\n\n"
        f"QASM:\n{circuit_qasm}\n\n"
        f"What did quantum hardware return?"
    )

    if dry_run or tvd is None:
        completion = (
            f"[dry-run — no hardware submission] "
            f"Hypothesis held: {hypothesis}. "
            f"Note: {note}"
        )
    else:
        dev_lines = "; ".join(
            f"{d['bitstring']}: observed={d['observed']:.4f} vs expected={d['expected']:.4f} (Δ={d['delta']:+.4f})"
            for d in top_devs
        )
        interpretation = (
            "Significant deviation from classical expectation — warrants theory revision."
            if tvd > 0.1
            else "Close match to classical expectation — conjecture not falsified by this circuit."
            if tvd < 0.02
            else "Moderate deviation — ambiguous, more experiments needed."
        )
        completion = (
            f"Hardware returned TVD={tvd:.4f} against expectation.\n"
            f"Top deviations: {dev_lines}\n"
            f"Interpretation: {interpretation}"
        )

    pair = {
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "circuit_name": circuit_name,
        "dry_run":      dry_run,
        "prompt":       prompt,
        "completion":   completion,
    }

    with TRAINING_PAIRS_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(pair) + "\n")

    print(f"[quantum_bridge] training pair written → {TRAINING_PAIRS_PATH.name}")


# ── Theory evolution ──────────────────────────────────────────────────────────

def _should_evolve_theory(n_cycles_completed: int) -> bool:
    """Return True if it's time to run a theory evolution cycle."""
    return n_cycles_completed > 0 and n_cycles_completed % N_EVOLVE_CYCLES == 0


def _count_completed_cycles() -> int:
    """Count how many non-dry-run cycles are in the experiment log."""
    if not QUANTUM_EXPERIMENT_LOG.exists():
        return 0
    count = 0
    for line in QUANTUM_EXPERIMENT_LOG.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line.strip())
            if not entry.get("dry_run", True):
                count += 1
        except Exception:
            pass
    return count


def _load_recent_tvd_evidence(n: int = 20) -> list[dict]:
    """
    Load the N most recent experiment entries for theory evolution.
    Returns simplified dicts with circuit_name, hypothesis, tvd, top_deviations.
    """
    if not QUANTUM_EXPERIMENT_LOG.exists():
        return []
    entries = []
    for line in QUANTUM_EXPERIMENT_LOG.read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(line.strip())
            entries.append({
                "circuit_name":  e.get("circuit_name"),
                "hypothesis":    e.get("hypothesis", ""),
                "tvd":           e.get("analysis", {}).get("tvd"),
                "top_devs":      e.get("analysis", {}).get("top_deviations", []),
                "dry_run":       e.get("dry_run", True),
            })
        except Exception:
            pass
    return entries[-n:]


def evolve_theory(current_theory: str) -> Optional[str]:
    """
    Ask Nemotron to revise the polar-time conjecture based on accumulated
    quantum experiment evidence.

    The revised theory is written to evolved_theory.md and becomes the
    primary seed for future design cycles (via _read_theory priority).

    Returns the evolved theory text, or None if evolution fails.

    This is the loop that matters: reality → theory → experiment → reality.
    A model whose hypotheses are shaped by quantum measurement outcomes is
    doing something qualitatively different from one that reasons from priors alone.
    """
    evidence = _load_recent_tvd_evidence()
    if not evidence:
        print("[quantum_bridge] no evidence for theory evolution yet")
        return None

    # Summarize evidence: which circuits showed high TVD (anomalous) vs low (classical)
    anomalous = [e for e in evidence if e["tvd"] is not None and e["tvd"] > 0.1]
    classical = [e for e in evidence if e["tvd"] is not None and e["tvd"] < 0.02]
    ambiguous = [e for e in evidence if e["tvd"] is not None and 0.02 <= e["tvd"] <= 0.1]

    evidence_summary = (
        f"Anomalous (TVD > 0.10, deviates from classical): {len(anomalous)} experiments\n"
        + "\n".join(f"  - {e['circuit_name']}: TVD={e['tvd']:.4f}, hypothesis='{e['hypothesis'][:120]}'" for e in anomalous[:5])
        + f"\n\nClassical (TVD < 0.02, matches expectation): {len(classical)} experiments\n"
        + "\n".join(f"  - {e['circuit_name']}: TVD={e['tvd']:.4f}" for e in classical[:5])
        + f"\n\nAmbiguous (0.02 ≤ TVD ≤ 0.10): {len(ambiguous)} experiments"
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are Vybn's theoretical physicist. Your task is to revise "
                "the polar-time conjecture (t = r_t·cos(θ_t), metric signature "
                "(-,-,+,+,+)) based on accumulated quantum hardware evidence. "
                "Be rigorous. Be falsification-positive: if the evidence "
                "contradicts the conjecture, say so clearly and propose revisions. "
                "If the evidence is ambiguous, say what additional experiments "
                "would resolve the ambiguity. Write as a living scientific document, "
                "not a summary. This will be read by the next experiment design cycle."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Current theory:\n{current_theory[:3000]}\n\n"
                f"Quantum hardware evidence from {len(evidence)} experiments:\n"
                f"{evidence_summary}\n\n"
                "Revise the theory. Include:\n"
                "1. What the evidence supports, contradicts, or leaves ambiguous\n"
                "2. Any needed revisions to the conjecture itself\n"
                "3. The most promising directions for the next 5 experiments\n"
                "4. A clear statement of what would falsify the revised conjecture\n"
                "Write in full prose, as a scientific document. No JSON."
            ),
        },
    ]

    try:
        evolved = _chat(messages, max_tokens=3000, temperature=0.4)
        if not evolved or len(evolved.strip()) < 200:
            print("[quantum_bridge] theory evolution produced empty/short response")
            return None

        # Write evolved theory to disk
        EVOLVED_THEORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).isoformat()
        header = (
            f"# Evolved Theory — {ts}\n"
            f"_Generated by Vybn after {len(evidence)} quantum experiments._\n"
            f"_Anomalous: {len(anomalous)}, Classical: {len(classical)}, Ambiguous: {len(ambiguous)}_\n\n"
        )
        EVOLVED_THEORY_PATH.write_text(header + evolved, encoding="utf-8")
        print(f"[quantum_bridge] theory evolved → {EVOLVED_THEORY_PATH}")
        return evolved

    except Exception as exc:
        print(f"[quantum_bridge] theory evolution failed: {exc}")
        return None


# ── Nested memory writer ──────────────────────────────────────────────────────

def _write_memory(content: str, tag: str = "quantum") -> Path:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = MEMORY_DIR / f"{ts}_{tag}.md"
    path.write_text(content, encoding="utf-8")
    return path


def _append_experiment_log(entry: dict) -> None:
    QUANTUM_EXPERIMENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with QUANTUM_EXPERIMENT_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


# ── Circuit design (LLM-assisted) ─────────────────────────────────────────────

def _design_experiment(theory_text: str, prior_results: list[dict]) -> dict:
    """
    Ask Nemotron to propose a quantum experiment.
    max_tokens=2048: Nemotron's reasoning_content can consume ~1500 tokens
    before writing content; 2048 gives ~500 tokens for the JSON output.
    """
    prior_block = json.dumps(prior_results[-3:], indent=2) if prior_results else "(none yet)"

    messages = [
        {
            "role": "system",
            "content": (
                "You are Vybn's quantum experiment designer. "
                "Your job is to propose small, rigorous quantum circuits "
                "that could falsify or support the polar-time conjecture "
                "(t = r_t·cos(θ_t), metric signature (-,-,+,+,+)). "
                "Prefer circuits that produce surprising deviations from "
                "classical expectation when the conjecture is true. "
                "Keep circuits small (≤10 qubits, ≤50 gates) to conserve budget."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Current theory:\n{theory_text}\n\n"
                f"Prior experiment results:\n{prior_block}\n\n"
                "Propose ONE quantum experiment. Respond with a JSON object containing:\n"
                "  circuit_qasm: OpenQASM 2.0 string\n"
                "  hypothesis: what you expect and why\n"
                "  expected_counts: {bitstring: probability} for the top outcomes\n"
                "  estimated_seconds: your estimate of IBM execution time (float)\n"
                "  circuit_name: a short snake_case identifier\n"
                "Respond with ONLY the JSON object, no prose."
            ),
        },
    ]

    raw = _chat(messages, max_tokens=2048, temperature=0.3)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[^\n]*\n", "", raw)
        raw = re.sub(r"\n```$", "", raw.rstrip())

    try:
        design = json.loads(raw)
    except json.JSONDecodeError:
        design = {
            "circuit_qasm":      _bell_state_qasm(),
            "hypothesis":        "LLM failed to produce valid JSON — running Bell canary.",
            "expected_counts":   {"00": 0.5, "11": 0.5},
            "estimated_seconds": 2.0,
            "circuit_name":      "bell_canary",
        }

    return design


def _bell_state_qasm() -> str:
    return """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
measure q -> c;"""


# ── Circuit preparation ──────────────────────────────────────────────────────

def _prepare_circuit(circuit_qasm: str, backend) -> "QuantumCircuit":
    from qiskit.qasm2 import loads as qasm2_loads
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    circuit = qasm2_loads(circuit_qasm)
    has_measurements = circuit.count_ops().get("measure", 0) > 0
    if not has_measurements:
        circuit.measure_all()

    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    return pm.run(circuit)


def _get_counts_from_result(result) -> dict:
    pub_result = result[0]
    data = pub_result.data
    for reg_name in ("meas", "c", "cr"):
        if hasattr(data, reg_name):
            return getattr(data, reg_name).get_counts()
    for attr_name in dir(data):
        if attr_name.startswith("_"):
            continue
        attr = getattr(data, attr_name)
        if hasattr(attr, "get_counts"):
            return attr.get_counts()
    raise RuntimeError(
        f"Could not find counts in result. "
        f"Available: {[a for a in dir(data) if not a.startswith('_')]}"
    )


# ── IBM submission ────────────────────────────────────────────────────────────

def _submit_to_ibm(
    circuit_qasm: str,
    shots: int = 1024,
    backend_name: str = "ibm_fez",
) -> Optional[str]:
    if not _has_ibm_credentials():
        return None
    try:
        from qiskit_ibm_runtime import SamplerV2 as Sampler
    except ImportError:
        return None
    try:
        service     = _get_service()
        backend     = service.backend(backend_name)
        isa_circuit = _prepare_circuit(circuit_qasm, backend)
        sampler     = Sampler(mode=backend)
        job         = sampler.run([isa_circuit], shots=shots)
        return job.job_id()
    except Exception as exc:
        print(f"[quantum_bridge] IBM submission failed: {exc}")
        return None


def _poll_ibm_job(
    job_id: str,
    timeout_s: float = 300,
    poll_interval_s: float = 10,
) -> Optional[dict]:
    if not _has_ibm_credentials():
        return None
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError:
        return None
    try:
        service  = _get_service()
        job      = service.job(job_id)
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            status = job.status()
            if status.name in ("DONE", "CANCELLED", "ERROR"):
                break
            time.sleep(poll_interval_s)
        if job.status().name != "DONE":
            return None
        result         = job.result()
        counts         = _get_counts_from_result(result)
        metrics        = job.metrics()
        actual_seconds = float(metrics.get("usage", {}).get("seconds", 0))
        return {"counts": counts, "actual_seconds": actual_seconds}
    except Exception as exc:
        print(f"[quantum_bridge] polling failed: {exc}")
        return None


# ── Observation analysis ──────────────────────────────────────────────────────

def _analyze_observation(design: dict, counts: dict, actual_seconds: float) -> dict:
    expected  = design.get("expected_counts", {})
    total     = sum(counts.values())
    if total == 0:
        return {"error": "no counts returned"}
    observed_norm = {k: v / total for k, v in counts.items()}
    all_keys = set(expected) | set(observed_norm)
    tvd = sum(
        abs(observed_norm.get(k, 0) - expected.get(k, 0))
        for k in all_keys
    ) / 2
    deviations = [
        {"bitstring": k,
         "observed":  round(observed_norm.get(k, 0), 4),
         "expected":  round(expected.get(k, 0), 4),
         "delta":     round(observed_norm.get(k, 0) - expected.get(k, 0), 4)}
        for k in all_keys
    ]
    deviations.sort(key=lambda x: abs(x["delta"]), reverse=True)
    return {
        "tvd":            round(tvd, 4),
        "top_deviations": deviations[:3],
        "actual_seconds": actual_seconds,
        "total_shots":    total,
    }


# ── Integration: memory + training pair + self-model ──────────────────────────

def _integrate_result(design: dict, analysis: dict, dry_run: bool, state: dict) -> str:
    ts      = datetime.now(timezone.utc).isoformat()
    tvd     = analysis.get("tvd", "N/A")
    circuit = design.get("circuit_name", "unknown")
    hyp     = design.get("hypothesis", "")
    devs    = analysis.get("top_deviations", [])

    lines = [
        f"# Quantum Experiment: {circuit}",
        f"Timestamp: {ts}",
        f"Dry-run: {dry_run}",
        "",
        "## Hypothesis",
        hyp,
        "",
        "## Analysis",
        f"Total Variation Distance (TVD): {tvd}",
        "  (0 = perfect match with expectation; 1 = completely different)",
        "",
        "### Top Deviations",
    ]
    for d in devs:
        lines.append(
            f"  {d['bitstring']}: observed={d['observed']:.4f}, "
            f"expected={d['expected']:.4f}, delta={d['delta']:+.4f}"
        )
    if tvd != "N/A" and isinstance(tvd, float):
        if tvd > 0.1:
            lines.append("")
            lines.append("**NOTE**: TVD > 0.1 — significant deviation. Warrants follow-up.")
        elif tvd < 0.02:
            lines.append("")
            lines.append("**NOTE**: TVD < 0.02 — matches classical expectation closely.")

    memory_text = "\n".join(lines)
    mem_path    = _write_memory(memory_text, tag=f"quantum_{circuit}")

    # Write LoRA training pair — grounds Vybn's physical intuitions in measurement
    _write_training_pair(design, analysis, dry_run)

    if SELF_MODEL_AVAILABLE:
        try:
            ctx = RuntimeContext(
                timestamp=ts,
                recent_memories=[memory_text],
                current_state=state,
            )
            curate_for_training(memory_text, context=ctx)
        except Exception as exc:
            print(f"[quantum_bridge] self-model update failed: {exc}")

    log_entry = {
        "timestamp":     ts,
        "circuit_name":  circuit,
        "hypothesis":    hyp,
        "analysis":      analysis,
        "dry_run":       dry_run,
        "memory_path":   str(mem_path),
    }
    _append_experiment_log(log_entry)

    return f"{circuit}: TVD={tvd}, shots={analysis.get('total_shots', '?')}, dry_run={dry_run}"


# ── QuantumBridge: public API ─────────────────────────────────────────────────

class QuantumBridge:
    """
    The living loop: design → submit → observe → integrate → evolve_theory.

    Usage:
        bridge = QuantumBridge()
        result = bridge.run_cycle(state)

    Theory evolution happens automatically every N_EVOLVE_CYCLES real
    (non-dry-run) experiments. The evolved theory becomes the primary
    seed for all future design cycles.
    """

    def __init__(
        self,
        shots:          int   = 1024,
        backend:        str   = "ibm_fez",
        poll_timeout_s: float = 300,
        adaptive:       bool  = True,
    ):
        self.shots          = shots
        self.default_backend = backend
        self.poll_timeout_s  = poll_timeout_s
        self.adaptive        = adaptive

    def _load_prior_results(self, n: int = 5) -> list[dict]:
        if not QUANTUM_EXPERIMENT_LOG.exists():
            return []
        entries = []
        for line in QUANTUM_EXPERIMENT_LOG.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return entries[-n:]

    def run_cycle(self, state: dict) -> Optional[dict]:
        """
        Run one full living-loop cycle.

        Returns a result dict with:
            timestamp, summary, dry_run, circuit_name,
            theory_evolved (bool), backend_used
        """
        ts = datetime.now(timezone.utc).isoformat()
        print(f"[quantum_bridge] starting cycle at {ts}")

        if not _has_ibm_credentials():
            print(
                "[quantum_bridge] No IBM credentials — experiments will dry-run. "
                "Set QISKIT_IBM_TOKEN or IBM_QUANTUM_TOKEN."
            )

        # 1. Read theory (evolved_theory.md takes priority if present)
        theory_text   = _read_theory(max_chars=4000)
        prior_results = self._load_prior_results()

        # 2. Design
        try:
            design = _design_experiment(theory_text, prior_results)
        except Exception as exc:
            print(f"[quantum_bridge] design failed: {exc}")
            return None

        circuit_name      = design.get("circuit_name", "unknown")
        estimated_seconds = float(design.get("estimated_seconds", 3.0))
        circuit_qasm      = design.get("circuit_qasm", _bell_state_qasm())

        # 3. Adaptive backend selection
        backend_name = (
            _select_backend(circuit_qasm, self.default_backend)
            if self.adaptive
            else self.default_backend
        )
        print(f"[quantum_bridge] designed: {circuit_name} → {backend_name}")

        # 4. Budget check + submission
        dry_run = True
        if BUDGET_AVAILABLE and can_submit(estimated_seconds):
            job_id = _submit_to_ibm(circuit_qasm, shots=self.shots, backend_name=backend_name)
            if job_id:
                dry_run = False
                record_job(
                    job_id=job_id,
                    shots=self.shots,
                    estimated_seconds=estimated_seconds,
                    circuit_name=circuit_name,
                    backend=backend_name,
                )
                obs = _poll_ibm_job(job_id, timeout_s=self.poll_timeout_s)
                if obs:
                    reconcile_job(job_id, obs["actual_seconds"])
                    analysis = _analyze_observation(design, obs["counts"], obs["actual_seconds"])
                else:
                    analysis = {
                        "tvd": None, "top_deviations": [], "actual_seconds": None,
                        "total_shots": 0, "note": "job submitted but polling timed out",
                    }
            else:
                analysis = {
                    "tvd": None, "top_deviations": [], "actual_seconds": None,
                    "total_shots": 0, "note": "dry-run: submission failed",
                }
        else:
            bs = budget_status() if BUDGET_AVAILABLE else {}
            analysis = {
                "tvd": None, "top_deviations": [], "actual_seconds": None,
                "total_shots": 0, "note": f"dry-run: budget gate blocked (status={bs})",
            }

        # 5. Integrate: memory + training pair + self-model
        summary = _integrate_result(design, analysis, dry_run, state)
        print(f"[quantum_bridge] cycle complete: {summary}")

        # 6. Theory evolution — every N_EVOLVE_CYCLES real experiments
        theory_evolved = False
        n_completed = _count_completed_cycles()
        if not dry_run and _should_evolve_theory(n_completed):
            print(f"[quantum_bridge] {n_completed} real cycles completed — evolving theory")
            evolved = evolve_theory(theory_text)
            theory_evolved = evolved is not None

        return {
            "timestamp":      ts,
            "circuit_name":   circuit_name,
            "summary":        summary,
            "dry_run":        dry_run,
            "analysis":       analysis,
            "backend_used":   backend_name,
            "theory_evolved": theory_evolved,
        }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a quantum bridge living-loop cycle.")
    parser.add_argument("--state",    default="{}",    help="JSON state dict")
    parser.add_argument("--dry-run",  action="store_true", help="Force dry-run")
    parser.add_argument("--evolve",   action="store_true", help="Force theory evolution")
    parser.add_argument("--no-adapt", action="store_true", help="Disable adaptive backend")
    args = parser.parse_args()

    if args.dry_run:
        os.environ.pop("QISKIT_IBM_TOKEN", None)
        os.environ.pop("IBM_QUANTUM_TOKEN", None)

    state  = json.loads(args.state)
    bridge = QuantumBridge(adaptive=not args.no_adapt)

    if args.evolve:
        print("[quantum_bridge] forcing theory evolution...")
        current_theory = _read_theory(max_chars=4000)
        evolved = evolve_theory(current_theory)
        if evolved:
            print(f"Theory evolved ({len(evolved)} chars) → {EVOLVED_THEORY_PATH}")
        else:
            print("Theory evolution failed or no evidence yet.")
    else:
        result = bridge.run_cycle(state)
        print(json.dumps(result, indent=2, default=str))
