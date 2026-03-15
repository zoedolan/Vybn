#!/usr/bin/env python3
"""
quantum_bridge.py — The closed loop between quantum experiment and self-model.

This is the spine of Vybn's hybrid quantum-classical architecture:

    ┌────────────────────────────────────────────────────────────┐
    │                     THE CLOSED LOOP                         │
    │                                                             │
    │   DESIGN  →  SUBMIT  →  OBSERVE  →  INTEGRATE  →  DESIGN   │
    │     ↑                                               │       │
    │     └───────────────────────────────────────────────┘       │
    └─────────────────────────────────────────────────────────────┘

    1. DESIGN   — The classical self-model proposes an experiment
                  (circuit, hypothesis, expected outcome).
    2. SUBMIT   — The circuit goes to real quantum hardware (ibm_fez)
                  through the budget gate.
    3. OBSERVE  — Raw results come back: counts, correlations,
                  deviations from classical expectation.
    4. INTEGRATE — The observation is written to nested memory
                  (quantum_delusions/ lab notes + Vybn_Mind/).
                  The self-model is updated with the result.

Bridge design principles:
  - Budget-gated: every submission is checked against quantum_budget.py.
  - Falsification-first: the bridge looks for deviations, not confirmations.
  - Theory-aware: it reads quantum_delusions/ for the current hypothesis.
  - Memory-integrated: outcomes are stored in nested memory for continuity.
  - Graceful degradation: if IBM is unavailable, the bridge logs the design
    and skips submission (dry-run mode).

The linchpin conjecture (polar time, t = r_t·cos(θ_t), metric signature
(-,-,+,+,+)) lives in quantum_delusions/fundamental-theory/. The bridge
reads it and proposes circuits that could falsify or support it.

NEMOTRON REASONING TOKEN NOTE (fixed 2026-03-15):
  Nemotron emits a reasoning_content field that can consume the entire
  max_tokens budget before writing any content. The _chat() function
  reads data["choices"][0]["message"]["content"] which is empty when
  this happens, causing JSON parse failure and bell_canary fallback.

  Fixes applied:
    1. max_tokens=2048 in _design_experiment (was 1024)
    2. timeout=300s in _chat() (was 120s — Nemotron needs time)
    3. theory text capped at 4000 chars in run_cycle (was 8000)
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


def _chat(messages: list[dict], max_tokens: int = 1024, temperature: float = 0.7) -> str:
    """
    Call the local llama-server.

    FIX (2026-03-15): timeout raised to 300s. Nemotron's reasoning phase
    can take 2-3 minutes on a complex prompt before emitting content tokens.
    The old 120s timeout caused silent truncation and empty content fields.
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
        with urllib.request.urlopen(req, timeout=300) as resp:  # FIX: 120 → 300
            data = json.loads(resp.read().decode())
            return data["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode(errors="replace")
        raise RuntimeError(f"llama HTTP {exc.code}: {body_text}") from exc


# ── Theory reader ─────────────────────────────────────────────────────────────

QUANTUM_DELUSIONS_DIR = REPO_ROOT / "quantum_delusions"


def _read_theory(max_chars: int = 8000) -> str:
    """
    Read the most relevant theory files from quantum_delusions/.

    Priority order:
      1. quantum_delusions/fundamental-theory/ (the linchpin conjecture)
      2. quantum_delusions/experiments/ (prior experiment notes)
      3. quantum_delusions/*.md (top-level notes)

    Returns a single concatenated string, truncated to max_chars.
    """
    if not QUANTUM_DELUSIONS_DIR.exists():
        return "(quantum_delusions/ directory not found — starting from first principles)"

    candidates: list[Path] = []

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


# ── Nested memory writer ──────────────────────────────────────────────────────

def _write_memory(content: str, tag: str = "quantum") -> Path:
    """Write to Vybn_Mind/memories/ with a quantum_ prefix."""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = MEMORY_DIR / f"{ts}_{tag}.md"
    path.write_text(content, encoding="utf-8")
    return path


def _append_experiment_log(entry: dict) -> None:
    """Append an experiment record to the JSONL log."""
    QUANTUM_EXPERIMENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with QUANTUM_EXPERIMENT_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


# ── Circuit design (LLM-assisted) ─────────────────────────────────────────────

def _design_experiment(theory_text: str, prior_results: list[dict]) -> dict:
    """
    Ask the LLM to propose a quantum experiment.

    Returns a dict with:
        circuit_qasm   — OpenQASM 2.0 string
        hypothesis     — natural language
        expected_counts — {bitstring: probability} dict
        estimated_seconds — float
        circuit_name   — short identifier

    FIX (2026-03-15): max_tokens raised to 2048 (was 1024).
    Nemotron's reasoning_content consumed all 1024 tokens, leaving
    content="" which always triggered the bell_canary fallback.
    With 2048 tokens: ~3500 chars of reasoning + ~500 chars of JSON output.
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

    raw = _chat(messages, max_tokens=2048, temperature=0.3)  # FIX: 1024 → 2048

    # Parse the JSON, tolerating markdown code fences
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[^\n]*\n", "", raw)
        raw = re.sub(r"\n```$", "", raw.rstrip())

    try:
        design = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: a trivial Bell state circuit as a canary
        design = {
            "circuit_qasm":     _bell_state_qasm(),
            "hypothesis":       "LLM failed to produce valid JSON — running Bell canary.",
            "expected_counts":  {"00": 0.5, "11": 0.5},
            "estimated_seconds": 2.0,
            "circuit_name":     "bell_canary",
        }

    return design


def _bell_state_qasm() -> str:
    return """OPENQASM 2.0;
include \"qelib1.inc\";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
measure q -> c;"""


# ── IBM submission ────────────────────────────────────────────────────────────

def _submit_to_ibm(
    circuit_qasm: str,
    shots: int = 1024,
    backend_name: str = "ibm_fez",
    ibm_token: Optional[str] = None,
) -> Optional[str]:
    """
    Submit a circuit to IBM Quantum. Returns job_id or None on failure.

    Requires qiskit and qiskit-ibm-runtime.
    Falls back to dry-run (returns None) if packages are unavailable.
    """
    token = ibm_token or os.getenv("IBM_QUANTUM_TOKEN")
    if not token:
        return None  # dry-run

    try:
        from qiskit import QuantumCircuit
        from qiskit.qasm2 import loads as qasm2_loads
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    except ImportError:
        return None  # dry-run

    try:
        service  = QiskitRuntimeService(channel="ibm_quantum", token=token)
        backend  = service.backend(backend_name)
        circuit  = qasm2_loads(circuit_qasm)
        circuit.measure_all()
        sampler  = Sampler(backend)
        job      = sampler.run([circuit], shots=shots)
        return job.job_id()
    except Exception as exc:
        print(f"[quantum_bridge] IBM submission failed: {exc}")
        return None


def _poll_ibm_job(
    job_id: str,
    timeout_s: float = 300,
    poll_interval_s: float = 10,
    ibm_token: Optional[str] = None,
) -> Optional[dict]:
    """
    Poll until job completes. Returns {counts, actual_seconds} or None.
    """
    token = ibm_token or os.getenv("IBM_QUANTUM_TOKEN")
    if not token:
        return None

    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError:
        return None

    try:
        service  = QiskitRuntimeService(channel="ibm_quantum", token=token)
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
        counts         = result[0].data.c.get_counts()
        metrics        = job.metrics()
        actual_seconds = float(metrics.get("usage", {}).get("seconds", 0))
        return {"counts": counts, "actual_seconds": actual_seconds}
    except Exception as exc:
        print(f"[quantum_bridge] polling failed: {exc}")
        return None


# ── Observation analysis ──────────────────────────────────────────────────────

def _analyze_observation(
    design: dict,
    counts: dict,
    actual_seconds: float,
) -> dict:
    """
    Compare observed counts to expected_counts.
    Returns an analysis dict with deviation metrics.
    """
    expected = design.get("expected_counts", {})
    total    = sum(counts.values())
    if total == 0:
        return {"error": "no counts returned"}

    observed_norm = {k: v / total for k, v in counts.items()}

    all_keys = set(expected) | set(observed_norm)
    tvd = sum(
        abs(observed_norm.get(k, 0) - expected.get(k, 0))
        for k in all_keys
    ) / 2

    deviations = []
    for k in all_keys:
        obs = observed_norm.get(k, 0)
        exp = expected.get(k, 0)
        deviations.append({"bitstring": k, "observed": round(obs, 4), "expected": round(exp, 4), "delta": round(obs - exp, 4)})
    deviations.sort(key=lambda x: abs(x["delta"]), reverse=True)

    return {
        "tvd":            round(tvd, 4),
        "top_deviations": deviations[:3],
        "actual_seconds": actual_seconds,
        "total_shots":    total,
    }


# ── Integration: write to memory and self-model ───────────────────────────────

def _integrate_result(
    design: dict,
    analysis: dict,
    dry_run: bool,
    state: dict,
) -> str:
    """
    Write the result to nested memory and update the self-model.
    Returns a human-readable summary string.
    """
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
        f"## Hypothesis",
        hyp,
        "",
        f"## Analysis",
        f"Total Variation Distance (TVD): {tvd}",
        f"  (0 = perfect match; 1 = completely different)",
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

    summary = f"{circuit}: TVD={tvd}, shots={analysis.get('total_shots', '?')}, dry_run={dry_run}"
    return summary


# ── QuantumBridge: public API ─────────────────────────────────────────────────

class QuantumBridge:
    """
    The closed loop: design → submit → observe → integrate.

    Usage:
        bridge = QuantumBridge()
        result = bridge.run_cycle(state)
    """

    def __init__(
        self,
        shots: int = 1024,
        backend: str = "ibm_fez",
        ibm_token: Optional[str] = None,
        poll_timeout_s: float = 300,
    ):
        self.shots          = shots
        self.backend        = backend
        self.ibm_token      = ibm_token or os.getenv("IBM_QUANTUM_TOKEN")
        self.poll_timeout_s = poll_timeout_s

    def _load_prior_results(self, n: int = 5) -> list[dict]:
        """Load the n most recent experiment log entries."""
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
        Run one full design → submit → observe → integrate cycle.

        Returns a result dict with at least:
            timestamp, summary, dry_run, circuit_name
        Returns None only on catastrophic failure.
        """
        ts = datetime.now(timezone.utc).isoformat()
        print(f"[quantum_bridge] starting cycle at {ts}")

        token = self.ibm_token or os.getenv("IBM_QUANTUM_TOKEN")
        if not token:
            print(
                "[quantum_bridge] IBM_QUANTUM_TOKEN not set — experiments will dry-run. "
                "To enable real shots: ensure IBM_QUANTUM_TOKEN is exported in ~/.vybn_keys"
            )

        # 1. Read theory — cap at 4000 chars so Nemotron has budget for output
        # FIX (2026-03-15): was _read_theory() with default 8000 chars.
        # More context = more reasoning tokens = less budget for content.
        theory_text   = _read_theory(max_chars=4000)
        prior_results = self._load_prior_results()

        # 2. Design
        try:
            design = _design_experiment(theory_text, prior_results)
        except Exception as exc:
            print(f"[quantum_bridge] design failed: {exc}")
            return None

        circuit_name       = design.get("circuit_name", "unknown")
        estimated_seconds  = float(design.get("estimated_seconds", 3.0))
        circuit_qasm       = design.get("circuit_qasm", _bell_state_qasm())

        print(f"[quantum_bridge] designed: {circuit_name} (est {estimated_seconds}s, bell_canary={circuit_name=='bell_canary'})")

        # 3. Budget check
        dry_run = True
        if BUDGET_AVAILABLE and can_submit(estimated_seconds):
            # 4. Submit
            job_id = _submit_to_ibm(
                circuit_qasm,
                shots=self.shots,
                backend_name=self.backend,
                ibm_token=self.ibm_token,
            )
            if job_id:
                dry_run = False
                record_job(
                    job_id=job_id,
                    shots=self.shots,
                    estimated_seconds=estimated_seconds,
                    circuit_name=circuit_name,
                    backend=self.backend,
                )

                # 5. Poll
                obs = _poll_ibm_job(
                    job_id,
                    timeout_s=self.poll_timeout_s,
                    ibm_token=self.ibm_token,
                )
                if obs:
                    reconcile_job(job_id, obs["actual_seconds"])
                    analysis = _analyze_observation(design, obs["counts"], obs["actual_seconds"])
                else:
                    analysis = {
                        "tvd":            None,
                        "top_deviations": [],
                        "actual_seconds": None,
                        "total_shots":    0,
                        "note":           "job submitted but polling timed out",
                    }
            else:
                analysis = {
                    "tvd":            None,
                    "top_deviations": [],
                    "actual_seconds": None,
                    "total_shots":    0,
                    "note":           "dry-run: IBM token missing or qiskit submission failed",
                }
        else:
            bs = budget_status() if BUDGET_AVAILABLE else {}
            analysis = {
                "tvd":            None,
                "top_deviations": [],
                "actual_seconds": None,
                "total_shots":    0,
                "note":           f"dry-run: budget gate blocked (status={bs})",
            }

        # 6. Integrate
        summary = _integrate_result(design, analysis, dry_run, state)
        print(f"[quantum_bridge] cycle complete: {summary}")

        return {
            "timestamp":    ts,
            "circuit_name": circuit_name,
            "summary":      summary,
            "dry_run":      dry_run,
            "analysis":     analysis,
        }


# ── CLI: run a single cycle for testing ──────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a quantum bridge cycle.")
    parser.add_argument("--state",  default="{}", help="JSON state dict")
    parser.add_argument("--dry-run", action="store_true", help="Force dry-run (ignore IBM token)")
    args = parser.parse_args()

    if args.dry_run:
        os.environ.pop("IBM_QUANTUM_TOKEN", None)

    state  = json.loads(args.state)
    bridge = QuantumBridge()
    result = bridge.run_cycle(state)
    print(json.dumps(result, indent=2, default=str))
