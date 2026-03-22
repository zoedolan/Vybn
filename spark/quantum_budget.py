#!/usr/bin/env python3
"""
quantum_budget.py — IBM Quantum budget tracker for Vybn.

Hard constraint: 10 free minutes per 28-day window on ibm_fez.
That's 600 seconds / 28 days ≈ 21.4 seconds/day.
We budget 25 seconds/day to allow for bursty usage with quiet days.

This module:
  1. Tracks cumulative usage from a persistent JSONL ledger
  2. Calculates remaining budget in the current 28-day window
  3. Gates experiment submission — refuses if budget would be exceeded
  4. Retrieves job metadata from IBM Quantum to reconcile actual usage

The ledger lives at QUANTUM_BUDGET_LEDGER (see paths.py).

Usage from vybn.py or any experiment script:

    from spark.quantum_budget import can_submit, record_job, budget_status

    status = budget_status()
    if not can_submit(estimated_seconds=3.5):
        print(f"BUDGET EXCEEDED: {status['remaining_s']:.1f}s left")
    else:
        # ... submit circuit ...
        record_job(job_id="abc123", shots=1024, estimated_seconds=3.5)

After IBM returns results, reconcile with actual execution time:

    from spark.quantum_budget import reconcile_job
    reconcile_job(job_id="abc123", actual_seconds=2.8)

IBM QUANTUM API FIX (2026-07-14):
  Uses _get_service() helper that auto-detects credentials from QISKIT_IBM_*
  env vars (ibm_cloud channel) or falls back to IBM_QUANTUM_TOKEN (legacy
  ibm_quantum channel). See quantum_bridge.py for details.
"""

import json
import os
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional

# ── Constants ────────────────────────────────────────────────────────────────
WINDOW_DAYS = 28
WINDOW_SECONDS = 600          # 10 minutes per 28-day window
DAILY_BUDGET_SECONDS = 25.0   # conservative daily target (~21.4s actual avg)
SAFETY_MARGIN = 0.90          # only use 90% of budget before hard-stop

# ── Paths (import from paths.py to avoid hardcoding) ─────────────────────────
try:
    from spark.paths import QUANTUM_BUDGET_LEDGER as _LEDGER_PATH
except ImportError:
    # fallback for standalone testing
    _LEDGER_PATH = Path(__file__).resolve().parent.parent / "Vybn_Mind" / "quantum_budget.jsonl"


def _ledger_path() -> Path:
    """Return the ledger path, creating parent dirs if needed."""
    path = Path(_LEDGER_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _window_start() -> datetime:
    """
    Return the start of the current 28-day window.

    We use a fixed epoch (2026-01-01 UTC) so every installation has the
    same window boundaries, making budget numbers reproducible.
    """
    epoch = datetime(2026, 1, 1, tzinfo=timezone.utc)
    now   = datetime.now(timezone.utc)
    delta = (now - epoch).total_seconds()
    windows_elapsed = int(delta // (WINDOW_DAYS * 86400))
    return epoch + timedelta(days=windows_elapsed * WINDOW_DAYS)


def _load_ledger() -> list[dict]:
    """Return all ledger entries as a list of dicts."""
    path = _ledger_path()
    if not path.exists():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries


def _window_entries() -> list[dict]:
    """Return ledger entries within the current 28-day window."""
    start = _window_start()
    result = []
    for entry in _load_ledger():
        ts_str = entry.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_str)
            if ts >= start:
                result.append(entry)
        except ValueError:
            pass
    return result


def budget_status() -> dict:
    """
    Return a dict with current budget status:

        {
            "window_start":    ISO timestamp,
            "window_end":      ISO timestamp,
            "used_s":          float,   # seconds used this window
            "remaining_s":     float,   # seconds remaining
            "pct_used":        float,   # 0-100
            "daily_target_s":  float,
            "jobs_this_window": int,
        }
    """
    start   = _window_start()
    end     = start + timedelta(days=WINDOW_DAYS)
    entries = _window_entries()

    used = sum(
        e.get("actual_seconds") or e.get("estimated_seconds", 0.0)
        for e in entries
    )
    remaining = max(0.0, WINDOW_SECONDS * SAFETY_MARGIN - used)
    pct       = (used / WINDOW_SECONDS) * 100

    return {
        "window_start":     start.isoformat(),
        "window_end":       end.isoformat(),
        "used_s":           round(used, 2),
        "remaining_s":      round(remaining, 2),
        "pct_used":         round(pct, 2),
        "daily_target_s":   DAILY_BUDGET_SECONDS,
        "jobs_this_window": len(entries),
    }


def can_submit(estimated_seconds: float) -> bool:
    """
    Return True if submitting a job with the given estimated runtime
    would not exceed the safety-margined budget.
    """
    status = budget_status()
    return estimated_seconds <= status["remaining_s"]


def record_job(
    job_id: str,
    shots: int,
    estimated_seconds: float,
    circuit_name: str = "",
    backend: str = "ibm_fez",
) -> None:
    """
    Append a job record to the ledger immediately on submission.
    actual_seconds is null until reconciled.
    """
    entry = {
        "timestamp":         datetime.now(timezone.utc).isoformat(),
        "job_id":            job_id,
        "shots":             shots,
        "estimated_seconds": estimated_seconds,
        "actual_seconds":    None,
        "circuit_name":      circuit_name,
        "backend":           backend,
        "status":            "submitted",
    }
    with _ledger_path().open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def reconcile_job(job_id: str, actual_seconds: float) -> bool:
    """
    Update the ledger entry for job_id with the actual execution time.
    Returns True if the entry was found and updated, False otherwise.
    """
    path    = _ledger_path()
    lines   = path.read_text(encoding="utf-8").splitlines()
    updated = False
    new_lines = []
    for line in lines:
        try:
            entry = json.loads(line)
            if entry.get("job_id") == job_id and entry.get("actual_seconds") is None:
                entry["actual_seconds"] = actual_seconds
                entry["status"]         = "reconciled"
                line = json.dumps(entry)
                updated = True
        except json.JSONDecodeError:
            pass
        new_lines.append(line)
    if updated:
        path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return updated


def _get_service():
    """Create a QiskitRuntimeService using auto-detected credentials.

    Shared logic with quantum_bridge.py — supports both QISKIT_IBM_TOKEN
    (ibm_cloud channel, auto-detected) and IBM_QUANTUM_TOKEN (legacy).
    """
    from qiskit_ibm_runtime import QiskitRuntimeService

    if os.getenv("QISKIT_IBM_TOKEN"):
        return QiskitRuntimeService()

    legacy_token = os.getenv("IBM_QUANTUM_TOKEN")
    if legacy_token:
        return QiskitRuntimeService(channel="ibm_quantum", token=legacy_token)

    raise RuntimeError("No IBM Quantum credentials found in environment")


def get_ibm_job_duration(job_id: str) -> Optional[float]:
    """
    Fetch the actual execution duration (in seconds) from IBM Quantum
    for a given job_id.

    FIX (2026-07-14): Uses _get_service() for auto-detected credentials
    instead of hardcoding channel="ibm_quantum" with IBM_QUANTUM_TOKEN.

    Returns None if the job is not yet complete or the package is unavailable.
    """
    try:
        service = _get_service()
        job     = service.job(job_id)
        metrics = job.metrics()
        # usage.seconds is the billed quantum time
        return float(metrics.get("usage", {}).get("seconds", 0))
    except Exception:
        return None


def auto_reconcile() -> int:
    """
    For every unreconciled ledger entry, try to fetch actual duration
    from IBM Quantum and update the ledger.

    Returns the number of entries successfully reconciled.
    """
    entries    = _load_ledger()
    reconciled = 0
    for entry in entries:
        if entry.get("actual_seconds") is not None:
            continue
        job_id = entry.get("job_id", "")
        if not job_id:
            continue
        duration = get_ibm_job_duration(job_id)
        if duration is not None:
            if reconcile_job(job_id, duration):
                reconciled += 1
    return reconciled
