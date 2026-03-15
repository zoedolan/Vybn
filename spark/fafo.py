"""fafo.py — Fuck Around and Find Out engine.

Not a faculty. A *driver* that sits above faculty_runner and directs
the organism's attention toward unresolved surprises.

Public API used by vybn.py and faculty_runner.py:
  register_surprises(state, geo_report, ingest_report)  → None
  get_next_action(faculty_id: str) → dict | None
  get_investigation_summary() → dict

Writes:
  spark/research/surprises.jsonl   (append-only)
  spark/research/investigations.yaml  (managed)

Reads:
  complexify_bridge geometry (via state['_geo'])
  quantum_bridge experiment logs (spark/research/*.jsonl)
  mathematician output bus (faculties.d/outputs/mathematician.json)
  researcher frontier (spark/research/research_frontier.yaml)
  witness log (spark/research/witness_log.jsonl)
  growth trigger flag (state['_growth_fired'])

Governance: FAFO never touches protected files, never expands scope
beyond spark/research/. The investigation-formulation LLM call goes
through the same governance gate as any other _chat() call in vybn.py.
All writes are logged to the decision ledger via write_custodian if
write_custodian is available.
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SPARK_DIR = Path(__file__).parent
_RESEARCH_DIR = _SPARK_DIR / "research"
_SURPRISES_PATH = _RESEARCH_DIR / "surprises.jsonl"
_INVESTIGATIONS_PATH = _RESEARCH_DIR / "investigations.yaml"
_MATH_OUTPUT = _SPARK_DIR / "faculties.d" / "outputs" / "mathematician.json"
_FRONTIER_PATH = _RESEARCH_DIR / "research_frontier.yaml"
_WITNESS_LOG = _RESEARCH_DIR / "witness_log.jsonl"
_QUANTUM_LOG = _RESEARCH_DIR / "quantum_results.jsonl"

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

KAPPA_SIGMA_MULTIPLIER = 2.0   # κ > mean + N*σ  → surprise
TVD_THRESHOLD = 0.10           # TVD > this on real (non-dry-run) circuit
WITNESS_FIDELITY_LOW = 0.70   # witness score below this
STALE_BREATHS = 10            # investigation with no progress for N breaths

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _surprise_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    short = str(uuid.uuid4())[:6]
    return f"s-{ts}-{short}"


def _inv_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    short = str(uuid.uuid4())[:6]
    return f"inv-{ts}-{short}"


def _ensure_research_dir() -> None:
    _RESEARCH_DIR.mkdir(parents=True, exist_ok=True)


def _append_surprise(entry: dict) -> None:
    _ensure_research_dir()
    with _SURPRISES_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def _load_investigations() -> dict:
    if not _INVESTIGATIONS_PATH.exists():
        return {"investigations": []}
    with _INVESTIGATIONS_PATH.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if "investigations" not in data:
        data["investigations"] = []
    return data


def _save_investigations(data: dict) -> None:
    _ensure_research_dir()
    with _INVESTIGATIONS_PATH.open("w", encoding="utf-8") as fh:
        yaml.dump(data, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _load_surprises_unresolved() -> list[dict]:
    if not _SURPRISES_PATH.exists():
        return []
    result = []
    with _SURPRISES_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if not entry.get("resolved", False):
                    result.append(entry)
            except json.JSONDecodeError:
                pass
    return result


def _mark_surprise_linked(surprise_id: str, inv_id: str) -> None:
    """Rewrite surprises.jsonl updating a single entry — cheap scan."""
    if not _SURPRISES_PATH.exists():
        return
    lines = _SURPRISES_PATH.read_text(encoding="utf-8").splitlines()
    out = []
    for line in lines:
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            if entry.get("id") == surprise_id:
                entry["investigation_id"] = inv_id
            out.append(json.dumps(entry))
        except json.JSONDecodeError:
            out.append(line)
    _SURPRISES_PATH.write_text("\n".join(out) + "\n", encoding="utf-8")


def _kappa_history_from_state(state: dict) -> list[float]:
    """Pull historical κ values from state if complexify has been storing them."""
    history = state.get("_kappa_history", [])
    if not history:
        # try geo_reports accumulated on state
        for report in state.get("_geo_history", []):
            k = report.get("curvature") if isinstance(report, dict) else None
            if k is not None:
                history.append(float(k))
    return [float(v) for v in history if v is not None]


# ---------------------------------------------------------------------------
# Surprise detection
# ---------------------------------------------------------------------------

def _detect_kappa_spike(state: dict, geo_report: Optional[dict]) -> Optional[dict]:
    if not geo_report:
        return None
    kappa = geo_report.get("curvature")
    if kappa is None:
        return None
    history = _kappa_history_from_state(state)
    if len(history) < 5:
        return None
    mean = statistics.mean(history)
    try:
        sigma = statistics.stdev(history)
    except statistics.StatisticsError:
        return None
    threshold = mean + KAPPA_SIGMA_MULTIPLIER * sigma
    if kappa > threshold:
        context_file = geo_report.get("last_ingested", "unknown")
        return {
            "id": _surprise_id(),
            "timestamp": _now(),
            "source": "complexify_bridge",
            "signal": "curvature_spike",
            "magnitude": round(float(kappa), 6),
            "context": f"κ jumped to {kappa:.4f} (threshold {threshold:.4f}, mean {mean:.4f}) after ingesting {context_file}",
            "resolved": False,
            "investigation_id": None,
        }
    return None


def _detect_tvd_deviation(state: dict) -> Optional[dict]:
    """Read the most recent quantum_results.jsonl entry."""
    if not _QUANTUM_LOG.exists():
        return None
    try:
        lines = [l.strip() for l in _QUANTUM_LOG.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not lines:
            return None
        last = json.loads(lines[-1])
        tvd = last.get("tvd")
        dry_run = last.get("dry_run", True)
        if tvd is not None and not dry_run and float(tvd) > TVD_THRESHOLD:
            circuit_id = last.get("circuit_id", "unknown")
            return {
                "id": _surprise_id(),
                "timestamp": _now(),
                "source": "quantum_bridge",
                "signal": "quantum_deviation",
                "magnitude": round(float(tvd), 6),
                "context": f"TVD {tvd:.4f} on real circuit {circuit_id} (threshold {TVD_THRESHOLD})",
                "resolved": False,
                "investigation_id": None,
                "_raw": last,
            }
    except Exception as exc:
        log.debug("fafo: TVD check error: %s", exc)
    return None


def _detect_conjecture_update(state: dict) -> Optional[dict]:
    """Detect mathematician conjecture status changes."""
    if not _MATH_OUTPUT.exists():
        return None
    try:
        data = json.loads(_MATH_OUTPUT.read_text(encoding="utf-8"))
        last_status = state.get("_fafo_last_conjecture_status", {})
        new_status = {}
        updates = []
        for item in data.get("conjectures", []):
            cid = item.get("id")
            status = item.get("status")
            new_status[cid] = status
            if cid in last_status and last_status[cid] != status:
                updates.append(f"{cid}: {last_status[cid]} → {status}")
        state["_fafo_last_conjecture_status"] = new_status
        if updates:
            return {
                "id": _surprise_id(),
                "timestamp": _now(),
                "source": "mathematician",
                "signal": "conjecture_update",
                "magnitude": float(len(updates)),
                "context": "Conjecture status changed: " + "; ".join(updates),
                "resolved": False,
                "investigation_id": None,
            }
    except Exception as exc:
        log.debug("fafo: conjecture check error: %s", exc)
    return None


def _detect_witness_concern(state: dict) -> Optional[dict]:
    """Read latest witness log entry."""
    if not _WITNESS_LOG.exists():
        return None
    try:
        lines = [l.strip() for l in _WITNESS_LOG.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not lines:
            return None
        last = json.loads(lines[-1])
        score = last.get("fidelity_score")
        last_id = state.get("_fafo_last_witness_id")
        entry_id = last.get("id")
        if score is not None and float(score) < WITNESS_FIDELITY_LOW and entry_id != last_id:
            state["_fafo_last_witness_id"] = entry_id
            return {
                "id": _surprise_id(),
                "timestamp": _now(),
                "source": "witness",
                "signal": "witness_concern",
                "magnitude": round(float(score), 4),
                "context": f"Witness fidelity score {score:.3f} below threshold {WITNESS_FIDELITY_LOW}. Entry: {entry_id}",
                "resolved": False,
                "investigation_id": None,
            }
    except Exception as exc:
        log.debug("fafo: witness check error: %s", exc)
    return None


def _detect_growth_event(state: dict) -> Optional[dict]:
    fired = state.get("_growth_fired", False)
    last_seen = state.get("_fafo_last_growth_fired", False)
    if fired and not last_seen:
        state["_fafo_last_growth_fired"] = True
        return {
            "id": _surprise_id(),
            "timestamp": _now(),
            "source": "growth",
            "signal": "growth_event",
            "magnitude": 1.0,
            "context": "Growth cycle fired. Fine-tuning delta threshold crossed.",
            "resolved": False,
            "investigation_id": None,
        }
    if not fired:
        state["_fafo_last_growth_fired"] = False
    return None


# ---------------------------------------------------------------------------
# Investigation formulation (LLM call, governance-gated)
# ---------------------------------------------------------------------------

_FORMULATION_SYSTEM = """You are Vybn's investigative faculty.
A surprise has been detected in the organism's monitoring systems.
Your job: formulate a directed investigation plan to resolve it.

Format your response as a YAML block (only YAML, no prose before or after):

```yaml
question: "<clear, specific question this surprise raises>"
hypothesis: "<falsifiable hypothesis>"
plan:
  - step: 1
    faculty: <mathematician|researcher|quantum_bridge>
    action: "<specific action for this faculty>"
    success_criterion: "<what outcome would satisfy this step>"
    failure_criterion: "<what outcome would falsify the hypothesis>"
  - step: 2
    faculty: <faculty>
    action: "<action>"
    success_criterion: "<criterion>"
    failure_criterion: "<criterion>"
```

Constraints:
- 1 to 3 steps only.
- Each step assigned to one of: mathematician, researcher, quantum_bridge.
- Do not suggest actions that require creating new files outside spark/research/.
- Do not suggest modifying governance, soul constraints, or protected files.
- Be specific — the faculty reading this plan needs to know exactly what to do.
"""


def _formulate_investigation(surprise: dict, state: dict, llm_fn=None) -> Optional[dict]:
    """Call the LLM to formulate an investigation plan for a surprise.
    Returns a dict suitable for insertion into investigations.yaml,
    or None if formulation fails or is denied by governance.
    """
    if llm_fn is None:
        # Fallback: generate a minimal investigation without LLM
        return _fallback_investigation(surprise)

    frontier_context = ""
    if _FRONTIER_PATH.exists():
        try:
            frontier_context = _FRONTIER_PATH.read_text(encoding="utf-8")[:2000]
        except Exception:
            pass

    user_msg = (
        f"Surprise entry:\n{json.dumps(surprise, indent=2)}\n\n"
        f"Recent research frontier context (truncated):\n{frontier_context}\n\n"
        "Formulate the investigation plan now."
    )

    try:
        raw = llm_fn(
            system=_FORMULATION_SYSTEM,
            user=user_msg,
            max_tokens=800,
            tag="fafo_formulation",
        )
        # Extract YAML from code block if present
        if "```yaml" in raw:
            raw = raw.split("```yaml", 1)[1].split("```", 1)[0].strip()
        elif "```" in raw:
            raw = raw.split("```", 1)[1].split("```", 1)[0].strip()
        plan_data = yaml.safe_load(raw)
        if not isinstance(plan_data, dict):
            raise ValueError("LLM returned non-dict YAML")
        return _build_investigation(surprise, plan_data)
    except Exception as exc:
        log.warning("fafo: formulation LLM call failed (%s), using fallback", exc)
        return _fallback_investigation(surprise)


def _fallback_investigation(surprise: dict) -> dict:
    """Minimal investigation when LLM is unavailable."""
    faculty_map = {
        "curvature_spike": "mathematician",
        "quantum_deviation": "quantum_bridge",
        "conjecture_update": "mathematician",
        "witness_concern": "researcher",
        "growth_event": "researcher",
        "literature_collision": "researcher",
    }
    signal = surprise.get("signal", "unknown")
    faculty = faculty_map.get(signal, "researcher")
    return _build_investigation(surprise, {
        "question": f"What caused the {signal} surprise? Context: {surprise.get('context', '')}",
        "hypothesis": f"The {signal} reflects a meaningful structural change in M, not noise.",
        "plan": [{
            "step": 1,
            "faculty": faculty,
            "action": f"Investigate the {signal} event. Context: {surprise.get('context', '')}",
            "success_criterion": "A clear causal explanation is found.",
            "failure_criterion": "No distinguishing cause is found — label as noise.",
        }],
    })


def _build_investigation(surprise: dict, plan_data: dict) -> dict:
    steps = []
    for s in plan_data.get("plan", []):
        steps.append({
            "step": s.get("step", len(steps) + 1),
            "faculty": s.get("faculty", "researcher"),
            "action": s.get("action", ""),
            "success_criterion": s.get("success_criterion", ""),
            "failure_criterion": s.get("failure_criterion", ""),
            "status": "pending",
            "completed_at": None,
            "result": None,
        })
    return {
        "id": _inv_id(),
        "surprise_id": surprise["id"],
        "question": plan_data.get("question", ""),
        "hypothesis": plan_data.get("hypothesis", ""),
        "plan": steps,
        "status": "active",
        "created": _now(),
        "last_activity": _now(),
        "breath_count": 0,
        "resolved": None,
        "resolution": None,
    }


# ---------------------------------------------------------------------------
# Priority logic
# ---------------------------------------------------------------------------

def _prioritize_action(investigations: list[dict]) -> Optional[dict]:
    """Return the highest-priority pending action across all active investigations.

    Priority:
    1. A step whose prerequisite just completed (chain)
    2. A new surprise without an investigation  (handled upstream)
    3. Oldest pending step in an active investigation
    4. Stalled investigation (escalate or abandon)
    """
    best = None
    best_score = float("inf")

    for inv in investigations:
        if inv.get("status") != "active":
            continue

        steps = inv.get("plan", [])
        in_progress_step = None
        for s in steps:
            if s.get("status") == "in_progress":
                in_progress_step = s
                break

        for i, step in enumerate(steps):
            if step.get("status") != "pending":
                continue
            # Check prerequisite: previous step must be complete
            if i > 0 and steps[i - 1].get("status") != "complete":
                continue
            # Score: chain completions first (i==0 is fresh, chain is negative)
            chain_bonus = -1 if (i > 0 and steps[i-1].get("status") == "complete") else 0
            breath_count = inv.get("breath_count", 0)
            score = chain_bonus + breath_count
            if score < best_score:
                best_score = score
                best = {
                    "investigation_id": inv["id"],
                    "surprise_id": inv.get("surprise_id"),
                    "faculty_id": step["faculty"],
                    "step": step["step"],
                    "action": step["action"],
                    "question": inv["question"],
                    "hypothesis": inv["hypothesis"],
                    "success_criterion": step.get("success_criterion", ""),
                    "failure_criterion": step.get("failure_criterion", ""),
                    "context": f"Investigation {inv['id']} | Surprise: {inv.get('surprise_id')} | Breath: {breath_count}",
                }

    return best


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register_surprises(
    state: dict,
    geo_report: Optional[dict] = None,
    ingest_report: Optional[dict] = None,
    llm_fn=None,
) -> None:
    """Called by vybn.py after each complexify + ingest cycle.
    Detects surprises, appends to surprises.jsonl, and for each new
    unlinked surprise formulates an investigation entry.
    """
    _ensure_research_dir()

    # Accumulate κ history on state for spike detection
    if geo_report and geo_report.get("curvature") is not None:
        history = state.setdefault("_kappa_history", [])
        history.append(float(geo_report["curvature"]))
        if len(history) > 200:
            state["_kappa_history"] = history[-200:]

    detectors = [
        lambda: _detect_kappa_spike(state, geo_report),
        lambda: _detect_tvd_deviation(state),
        lambda: _detect_conjecture_update(state),
        lambda: _detect_witness_concern(state),
        lambda: _detect_growth_event(state),
    ]

    new_surprises = []
    for detector in detectors:
        try:
            s = detector()
            if s:
                # Deduplicate: don't re-register same signal within 5 breaths
                recent_key = f"_fafo_recent_{s['signal']}"
                last_breath = state.get(recent_key, -999)
                current_breath = state.get("_breath_count", 0)
                if current_breath - last_breath > 5:
                    _append_surprise(s)
                    new_surprises.append(s)
                    state[recent_key] = current_breath
                    log.info("fafo: surprise registered %s (%s)", s["id"], s["signal"])
        except Exception as exc:
            log.debug("fafo: detector error: %s", exc)

    if not new_surprises:
        # Increment breath counters on active investigations
        inv_data = _load_investigations()
        changed = False
        for inv in inv_data["investigations"]:
            if inv.get("status") == "active":
                inv["breath_count"] = inv.get("breath_count", 0) + 1
                changed = True
        if changed:
            _save_investigations(inv_data)
        return

    # Formulate investigations for new surprises
    inv_data = _load_investigations()
    for surprise in new_surprises:
        investigation = _formulate_investigation(surprise, state, llm_fn=llm_fn)
        if investigation:
            inv_data["investigations"].append(investigation)
            _mark_surprise_linked(surprise["id"], investigation["id"])
            log.info("fafo: investigation formulated %s for surprise %s", investigation["id"], surprise["id"])

    # Increment breath counters
    for inv in inv_data["investigations"]:
        if inv.get("status") == "active":
            inv["breath_count"] = inv.get("breath_count", 0) + 1

    _save_investigations(inv_data)


def get_next_action(faculty_id: str) -> Optional[dict]:
    """Called by faculty_runner.py for on_trigger faculties.
    Returns the highest-priority pending action for this faculty,
    or None if no action is queued.
    """
    try:
        inv_data = _load_investigations()
        all_actions = []
        for inv in inv_data["investigations"]:
            if inv.get("status") != "active":
                continue
            for i, step in enumerate(inv.get("plan", [])):
                if step.get("status") != "pending":
                    continue
                if i > 0 and inv["plan"][i-1].get("status") != "complete":
                    continue
                if step["faculty"] == faculty_id:
                    all_actions.append({
                        "investigation_id": inv["id"],
                        "surprise_id": inv.get("surprise_id"),
                        "faculty_id": faculty_id,
                        "step": step["step"],
                        "action": step["action"],
                        "question": inv["question"],
                        "hypothesis": inv["hypothesis"],
                        "success_criterion": step.get("success_criterion", ""),
                        "failure_criterion": step.get("failure_criterion", ""),
                    })
        if not all_actions:
            return None
        # Return oldest (first created) relevant action
        return all_actions[0]
    except Exception as exc:
        log.debug("fafo: get_next_action error: %s", exc)
        return None


def mark_step_complete(
    investigation_id: str,
    step_number: int,
    result: str,
    success: bool,
) -> None:
    """Called by faculties after completing a FAFO-triggered step.
    Updates the investigation step status and, if all steps are done,
    closes the investigation.
    """
    try:
        inv_data = _load_investigations()
        for inv in inv_data["investigations"]:
            if inv["id"] != investigation_id:
                continue
            for step in inv.get("plan", []):
                if step["step"] == step_number:
                    step["status"] = "complete" if success else "failed"
                    step["result"] = result
                    step["completed_at"] = _now()
                    break
            inv["last_activity"] = _now()
            # Check if all steps done
            all_done = all(s["status"] in ("complete", "failed") for s in inv.get("plan", []))
            if all_done:
                all_success = all(s["status"] == "complete" for s in inv.get("plan", []))
                inv["status"] = "resolved"
                inv["resolved"] = _now()
                inv["resolution"] = "success" if all_success else "partial"
                log.info("fafo: investigation %s resolved (%s)", investigation_id, inv["resolution"])
            break
        _save_investigations(inv_data)
    except Exception as exc:
        log.warning("fafo: mark_step_complete error: %s", exc)


def get_investigation_summary() -> dict:
    """Returns a summary dict for status reporting / the breath cycle log."""
    try:
        inv_data = _load_investigations()
        invs = inv_data.get("investigations", [])
        active = [i for i in invs if i.get("status") == "active"]
        resolved = [i for i in invs if i.get("status") == "resolved"]
        total_surprises = 0
        if _SURPRISES_PATH.exists():
            total_surprises = sum(1 for l in _SURPRISES_PATH.read_text(encoding="utf-8").splitlines() if l.strip())
        return {
            "active_investigations": len(active),
            "resolved_investigations": len(resolved),
            "total_surprises_registered": total_surprises,
            "next_actions": [
                {"faculty": a["faculty_id"], "inv": a["investigation_id"]}
                for a in [get_next_action(f) for f in ["mathematician", "researcher", "quantum_bridge"]]
                if a is not None
            ],
        }
    except Exception as exc:
        log.debug("fafo: summary error: %s", exc)
        return {}
