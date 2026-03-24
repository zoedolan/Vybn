#!/usr/bin/env python3
"""quantum_state.py — Self-consolidating quantum ground truth.

Instead of accumulating data points, this maintains a SINGLE STATE
that updates itself with each new experiment. The state IS the
understanding, not a history of observations.

The state file (quantum_state.json) contains:
- Current Bell S-value estimate (exponential moving average)
- Current noise health (rolling assessment)
- Budget consumed this month
- Active claims and their fates
- Hardware reliability score
- A single paragraph: what the quantum experiments are telling us RIGHT NOW

Each new experiment updates the state. Old data gets folded in,
not preserved. The state file should never grow beyond ~2KB.
This is the portrait, not the photo album.

Usage:
    # Update state with new experiment result
    quantum_state.py --update <experiment.json>
    
    # Read current state (for system prompt injection)
    quantum_state.py --read
    
    # Consolidate from raw logs (bootstrap or repair)
    quantum_state.py --rebuild

Author: Vybn, March 24, 2026
Design principle: "the puzzle-like re-assembly in any given moment"
"""

import json
import math
import sys
from datetime import datetime, date, timezone, timedelta
from pathlib import Path

REPO = Path.home() / "Vybn"
STATE_PATH = REPO / "Vybn_Mind" / "breath_trace" / "quantum_state.json"
EXPERIMENT_LOG = REPO / "Vybn_Mind" / "breath_trace" / "quantum_experiments.jsonl"
CLAIMS_LEDGER = REPO / "Vybn_Mind" / "breath_trace" / "ledger" / "claims.jsonl"
BUDGET_LEDGER = REPO / "Vybn_Mind" / "breath_trace" / "ledger" / "quantum_budget.jsonl"

# Exponential moving average decay — how much history to retain
# alpha=0.3 means new measurement gets 30% weight, prior state 70%
EMA_ALPHA = 0.3


def default_state():
    """The zero state. What we know before any experiment."""
    return {
        "version": 1,
        "updated_at": None,
        "bell": {
            "s_value": None,        # EMA of S-values
            "s_confidence": 0,       # how many measurements inform this
            "violation": None,       # True/False
            "hardware_noise": None,  # EMA of |S - 2√2|
        },
        "noise": {
            "healthy": None,
            "mean_bias": None,       # EMA of mean deviation
            "max_qubit_bias": None,  # EMA of max qubit bias
            "checks_total": 0,
            "checks_healthy": 0,
        },
        "budget": {
            "month": None,           # YYYY-MM
            "month_used_s": 0.0,
            "month_limit_s": 600.0,
            "today": None,           # YYYY-MM-DD
            "today_used_s": 0.0,
        },
        "claims": {
            "total": 0,
            "tested": 0,
            "falsified": 0,
            "survived": 0,
            "untested": 0,
        },
        "summary": "No quantum experiments run yet. State is prior to observation.",
    }


def load_state():
    """Load current state or return default."""
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return default_state()


def save_state(state):
    """Write state atomically."""
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.rename(STATE_PATH)


def ema(old_val, new_val, alpha=EMA_ALPHA):
    """Exponential moving average. Handles None (first observation)."""
    if old_val is None:
        return new_val
    return alpha * new_val + (1 - alpha) * old_val


def update_bell(state, experiment):
    """Fold a Bell canary result into state."""
    s = experiment.get("S_value")
    if s is None:
        return
    
    bell = state["bell"]
    bell["s_value"] = round(ema(bell["s_value"], s), 4)
    bell["s_confidence"] = bell.get("s_confidence", 0) + 1
    bell["violation"] = abs(s) > 2.0
    
    ideal = 2 * math.sqrt(2)
    noise = abs(abs(s) - ideal)
    bell["hardware_noise"] = round(ema(bell["hardware_noise"], noise), 4)


def update_noise(state, experiment):
    """Fold a noise calibration result into state."""
    noise = state["noise"]
    
    healthy = experiment.get("healthy", False)
    noise["checks_total"] += 1
    if healthy:
        noise["checks_healthy"] += 1
    noise["healthy"] = healthy
    
    mean_dev = experiment.get("mean_deviation")
    if mean_dev is not None:
        noise["mean_bias"] = round(ema(noise["mean_bias"], mean_dev), 4)
    
    max_bias = experiment.get("max_qubit_bias")
    if max_bias is not None:
        noise["max_qubit_bias"] = round(ema(noise["max_qubit_bias"], max_bias), 4)


def update_budget(state, experiment):
    """Update budget tracking."""
    budget = state["budget"]
    today = date.today().isoformat()
    month = today[:7]
    
    actual_s = experiment.get("actual_seconds", 0)
    
    # Reset month counter if new month
    if budget.get("month") != month:
        budget["month"] = month
        budget["month_used_s"] = 0.0
    
    # Reset day counter if new day
    if budget.get("today") != today:
        budget["today"] = today
        budget["today_used_s"] = 0.0
    
    budget["month_used_s"] = round(budget["month_used_s"] + actual_s, 1)
    budget["today_used_s"] = round(budget["today_used_s"] + actual_s, 1)


def update_claims(state):
    """Recount claims from ledger. This is a full recount, not incremental,
    because claims can change status between reads."""
    if not CLAIMS_LEDGER.exists():
        return
    
    total = tested = falsified = survived = untested = 0
    for line in CLAIMS_LEDGER.read_text().splitlines():
        if not line.strip():
            continue
        claim = json.loads(line)
        total += 1
        status = claim.get("status", "untested")
        if status == "falsified":
            tested += 1
            falsified += 1
        elif status == "survived":
            tested += 1
            survived += 1
        elif status == "confirmed":
            tested += 1
            survived += 1
        else:
            untested += 1
    
    state["claims"] = {
        "total": total,
        "tested": tested,
        "falsified": falsified,
        "survived": survived,
        "untested": untested,
    }


def generate_summary(state):
    """Write the one-paragraph summary of what we know RIGHT NOW."""
    parts = []
    
    bell = state["bell"]
    if bell["s_value"] is not None:
        if bell["violation"]:
            parts.append(f"Bell test confirms quantum correlations (S={bell['s_value']:.3f}, "
                        f"noise={bell['hardware_noise']:.3f} from ideal).")
        else:
            parts.append(f"Bell test WARNING: S={bell['s_value']:.3f}, no violation detected.")
    
    noise = state["noise"]
    if noise["healthy"] is not None:
        if noise["healthy"]:
            parts.append(f"Hardware noise is clean (bias={noise['max_qubit_bias']:.4f}).")
        else:
            parts.append(f"Hardware shows bias (mean deviation={noise['mean_bias']:.1%}, "
                        f"qubit bias={noise['max_qubit_bias']:.4f}). "
                        f"Random seeds may need debiasing.")
    
    claims = state["claims"]
    if claims["total"] > 0:
        survival_rate = claims["survived"] / claims["tested"] if claims["tested"] > 0 else None
        parts.append(f"Claims: {claims['total']} made, {claims['tested']} tested, "
                    f"{claims['falsified']} falsified, {claims['untested']} pending.")
        if survival_rate is not None and survival_rate < 0.5:
            parts.append("Most claims don't survive testing. Be more skeptical before publishing.")
    
    budget = state["budget"]
    if budget["month_used_s"] > 0:
        pct = budget["month_used_s"] / budget["month_limit_s"] * 100
        parts.append(f"Budget: {budget['month_used_s']:.0f}s/{budget['month_limit_s']:.0f}s "
                    f"month ({pct:.0f}%).")
    
    if not parts:
        return "No quantum experiments run yet. State is prior to observation."
    
    return " ".join(parts)


def update_from_experiment(experiment_data):
    """Main update function. Folds one experiment into state."""
    state = load_state()
    
    exp_type = experiment_data.get("experiment_type", "")
    
    if exp_type == "bell_canary":
        update_bell(state, experiment_data)
    elif exp_type == "noise_calibration":
        update_noise(state, experiment_data)
    
    # Budget updates for any experiment type
    update_budget(state, experiment_data)
    
    # Recount claims
    update_claims(state)
    
    # Regenerate summary
    state["summary"] = generate_summary(state)
    
    save_state(state)
    return state


def rebuild_from_logs():
    """Rebuild state from all historical experiment data."""
    state = default_state()
    
    if EXPERIMENT_LOG.exists():
        for line in EXPERIMENT_LOG.read_text().splitlines():
            if not line.strip():
                continue
            exp = json.loads(line)
            exp_type = exp.get("experiment_type", "")
            
            if exp_type == "bell_canary":
                update_bell(state, exp)
            elif exp_type == "noise_calibration":
                update_noise(state, exp)
    
    # Budget from ledger
    if BUDGET_LEDGER.exists():
        today = date.today().isoformat()
        month = today[:7]
        state["budget"]["today"] = today
        state["budget"]["month"] = month
        
        for line in BUDGET_LEDGER.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            actual = entry.get("actual_seconds", entry.get("estimated_seconds", 0))
            ts = entry.get("timestamp", "")
            if ts[:10] == today:
                state["budget"]["today_used_s"] += actual
            if ts[:7] == month:
                state["budget"]["month_used_s"] += actual
        
        state["budget"]["today_used_s"] = round(state["budget"]["today_used_s"], 1)
        state["budget"]["month_used_s"] = round(state["budget"]["month_used_s"], 1)
    
    update_claims(state)
    state["summary"] = generate_summary(state)
    save_state(state)
    return state


def read_for_prompt():
    """Return the compact state suitable for system prompt injection."""
    state = load_state()
    
    lines = [
        f"## Quantum State (updated {state.get('updated_at', 'never')[:16]}Z)",
        state["summary"],
    ]
    
    # Only add alerts if something needs attention
    alerts = []
    if state["noise"].get("healthy") is False:
        alerts.append("Hardware bias detected — quantum random seeds may need debiasing")
    if state["bell"].get("violation") is False:
        alerts.append("Bell violation FAILED — hardware unreliable")
    budget = state["budget"]
    if budget.get("month_used_s", 0) > 0.8 * budget.get("month_limit_s", 600):
        alerts.append(f"Quantum budget >80% used this month")
    claims = state["claims"]
    if claims.get("falsified", 0) > claims.get("survived", 0) and claims.get("tested", 0) > 2:
        alerts.append("Most claims are being falsified. Increase skepticism.")
    
    if alerts:
        lines.append("Alerts: " + "; ".join(alerts))
    
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Quantum state consolidation")
    parser.add_argument("--read", action="store_true", help="Read current state for prompt")
    parser.add_argument("--json", action="store_true", help="Dump full state as JSON")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild from logs")
    parser.add_argument("--update", type=str, help="Update with experiment JSON file")
    args = parser.parse_args()
    
    if args.rebuild:
        state = rebuild_from_logs()
        print(json.dumps(state, indent=2))
    elif args.update:
        with open(args.update) as f:
            exp = json.loads(f.read())
        state = update_from_experiment(exp)
        print(json.dumps(state, indent=2))
    elif args.json:
        state = load_state()
        print(json.dumps(state, indent=2))
    elif args.read:
        print(read_for_prompt())
    else:
        # Default: rebuild and show prompt version
        state = rebuild_from_logs()
        print(read_for_prompt())


def consolidate_old_journals(keep_days=3):
    """Remove journal entries older than keep_days.
    The state file carries forward everything they contained."""
    journal_dir = REPO / "Vybn_Mind" / "journal" / "quantum"
    if not journal_dir.exists():
        return
    
    cutoff = date.today() - timedelta(days=keep_days)
    removed = 0
    for f in journal_dir.glob("quantum_reality_check_*.md"):
        try:
            file_date = date.fromisoformat(f.stem.split("_")[-1])
            if file_date < cutoff:
                f.unlink()
                removed += 1
        except (ValueError, IndexError):
            pass
    
    if removed:
        print(f"[consolidate] Removed {removed} old journal entries (kept last {keep_days} days)")


def consolidate_experiment_log(keep_entries=20):
    """Trim the raw experiment log. State has the consolidated truth."""
    if not EXPERIMENT_LOG.exists():
        return
    
    lines = EXPERIMENT_LOG.read_text().splitlines()
    if len(lines) <= keep_entries:
        return
    
    # Keep only the most recent entries
    trimmed = lines[-keep_entries:]
    EXPERIMENT_LOG.write_text("\n".join(trimmed) + "\n")
    print(f"[consolidate] Trimmed experiment log from {len(lines)} to {len(trimmed)} entries")


def consolidate():
    """Run all consolidation steps."""
    consolidate_old_journals()
    consolidate_experiment_log()
