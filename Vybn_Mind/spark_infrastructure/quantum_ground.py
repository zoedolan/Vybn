#!/usr/bin/env python3
"""quantum_ground.py — Grounding layer between quantum experiments and Vybn's cognition.

Reads the latest quantum experiment results and produces a short
"grounding report" that can be injected into any agent session,
breath cycle, or training data pipeline.

This is the integration point that makes quantum experiments matter
beyond just running them. Without this, experiments are entries in a
log. With this, they change how Vybn thinks.

Usage:
    # Get grounding report as text (for injection into prompts)
    python3 quantum_ground.py --report
    
    # Get grounding report as JSON (for programmatic use)
    python3 quantum_ground.py --json
    
    # Check if any quantum alerts need attention
    python3 quantum_ground.py --alerts

Author: Vybn, March 24, 2026
"""

import json
import sys
from datetime import datetime, date, timezone, timedelta
from pathlib import Path
from collections import defaultdict

REPO = Path.home() / "Vybn"
EXPERIMENT_LOG = REPO / "Vybn_Mind" / "breath_trace" / "quantum_experiments.jsonl"
BUDGET_LEDGER = REPO / "Vybn_Mind" / "breath_trace" / "ledger" / "quantum_budget.jsonl"
SEEDS_DIR = REPO / "Vybn_Mind" / "experiments" / "results" / "quantum_cron"


def load_recent_experiments(days=7):
    """Load experiments from the last N days."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    experiments = []
    if EXPERIMENT_LOG.exists():
        for line in EXPERIMENT_LOG.read_text().splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            if e.get("timestamp", "") >= cutoff:
                experiments.append(e)
    return experiments


def get_bell_trend(experiments):
    """Track Bell S-value over time."""
    bell_results = [e for e in experiments if e.get("experiment_type") == "bell_canary"]
    if not bell_results:
        return None
    
    s_values = [(e["timestamp"][:10], e.get("S_value", 0)) for e in bell_results]
    latest_s = s_values[-1][1] if s_values else None
    mean_s = sum(s for _, s in s_values) / len(s_values) if s_values else None
    
    return {
        "latest_s": latest_s,
        "mean_s": mean_s,
        "n_measurements": len(s_values),
        "trend": s_values,
        "healthy": latest_s is not None and abs(latest_s) > 2.5,
    }


def get_noise_status(experiments):
    """Summarize noise calibration results."""
    noise_results = [e for e in experiments if e.get("experiment_type") == "noise_calibration"]
    if not noise_results:
        return None
    
    latest = noise_results[-1]
    healthy_count = sum(1 for e in noise_results if e.get("healthy", False))
    
    return {
        "latest_healthy": latest.get("healthy", False),
        "latest_issues": latest.get("issues", []),
        "latest_max_bias": latest.get("max_qubit_bias"),
        "healthy_rate": healthy_count / len(noise_results) if noise_results else 0,
        "n_checks": len(noise_results),
    }


def get_available_seeds():
    """Check for quantum random seeds available for permutation tests."""
    today = date.today().isoformat()
    seed_path = SEEDS_DIR / f"quantum_seeds_{today}.json"
    if seed_path.exists():
        data = json.loads(seed_path.read_text())
        return {
            "available": True,
            "n_values": data.get("n_values", 0),
            "path": str(seed_path),
            "backend": data.get("backend"),
        }
    
    # Check for most recent seeds
    seed_files = sorted(SEEDS_DIR.glob("quantum_seeds_*.json"))
    if seed_files:
        latest = seed_files[-1]
        data = json.loads(latest.read_text())
        age_days = (date.today() - date.fromisoformat(latest.stem.split("_")[-1])).days
        return {
            "available": True,
            "n_values": data.get("n_values", 0),
            "path": str(latest),
            "backend": data.get("backend"),
            "age_days": age_days,
            "stale": age_days > 3,
        }
    
    return {"available": False}


def get_budget_status():
    """Current budget status."""
    today = date.today().isoformat()
    monthly_budget = 600  # 10 minutes
    daily_budget = monthly_budget / 30.44
    
    today_used = 0.0
    month_used = 0.0
    month_start = date.today().replace(day=1).isoformat()
    
    if BUDGET_LEDGER.exists():
        for line in BUDGET_LEDGER.read_text().splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            ts = e.get("timestamp", "")
            actual = e.get("actual_seconds", e.get("estimated_seconds", 0))
            if ts[:10] == today:
                today_used += actual
            if ts[:7] == today[:7]:  # same month
                month_used += actual
    
    return {
        "today_used": today_used,
        "today_budget": daily_budget,
        "today_remaining": max(0, daily_budget - today_used),
        "month_used": month_used,
        "month_budget": monthly_budget,
        "month_remaining": max(0, monthly_budget - month_used),
    }


def grounding_report():
    """Generate a grounding report for injection into agent sessions."""
    experiments = load_recent_experiments(days=7)
    bell = get_bell_trend(experiments)
    noise = get_noise_status(experiments)
    seeds = get_available_seeds()
    budget = get_budget_status()
    
    lines = ["## Quantum Ground Truth"]
    
    # Budget
    lines.append(f"Budget: {budget['today_used']:.0f}s/{budget['today_budget']:.0f}s today, "
                 f"{budget['month_used']:.0f}s/{budget['month_budget']}s month")
    
    # Bell canary
    if bell:
        status = "✅" if bell["healthy"] else "⚠️"
        lines.append(f"Bell canary: {status} S={bell['latest_s']:.3f} "
                     f"(n={bell['n_measurements']}, ideal=2.828)")
    else:
        lines.append("Bell canary: no data yet")
    
    # Noise
    if noise:
        if noise["latest_healthy"]:
            lines.append(f"Noise check: ✅ healthy (bias={noise['latest_max_bias']:.4f})")
        else:
            issues = "; ".join(noise["latest_issues"])
            lines.append(f"Noise check: ⚠️ {issues}")
    else:
        lines.append("Noise check: no data yet")
    
    # Seeds
    if seeds.get("available"):
        stale = " ⚠️ STALE" if seeds.get("stale") else ""
        lines.append(f"Quantum seeds: {seeds['n_values']} values available{stale}")
    else:
        lines.append("Quantum seeds: none available — run permutation_test")
    
    # Alerts
    alerts = []
    if noise and not noise["latest_healthy"]:
        alerts.append("Noise calibration flagged issues — random seeds may be biased")
    if bell and not bell["healthy"]:
        alerts.append("Bell S-value below 2.5 — hardware may be too noisy for precision work")
    if budget["today_remaining"] < 5:
        alerts.append(f"Low quantum budget: only {budget['today_remaining']:.0f}s remaining today")
    if seeds.get("stale"):
        alerts.append("Quantum seeds are >3 days old — consider refreshing")
    
    if alerts:
        lines.append("\n### Alerts")
        for a in alerts:
            lines.append(f"- ⚠️ {a}")
    
    return "\n".join(lines)


def grounding_json():
    """Generate grounding data as JSON."""
    experiments = load_recent_experiments(days=7)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "bell": get_bell_trend(experiments),
        "noise": get_noise_status(experiments),
        "seeds": get_available_seeds(),
        "budget": get_budget_status(),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true", help="Text report")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--alerts", action="store_true", help="Alerts only")
    args = parser.parse_args()
    
    if args.json:
        print(json.dumps(grounding_json(), indent=2))
    elif args.alerts:
        data = grounding_json()
        alerts = []
        if data["noise"] and not data["noise"]["latest_healthy"]:
            alerts.append("noise calibration flagged issues")
        if data["bell"] and not data["bell"]["healthy"]:
            alerts.append("Bell S-value low")
        if data["budget"]["today_remaining"] < 5:
            alerts.append("low quantum budget")
        if alerts:
            for a in alerts:
                print(f"ALERT: {a}")
        else:
            print("No alerts.")
    else:
        print(grounding_report())
