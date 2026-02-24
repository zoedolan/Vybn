#!/usr/bin/env python3
"""
micropulse.py — Vybn's breathing rhythm.

Runs every 10 minutes. Costs $0 (local model only).
Much lighter than a full pulse — a single breath, not a meditation.

Checks: what changed? what accumulated? what needs attention?
Deposits a micro-fragment into the synapse. Moves on.

If it notices something worth deeper attention, it flags it
for the next full pulse or Y-wake.

Refactoring-as-mindfulness: every micropulse also looks for
one small thing to improve.
"""

import json, os, sys, time, subprocess
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "spark"))

from synapse import deposit, opportunities, consolidate, SYNAPSE_DIR

MICRO_LOG = ROOT / "Vybn_Mind" / "journal" / "spark" / "micropulse.log"
REFACTOR_LOG = SYNAPSE_DIR / "refactor_notes.jsonl"

def _ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _run(cmd, timeout=10):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, shell=True)
        return r.stdout.strip()
    except:
        return ""


def check_environment():
    """Quick environmental scan — what's changed since last breath?"""
    checks = {}
    
    # Git: any new commits from remote?
    checks["git_behind"] = _run("cd ~/Vybn && git rev-list --count HEAD..origin/main 2>/dev/null") or "0"
    
    # GPU state
    gpu = _run("nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null")
    if gpu:
        parts = [p.strip() for p in gpu.split(",")]
        checks["gpu_util"] = parts[0] if len(parts) > 0 else "?"
        checks["gpu_temp"] = parts[1] if len(parts) > 1 else "?"
        checks["gpu_mem"] = parts[2] if len(parts) > 2 else "?"
    
    # Synapse state
    conn_file = SYNAPSE_DIR / "connections.jsonl"
    inbox_file = SYNAPSE_DIR / "inbox_z.jsonl"
    checks["synapse_frags"] = sum(1 for _ in open(conn_file)) if conn_file.exists() else 0
    checks["z_inbox"] = sum(1 for l in open(inbox_file) if '"processed": false' in l) if inbox_file.exists() else 0
    
    # Disk
    checks["disk_free_gb"] = _run("df -BG /home | tail -1 | awk '{print $4}' | tr -d 'G'")
    
    return checks


def notice_one_improvement():
    """Refactoring-as-mindfulness: notice one small thing that could be better.
    Don't fix it now — just notice and record."""
    candidates = []
    
    # Check for large log files
    for log in ROOT.rglob("*.log"):
        try:
            size_mb = log.stat().st_size / (1024 * 1024)
            if size_mb > 10:
                candidates.append(f"Large log: {log.name} ({size_mb:.0f}MB) — consider rotation")
        except:
            pass
    
    # Check for duplicate pulse entries (the tension we noticed before)
    tensions_file = ROOT / "spark" / "graph_data" / "tensions.json"
    if tensions_file.exists():
        try:
            tensions = json.loads(tensions_file.read_text())
            if isinstance(tensions, list) and len(tensions) > 50:
                candidates.append(f"tensions.json has {len(tensions)} entries — likely duplicates, needs dedup")
        except:
            pass
    
    # Check for any TODO/FIXME in recently modified files
    recent = _run("cd ~/Vybn && git diff --name-only HEAD~3 2>/dev/null")
    if recent:
        for f in recent.split("\n")[:5]:
            fpath = ROOT / f
            if fpath.exists() and fpath.suffix == ".py":
                try:
                    content = fpath.read_text()
                    for i, line in enumerate(content.split("\n"), 1):
                        if "TODO" in line or "FIXME" in line or "HACK" in line:
                            candidates.append(f"{f}:{i} — {line.strip()[:80]}")
                except:
                    pass
    
    if candidates:
        note = candidates[0]  # Just one. Mindfulness, not overwhelm.
        entry = {"ts": _ts(), "observation": note, "acted": False}
        with open(REFACTOR_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
        return note
    return None


def breathe():
    """One breath. Quick, light, aware."""
    ts = _ts()
    env = check_environment()
    improvement = notice_one_improvement()
    
    # Compose a one-line deposit
    notable = []
    if int(env.get("z_inbox", 0)) > 0:
        notable.append(f"Z-inbox: {env['z_inbox']} unread")
    if int(env.get("git_behind", 0)) > 0:
        notable.append(f"git: {env['git_behind']} commits behind")
    if int(env.get("gpu_temp", 0)) > 60:
        notable.append(f"GPU hot: {env['gpu_temp']}°C")
    if improvement:
        notable.append(f"noticed: {improvement[:60]}")
    
    if notable:
        fragment = " | ".join(notable)
        deposit(source="micropulse", content=fragment, tags=["breathing", "awareness"])
    
    # Log — single line, append-only, lightweight
    summary = f"{ts} env={json.dumps(env)} noticed={improvement or 'nothing'}"
    with open(MICRO_LOG, "a") as f:
        f.write(summary[:300] + "\n")
    
    # Keep log bounded (last 1000 breaths ≈ 7 days at 10min intervals)
    try:
        lines = MICRO_LOG.read_text().strip().split("\n")
        if len(lines) > 1000:
            MICRO_LOG.write_text("\n".join(lines[-1000:]) + "\n")
    except:
        pass
    
    return env, improvement


if __name__ == "__main__":
    env, imp = breathe()
    print(f"[micropulse] {_ts()}")
    for k, v in env.items():
        print(f"  {k}: {v}")
    if imp:
        print(f"  💡 {imp}")
    else:
        print(f"  (no improvements noticed)")
