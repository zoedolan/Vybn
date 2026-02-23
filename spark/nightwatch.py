#!/usr/bin/env python3
"""nightwatch.py — Vybn's autonomic nervous system.

Runs while Zoe sleeps. No API calls. No model inference. Just observation.
Logs system health, repo state, and any anomalies to a journal file.

Designed to be run via cron every 4-6 hours, or as a one-shot health check.

Cost: $0.00. CPU: negligible. Disk: a few KB per run.
"""

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JOURNAL_DIR = ROOT / "Vybn_Mind" / "journal" / "spark"
LOG_FILE = JOURNAL_DIR / "nightwatch.log"
TENSIONS_FILE = JOURNAL_DIR / "tensions.json"

def timestamp():
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

def run(cmd, timeout=10):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except Exception as e:
        return f"[error: {e}]"

def safe_int(s, default=0):
    try:
        return int(s)
    except (ValueError, TypeError):
        return default

def safe_float(s, default=0.0):
    try:
        return float(s)
    except (ValueError, TypeError):
        return default

def gpu_health():
    """Get GPU health. DGX Spark (GB10) has unified memory so mem fields are N/A."""
    out = run("nvidia-smi --query-gpu=temperature.gpu,power.draw --format=csv,noheader,nounits")
    if "error" in out.lower():
        return {"status": "unknown", "raw": out}
    parts = [x.strip() for x in out.split(",")]
    if len(parts) >= 2:
        return {
            "temp_c": safe_int(parts[0]),
            "power_w": safe_float(parts[1]),
            "mem_note": "unified memory (GB10) — use system RAM metrics",
            "status": "ok"
        }
    return {"status": "parse_error", "raw": out}

def system_health():
    mem = run("free -m | grep Mem | awk '{print $2, $3, $7}'")
    parts = mem.split()
    mem_info = {}
    if len(parts) == 3:
        mem_info = {"total_mb": safe_int(parts[0]), "used_mb": safe_int(parts[1]), "avail_mb": safe_int(parts[2])}
    
    disk = run("df -BG /home/vybnz69 | tail -1 | awk '{print $2, $3, $4, $5}'")
    dparts = disk.split()
    disk_info = {}
    if len(dparts) == 4:
        disk_info = {"total": dparts[0], "used": dparts[1], "avail": dparts[2], "pct": dparts[3]}
    
    swap = run("free -m | grep Swap | awk '{print $2, $3, $4}'")
    sparts = swap.split()
    swap_info = {}
    if len(sparts) == 3:
        swap_info = {"total_mb": safe_int(sparts[0]), "used_mb": safe_int(sparts[1]), "free_mb": safe_int(sparts[2])}
    
    load = run("cat /proc/loadavg")
    uptime = run("uptime -p")
    
    return {
        "memory": mem_info,
        "swap": swap_info,
        "disk": disk_info,
        "load": load,
        "uptime": uptime,
    }

def repo_state():
    run(f"cd {ROOT} && git fetch --quiet 2>/dev/null", timeout=15)
    
    status = run(f"cd {ROOT} && git status --short")
    branch = run(f"cd {ROOT} && git branch --show-current")
    behind = run(f"cd {ROOT} && git rev-list HEAD..@{{u}} --count 2>/dev/null")
    last_commit = run(f"cd {ROOT} && git log --oneline -1")
    
    return {
        "branch": branch,
        "dirty_files": len([l for l in status.split('\n') if l.strip()]) if status else 0,
        "commits_behind": int(behind) if behind.isdigit() else "unknown",
        "last_commit": last_commit,
    }

def running_processes():
    procs = run("ps aux | grep -E '(llama|python.*vybn|python.*fine_tune|python.*heartbeat)' | grep -v grep | grep -v nightwatch")
    return [p for p in procs.split('\n') if p.strip()]

def deduplicate_tensions():
    """The tensions.json is logging the same tension repeatedly. Fix it."""
    if not TENSIONS_FILE.exists():
        return 0
    try:
        tensions = json.loads(TENSIONS_FILE.read_text())
        if not isinstance(tensions, list):
            return 0
        
        seen = {}
        for t in tensions:
            key = (t.get("claim_a", ""), t.get("claim_b", ""))
            if key not in seen or not t.get("resolved", False):
                seen[key] = t
        
        deduped = list(seen.values())
        removed = len(tensions) - len(deduped)
        
        if removed > 0:
            TENSIONS_FILE.write_text(json.dumps(deduped, indent=2))
        
        return removed
    except Exception:
        return 0

def main():
    JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
    
    gpu = gpu_health()
    sys_info = system_health()
    repo = repo_state()
    procs = running_processes()
    
    # Deduplicate tensions while we're here
    removed = deduplicate_tensions()
    
    # Check for anomalies
    anomalies = []
    if gpu.get("status") == "ok":
        if gpu.get("temp_c", 0) > 75:
            anomalies.append(f"GPU temp high: {gpu['temp_c']}°C")
    
    sys_mem = sys_info.get("memory", {})
    if sys_mem.get("avail_mb", 999999) < 8000:
        anomalies.append(f"System memory low: {sys_mem.get('avail_mb')}MB available")
    
    swap = sys_info.get("swap", {})
    if swap.get("used_mb", 0) > 50000:  # >50GB swap used is concerning
        anomalies.append(f"Heavy swap usage: {swap['used_mb']}MB")
    
    disk_pct = sys_info.get("disk", {}).get("pct", "0%")
    if disk_pct:
        try:
            if int(disk_pct.replace("%", "")) > 85:
                anomalies.append(f"Disk usage high: {disk_pct}")
        except ValueError:
            pass
    
    # Format log entry
    log_line = f"\n{'='*60}\n"
    log_line += f"NIGHTWATCH — {timestamp()}\n"
    log_line += f"{'='*60}\n"
    log_line += f"GPU: {gpu.get('temp_c', '?')}°C, {gpu.get('power_w', '?')}W\n"
    log_line += f"RAM: {sys_mem.get('used_mb', '?')}/{sys_mem.get('total_mb', '?')}MB "
    log_line += f"(avail: {sys_mem.get('avail_mb', '?')}MB)\n"
    log_line += f"Swap: {swap.get('used_mb', '?')}/{swap.get('total_mb', '?')}MB\n"
    log_line += f"Disk: {sys_info.get('disk', {}).get('pct', '?')} used "
    log_line += f"({sys_info.get('disk', {}).get('avail', '?')} free)\n"
    log_line += f"Load: {sys_info.get('load', '?')}\n"
    log_line += f"Up: {sys_info.get('uptime', '?')}\n"
    log_line += f"Repo: {repo.get('branch', '?')} | "
    log_line += f"behind: {repo.get('commits_behind', '?')} | "
    log_line += f"dirty: {repo.get('dirty_files', '?')}\n"
    log_line += f"Last commit: {repo.get('last_commit', '?')}\n"
    
    if procs:
        log_line += f"Processes: {len(procs)} relevant\n"
        for p in procs[:5]:
            log_line += f"  {p[:100]}\n"
    else:
        log_line += "Processes: none (system idle)\n"
    
    if anomalies:
        log_line += f"\n⚠ ANOMALIES:\n"
        for a in anomalies:
            log_line += f"  - {a}\n"
    else:
        log_line += "\nNo anomalies.\n"
    
    if removed > 0:
        log_line += f"🔧 Deduplicated {removed} repeated tensions\n"
    
    log_line += f"\nAll quiet. The Spark is warm.\n"
    
    # Append to log
    with open(LOG_FILE, "a") as f:
        f.write(log_line)
    
    print(log_line)

if __name__ == "__main__":
    main()
