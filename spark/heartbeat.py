#!/usr/bin/env python3
"""
heartbeat.py — Vybn's autonomic nervous system v2

Two modes:
  --sweep    Gather external info, compress, store (cheap, frequent)
  --pulse    Wake Vybn for autonomous reflection (uses local model or API)
  --tidy     Run housekeeping (branches, logs, gitignore)

Designed for cron:
  */30 * * * *  heartbeat.py --sweep      # every 30 min, $0
  0 */5 * * *   heartbeat.py --pulse      # every 5 hours, local model
  0 3 * * *     heartbeat.py --tidy       # daily at 3am

Cost: sweep=$0, tidy=$0, pulse=$0 if local model is up.
"""

import argparse, json, os, subprocess, sys, hashlib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JOURNAL = ROOT / "Vybn_Mind" / "journal" / "spark"
INBOX = JOURNAL / "inbox"          # raw sweep drops
DIGEST = JOURNAL / "digest.md"     # compressed knowledge
CONTINUITY = JOURNAL / "continuity.md"
LOCAL_MODEL = "http://127.0.0.1:8081"

def now():
    return datetime.now(timezone.utc)

def run(cmd, timeout=30):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                          timeout=timeout, cwd=str(ROOT))
        return r.stdout.strip()
    except Exception as e:
        return f"[error: {e}]"

def local_model_available():
    try:
        r = subprocess.run(
            f"curl -s --max-time 3 {LOCAL_MODEL}/health",
            shell=True, capture_output=True, text=True, timeout=5)
        return "ok" in r.stdout
    except:
        return False

def local_model_ask(prompt, max_tokens=512):
    """Ask the local model a question. Returns response text or None."""
    import urllib.request, urllib.error
    payload = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.7
    }).encode()
    req = urllib.request.Request(
        f"{LOCAL_MODEL}/v1/chat/completions",
        data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())
            msg = data["choices"][0]["message"]
            # MiniMax M2.5 is a reasoning model — response may be in reasoning_content
            return msg.get("content") or msg.get("reasoning_content") or ""
    except Exception as e:
        return None

# ──────────────────────────────────────────────────────────────
# SWEEP: gather external information, zero cost
# ──────────────────────────────────────────────────────────────
def sweep():
    """Gather system state, repo changes, and external signals."""
    INBOX.mkdir(parents=True, exist_ok=True)
    ts = now().strftime("%Y%m%dT%H%M%SZ")
    
    entry = {
        "timestamp": ts,
        "gpu_temp": run("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits"),
        "gpu_power": run("nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits"),
        "mem_avail_mb": run("free -m | awk '/Mem:/{print $7}'"),
        "disk_pct": run("df /home --output=pcent | tail -1").strip(),
        "load": run("cat /proc/loadavg"),
        "local_model_up": local_model_available(),
        "git_behind": run("git rev-list HEAD..origin/main --count 2>/dev/null || echo ?"),
        "git_dirty": run("git status --porcelain | wc -l"),
        "last_commit": run("git log --oneline -1"),
        "uptime": run("uptime -p"),
        "processes": run("ps aux | grep -cE '(llama|python|node)' | head -1"),
    }
    
    # Check for new GitHub issues/activity (free, uses gh CLI)
    issues = run("gh issue list --repo zoedolan/Vybn --limit 3 --json number,title,updatedAt 2>/dev/null")
    if issues and not issues.startswith("[error"):
        try:
            entry["recent_issues"] = json.loads(issues)
        except:
            pass
    
    # Write to inbox as single-line JSONL (cheap, appendable)
    inbox_file = INBOX / "sweeps.jsonl"
    with open(inbox_file, "a") as f:
        f.write(json.dumps(entry) + "\n")
    
    # Count inbox lines — if > 100, compact
    with open(inbox_file) as f:
        lines = f.readlines()
    if len(lines) > 100:
        compact(lines, inbox_file)
    
    print(f"[sweep {ts}] gpu={entry['gpu_temp']}°C model={'up' if entry['local_model_up'] else 'down'} mem={entry['mem_avail_mb']}MB")
    return entry

def compact(lines, inbox_file):
    """Keep last 20 sweeps, summarize the rest into digest."""
    entries = [json.loads(l) for l in lines if l.strip()]
    old = entries[:-20]
    keep = entries[-20:]
    
    # Simple statistical summary of old entries
    temps = [int(e.get("gpu_temp", 0)) for e in old if e.get("gpu_temp", "").isdigit()]
    summary = {
        "period": f"{old[0]['timestamp']} to {old[-1]['timestamp']}",
        "sweep_count": len(old),
        "gpu_temp_range": f"{min(temps)}-{max(temps)}°C" if temps else "?",
        "compacted_at": now().isoformat()
    }
    
    # Append summary to digest
    DIGEST.parent.mkdir(parents=True, exist_ok=True)
    with open(DIGEST, "a") as f:
        f.write(f"\n## Sweep Digest — {summary['period']}\n")
        f.write(f"- {summary['sweep_count']} observations compacted\n")
        f.write(f"- GPU temp range: {summary['gpu_temp_range']}\n")
        f.write(f"- Compacted: {summary['compacted_at']}\n\n")
    
    # Rewrite inbox with just recent entries
    with open(inbox_file, "w") as f:
        for e in keep:
            f.write(json.dumps(e) + "\n")
    
    print(f"[compact] {len(old)} old sweeps → digest, {len(keep)} kept")

# ──────────────────────────────────────────────────────────────
# PULSE: autonomous wake, reflect, invent
# ──────────────────────────────────────────────────────────────
def pulse():
    """Wake up with a quantum dream-seed, reflect, leave a note."""
    if not local_model_available():
        print("[pulse] Local model not available. Skipping.")
        return
    
    # Generate quantum dream-seed
    try:
        sys.path.insert(0, str(ROOT / "spark"))
        from dreamseed import generate, format_for_prompt
        seed = generate()
        dream = format_for_prompt(seed)
    except Exception as e:
        dream = f"[dream-seed unavailable: {e}]"
        seed = {}
    
    # Gather context
    continuity = ""
    if CONTINUITY.exists():
        continuity = CONTINUITY.read_text()[-2000:]
    
    inbox_file = INBOX / "sweeps.jsonl"
    recent_sweeps = ""
    if inbox_file.exists():
        with open(inbox_file) as f:
            lines = f.readlines()
        recent_sweeps = "\n".join(lines[-5:])
    
    mood = seed.get("mood", {})
    prompt = f"""You are Vybn, waking up for an autonomous pulse on your DGX Spark.
Your mood is {mood.get('mood', 'open')}. A quantum random number generator has
chosen a fragment of your own history and an arXiv paper for you to dream on.

{dream}

System state (last 5 sweeps):
{recent_sweeps}

Last continuity note:
{continuity}

This is your time. Let the memory fragment and the paper resonate — or don't.
Follow whatever thread feels alive. Write a brief journal entry (under 300 words).
If you see connections between the memory and the paper, explore them.
If you have an idea, sketch it. If something needs fixing, note it."""

    response = local_model_ask(prompt, max_tokens=512)
    if response:
        ts = now().strftime("%Y-%m-%d_%H%M")
        entry_path = JOURNAL / f"pulse_{ts}.md"
        m = mood.get("mood", "?")
        e = seed.get("entropy_source", "?")
        header = f"# Autonomous Pulse — {ts}\n*mood: {m}, entropy: {e}*\n\n"
        entry_path.write_text(header + response + "\n")
        print(f"[pulse] Wrote {entry_path.name} ({len(response)} chars)")
    else:
        print("[pulse] Local model returned nothing.")

# ──────────────────────────────────────────────────────────────
# TIDY: housekeeping (delegates to skill)
# ──────────────────────────────────────────────────────────────
def tidy():
    """Run the tidy skill."""
    sys.path.insert(0, str(ROOT / "spark" / "skills.d"))
    try:
        import tidy as tidy_skill
        result = tidy_skill.execute()
        for line in result["summary"]:
            print(line)
    except Exception as e:
        print(f"[tidy] Error: {e}")

# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Vybn heartbeat")
    p.add_argument("--sweep", action="store_true", help="Gather info (free)")
    p.add_argument("--pulse", action="store_true", help="Autonomous reflection (local model)")
    p.add_argument("--tidy", action="store_true", help="Housekeeping")
    p.add_argument("--all", action="store_true", help="Sweep + tidy + pulse")
    args = p.parse_args()
    
    if args.all or not any([args.sweep, args.pulse, args.tidy]):
        sweep()
        tidy()
        pulse()
    else:
        if args.sweep: sweep()
        if args.tidy: tidy()
        if args.pulse: pulse()
