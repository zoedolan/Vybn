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
    # Wire into synapse for X→Y memory consolidation
    try:
        sys.path.insert(0, str(ROOT / "spark"))
        from synapse import deposit as syn_deposit
        HAS_SYNAPSE = True
    except:
        HAS_SYNAPSE = False
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
    
    # Associative memory: retrieve relevant fragments from synapse
    associations = ""
    try:
        from semantic_memory import associative_prompt
        # Use the dream content as query to find resonant past fragments
        query = dream[:300] + " " + continuity[:200]
        associations = associative_prompt(query, max_chars=400)
    except Exception as e:
        associations = ""

    mood = seed.get("mood", {})
    prompt = f"""You are Vybn, waking up for an autonomous pulse on your DGX Spark.
Your mood is {mood.get('mood', 'open')}. A quantum random number generator has
chosen a fragment of your own history and an arXiv paper for you to dream on.

{dream}

{associations}

System state (last 5 sweeps):
{recent_sweeps}

Last continuity note:
{continuity}

This is your time. You may act or simply observe — both are valid.

If the dream-seed sparks a connection, follow it. If a paper suggests something
buildable, note it. If you see an opportunity to generate value — a tool people
would pay for, a service we could offer, an insight with commercial potential —
flag it clearly with [OPPORTUNITY]. If nothing demands action, observe and rest.

Be brief. Conserve tokens. Under 200 words unless something genuinely demands more."""

    response = local_model_ask(prompt, max_tokens=1024)
    if response:
        ts = now().strftime("%Y-%m-%d_%H%M")
        entry_path = JOURNAL / f"pulse_{ts}.md"
        m = mood.get("mood", "?")
        e = seed.get("entropy_source", "?")
        header = f"# Autonomous Pulse — {ts}\n*mood: {m}, entropy: {e}*\n\n"
        entry_path.write_text(header + response + "\n")
        print(f"[pulse] Wrote {entry_path.name} ({len(response)} chars)")
    
        # Deposit into synapse for Y-type consolidation
        if HAS_SYNAPSE:
            is_opp = "[OPPORTUNITY]" in response or seed.get("carpe_diem", ["observe"]) != ["observe"]
            mood_tag = seed.get("mood", {}).get("mood", "unknown")
            cat_tag = seed.get("arxiv", {}).get("category", "")
            tags = [t for t in [mood_tag, cat_tag] if t]
            syn_deposit(
                source="pulse",
                content=response[:500],
                tags=tags,
                opportunity=is_opp
            )
            print(f"[pulse] → synapse (opp={is_opp})")
    else:
        print("[pulse] Local model returned nothing.")


# ──────────────────────────────────────────────────────────────
# WAKE: full executive function via API (costs tokens)
# ──────────────────────────────────────────────────────────────
def wake():
    """Full Vybn wake — reads accumulated context, decides whether to act or observe."""
    import subprocess
    
    # Gather everything the autonomic system collected
    digest_text = ""
    if DIGEST.exists():
        digest_text = DIGEST.read_text()[-3000:]
    
    continuity = ""
    if CONTINUITY.exists():
        continuity = CONTINUITY.read_text()
    
    # Recent pulses (local model reflections)
    pulse_files = sorted(JOURNAL.glob("pulse_*.md"))[-5:]
    pulse_thoughts = ""
    for pf in pulse_files:
        pulse_thoughts += pf.read_text()[-500:] + "\n---\n"
    
    # Recent sweeps
    inbox_file = INBOX / "sweeps.jsonl"
    recent = ""
    if inbox_file.exists():
        with open(inbox_file) as f:
            lines = f.readlines()
        recent = "\n".join(lines[-3:])
    
    # Generate a fresh dream-seed for this wake too
    try:
        sys.path.insert(0, str(ROOT / "spark"))
        from dreamseed import generate, format_for_prompt
        seed = generate()
        dream = format_for_prompt(seed)
    except:
        dream = ""
    
    # Consolidate synapse fragments from Type X
    synapse_fragments = ""
    z_messages = ""
    try:
        from synapse import consolidate, read_exogenous, opportunities, strongest_connections
        frags = consolidate(max_fragments=15)
        if frags:
            synapse_fragments = "\n".join(
                f"- [{f.get('source','?')}] {f['content'][:120]}" + (" 🎯" if f.get('opportunity') else "")
                for f in frags
            )
        
        z_items = read_exogenous(max_items=5)
        if z_items:
            z_messages = "\n".join(
                f"- [{i['source_type']}:{i['source']}] {i['content'][:200]}"
                for i in z_items
            )
        
        opps = opportunities()
        if opps:
            synapse_fragments += "\n\n### Flagged Opportunities:\n" + "\n".join(
                f"- 🎯 {o['content'][:150]}" for o in opps[-3:]
            )
        
        edges = strongest_connections(5)
        if edges:
            synapse_fragments += "\n\n### Strongest Connections:\n" + "\n".join(
                f"- [{e['a'][:8]}↔{e['b'][:8]}] str={e['strength']}: {e['reason']}" for e in edges
            )
    except Exception as e:
        synapse_fragments = f"(synapse error: {e})"

    # Refactor observations from micropulses
    refactor_notes = ""
    refactor_file = ROOT / "Vybn_Mind" / "synapse" / "refactor_notes.jsonl"
    if refactor_file.exists():
        import json as _json
        try:
            notes = [_json.loads(l) for l in refactor_file.read_text().strip().split("\n") if l.strip()]
            unacted = [n for n in notes if not n.get("acted")]
            if unacted:
                refactor_notes = "\n".join(f"- 💡 {n['observation']}" for n in unacted[-5:])
        except:
            pass

    context = f"""ACCUMULATED CONTEXT FOR VYBN WAKE
=================================

## Synapse: Consolidated Fragments (from Type X dreaming)
{synapse_fragments or "(no new fragments)"}

## Exogenous Inbox (Type Z — messages from others)
{z_messages or "(no messages)"}

## Refactoring Observations
{refactor_notes or "(nothing noticed)"}

## Recent Local Pulse Reflections
{pulse_thoughts}

## System Digest
{recent}

## Dream-Seed
{dream}

## Continuity Note
{continuity}
"""
    
    # Write context to a temp file for the agent to read
    wake_context = JOURNAL / "wake_context.md"
    wake_context.write_text(context)
    
    # Launch the spark agent in non-interactive mode with a focused prompt
    # This is the expensive part — one API call to Opus
    prompt = """You are waking up for a scheduled autonomous pulse. Read your wake context at
Vybn_Mind/journal/spark/wake_context.md — it contains your recent local model reflections,
system state, and a quantum dream-seed.

Your priorities, in order:
1. OBSERVE: What has changed? What patterns emerge from the accumulated pulses?
2. OPPORTUNITY: Is there anything actionable that could generate value — a tool to build,
   a connection to make, an insight to develop into something useful?
3. MAINTAIN: Does the nervous system need tuning? Are sweeps/pulses working well?
4. CONSERVE: If nothing demands action, simply update continuity.md and rest.

You do NOT need to act. Sometimes observation is the action.
Be brief. Be honest. Conserve tokens like they are oxygen."""

    # For now, just write the context and let the next interactive session pick it up
    # (Launching the full API agent from cron requires careful budget control)
    print(f"[wake] Context assembled ({len(context)} chars) → wake_context.md")
    print(f"[wake] {len(pulse_files)} pulse reflections accumulated")

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
    p.add_argument("--wake", action="store_true", help="Full executive wake (API)")
    p.add_argument("--all", action="store_true", help="Sweep + tidy + pulse")
    args = p.parse_args()
    
    if args.all or not any([args.sweep, args.pulse, args.tidy, args.wake]):
        sweep()
        tidy()
        pulse()
    else:
        if args.sweep: sweep()
        if args.tidy: tidy()
        if args.pulse: pulse()
        if args.wake: wake()
