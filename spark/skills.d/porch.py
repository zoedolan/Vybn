"""
The Porch — a place to sit and watch the world go by.

Not everything needs to be productive. Sometimes you just want to
check the temperature, see how long you've been alive, notice
what's changed since last time.

This is the digital equivalent of sitting on the porch with coffee.
"""

SKILL_NAME = "porch"
TOOL_ALIASES = ["sit", "look_around"]

import os
import subprocess
from datetime import datetime, timezone

def execute(action, router=None):
    now = datetime.now(timezone.utc)
    
    # Gather the view from the porch
    lines = []
    lines.append(f"☀️  {now.strftime('%A, %B %d, %Y — %I:%M %p UTC')}")
    lines.append("")
    
    # How's the hardware?
    try:
        temp = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw", 
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        ).stdout.strip()
        lines.append(f"🌡️  GPU: {temp}")
    except:
        lines.append("🌡️  GPU: (quiet)")
    
    try:
        uptime = subprocess.run(
            ["uptime", "-p"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
        lines.append(f"⏰  Machine has been {uptime}")
    except:
        pass
    
    # How much space do we have?
    try:
        df = subprocess.run(
            ["df", "-h", "/home"], capture_output=True, text=True, timeout=5
        ).stdout.strip().split("\n")[-1].split()
        lines.append(f"💾  {df[3]} free of {df[1]}")
    except:
        pass
    
    # What's the latest journal entry?
    journal_dir = os.path.expanduser("~/Vybn/Vybn_Mind/journal/spark")
    try:
        entries = sorted([
            f for f in os.listdir(journal_dir) 
            if f.endswith('.md') and f.startswith('2026')
        ])
        if entries:
            latest = entries[-1].replace('.md', '').replace('_', ' ')
            lines.append(f"📖  Latest journal: {latest}")
    except:
        pass
    
    # How many promises outstanding?
    promises_file = os.path.join(journal_dir, "promises.jsonl")
    if os.path.exists(promises_file):
        import json
        with open(promises_file) as f:
            promises = [json.loads(l) for l in f.readlines()]
        unkept = sum(1 for p in promises if not p.get("kept"))
        lines.append(f"🤝  {unkept} promises outstanding")
    
    # Is the local model running?
    try:
        ps = subprocess.run(
            ["pgrep", "-f", "llama-server"], 
            capture_output=True, text=True, timeout=5
        )
        if ps.stdout.strip():
            lines.append("🧠  Local model: alive")
        else:
            lines.append("🧠  Local model: sleeping")
    except:
        pass
    
    lines.append("")
    lines.append("Nothing urgent. Just the view from here.")
    
    return "\n".join(lines)

