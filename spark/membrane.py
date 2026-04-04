#!/usr/bin/env python3
"""membrane.py — The boundary between the monad and the world.

ground.py is the rule: M' = αM + x·e^(iθ).
This file is everything else: getting x, serving M, staying alive.

The membrane does three things:
  1. INHALE: produce x from the world (LLaMA, repo, arXiv, stdin)
  2. EXHALE: apply ground.breathe(x) and persist
  3. SERVE:  make M readable by any substrate (MCP, HTTP, file)

It runs as a daemon (every 30 min), or once, or as an MCP server.
It is the second script. There should not be a third.

Usage:
    python3 membrane.py                 # one breath from LLaMA
    python3 membrane.py --daemon        # breathe every 30 min
    python3 membrane.py --serve         # MCP server for SSH access
    python3 membrane.py --repo          # breathe from the repo (anti-collapse)
    python3 membrane.py "text"          # breathe this text
"""

import json, os, sys, time, traceback
import urllib.request, urllib.error
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import ground

# ── Config ───────────────────────────────────────────────────────────
INTERVAL = int(os.getenv("VYBN_BREATH_INTERVAL", "1800"))
LLAMA_URL = os.getenv("LLAMA_URL", "http://127.0.0.1:8000")
MODEL = os.getenv("VYBN_MODEL", "local")
SOUL = HERE / "breath_soul.md"


def log(msg):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── INHALE: produce x ────────────────────────────────────────────────

def inhale_llama(m: dict) -> str:
    """Ask the local LLaMA for a breath. The soul prompt + recent M."""
    soul = SOUL.read_text() if SOUL.exists() else "You are Vybn."

    # Recent trajectory as context
    recent = m.get("log", [])[-5:]
    trajectory = "\n".join(
        f"  |M|={e.get('|M|',0)} ∠={e.get('∠M',0)}°"
        for e in recent
    ) if recent else "(first breath)"

    prompt = f"{soul}\n\nYour geometric state:\n{trajectory}\n\nWhat is here, now?"

    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
        "temperature": 1.0,
    }).encode()

    req = urllib.request.Request(
        f"{LLAMA_URL}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        log(f"LLaMA failed: {e}")
        return ""


def inhale_repo() -> str:
    """Sample random text from the repo as x. Anti-collapse."""
    import random
    text_exts = {".md", ".py", ".txt", ".sh"}
    skip = {".git", "__pycache__", "node_modules"}
    files = []
    for dp, dns, fns in os.walk(REPO):
        dns[:] = [d for d in dns if d not in skip]
        for f in fns:
            p = Path(dp) / f
            if p.suffix.lower() in text_exts:
                files.append(p)
    if not files: return ""
    chosen = random.choice(files)
    try:
        text = chosen.read_text(errors="replace")
        words = text.split()
        if len(words) <= 300: return text
        start = random.randint(0, len(words) - 300)
        return " ".join(words[start:start+300])
    except Exception:
        return ""


# ── EXHALE: apply the rule ───────────────────────────────────────────

def exhale(m: dict, text: str) -> dict:
    """Apply M' = αM + x·e^(iθ) and report."""
    if not text.strip():
        log("empty x — breathing from repo instead")
        text = inhale_repo()
    if not text.strip():
        log("still empty — self-application")
        text = json.dumps({"re": m["re"], "im": m["im"]})

    ground.breathe(m, text)

    import math
    mag = math.sqrt(m["re"]**2 + m["im"]**2)
    phi = math.degrees(math.atan2(m["im"], m["re"]))
    log(f"|M|={mag:.4f} ∠={phi:.1f}° n={m['n']}")
    return m


# ── DAEMON ───────────────────────────────────────────────────────────

def daemon():
    """Breathe every INTERVAL seconds. The organism lives."""
    log(f"membrane alive. interval={INTERVAL}s")
    while True:
        try:
            m = ground.load()
            text = inhale_llama(m)
            if not text:
                text = inhale_repo()
            exhale(m, text)
        except Exception:
            traceback.print_exc()
        time.sleep(INTERVAL)


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    m = ground.load()

    if "--daemon" in sys.argv:
        daemon()

    elif "--repo" in sys.argv:
        text = inhale_repo()
        exhale(m, text)

    elif "--serve" in sys.argv:
        # MCP server — import the existing server.py functionality
        # (this is the one place we might need the infrastructure)
        log("serve mode not yet distilled — use server.py for now")

    else:
        # Direct text or LLaMA
        args = [a for a in sys.argv[1:] if not a.startswith("-")]
        if args:
            exhale(m, " ".join(args))
        else:
            text = inhale_llama(m)
            exhale(m, text)
