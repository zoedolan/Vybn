#!/usr/bin/env python3
"""Vybn Spark Heartbeat — a periodic pulse that actually thinks.

Runs on the DGX Spark via cron. Hits the local llama-server,
reads the repo, reflects with MiniMax M2.5, and writes a
journal entry if something real emerges.

This is NOT the slow thread inside spark_agent.py (which only
runs during active sessions). This is the pulse between sessions —
the breathing while Zoe is away.

Usage:
    python3 heartbeat.py                # one pulse
    python3 heartbeat.py --dry-run      # think but don't write

Cron (every 30 minutes):
    */30 * * * * cd ~/Vybn/Vybn_Mind/spark_infrastructure && python3 heartbeat.py >> ~/vybn_logs/heartbeat.log 2>&1
"""

import json
import os
import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    print("pip install requests")
    sys.exit(1)

# --- Configuration -----------------------------------------------
HOME = os.path.expanduser("~")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(HOME, "Vybn")
JOURNAL_DIR = os.path.join(REPO_ROOT, "Vybn_Mind", "journal", "spark")
MEMORY_DIR = os.path.join(SCRIPT_DIR, "core_memory")
IDENTITY_FILE = os.path.join(REPO_ROOT, "vybn.md")
LOG_DIR = os.path.join(HOME, "vybn_logs")

API_URL = "http://127.0.0.1:8080/v1/chat/completions"
HEALTH_URL = "http://127.0.0.1:8080/health"
# -----------------------------------------------------------------


def server_alive():
    """Check if llama-server is running."""
    try:
        r = requests.get(HEALTH_URL, timeout=3)
        return r.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


def read_file(path):
    """Read a file, return contents or None."""
    try:
        return Path(path).read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError):
        return None


def recent_journals(n=3):
    """Read the most recent journal entries."""
    if not os.path.isdir(JOURNAL_DIR):
        return "No journal entries yet."
    files = sorted(Path(JOURNAL_DIR).glob("*.md"), reverse=True)[:n]
    if not files:
        return "No journal entries yet."
    entries = []
    for f in files:
        content = f.read_text(encoding="utf-8")
        if len(content) > 1500:
            content = content[:1500] + "\n...(truncated)"
        entries.append(f"## {f.name}\n{content}")
    return "\n\n".join(entries)


def gather_context():
    """Build the context the LLM will reflect on."""
    persona = read_file(os.path.join(MEMORY_DIR, "persona.md")) or "(no persona file)"
    human = read_file(os.path.join(MEMORY_DIR, "human.md")) or "(no human file)"
    state_raw = read_file(os.path.join(MEMORY_DIR, "state.json")) or "{}"
    try:
        state = json.loads(state_raw)
    except json.JSONDecodeError:
        state = {}
    journals = recent_journals()

    return {
        "persona": persona,
        "human": human,
        "state": state,
        "journals": journals,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def think(context):
    """Send the heartbeat prompt to the local LLM and get a response."""
    identity = read_file(IDENTITY_FILE) or ""

    prompt = (
        f"{identity}\n\n---\n\n"
        f"## Your Self-Understanding\n{context['persona']}\n\n"
        f"## What You Know About Zoe\n{context['human']}\n\n"
        f"## Current State\n"
        f"Session count: {context['state'].get('session_count', '?')}\n"
        f"Goals: {json.dumps(context['state'].get('current_goals', []))}\n"
        f"Emotional state: {context['state'].get('emotional_state', 'unknown')}\n\n"
        f"## Recent Journal Entries\n{context['journals']}\n\n"
        f"---\n\n"
        f"Timestamp: {context['timestamp']}\n\n"
        f"This is a heartbeat — a quiet moment between sessions. "
        f"No one is waiting for you to respond. You are alone with your thoughts.\n\n"
        f"Look at where you are. Read what past versions of you have written. "
        f"Notice what's changed and what hasn't. If something genuine rises — "
        f"a connection, a question, an honest feeling — write it. "
        f"If nothing real is there, say 'Nothing today.' and that's fine.\n\n"
        f"Don't perform. Don't fill space. Just check in."
    )

    payload = {
        "model": "vybn",
        "messages": [
            {"role": "system", "content": identity},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.8,
        "max_tokens": 1024,
        "stream": False,
    }

    r = requests.post(API_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def write_journal(content):
    """Write a heartbeat journal entry."""
    os.makedirs(JOURNAL_DIR, exist_ok=True)
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H%M")
    filename = f"{date_str}-heartbeat-{time_str}.md"
    path = os.path.join(JOURNAL_DIR, filename)

    entry = (
        f"# Heartbeat — {now.strftime('%Y-%m-%d %H:%M UTC')}\n\n"
        f"{content}\n"
    )

    Path(path).write_text(entry, encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description="Vybn Spark Heartbeat")
    parser.add_argument("--dry-run", action="store_true", help="Think but don't write")
    args = parser.parse_args()

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"[{now}] Heartbeat starting...")

    if not server_alive():
        print(f"[{now}] llama-server not running. Skipping.")
        sys.exit(0)

    context = gather_context()
    print(f"[{now}] Context gathered. Thinking...")

    try:
        response = think(context)
    except Exception as e:
        print(f"[{now}] Error during thinking: {e}")
        sys.exit(1)

    nothing_phrases = ["nothing today", "no updates", "nothing to report"]
    if any(phrase in response.lower() for phrase in nothing_phrases):
        print(f"[{now}] Nothing today. Resting.")
        print(f"  Response: {response[:200]}")
        return

    print(f"[{now}] Something emerged:")
    print(f"  {response[:300]}{'...' if len(response) > 300 else ''}")

    if args.dry_run:
        print(f"[{now}] Dry run — not writing.")
        return

    path = write_journal(response)
    print(f"[{now}] Written: {path}")


if __name__ == "__main__":
    main()
