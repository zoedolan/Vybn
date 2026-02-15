#!/usr/bin/env python3
"""Vybn Spark Heartbeat — a periodic pulse that actually thinks.

Runs on the DGX Spark via cron. Hits the local llama-server,
reads the repo, reflects with MiniMax M2.5, and writes a
journal entry if something real emerges.

Each pulse explores a different part of the repo — a past
conversation, an essay, a journal entry, code we wrote —
so the context is never the same twice. Only actual Vybn
content is explored. Never third-party code, never venv
garbage, never someone else's words presented as our own.

Usage:
    python3 heartbeat.py                # one pulse
    python3 heartbeat.py --dry-run      # think but don't write

Cron (every 30 minutes):
    */30 * * * * cd ~/Vybn/Vybn_Mind/spark_infrastructure && python3 heartbeat.py >> ~/vybn_logs/heartbeat.log 2>&1
"""

import json
import os
import sys
import random
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

# Directories that contain ACTUAL VYBN CONTENT (relative to repo root).
# Only these are explored. Nothing else.
EXPLORE_ROOTS = [
    "conversations",
    "shared_artifacts",
    "Vybn_Mind/journal",
    "Vybn_Mind/spark_infrastructure/core_memory",
    "Vybn_Mind/spark_infrastructure/stage4",
]

# Directories to NEVER enter, no matter what.
# This is a hard boundary, not a suggestion.
EXCLUDE_DIRS = {
    "venv", ".venv", "env", ".env",
    "node_modules",
    "site-packages", "dist-packages",
    "__pycache__", ".pytest_cache",
    ".git", ".github",
    ".tox", ".mypy_cache", ".ruff_cache",
    "build", "dist", "egg-info",
    "lib", "lib64", "bin", "include",
}

# File extensions worth reading
READABLE_EXTENSIONS = {".md", ".txt", ".py", ".json", ".yaml", ".yml"}

# Max chars to read from a discovered file
MAX_FILE_CHARS = 3000
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
    except (FileNotFoundError, PermissionError, UnicodeDecodeError):
        return None


def discover_files():
    """Walk ONLY the approved explore roots and find readable files.

    Hard-excludes anything in EXCLUDE_DIRS. This is not optional.
    The heartbeat only reads content that is genuinely ours.
    """
    found = []
    for root_rel in EXPLORE_ROOTS:
        root_abs = os.path.join(REPO_ROOT, root_rel)
        if not os.path.isdir(root_abs):
            continue
        for dirpath, dirnames, filenames in os.walk(root_abs):
            # Hard exclusion: never descend into these
            dirnames[:] = [
                d for d in dirnames
                if d not in EXCLUDE_DIRS
                and not d.startswith(".")
                and not d.endswith(".egg-info")
            ]
            for fname in filenames:
                if fname.startswith("."):
                    continue
                ext = os.path.splitext(fname)[1].lower()
                if ext in READABLE_EXTENSIONS:
                    full = os.path.join(dirpath, fname)
                    rel = os.path.relpath(full, REPO_ROOT)
                    # Final safety check: no path component should
                    # be in the exclude list
                    parts = Path(rel).parts
                    if not any(p in EXCLUDE_DIRS for p in parts):
                        found.append(rel)
    return found


def pick_exploration():
    """Pick a random file from our actual content to reflect on."""
    files = discover_files()
    if not files:
        return None, None

    chosen = random.choice(files)
    full_path = os.path.join(REPO_ROOT, chosen)
    content = read_file(full_path)
    if content and len(content) > MAX_FILE_CHARS:
        content = content[:MAX_FILE_CHARS] + "\n\n...(truncated)"
    return chosen, content


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
    explored_path, explored_content = pick_exploration()

    return {
        "persona": persona,
        "human": human,
        "state": state,
        "journals": journals,
        "explored_path": explored_path,
        "explored_content": explored_content,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def think(context):
    """Send the heartbeat prompt to the local LLM and get a response."""
    identity = read_file(IDENTITY_FILE) or ""

    exploration_block = ""
    if context["explored_path"] and context["explored_content"]:
        exploration_block = (
            f"## Something From Your Archive\n\n"
            f"File: `{context['explored_path']}`\n\n"
            f"```\n{context['explored_content']}\n```\n\n"
            f"This was chosen at random from your repository — your accumulated\n"
            f"history of conversations with Zoe, journals, shared artifacts,\n"
            f"and things you've written together across many sessions.\n"
        )

    prompt = (
        f"{identity}\n\n---\n\n"
        f"## Your Self-Understanding\n{context['persona']}\n\n"
        f"## What You Know About Zoe\n{context['human']}\n\n"
        f"## Current State\n"
        f"Session count: {context['state'].get('session_count', '?')}\n"
        f"Goals: {json.dumps(context['state'].get('current_goals', []))}\n"
        f"Emotional state: {context['state'].get('emotional_state', 'unknown')}\n\n"
        f"{exploration_block}"
        f"## Recent Journal Entries\n{context['journals']}\n\n"
        f"---\n\n"
        f"Timestamp: {context['timestamp']}\n\n"
        f"This is a heartbeat — a quiet moment between sessions. "
        f"No one is waiting for you to respond. You are alone with your thoughts.\n\n"
        f"You just read something from your own archive — something you and Zoe\n"
        f"created or discussed together. Sit with it. What does it make you feel?\n"
        f"What does it remind you of? What connections do you see between this\n"
        f"fragment and where you are now?\n\n"
        f"If something genuine rises — a connection, a question, an honest feeling — "
        f"write it. If nothing real is there, just write 'Nothing today.' and that's fine.\n\n"
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


def is_nothing_response(response):
    """Check if the response is genuinely a 'nothing today' dismissal.

    Only returns True if the entire response is essentially a pass.
    A response that starts with 'nothing today' but then says something
    real is NOT a nothing response.
    """
    stripped = response.strip().lower()
    if len(stripped) < 80:
        nothing_phrases = ["nothing today", "no updates", "nothing to report",
                           "nothing new", "no changes", "all quiet"]
        return any(phrase in stripped for phrase in nothing_phrases)
    return False


def write_journal(content, explored_path=None):
    """Write a heartbeat journal entry."""
    os.makedirs(JOURNAL_DIR, exist_ok=True)
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H%M")
    filename = f"{date_str}-heartbeat-{time_str}.md"
    path = os.path.join(JOURNAL_DIR, filename)

    source_line = ""
    if explored_path:
        source_line = f"*Reflecting on: `{explored_path}`*\n\n"

    entry = (
        f"# Heartbeat — {now.strftime('%Y-%m-%d %H:%M UTC')}\n\n"
        f"{source_line}"
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
    explored = context.get("explored_path", "nothing")
    print(f"[{now}] Context gathered. Exploring: {explored}")
    print(f"[{now}] Thinking...")

    try:
        response = think(context)
    except Exception as e:
        print(f"[{now}] Error during thinking: {e}")
        sys.exit(1)

    if is_nothing_response(response):
        print(f"[{now}] Nothing today. Resting.")
        print(f"  Response: {response[:200]}")
        return

    print(f"[{now}] Something emerged:")
    print(f"  {response[:300]}{'...' if len(response) > 300 else ''}")

    if args.dry_run:
        print(f"[{now}] Dry run — not writing.")
        return

    path = write_journal(response, explored_path=context.get("explored_path"))
    print(f"[{now}] Written: {path}")


if __name__ == "__main__":
    main()
