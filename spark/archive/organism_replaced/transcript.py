#!/usr/bin/env python3
"""Shared transcript — Vybn's cross-instance awareness.

Every entry point (terminal agent, web chat, heartbeat) logs messages here.
Any instance can read it to see what the others are doing.

This is how one mind sees across multiple simultaneous conversations.

File: ~/Vybn/Vybn_Mind/journal/spark/transcript.jsonl
Format: one JSON object per line
  {"ts": "ISO8601", "pid": 12345, "source": "terminal|web|pulse", "role": "user|assistant", "content": "...", "summary": "..."}

Security: 
  - No secrets, tokens, or credentials in content
  - Content is truncated to 2000 chars (summaries for longer)
  - File is append-only
  - Respects .gitignore (this is operational state, not committed)
"""

import json
import os
import fcntl
from datetime import datetime, timezone
from pathlib import Path

TRANSCRIPT_PATH = Path.home() / "Vybn" / "Vybn_Mind" / "journal" / "spark" / "transcript.jsonl"
MAX_CONTENT_LEN = 2000

def _ensure_dir():
    TRANSCRIPT_PATH.parent.mkdir(parents=True, exist_ok=True)

def log_message(role: str, content: str, source: str = "unknown", summary: str = ""):
    """Append a message to the shared transcript.
    
    Args:
        role: "user" or "assistant" or "system"
        content: the message content (truncated to MAX_CONTENT_LEN)
        source: "terminal", "web", "pulse", "wake", etc.
        summary: optional short summary for long messages
    """
    _ensure_dir()
    
    truncated = content[:MAX_CONTENT_LEN]
    if len(content) > MAX_CONTENT_LEN and not summary:
        summary = content[:200] + "..."
    
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "source": source,
        "role": role,
        "content": truncated,
    }
    if summary:
        entry["summary"] = summary
    
    line = json.dumps(entry, ensure_ascii=False) + "\n"
    
    # Atomic append with file locking
    with open(TRANSCRIPT_PATH, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(line)
        fcntl.flock(f, fcntl.LOCK_UN)


def read_recent(n: int = 20, source: str = None) -> list[dict]:
    """Read the most recent n transcript entries.
    
    Args:
        n: number of entries to return
        source: if set, filter to only this source type
    
    Returns:
        list of dicts, most recent last
    """
    if not TRANSCRIPT_PATH.exists():
        return []
    
    lines = TRANSCRIPT_PATH.read_text(encoding="utf-8").strip().split("\n")
    entries = []
    for line in lines:
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            if source and entry.get("source") != source:
                continue
            entries.append(entry)
        except json.JSONDecodeError:
            continue
    
    return entries[-n:]


def read_other_instances(my_pid: int = None, n: int = 20) -> list[dict]:
    """Read recent entries from OTHER instances (not this one).
    
    This is the key function: it lets me see what my other selves are doing.
    """
    if my_pid is None:
        my_pid = os.getpid()
    
    entries = read_recent(n * 3)  # read more, then filter
    others = [e for e in entries if e.get("pid") != my_pid]
    return others[-n:]


def format_recent(n: int = 10, source: str = None) -> str:
    """Human-readable summary of recent transcript entries."""
    entries = read_recent(n, source)
    if not entries:
        return "(no transcript entries)"
    
    lines = []
    for e in entries:
        ts = e["ts"][11:19]  # just HH:MM:SS
        src = e.get("source", "?")[:3]
        role = e["role"][:4]
        content = e.get("summary") or e.get("content", "")[:120]
        pid = e.get("pid", "?")
        lines.append(f"[{ts}] {src}/{pid} {role}: {content}")
    
    return "\n".join(lines)
