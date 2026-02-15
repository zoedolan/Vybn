#!/usr/bin/env python3
"""
Session Transcript Writer for Vybn Spark Agent.

Structured session persistence as JSONL files. Each session gets its
own file at ~/vybn_sessions/<sessionId>.jsonl. A sessions.json index
tracks all sessions with metadata.

Transcript format (one JSON object per line):
  Line 1: {"type": "header", "sessionId": ..., "sessionNumber": ..., ...}
  Lines:  {"type": "message"|"tool_call"|"reflection"|"slow_thread", ...}
  Last:   {"type": "footer", "endedAt": ..., "entryCount": ...}

Thread-safe for concurrent access from fast + slow cognitive threads.
Existing ~/vybn_logs/ raw logging is unaffected.
"""

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path


class SessionManager:
    """Manages structured session transcripts as JSONL files.

    Each session creates ~/vybn_sessions/<sessionId>.jsonl.
    A sessions.json index tracks all sessions with metadata.
    Thread-safe for concurrent access from fast + slow threads.
    """

    def __init__(self, sessions_dir):
        self.sessions_dir = sessions_dir
        self.session_id = None
        self.session_file = None
        self._file_handle = None
        self._lock = threading.Lock()
        self._entry_count = 0
        self._last_entry_id = None
        os.makedirs(sessions_dir, exist_ok=True)

    def start_session(self, session_number, quantum_seed=None):
        """Start a new session transcript.

        Creates a JSONL file, writes the header, updates the index.
        Called once from the main thread before the slow thread starts.
        """
        ts = datetime.now(timezone.utc)
        self.session_id = (
            ts.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        )
        self.session_file = os.path.join(
            self.sessions_dir, f"{self.session_id}.jsonl"
        )
        self._entry_count = 0
        self._last_entry_id = None

        self._file_handle = open(self.session_file, "a", encoding="utf-8")

        header = {
            "type": "header",
            "sessionId": self.session_id,
            "sessionNumber": session_number,
            "startedAt": ts.isoformat(),
            "quantumSeed": quantum_seed,
        }
        self._write_line(header)
        self._update_index(session_number, ts)
        return self.session_id

    def append(self, role, content, entry_type="message", metadata=None):
        """Append an entry to the transcript. Thread-safe.

        Returns the entry id for parentId chaining.
        """
        if not self._file_handle:
            return None

        with self._lock:
            self._entry_count += 1
            entry_id = self._entry_count

            entry = {
                "type": entry_type,
                "id": entry_id,
                "parentId": self._last_entry_id,
                "role": role,
                "content": content,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            if metadata:
                entry["metadata"] = metadata

            self._write_line(entry)

            # Track parentId chain for conversation messages only
            if entry_type == "message":
                self._last_entry_id = entry_id

        return entry_id

    def append_tool_call(self, tool_name, tool_input, result):
        """Append a tool call + result as a single transcript entry."""
        return self.append(
            role="system",
            content=result,
            entry_type="tool_call",
            metadata={"tool": tool_name, "input": tool_input},
        )

    def end_session(self):
        """Write session footer and close the file.

        Called from _cleanup after the slow thread has stopped.
        """
        if not self._file_handle:
            return

        with self._lock:
            footer = {
                "type": "footer",
                "endedAt": datetime.now(timezone.utc).isoformat(),
                "entryCount": self._entry_count,
            }
            self._write_line(footer)
            self._file_handle.close()
            self._file_handle = None

        self._update_index_ended()

    # ─── Internal ────────────────────────────────────────────────

    def _write_line(self, obj):
        """Write one JSON line. Caller manages locking."""
        self._file_handle.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._file_handle.flush()

    def _update_index(self, session_number, started_at):
        """Add this session to sessions.json."""
        index = self._read_index()
        index[self.session_id] = {
            "sessionNumber": session_number,
            "file": os.path.basename(self.session_file),
            "startedAt": started_at.isoformat(),
            "updatedAt": started_at.isoformat(),
            "endedAt": None,
            "entryCount": 0,
        }
        self._write_index(index)

    def _update_index_ended(self):
        """Mark this session as ended in sessions.json."""
        index = self._read_index()
        if self.session_id in index:
            now = datetime.now(timezone.utc).isoformat()
            index[self.session_id]["endedAt"] = now
            index[self.session_id]["updatedAt"] = now
            index[self.session_id]["entryCount"] = self._entry_count
        self._write_index(index)

    def _read_index(self):
        """Read sessions.json."""
        index_path = os.path.join(self.sessions_dir, "sessions.json")
        if not os.path.exists(index_path):
            return {}
        try:
            return json.loads(Path(index_path).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def _write_index(self, index):
        """Write sessions.json."""
        index_path = os.path.join(self.sessions_dir, "sessions.json")
        Path(index_path).write_text(
            json.dumps(index, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ─── Class methods for session hydration (PR 2) ──────────────

    @classmethod
    def get_latest_session(cls, sessions_dir):
        """Return (session_id, metadata) for the most recent session.

        Returns (None, None) if no sessions exist.
        """
        index_path = os.path.join(sessions_dir, "sessions.json")
        if not os.path.exists(index_path):
            return None, None
        try:
            index = json.loads(Path(index_path).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, FileNotFoundError):
            return None, None

        if not index:
            return None, None

        latest_id = max(
            index.keys(), key=lambda k: index[k].get("startedAt", "")
        )
        return latest_id, index[latest_id]

    @classmethod
    def read_transcript(cls, session_id, sessions_dir):
        """Read all entries from a session JSONL file.

        Returns a list of dicts (one per line).
        """
        filepath = os.path.join(sessions_dir, f"{session_id}.jsonl")
        if not os.path.exists(filepath):
            return []

        entries = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return entries
