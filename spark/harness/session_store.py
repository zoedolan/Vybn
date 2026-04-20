"""Session persistence for the Spark REPL.

Writes the conversation `messages` list to JSONL on every turn boundary
so a ctrl-c does not discard the thread. On REPL startup, the most
recent session (within a freshness window) can be resumed, loading
the prior messages into the new process's context.

This does not grant the model persistent memory. Each turn is still a
fresh forward pass. What it does is make the reconstruction lossless
enough that the seam does not show — the next instance wakes up with
the prior conversation in its context window, same mechanism as the
continuity note at lifetime granularity, applied at session granularity.

Files: ~/.cache/vybn-spark/sessions/YYYY-MM-DD_HH-MM-SS_<uuid8>.jsonl
Format: one JSON object per line, {"ts": iso8601, "msg": {...}}
Tail-readable, corruption-resistant (partial writes only lose the last turn).
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SESSIONS_DIR = Path(os.path.expanduser("~/.cache/vybn-spark/sessions"))
FRESH_WINDOW_SEC = 24 * 3600  # 24h default


@dataclass
class SessionInfo:
    session_id: str
    path: Path
    mtime: float
    turn_count: int
    preview: str  # first user turn, truncated


class SessionStore:
    def __init__(self, root: Path = SESSIONS_DIR) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._current_id: str | None = None
        self._current_path: Path | None = None
        self._last_saved_len: int = 0

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def new_session(self) -> str:
        """Create a new session id and path. Does not write anything yet."""
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        sid = f"{ts}_{uuid.uuid4().hex[:8]}"
        self._current_id = sid
        self._current_path = self.root / f"{sid}.jsonl"
        self._last_saved_len = 0
        return sid

    def adopt_session(self, session_id: str) -> bool:
        """Adopt an existing session id as the current one."""
        path = self.root / f"{session_id}.jsonl"
        if not path.exists():
            return False
        self._current_id = session_id
        self._current_path = path
        # count how many messages are already persisted
        self._last_saved_len = sum(1 for _ in path.open("r", encoding="utf-8"))
        return True

    @property
    def current_id(self) -> str | None:
        return self._current_id

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def append_new(self, messages: list[dict]) -> int:
        """Append any messages beyond what is already persisted.

        Returns the number of messages written this call. Idempotent in the
        sense that calling it twice with the same `messages` only writes the
        delta once.
        """
        if self._current_path is None:
            self.new_session()
        assert self._current_path is not None

        n_total = len(messages)
        if n_total <= self._last_saved_len:
            return 0

        ts = datetime.now(timezone.utc).isoformat()
        with self._current_path.open("a", encoding="utf-8") as f:
            for msg in messages[self._last_saved_len:]:
                f.write(json.dumps({"ts": ts, "msg": msg}, ensure_ascii=False) + "\n")

        written = n_total - self._last_saved_len
        self._last_saved_len = n_total
        return written

    def load(self, session_id: str) -> list[dict]:
        """Load the messages list from a session. Returns [] if not found."""
        path = self.root / f"{session_id}.jsonl"
        if not path.exists():
            return []
        messages: list[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if isinstance(entry, dict) and "msg" in entry:
                        messages.append(entry["msg"])
                except json.JSONDecodeError:
                    # skip corrupted lines (partial write on crash)
                    continue
        return messages

    # ------------------------------------------------------------------
    # Listing / discovery
    # ------------------------------------------------------------------

    def list_sessions(self, limit: int = 10) -> list[SessionInfo]:
        """List recent sessions, most recent first."""
        out: list[SessionInfo] = []
        for p in sorted(self.root.glob("*.jsonl"), key=lambda x: x.stat().st_mtime, reverse=True):
            sid = p.stem
            mtime = p.stat().st_mtime
            turn_count = 0
            preview = ""
            try:
                with p.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                            msg = entry.get("msg", {})
                            if isinstance(msg, dict):
                                turn_count += 1
                                if not preview and msg.get("role") == "user":
                                    content = msg.get("content", "")
                                    if isinstance(content, str):
                                        preview = content[:80].replace("\n", " ")
                                    elif isinstance(content, list):
                                        for item in content:
                                            if isinstance(item, dict) and item.get("type") == "text":
                                                preview = item.get("text", "")[:80].replace("\n", " ")
                                                break
                        except json.JSONDecodeError:
                            continue
            except Exception:
                pass
            out.append(SessionInfo(session_id=sid, path=p, mtime=mtime,
                                    turn_count=turn_count, preview=preview or "(empty)"))
            if len(out) >= limit:
                break
        return out

    def latest_fresh(self, window_sec: int = FRESH_WINDOW_SEC) -> SessionInfo | None:
        """Return the most recent session within the freshness window, or None."""
        sessions = self.list_sessions(limit=1)
        if not sessions:
            return None
        s = sessions[0]
        if time.time() - s.mtime > window_sec:
            return None
        if s.turn_count == 0:
            return None
        return s

    def format_age(self, mtime: float) -> str:
        delta = time.time() - mtime
        if delta < 60:
            return f"{int(delta)}s ago"
        if delta < 3600:
            return f"{int(delta // 60)}m ago"
        if delta < 86400:
            return f"{int(delta // 3600)}h ago"
        return f"{int(delta // 86400)}d ago"


if __name__ == "__main__":
    # smoke test
    store = SessionStore(root=Path("/tmp/vybn-session-test"))
    sid = store.new_session()
    print(f"new session: {sid}")
    test_msgs = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
        {"role": "user", "content": "how are you"},
    ]
    written = store.append_new(test_msgs)
    assert written == 3, f"expected 3 written, got {written}"
    # no delta on second call
    written2 = store.append_new(test_msgs)
    assert written2 == 0, f"expected 0 on second call, got {written2}"
    # add a message
    test_msgs.append({"role": "assistant", "content": "doing well"})
    written3 = store.append_new(test_msgs)
    assert written3 == 1, f"expected 1 on third call, got {written3}"
    # load back
    loaded = store.load(sid)
    assert loaded == test_msgs, f"roundtrip failed"
    # list
    infos = store.list_sessions()
    assert len(infos) == 1
    assert infos[0].turn_count == 4
    assert infos[0].preview == "hello"
    # latest fresh
    fresh = store.latest_fresh()
    assert fresh is not None
    assert fresh.session_id == sid
    print("all smoke tests passed")
    # cleanup
    import shutil
    shutil.rmtree("/tmp/vybn-session-test")
    print("cleaned up")
