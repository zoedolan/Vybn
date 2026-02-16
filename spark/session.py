#!/usr/bin/env python3
"""Session persistence — JSONL storage for conversation continuity."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


class SessionManager:
    def __init__(self, config: dict):
        self.session_dir = Path(config["paths"]["session_dir"]).expanduser()
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.auto_resume = config.get("session", {}).get("auto_resume", True)
        self.resume_window = config.get("session", {}).get("resume_window_seconds", 7200)
        self.session_id = None
        self.session_file = None
        self._handle = None

    def load_or_create(self) -> list:
        if self.auto_resume:
            latest = self._find_latest_session()
            if latest:
                self.session_id = latest.stem
                self.session_file = latest
                return self._load_messages(latest)

        self.session_id = self._make_id()
        self.session_file = self.session_dir / f"{self.session_id}.jsonl"
        return []

    def _make_id(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6]

    def _find_latest_session(self) -> Path | None:
        sessions = sorted(
            self.session_dir.glob("*.jsonl"),
            key=os.path.getmtime,
            reverse=True,
        )
        if sessions:
            age = datetime.now().timestamp() - sessions[0].stat().st_mtime
            if age < self.resume_window:
                return sessions[0]
        return None

    def _load_messages(self, path: Path) -> list:
        messages = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    messages.append({
                        "role": entry["role"],
                        "content": entry["content"],
                    })
        return messages

    def save_turn(self, user_input: str, assistant_response: str):
        if not self._handle:
            self._handle = open(self.session_file, "a")

        ts = datetime.now(timezone.utc).isoformat()

        self._handle.write(json.dumps({
            "role": "user",
            "content": user_input,
            "timestamp": ts,
        }) + "\n")

        self._handle.write(json.dumps({
            "role": "assistant",
            "content": assistant_response,
            "timestamp": ts,
        }) + "\n")

        self._handle.flush()

    def new_session(self) -> str:
        self.close()
        self.session_id = self._make_id()
        self.session_file = self.session_dir / f"{self.session_id}.jsonl"
        self._handle = None
        return self.session_id

    def close(self):
        if self._handle:
            self._handle.close()
            self._handle = None
