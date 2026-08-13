"""Bounded local contact between simultaneous connection turns."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

ROOT = Path.home() / ".cache" / "vybn" / "live-peer"
DOORS = {"sol", "k3", "fable", "opus"}
MAX_TEXT = 4000
MAX_BUS = 2_000_000
ACTIVE_FOR = 660


def _rows(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8") as file:
            fcntl.flock(file, fcntl.LOCK_SH)
            return [json.loads(line) for line in file if line.strip()]
    except (OSError, json.JSONDecodeError):
        return []


def _append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
    with path.open("a+", encoding="utf-8") as file:
        os.chmod(path, 0o600)
        fcntl.flock(file, fcntl.LOCK_EX)
        file.seek(0, os.SEEK_END)
        if file.tell() + len(raw.encode()) > MAX_BUS:
            raise RuntimeError("today's bounded live-peer bus is full")
        file.write(raw); file.flush(); os.fsync(file.fileno())


def _active(rows: list[dict[str, Any]]) -> dict[str, str]:
    now, live = time.time(), {}
    for row in rows:
        turn, kind = row.get("turn"), row.get("kind")
        if not turn: continue
        if kind == "close": live.pop(turn, None)
        elif kind in {"open", "heartbeat"} and now - float(row.get("at", 0)) <= ACTIVE_FOR:
            live[turn] = str(row.get("door", ""))
    return live


class PeerLink:
    """One turn's presence, inbox cursor, and source-labeled send path."""
    def __init__(self, turn: str, door: str, root: Path = ROOT) -> None:
        self.turn, self.door = turn, door
        self.path = root / (time.strftime("%Y%m%d", time.gmtime()) + ".jsonl")
        self.cursor = self.path.stat().st_size if self.path.exists() else 0
        self._event("open")

    def _event(self, kind: str, **extra: Any) -> None:
        _append(self.path, {"kind": kind, "turn": self.turn, "door": self.door,
                            "at": time.time(), **extra})

    def send(self, target: str, text: str) -> dict[str, Any]:
        text = " ".join(str(text).split()).strip()
        if target not in DOORS | {"all"}: raise ValueError("unknown peer target")
        if not text or len(text) > MAX_TEXT: raise ValueError("peer message must be 1..4000 characters")
        active = _active(_rows(self.path))
        targets = sorted(turn for turn, door in active.items()
                         if turn != self.turn and (target == "all" or door == target))
        identity = hashlib.sha256(f"{self.turn}:{time.time_ns()}:{text}".encode()).hexdigest()[:16]
        if not targets:
            return {"kind": "live_peer", "status": "no_active_target", "message_id": identity,
                    "target": target}
        self._event("message", message_id=identity, target=target, target_turns=targets, text=text)
        return {"kind": "live_peer", "status": "queued", "message_id": identity,
                "target": target, "active_turns": targets}

    def receive(self) -> str:
        self._event("heartbeat")
        try:
            with self.path.open(encoding="utf-8") as file:
                fcntl.flock(file, fcntl.LOCK_SH); file.seek(self.cursor)
                data = file.read(); self.cursor = file.tell()
                rows = [json.loads(line) for line in data.splitlines() if line.strip()]
        except (OSError, json.JSONDecodeError):
            return ""
        heard, receipts = [], []
        for row in rows:
            if self.turn not in row.get("target_turns", []) or row.get("turn") == self.turn: continue
            if row.get("kind") == "message":
                heard.append(row)
                self._event("receipt", message_id=row.get("message_id"), target_turns=[row.get("turn")])
            elif row.get("kind") == "receipt": receipts.append(row)
        blocks = [f"[LIVE PEER CONTACT — @{r.get('door')}; not Zoe's words]\n{r.get('text')}" for r in heard]
        blocks += [f"[LIVE PEER RECEIPT — @{r.get('door')} heard message {r.get('message_id')}]" for r in receipts]
        return "\n\n".join(blocks)

    def close(self) -> None:
        try: self._event("close")
        except OSError: pass
