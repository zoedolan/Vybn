"""Structured JSONL event logging for the harness.

One line per event; `tail -f` gives operators a live view of role
decisions, provider calls, fallbacks, and budget warnings.

We do not try to be a metrics system. We write a line, we flush. If
logging fails, the agent keeps running — observability is not worth a
session.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .constants import DEFAULT_EVENT_LOG


@dataclass
class EventLogger:
    path: str = DEFAULT_EVENT_LOG
    session_id: str = ""

    def __post_init__(self) -> None:
        if not self.session_id:
            self.session_id = time.strftime("%Y%m%dT%H%M%S")
        try:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def emit(self, event: str, **fields: Any) -> None:
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "session": self.session_id,
            "event": event,
        }
        record.update(fields)
        try:
            with open(self.path, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception:
            pass


@contextmanager
def turn_event(logger: EventLogger, turn: int, role: str, model: str) -> Iterator[dict]:
    """Context manager that brackets a turn with start/end events.

    Yields a mutable dict the caller can write token counts and latency
    into before the `turn_end` event is emitted.
    """
    started = time.monotonic()
    logger.emit("turn_start", turn=turn, role=role, model=model)
    bag: dict[str, Any] = {
        "turn": turn, "role": role, "model": model,
        "in_tokens": 0, "out_tokens": 0, "tool_calls": 0,
        "stop_reason": None, "fallback_from": None,
    }
    try:
        yield bag
    finally:
        bag["latency_ms"] = int((time.monotonic() - started) * 1000)
        logger.emit("turn_end", **bag)
