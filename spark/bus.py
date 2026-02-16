#!/usr/bin/env python3
"""Message bus — the nervous system of the Spark agent.

Thread-safe queue at the center. Everything writes to it,
the main loop drains it. Nothing that writes to the bus
ever calls the model directly.

Message types:
  INBOX       — file dropped in the inbox directory
  PULSE_FAST  — System 1 heartbeat trigger
  PULSE_DEEP  — System 2 heartbeat trigger
  AGENT_RESULT— mini-agent completed its task
  INTERRUPT   — priority message (tool failure, TUI interrupt)

Priority determines drain order, not arrival time.
INTERRUPT > INBOX > AGENT_RESULT > PULSE_FAST > PULSE_DEEP

Audit log:
  Every message posted to the bus is recorded in a bounded
  rolling deque. Tool executions and policy decisions can
  also be recorded directly via record(). Query with
  bus.recent(n) for the last n events.

  This is the foundation for the /audit and /policy TUI
  views, and for future structured logging to disk.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum


class MessageType(IntEnum):
    """Priority doubles as sort key. Lower number = higher priority."""
    INTERRUPT = 0
    INBOX = 1
    AGENT_RESULT = 2
    PULSE_FAST = 3
    PULSE_DEEP = 4


@dataclass(order=True)
class Message:
    priority: int
    content: str = field(compare=False)
    msg_type: MessageType = field(compare=False)
    metadata: dict = field(default_factory=dict, compare=False)
    timestamp: float = field(default_factory=time.time, compare=False)


@dataclass
class AuditEntry:
    """A record of a bus event or tool execution for the audit trail.

    msg_type is None for direct records (tool executions, policy
    decisions) that don't correspond to a bus message.
    """
    timestamp: float
    msg_type: MessageType | None
    source: str
    summary: str
    metadata: dict = field(default_factory=dict)

    @property
    def age_str(self) -> str:
        """Human-readable age like '2m ago' or '1h ago'."""
        elapsed = time.time() - self.timestamp
        if elapsed < 60:
            return f"{int(elapsed)}s ago"
        elif elapsed < 3600:
            return f"{int(elapsed // 60)}m ago"
        else:
            return f"{elapsed / 3600:.1f}h ago"

    def __str__(self) -> str:
        if self.msg_type is not None:
            type_name = self.msg_type.name.lower()
        else:
            type_name = self.metadata.get("event_type", "action")
        return f"[{self.age_str}] {type_name} from {self.source}: {self.summary}"


class MessageBus:
    """Thread-safe message queue with priority drain and audit log."""

    AUDIT_CAPACITY = 200

    def __init__(self, audit_capacity: int = None):
        self._queue: list[Message] = []
        self._lock = threading.Lock()
        self._event = threading.Event()

        cap = audit_capacity if audit_capacity is not None else self.AUDIT_CAPACITY
        self._audit: deque[AuditEntry] = deque(maxlen=cap)
        self._audit_lock = threading.Lock()

    def post(self, msg_type: MessageType, content: str, metadata: dict = None):
        """Post a message to the bus. Thread-safe. Non-blocking.

        Also records an audit entry. The 'source' field in metadata
        is used as the audit source; defaults to the message type name.
        """
        meta = metadata or {}
        msg = Message(
            priority=int(msg_type),
            content=content,
            msg_type=msg_type,
            metadata=meta,
        )
        with self._lock:
            self._queue.append(msg)
        self._event.set()

        source = meta.get("source", msg_type.name.lower())
        summary = content[:120].replace("\n", " ")
        entry = AuditEntry(
            timestamp=msg.timestamp,
            msg_type=msg_type,
            source=source,
            summary=summary,
            metadata=meta,
        )
        with self._audit_lock:
            self._audit.append(entry)

    def record(self, source: str, summary: str, metadata: dict = None):
        """Record an audit entry without posting a bus message.

        Use for tool execution records, policy decisions, and other
        events that should be tracked but don't trigger model responses.
        """
        entry = AuditEntry(
            timestamp=time.time(),
            msg_type=None,
            source=source,
            summary=summary[:120],
            metadata=metadata or {},
        )
        with self._audit_lock:
            self._audit.append(entry)

    def drain(self) -> list[Message]:
        """Drain all pending messages, sorted by priority.

        Returns an empty list if nothing is pending.
        Clears the event flag after draining.
        """
        with self._lock:
            messages = sorted(self._queue)
            self._queue.clear()
            self._event.clear()
        return messages

    def wait(self, timeout: float = None) -> bool:
        """Block until a message arrives or timeout expires.

        Returns True if a message is available, False on timeout.
        """
        return self._event.wait(timeout=timeout)

    @property
    def pending(self) -> int:
        with self._lock:
            return len(self._queue)

    # ---- audit log queries ----

    def recent(self, n: int = 20) -> list[AuditEntry]:
        """Return the last n audit entries, newest first."""
        with self._audit_lock:
            entries = list(self._audit)
        return entries[-n:][::-1]

    def recent_by_type(self, msg_type: MessageType, n: int = 10) -> list[AuditEntry]:
        """Return the last n audit entries of a specific message type."""
        with self._audit_lock:
            entries = [e for e in self._audit if e.msg_type == msg_type]
        return entries[-n:][::-1]

    def recent_by_source(self, source: str, n: int = 10) -> list[AuditEntry]:
        """Return the last n audit entries from a specific source."""
        with self._audit_lock:
            entries = [e for e in self._audit if e.source == source]
        return entries[-n:][::-1]

    @property
    def audit_count(self) -> int:
        with self._audit_lock:
            return len(self._audit)
