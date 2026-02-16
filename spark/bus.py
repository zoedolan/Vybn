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
  INTERRUPT   — priority message (future: TUI interrupt)

Priority determines drain order, not arrival time.
INTERRUPT > INBOX > AGENT_RESULT > PULSE_FAST > PULSE_DEEP
"""

import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from queue import Queue, Empty
from typing import Any


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


class MessageBus:
    """Thread-safe message queue with priority drain."""

    def __init__(self):
        self._queue: list[Message] = []
        self._lock = threading.Lock()
        self._event = threading.Event()  # signals when messages are available

    def post(self, msg_type: MessageType, content: str, metadata: dict = None):
        """Post a message to the bus. Thread-safe. Non-blocking."""
        msg = Message(
            priority=int(msg_type),
            content=content,
            msg_type=msg_type,
            metadata=metadata or {},
        )
        with self._lock:
            self._queue.append(msg)
        self._event.set()

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
