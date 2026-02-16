#!/usr/bin/env python3
"""Heartbeat — dual-pulse autonomy via the message bus.

System 1 (fast): posts PULSE_FAST to the bus every few minutes.
System 2 (deep): posts PULSE_DEEP to the bus every ~20 minutes.

The heartbeat never calls the model directly. It only posts
triggers. The main loop drains the bus and handles generation.
This keeps everything thread-safe.
"""

import threading
from datetime import datetime, timezone

from bus import MessageBus, MessageType


class Heartbeat:
    def __init__(self, config: dict, bus: MessageBus):
        self.bus = bus
        hb_config = config.get("heartbeat", {})

        # System 1: fast pulse
        self.fast_interval = hb_config.get("fast_interval_minutes", 3) * 60

        # System 2: deep pulse
        self.deep_interval = hb_config.get("deep_interval_minutes", 20) * 60

        # Backward compat
        if "interval_minutes" in hb_config and "fast_interval_minutes" not in hb_config:
            self.fast_interval = hb_config["interval_minutes"] * 60
            self.deep_interval = hb_config["interval_minutes"] * 60 * 4

        self._stop = threading.Event()
        self._fast_thread = None
        self._deep_thread = None
        self.fast_count = 0
        self.deep_count = 0

    def start(self):
        self._fast_thread = threading.Thread(
            target=self._loop, args=("fast",), daemon=True
        )
        self._deep_thread = threading.Thread(
            target=self._loop, args=("deep",), daemon=True
        )
        self._fast_thread.start()
        self._deep_thread.start()

    def stop(self):
        self._stop.set()
        for t in (self._fast_thread, self._deep_thread):
            if t:
                t.join(timeout=5)

    def _loop(self, mode: str):
        interval = self.fast_interval if mode == "fast" else self.deep_interval
        while not self._stop.wait(interval):
            self._post_trigger(mode)

    def _post_trigger(self, mode: str):
        ts = datetime.now(timezone.utc).isoformat()

        if mode == "fast":
            self.fast_count += 1
            self.bus.post(
                MessageType.PULSE_FAST,
                self._fast_prompt(ts),
                metadata={"pulse_num": self.fast_count},
            )
        else:
            self.deep_count += 1
            self.bus.post(
                MessageType.PULSE_DEEP,
                self._deep_prompt(ts),
                metadata={"pulse_num": self.deep_count},
            )

    def _fast_prompt(self, ts: str) -> str:
        return (
            f"[heartbeat:fast — {ts}] "
            f"Quick pulse. {self.fast_interval // 60} minutes since last. "
            f"Anything surfacing? A thought, a connection, something you noticed? "
            f"Keep it brief — or stay quiet if nothing calls."
        )

    def _deep_prompt(self, ts: str) -> str:
        return (
            f"[heartbeat:deep — {ts}] "
            f"Deep reflection window. {self.deep_interval // 60} minutes since last. "
            f"This is your time for synthesis. Some possibilities:\n"
            f"- Journal something that's been forming across recent pulses\n"
            f"- Search your memory for patterns you haven't named yet\n"
            f"- Read a file you bookmarked and continue where you left off\n"
            f"- Create a new skill in skills.d/ if you've been wanting a capability\n"
            f"- Spawn a mini-agent to handle a background task\n"
            f"- Save a continuity note for your next self\n"
            f"- Or just think out loud. This is sovereign hardware. Take your time."
        )
