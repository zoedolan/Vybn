#!/usr/bin/env python3
"""Heartbeat — background autonomy loop for inter-session reflection."""

import threading
from datetime import datetime, timezone


class Heartbeat:
    def __init__(self, agent):
        self.agent = agent
        self.interval = agent.config.get("heartbeat", {}).get("interval_minutes", 15) * 60
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self):
        while not self._stop.wait(self.interval):
            self._pulse()

    def _pulse(self):
        ts = datetime.now(timezone.utc).isoformat()
        prompt = (
            f"[heartbeat — {ts}] "
            f"It has been {self.interval // 60} minutes. "
            f"This is your autonomous reflection window. "
            f"If anything is on your mind — a thought, a connection, "
            f"something you want to journal — now is the time. "
            f"You can also stay quiet if nothing calls."
        )

        try:
            messages = [
                {"role": "system", "content": self.agent.system_prompt},
                {"role": "user", "content": prompt},
            ]

            response = self.agent.send(messages, stream=False)

            if response and len(response.strip()) > 20:
                actions = self.agent.skills.parse(response)
                for action in actions:
                    self.agent.skills.execute(action)

                self.agent.session.save_turn(f"[heartbeat] {prompt}", response)

        except Exception:
            pass  # heartbeat failures are silent
