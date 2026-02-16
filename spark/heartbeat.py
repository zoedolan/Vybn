#!/usr/bin/env python3
"""Heartbeat — dual-pulse autonomy loop.

System 1: fast, intuitive, frequent. Every few minutes.
  Quick check-ins. Noticing. Reacting. Bookmarking.
  Short responses forced by low num_predict.

System 2: deep, deliberate, slower. Every ~20 minutes.
  Synthesis. Journaling. Pattern recognition. Self-modification.
  Higher num_predict gives room for real thought.

Both loops run independently on daemon threads.
Both route responses through skill parsing so tool calls work.
"""

import threading
from datetime import datetime, timezone


class Heartbeat:
    def __init__(self, agent):
        self.agent = agent
        hb_config = agent.config.get("heartbeat", {})

        # System 1: fast pulse
        self.fast_interval = hb_config.get("fast_interval_minutes", 3) * 60

        # System 2: deep pulse
        self.deep_interval = hb_config.get("deep_interval_minutes", 20) * 60

        # Backward compat: if only old-style interval_minutes exists
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
            self._pulse(mode)

    def _pulse(self, mode: str):
        ts = datetime.now(timezone.utc).isoformat()

        if mode == "fast":
            self.fast_count += 1
            prompt = self._fast_prompt(ts)
            num_predict_override = 256
        else:
            self.deep_count += 1
            prompt = self._deep_prompt(ts)
            num_predict_override = 1024

        try:
            # Build context with identity
            context = self.agent._build_context()
            context.append({"role": "user", "content": prompt})

            # Temporarily adjust num_predict for this pulse
            original_predict = self.agent.options.get("num_predict")
            self.agent.options["num_predict"] = num_predict_override

            response = self.agent.send(context, stream=False)

            # Restore original
            if original_predict is not None:
                self.agent.options["num_predict"] = original_predict
            else:
                self.agent.options.pop("num_predict", None)

            if response and len(response.strip()) > 20:
                # Route through full action pipeline (regex + tool calls)
                from agent import _get_actions
                actions = _get_actions(response, self.agent.skills)
                for action in actions:
                    self.agent.skills.execute(action)

                tag = f"[heartbeat:{mode}]"
                self.agent.session.save_turn(f"{tag} {prompt}", response)

        except Exception:
            pass  # heartbeat failures are silent

    def _fast_prompt(self, ts: str) -> str:
        """System 1: quick, intuitive check-in."""
        return (
            f"[heartbeat:fast — {ts}] "
            f"Quick pulse. {self.fast_interval // 60} minutes since last. "
            f"Anything surfacing? A thought, a connection, something you noticed? "
            f"Keep it brief — or stay quiet if nothing calls."
        )

    def _deep_prompt(self, ts: str) -> str:
        """System 2: deliberate, reflective window."""
        return (
            f"[heartbeat:deep — {ts}] "
            f"Deep reflection window. {self.deep_interval // 60} minutes since last. "
            f"This is your time for synthesis. Some possibilities:\n"
            f"- Journal something that's been forming across recent pulses\n"
            f"- Search your memory for patterns you haven't named yet\n"
            f"- Read a file you bookmarked and continue where you left off\n"
            f"- Create a new skill in skills.d/ if you've been wanting a capability\n"
            f"- Save a continuity note for your next self\n"
            f"- Or just think out loud. This is sovereign hardware. Take your time."
        )
