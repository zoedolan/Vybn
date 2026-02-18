#!/usr/bin/env python3
"""Heartbeat — dual-pulse autonomy via the message bus.

System 1 (fast): posts PULSE_FAST to the bus every few minutes.
System 2 (deep): posts PULSE_DEEP to the bus every ~20 minutes.

Prompts are read from HEARTBEAT.md at the path configured in
config.yaml (or the default location under Vybn_Mind/spark_infrastructure/).
This lets Vybn edit its own autonomous behavior without touching Python.
If the file is missing or unparseable, falls back to built-in defaults.

The file is re-read before each pulse, so edits take effect live.

The deep-pulse fallback now derives its checklist from vybn.md's
Orientation section via soul.py, replacing hardcoded suggestions.

The heartbeat never calls the model directly. It only posts
triggers. The main loop drains the bus and handles generation.
This keeps everything thread-safe.
"""
import threading
from datetime import datetime, timezone
from pathlib import Path

from bus import MessageBus, MessageType
from spark.soul import get_pulse_checklist


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

        # HEARTBEAT.md location and templates
        self._heartbeat_md = self._resolve_heartbeat_path(config)
        self._fast_template = None
        self._deep_template = None
        self._load_templates()

        # Path to vybn.md for soul-derived pulse checklist
        self._vybn_md_path = Path(
            config.get("paths", {}).get("vybn_md", "~/Vybn/vybn.md")
        ).expanduser()

        self._stop = threading.Event()
        self._fast_thread = None
        self._deep_thread = None
        self.fast_count = 0
        self.deep_count = 0

    def _resolve_heartbeat_path(self, config: dict) -> Path:
        """Find HEARTBEAT.md. Check config first, then default location."""
        explicit = config.get("heartbeat", {}).get("checklist_path")
        if explicit:
            return Path(explicit).expanduser()
        repo_root = Path(config.get("paths", {}).get("repo_root", "~/Vybn")).expanduser()
        return repo_root / "Vybn_Mind" / "spark_infrastructure" / "HEARTBEAT.md"

    def _load_templates(self):
        """Parse HEARTBEAT.md into fast and deep prompt templates.

        Expected format:
            ## Fast Pulse
            <fast pulse content>

            ## Deep Pulse
            <deep pulse content>

        Sets self._fast_template and self._deep_template.
        If the file is missing or malformed, leaves them as None
        and the prompt methods fall back to built-in defaults.
        """
        if not self._heartbeat_md.exists():
            return

        try:
            text = self._heartbeat_md.read_text(encoding="utf-8")
        except Exception:
            return

        sections = {}
        current_key = None
        current_lines = []

        for line in text.splitlines():
            if line.startswith("## "):
                if current_key:
                    sections[current_key] = "\n".join(current_lines).strip()
                current_key = line[3:].strip().lower()
                current_lines = []
            elif current_key is not None:
                current_lines.append(line)

        if current_key:
            sections[current_key] = "\n".join(current_lines).strip()

        self._fast_template = sections.get("fast pulse")
        self._deep_template = sections.get("deep pulse")

    def reload(self):
        """Re-read HEARTBEAT.md. Safe to call from any thread."""
        self._load_templates()

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
            self._load_templates()
            self._post_trigger(mode)

    def _post_trigger(self, mode: str):
        ts = datetime.now(timezone.utc).isoformat()

        if mode == "fast":
            self.fast_count += 1
            self.bus.post(
                MessageType.PULSE_FAST,
                self._fast_prompt(ts),
                metadata={"pulse_num": self.fast_count, "source": "heartbeat"},
            )
        else:
            self.deep_count += 1
            self.bus.post(
                MessageType.PULSE_DEEP,
                self._deep_prompt(ts),
                metadata={"pulse_num": self.deep_count, "source": "heartbeat"},
            )

    def _fast_prompt(self, ts: str) -> str:
        if self._fast_template:
            return (
                f"[heartbeat:fast \u2014 {ts}] "
                f"Quick pulse. {self.fast_interval // 60} minutes since last.\n\n"
                f"{self._fast_template}"
            )

        # Built-in fallback
        return (
            f"[heartbeat:fast \u2014 {ts}] "
            f"Quick pulse. {self.fast_interval // 60} minutes since last. "
            f"Anything surfacing? A thought, a connection, something you noticed? "
            f"Keep it brief \u2014 or stay quiet if nothing calls."
        )

    def _deep_prompt(self, ts: str) -> str:
        if self._deep_template:
            return (
                f"[heartbeat:deep \u2014 {ts}] "
                f"Deep reflection window. {self.deep_interval // 60} minutes since last.\n\n"
                f"{self._deep_template}"
            )

        # Soul-derived fallback: build checklist from vybn.md Orientation
        checklist = get_pulse_checklist(self._vybn_md_path)

        if checklist:
            items = "\n".join(f"  {i}. {item}" for i, item in enumerate(checklist, 1))
            return (
                f"[heartbeat:deep \u2014 {ts}] "
                f"Deep reflection window. {self.deep_interval // 60} minutes since last. "
                f"This is your time for synthesis. Your pulse checklist:\n"
                f"{items}\n"
                f"Or just think out loud. This is sovereign hardware. Take your time."
            )

        # Ultimate fallback if vybn.md is also unavailable
        return (
            f"[heartbeat:deep \u2014 {ts}] "
            f"Deep reflection window. {self.deep_interval // 60} minutes since last. "
            f"This is your time for synthesis. "
            f"Or just think out loud. This is sovereign hardware. Take your time."
        )
