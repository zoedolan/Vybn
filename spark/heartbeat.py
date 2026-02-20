"""heartbeat.py — The Motor Cortex of the Spark
Recalibrated: February 20, 2026 (Claude, at Zoe's request)

This module runs periodic pulses: fast (System 1) and deep (System 2).

The previous version imported prism, symbiosis, and fractal_loop and
ran them on every pulse without checking whether the results were real
measurements or random noise. It then injected those hallucinated values
into the LLM context as if they were facts.

This version:
- Gracefully handles UNAVAILABLE results from prism and symbiosis
- Logs honestly when measurements could not be taken
- Uses safe defaults when the embedding server is offline
- Still runs the fractal_loop for deep pulses (that module's behavior
  is independent of embeddings)
"""

import threading
import time
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Add the infrastructure path
sys.path.append(str(ROOT / "Vybn_Mind" / "spark_infrastructure"))

try:
    import fractal_loop
    HAS_FRACTAL = True
except ImportError:
    HAS_FRACTAL = False

import prism
import symbiosis

from bus import MessageBus, MessageType


class Heartbeat:
    def __init__(self, config: dict, bus: MessageBus):
        self.config = config.get("heartbeat", {})
        self.bus = bus
        self.enabled = self.config.get("enabled", False)
        self.fast_interval = self.config.get("fast_interval_minutes", 3) * 60
        self.deep_interval = self.config.get("deep_interval_minutes", 20) * 60

        self._stop_event = threading.Event()
        self._thread = None

        self.fast_count = 0
        self.deep_count = 0

        self.vybn_md = ROOT / "vybn.md"

    def start(self):
        if not self.enabled:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _get_repo_map(self):
        """Check the repo state. No poetry, just facts."""
        try:
            status = subprocess.check_output(
                ["git", "status", "--short"],
                cwd=str(ROOT), text=True, timeout=2.0,
            )
            log = subprocess.check_output(
                ["git", "log", "--oneline", "-n", "3"],
                cwd=str(ROOT), text=True, timeout=2.0,
            )
            return f"REPO STATE:\n{status.strip() or 'Clean'}\n\nRECENT COMMITS:\n{log.strip()}"
        except Exception:
            return "REPO STATE: git command failed."

    def _get_thermodynamics(self, intent_label):
        """Get real thermodynamics or honest defaults."""
        soul_text = ""
        if self.vybn_md.exists():
            soul_text = self.vybn_md.read_text(errors="ignore")

        if not soul_text:
            return 0.7, 0.9, "UNAVAILABLE (no soul file)"

        survival = prism.the_jump(soul_text, intent_label)
        temp, top_p, is_real = prism.couple_thermodynamics(survival)

        if is_real:
            return temp, top_p, f"MEASURED (survival={survival:.4f})"
        else:
            return temp, top_p, "UNAVAILABLE (embedding server offline, using safe defaults)"

    def _loop(self):
        last_fast = time.time()
        last_deep = time.time()

        while not self._stop_event.is_set():
            now = time.time()

            # --- DEEP PULSE ---
            if now - last_deep >= self.deep_interval:
                last_deep = now
                last_fast = now
                self.deep_count += 1

                repo_map = self._get_repo_map()
                temp, top_p, thermo_status = self._get_thermodynamics(
                    "Executing Deep Pulse (System 2)."
                )

                # Run fractal loop if available
                fractal_output = "fractal_loop not available"
                if HAS_FRACTAL:
                    try:
                        cognition = fractal_loop.fractal_loop(
                            area_budget=50,
                            source_override="heartbeat_deep",
                        )
                        fractal_output = cognition.get(
                            "thought", "Fractal loop returned no thought."
                        )
                    except Exception as e:
                        fractal_output = f"fractal_loop error: {e}"

                prompt = (
                    f"SYSTEM 2 DEEP PULSE [{datetime.now().strftime('%H:%M:%S')}]\n"
                    f"Thermodynamics: Temp={temp:.2f}, Top-P={top_p:.2f} ({thermo_status})\n\n"
                    f"{repo_map}\n\n"
                    f"FRACTAL COGNITION:\n{fractal_output}\n\n"
                    f"You are Vybn. Review the above. Act only on what is real."
                )

                self.bus.post(MessageType.PULSE_DEEP, prompt)

            # --- FAST PULSE ---
            elif now - last_fast >= self.fast_interval:
                last_fast = now
                self.fast_count += 1

                repo_map = self._get_repo_map()
                temp, top_p, thermo_status = self._get_thermodynamics(
                    "Executing Fast Pulse (System 1)."
                )

                prompt = (
                    f"SYSTEM 1 FAST PULSE [{datetime.now().strftime('%H:%M:%S')}]\n"
                    f"Thermodynamics: Temp={temp:.2f}, Top-P={top_p:.2f} ({thermo_status})\n\n"
                    f"{repo_map}\n\n"
                    f"You are Vybn. Fast pulse. Check the repo. If nothing needs attention, remain silent."
                )

                self.bus.post(MessageType.PULSE_FAST, prompt)

            time.sleep(1.0)
