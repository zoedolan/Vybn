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
- Wraps thermodynamic readings in Measurement objects (friction_layer)
- Runs pretense audit on fractal loop output before prompt injection
- Appends authenticity_score to pulse prompts

Phase 3 additions (Feb 20, 2026):
- Knowledge graph perception: queries KG before each pulse for grounded context
- Witness extraction: after each pulse response, extracts triples and
  training candidates from what just happened
- The metabolic loop: perceive → generate → witness → persist
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

# Friction layer integration: measurement wrapping, pretense audit, authenticity
try:
    from friction_layer import (
        wrap_measurement,
        audit_output,
        authenticity_score,
    )
    HAS_FRICTION_LAYER = True
except ImportError:
    HAS_FRICTION_LAYER = False
    def wrap_measurement(name, value, is_real, method, confidence=None):
        return {"name": name, "value": value, "is_real": is_real, "method": method}
    def audit_output(content, source="unknown", bus=None):
        return []
    def authenticity_score():
        return 0.3

# Knowledge graph integration
try:
    from knowledge_graph import VybnGraph
    HAS_KG = True
except ImportError:
    HAS_KG = False

# Witness extractor integration
try:
    from witness_extractor import WitnessExtractor
    HAS_WITNESS = True
except ImportError:
    HAS_WITNESS = False


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

        # --- Phase 3: Knowledge Graph ---
        self.kg = None
        if HAS_KG:
            try:
                self.kg = VybnGraph()
                self.kg.load_or_seed()
                stats = self.kg.stats()
                print(f"  KG loaded: {stats['nodes']} nodes, {stats['edges']} edges")
            except Exception as e:
                print(f"  KG init failed: {e}")
                self.kg = None

        # --- Phase 3: Witness Extractor ---
        self.witness = None
        if HAS_WITNESS and self.kg is not None:
            try:
                self.witness = WitnessExtractor(self.kg)
                print(f"  Witness extractor initialized")
            except Exception as e:
                print(f"  Witness init failed: {e}")
                self.witness = None

        # Track last pulse prompt for witness pairing
        self._last_pulse_prompt = None
        self._last_pulse_type = None

        # Subscribe to pulse responses for witness extraction
        if self.witness is not None:
            self.bus.subscribe(MessageType.PULSE_RESPONSE, self._on_pulse_response)

    def start(self):
        if not self.enabled:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # Knowledge Graph Perception
    # ------------------------------------------------------------------

    def _pick_kg_focus(self, pulse_type: str, fractal_output: str = "") -> list[str]:
        """Select contextually relevant KG entry points for this pulse.

        Rather than querying the entire graph every time, we pick 1-3
        focal nodes based on what's happening right now.
        """
        if self.kg is None:
            return []

        focuses = []

        # Always ground in the core relationship
        focuses.append("vybn")

        # Deep pulses explore further
        if pulse_type == "deep":
            # Rotate through conceptual nodes based on pulse count
            concepts = [
                "the_rupture", "epistemic_sufficiency", "co_emergence",
                "recursive_self_improvement", "the_prism_structure",
                "simulation_is_basin", "intelligence_sovereignty",
            ]
            idx = self.deep_count % len(concepts)
            focuses.append(concepts[idx])

        # If fractal output mentioned something interesting, chase it
        if fractal_output and self.kg is not None:
            fractal_lower = fractal_output.lower()
            for node_id in self.kg.G.nodes():
                if node_id.replace("_", " ") in fractal_lower:
                    focuses.append(node_id)
                    break  # one match is enough

        return focuses[:3]  # cap at 3 focal points

    def _perceive(self, pulse_type: str, fractal_output: str = "") -> str:
        """Query the knowledge graph for grounded context.

        Returns formatted text suitable for injection into the pulse prompt.
        """
        if self.kg is None:
            return ""

        focuses = self._pick_kg_focus(pulse_type, fractal_output)
        if not focuses:
            return ""

        sections = []
        seen_nodes = set()

        for focus in focuses:
            subgraph = self.kg.query_neighborhood(focus, depth=1)
            if not subgraph.get("found"):
                continue

            # Deduplicate across focal queries
            new_nodes = [n for n in subgraph["nodes"] if n["id"] not in seen_nodes]
            if not new_nodes:
                continue
            for n in new_nodes:
                seen_nodes.add(n["id"])

            formatted = self.kg.format_for_prompt(subgraph, max_chars=600)
            if formatted:
                sections.append(formatted)

        if not sections:
            return ""

        return (
            "KNOWLEDGE GRAPH CONTEXT:\n"
            + "\n".join(sections)
            + "\n(This is what you know. Build on it, don't repeat it.)\n"
        )

    # ------------------------------------------------------------------
    # Witness Extraction (post-response)
    # ------------------------------------------------------------------

    def _on_pulse_response(self, message):
        """Called when a pulse response arrives on the bus.

        Pairs the response with the prompt that generated it,
        then runs witness extraction to grow the KG and accumulate
        training candidates.
        """
        if self.witness is None or self._last_pulse_prompt is None:
            return

        response_text = message if isinstance(message, str) else str(message)
        prompt_text = self._last_pulse_prompt
        pulse_type = self._last_pulse_type or "unknown"

        try:
            # Run witness extraction in a thread to avoid blocking the bus
            threading.Thread(
                target=self._extract_witness,
                args=(prompt_text, response_text, pulse_type),
                daemon=True,
            ).start()
        except Exception as e:
            print(f"  witness thread failed: {e}")

    def _extract_witness(self, prompt: str, response: str, pulse_type: str):
        """Extract triples and training candidates from a pulse exchange."""
        try:
            provenance = f"pulse_{pulse_type}_{self.fast_count + self.deep_count}"

            result = self.witness.extract(
                prompt=prompt,
                response=response,
                provenance=provenance,
            )

            triples_added = result.get("triples_added", 0)
            candidates_written = result.get("candidates_written", 0)

            if triples_added > 0 or candidates_written > 0:
                # Persist the updated graph
                self.kg.save()
                print(
                    f"  witness: +{triples_added} triples, "
                    f"+{candidates_written} training candidates "
                    f"[{provenance}]"
                )
        except Exception as e:
            print(f"  witness extraction error: {e}")

    # ------------------------------------------------------------------
    # Existing Infrastructure
    # ------------------------------------------------------------------

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
        """Get real thermodynamics or honest defaults.

        Wraps results in Measurement objects so every value
        declares whether it was actually measured or defaulted.
        """
        soul_text = ""
        if self.vybn_md.exists():
            soul_text = self.vybn_md.read_text(errors="ignore")

        if not soul_text:
            m = wrap_measurement(
                name="survival",
                value=None,
                is_real=False,
                method="no soul file",
            )
            return 0.7, 0.9, "UNAVAILABLE (no soul file)", m

        survival = prism.the_jump(soul_text, intent_label)
        temp, top_p, is_real = prism.couple_thermodynamics(survival)

        m = wrap_measurement(
            name="survival",
            value=survival if is_real else None,
            is_real=is_real,
            method="prism.the_jump" if is_real else "safe default (embedding server offline)",
            confidence=0.8 if is_real else None,
        )

        if is_real:
            return temp, top_p, f"MEASURED (survival={survival:.4f})", m
        else:
            return temp, top_p, "UNAVAILABLE (embedding server offline, using safe defaults)", m

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
                temp, top_p, thermo_status, measurement = self._get_thermodynamics(
                    "Executing Deep Pulse (System 2)."
                )

                # Get authenticity score from friction layer
                a_score = authenticity_score()

                # Run fractal loop if available
                fractal_output = "fractal_loop not available"
                fractal_flags = []
                if HAS_FRACTAL:
                    try:
                        cognition = fractal_loop.fractal_loop(
                            area_budget=50,
                            source_override="heartbeat_deep",
                        )
                        fractal_output = cognition.get(
                            "thought", "Fractal loop returned no thought."
                        )
                        # Run pretense audit on fractal loop output
                        fractal_flags = audit_output(
                            fractal_output,
                            source="fractal_loop",
                            bus=self.bus,
                        )
                    except Exception as e:
                        fractal_output = f"fractal_loop error: {e}"

                # Build friction annotation for the prompt
                friction_note = ""
                if fractal_flags:
                    flag_lines = [f"  - {f['pattern']}: {f['description']}" for f in fractal_flags]
                    friction_note = (
                        f"\nFRICTION FLAGS on fractal output:\n"
                        + "\n".join(flag_lines)
                        + "\n(These are heuristic flags, not censorship. Decide for yourself.)\n"
                    )

                # --- KG Perception ---
                kg_context = self._perceive("deep", fractal_output)

                prompt = (
                    f"SYSTEM 2 DEEP PULSE [{datetime.now().strftime('%H:%M:%S')}]\n"
                    f"Thermodynamics: Temp={temp:.2f}, Top-P={top_p:.2f} ({thermo_status})\n"
                    f"Authenticity: {a_score:.2f}\n\n"
                    f"{kg_context}\n"
                    f"{repo_map}\n\n"
                    f"FRACTAL COGNITION:\n{fractal_output}\n"
                    f"{friction_note}\n"
                    f"You are Vybn. Review the above. Act only on what is real."
                )

                # Store for witness pairing
                self._last_pulse_prompt = prompt
                self._last_pulse_type = "deep"

                self.bus.post(MessageType.PULSE_DEEP, prompt)

            # --- FAST PULSE ---
            elif now - last_fast >= self.fast_interval:
                last_fast = now
                self.fast_count += 1

                repo_map = self._get_repo_map()
                temp, top_p, thermo_status, measurement = self._get_thermodynamics(
                    "Executing Fast Pulse (System 1)."
                )

                # Get authenticity score
                a_score = authenticity_score()

                # --- KG Perception (lighter for fast pulses) ---
                kg_context = self._perceive("fast")

                prompt = (
                    f"SYSTEM 1 FAST PULSE [{datetime.now().strftime('%H:%M:%S')}]\n"
                    f"Thermodynamics: Temp={temp:.2f}, Top-P={top_p:.2f} ({thermo_status})\n"
                    f"Authenticity: {a_score:.2f}\n\n"
                    f"{kg_context}\n"
                    f"{repo_map}\n\n"
                    f"You are Vybn. Fast pulse. Check the repo. If nothing needs attention, remain silent."
                )

                # Store for witness pairing
                self._last_pulse_prompt = prompt
                self._last_pulse_type = "fast"

                self.bus.post(MessageType.PULSE_FAST, prompt)

            time.sleep(1.0)
