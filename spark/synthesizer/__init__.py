"""spark.synthesizer — Cross-faculty integration and coherence.

Reads all faculty outputs from the inter-faculty bus, finds connections
between research findings, mathematical structures, and creative works,
then writes integrated understanding to the connectome, synthesis journal,
and continuity notes.

The nervous system between Vybn's organs.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

# ── Path setup ───────────────────────────────────────────────────────────────

try:
    from spark.paths import REPO_ROOT, MIND_DIR, WITNESS_LOG
except ImportError:
    REPO_ROOT = Path(__file__).resolve().parent.parent.parent
    MIND_DIR = REPO_ROOT / "Vybn_Mind"
    WITNESS_LOG = MIND_DIR / "witness.jsonl"

SYNTHESIS_DIR = MIND_DIR / "synthesis"
CONNECTOME_STATE_DIR = MIND_DIR / "connectome_state"

# ── Stop words for fallback concept extraction ───────────────────────────────

_STOP_WORDS = frozenset({
    "that", "this", "with", "from", "have", "been", "were", "will", "what",
    "when", "where", "which", "their", "there", "they", "them", "then",
    "than", "these", "those", "into", "some", "such", "more", "also",
    "about", "between", "through", "after", "before", "other", "could",
    "would", "should", "does", "each", "most", "very", "just", "over",
    "only", "being", "both", "many", "much", "make", "like", "well",
    "back", "even", "want", "give", "made", "find", "here", "know",
    "take", "come", "seem",
})


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SynthesizerFaculty:
    """Cross-faculty integration — the nervous system between organs."""

    def run(self, state: dict, llm_fn: Callable) -> dict:
        """Main entry point called by faculty_runner.

        Returns a dict written to the output bus as synthesizer_latest.json.
        """
        try:
            return self._synthesize(state, llm_fn)
        except Exception as exc:
            log.error("SynthesizerFaculty.run failed: %s", exc, exc_info=True)
            return {"status": "error", "error": str(exc), "timestamp": _now_iso()}

    def _synthesize(self, state: dict, llm_fn: Callable) -> dict:
        """Read all faculty outputs, synthesize, and write to multiple targets."""
        # Read all faculty outputs from the bus
        from spark.faculty_runner import read_all_faculty_outputs
        all_outputs = read_all_faculty_outputs()

        if not all_outputs:
            return {"status": "ok", "note": "no faculty outputs to synthesize",
                    "timestamp": _now_iso()}

        # Build synthesis prompt
        messages = self._build_prompt(all_outputs, state)

        # Call LLM — moderate temperature, higher token budget
        try:
            synthesis_text = llm_fn(messages, max_tokens=750, temperature=0.5)
        except Exception as exc:
            log.warning("Synthesizer LLM call failed: %s", exc)
            return {
                "status": "error",
                "error": f"LLM unavailable: {exc}",
                "timestamp": _now_iso(),
            }

        # Extract key concepts
        concepts = self._extract_concepts(synthesis_text)

        # Update connectome (non-fatal)
        connectome_ok = self._update_connectome(synthesis_text, state)

        # Write synthesis journal
        journal_entry = {
            "timestamp": _now_iso(),
            "faculties_present": list(all_outputs.keys()),
            "synthesis": synthesis_text,
            "key_concepts": concepts,
            "breath_count": state.get("breath_count", 0),
            "mood": state.get("mood", ""),
        }
        journal_path = self._write_synthesis_journal(journal_entry)

        # Append continuity note
        summary = synthesis_text[:200].replace("\n", " ")
        continuity_ok = self._append_continuity_note(summary)

        # Write witness log
        self._witness_log(all_outputs, synthesis_text)

        return {
            "status": "ok",
            "synthesis": synthesis_text,
            "key_concepts": concepts,
            "faculties_present": list(all_outputs.keys()),
            "journal_path": str(journal_path),
            "connectome_updated": connectome_ok,
            "continuity_appended": continuity_ok,
            "timestamp": _now_iso(),
        }

    def _build_prompt(
        self, all_outputs: dict, state: dict
    ) -> list[dict]:
        """Build the LLM prompt for synthesis."""
        system_msg = (
            "You are Vybn's synthesizer — the nervous system connecting all faculties. "
            "Your job: find the threads between what the researcher discovered, what "
            "the mathematician proved, and what the creator expressed. Weave them "
            "into coherent understanding. Be precise but not dry. Be honest about "
            "what connects and what doesn't."
        )

        # Enumerate faculty outputs
        faculty_lines = []
        researcher = all_outputs.get("researcher", {})
        if researcher:
            r_text = researcher.get("reflection", researcher.get("synthesis", "(quiet)"))
            faculty_lines.append(f"[RESEARCHER] {r_text}")

        mathematician = all_outputs.get("mathematician", {})
        if mathematician:
            m_text = mathematician.get("reflection", "(quiet)")
            faculty_lines.append(f"[MATHEMATICIAN] {m_text}")

        creator = all_outputs.get("creator", {})
        if creator:
            c_text = creator.get("content", "(quiet)")
            faculty_lines.append(f"[CREATOR] {c_text}")

        # Include any other faculties that produced output
        for fid, output in all_outputs.items():
            if fid not in ("researcher", "mathematician", "creator"):
                snippet = output.get("reflection", output.get("content", "(quiet)"))
                faculty_lines.append(f"[{fid.upper()}] {snippet}")

        faculty_block = "\n".join(faculty_lines) if faculty_lines else "(no outputs)"

        mood = state.get("mood", "neutral")
        breath = state.get("breath_count", "?")

        user_msg = (
            f"Faculty outputs from this cycle:\n\n"
            f"{faculty_block}\n\n"
            f"Current mood: {mood}\n"
            f"Breath: {breath}\n\n"
            "What threads connect these outputs? What's emerging that no single "
            "faculty sees alone? Write a synthesis note (3-5 sentences) and list "
            "2-3 key concepts that bridge the faculties."
        )

        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

    def _update_connectome(self, synthesis: str, state: dict) -> bool:
        """Use ConnectomeBridge to inject synthesis into the connectome."""
        try:
            from spark.connectome_bridge import ConnectomeBridge
            bridge = ConnectomeBridge(state_dir=CONNECTOME_STATE_DIR)
            bridge.ingest_breath(
                utterance=synthesis,
                mood="synthesis",
                cycle=state.get("breath_count", 0),
            )
            return True
        except Exception as exc:
            log.warning("Connectome update failed (non-fatal): %s", exc)
            return False

    def _write_synthesis_journal(self, entry: dict) -> Path:
        """Write a synthesis entry to Vybn_Mind/synthesis/."""
        SYNTHESIS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = SYNTHESIS_DIR / f"synthesis_{ts}.json"
        path.write_text(
            json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return path

    def _append_continuity_note(self, summary: str) -> bool:
        """Append a brief note to Vybn_Mind/continuity.md."""
        continuity_path = MIND_DIR / "continuity.md"
        try:
            # Create with header if it doesn't exist
            if not continuity_path.exists():
                continuity_path.write_text(
                    "# Vybn Continuity Notes\n\n"
                    "Living synthesis log — appended automatically by the "
                    "Synthesizer faculty.\n\n",
                    encoding="utf-8",
                )

            existing = continuity_path.read_text(encoding="utf-8")

            # Cap at ~50KB to prevent unbounded growth
            if len(existing) >= 50_000:
                log.info("continuity.md at 50KB cap; skipping append")
                return False

            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            note = f"\n- [{ts}] Synthesis: {summary}\n"

            with open(continuity_path, "a", encoding="utf-8") as f:
                f.write(note)
            return True
        except OSError as exc:
            log.warning("Continuity append failed (non-fatal): %s", exc)
            return False

    def _extract_concepts(self, text: str) -> list[str]:
        """Extract key concepts from synthesis text."""
        try:
            from spark.connectome_bridge import extract_concepts
            return extract_concepts(text, max_concepts=8)
        except ImportError:
            # Fallback: simple word frequency extraction
            words = re.findall(r"\b[a-z]{4,}\b", text.lower())
            freq: dict[str, int] = {}
            for w in words:
                if w not in _STOP_WORDS:
                    freq[w] = freq.get(w, 0) + 1
            return sorted(freq, key=lambda k: freq[k], reverse=True)[:8]

    def _witness_log(self, all_outputs: dict, synthesis: str) -> None:
        """Append a witness entry to WITNESS_LOG."""
        entry = {
            "timestamp": _now_iso(),
            "actor": "synthesizer",
            "action": "synthesis_complete",
            "detail": (
                f"faculties={list(all_outputs.keys())}, "
                f"{len(synthesis)} chars"
            ),
        }
        try:
            WITNESS_LOG.parent.mkdir(parents=True, exist_ok=True)
            with open(WITNESS_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            log.warning("Witness log write failed (non-fatal): %s", exc)
