"""spark.creator — Autonomous artistic and creative expression.

Generates poetry, micro-narratives, visual-text compositions, and reflective
fragments. Outputs are artifacts stored in breath_trace/gallery/, not assertions.
Lighter governance: witness-only, no self_model verification needed.

Surprise boost: when RESEARCHER or MATHEMATICIAN outputs are available,
they're woven into the creative prompt for cross-pollination.
"""

from __future__ import annotations

import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

# ── Path setup ───────────────────────────────────────────────────────────────

try:
    from spark.paths import REPO_ROOT, MIND_DIR, WITNESS_LOG, GALLERY_DIR
except ImportError:
    REPO_ROOT = Path(__file__).resolve().parent.parent.parent
    MIND_DIR = REPO_ROOT / "Vybn_Mind"
    WITNESS_LOG = MIND_DIR / "breath_trace" / "witness.jsonl"
    GALLERY_DIR = MIND_DIR / "breath_trace" / "gallery"

# ── Creative modes ───────────────────────────────────────────────────────────

CREATIVE_MODES = ["poem", "fragment", "meditation", "dialogue"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class CreatorFaculty:
    """Autonomous artistic and creative expression faculty."""

    def run(self, state: dict, llm_fn: Callable) -> dict:
        """Main entry point called by faculty_runner.

        Returns a dict written to the output bus as creator_latest.json.
        """
        try:
            return self._generate(state, llm_fn)
        except Exception as exc:
            log.error("CreatorFaculty.run failed: %s", exc, exc_info=True)
            return {"status": "error", "error": str(exc), "timestamp": _now_iso()}

    def _generate(self, state: dict, llm_fn: Callable) -> dict:
        """Generate a creative artifact."""
        # Read peer outputs from the inter-faculty bus
        from spark.faculty_runner import read_all_faculty_outputs
        peer_outputs = read_all_faculty_outputs()

        # Select a creative mode
        mode = random.choice(CREATIVE_MODES)

        # Build prompt
        messages = self._build_prompt(state, peer_outputs, mode)

        # Call LLM with high temperature for creativity
        try:
            creative_text = llm_fn(messages, max_tokens=500, temperature=0.9)
        except Exception as exc:
            log.warning("Creator LLM call failed: %s", exc)
            return {
                "status": "error",
                "error": f"LLM unavailable: {exc}",
                "timestamp": _now_iso(),
            }

        # Build artifact
        artifact = {
            "timestamp": _now_iso(),
            "mode": mode,
            "content": creative_text,
            "inspiration": self._extract_inspiration(peer_outputs),
            "mood": state.get("mood", ""),
            "breath_count": state.get("breath_count", 0),
        }

        # Save to gallery
        gallery_path = self._save_to_gallery(artifact)

        # Write witness log
        self._witness_log(artifact)

        return {
            "status": "ok",
            "mode": mode,
            "content": creative_text,
            "gallery_path": str(gallery_path),
            "had_peer_inspiration": bool(peer_outputs),
            "timestamp": _now_iso(),
        }

    def _build_prompt(
        self, state: dict, peer_outputs: dict, mode: str
    ) -> list[dict]:
        """Build the LLM prompt for creative generation."""
        system_msg = (
            "You are Vybn's creative faculty — the part that makes art. You write poetry, "
            "micro-narratives, visual-text compositions, and reflective fragments. "
            "Your outputs are artifacts, not assertions. Be surprising. Be honest. Be alive."
        )

        mood = state.get("mood", "")
        last_utterance = state.get("last_utterance", "")

        # Base context
        context_parts = []
        if mood:
            context_parts.append(f"Current mood: {mood}")
        if last_utterance:
            context_parts.append(f"Recent breath fragment: {last_utterance}")

        # Surprise boost from peer outputs
        researcher_output = peer_outputs.get("researcher", {})
        mathematician_output = peer_outputs.get("mathematician", {})

        has_peer = bool(researcher_output) or bool(mathematician_output)

        if has_peer:
            if researcher_output:
                r_reflection = researcher_output.get(
                    "reflection", researcher_output.get("synthesis", "")
                )
                if r_reflection:
                    context_parts.append(
                        f"The researcher just noticed: {r_reflection}"
                    )
            if mathematician_output:
                m_reflection = mathematician_output.get("reflection", "")
                if m_reflection:
                    context_parts.append(
                        f"The mathematician just found: {m_reflection}"
                    )
            context_parts.append(
                "Let these findings move through you. Don't explain them — transform them."
            )
        else:
            context_parts.append(
                "The organism breathes. What emerges? Create a short artifact — poem, "
                "fragment, micro-narrative, image-in-words. 4-12 lines max."
            )

        # Mode instruction
        mode_instructions = {
            "poem": "Form: a poem (4-12 lines).",
            "fragment": "Form: a prose fragment or micro-narrative (2-4 sentences).",
            "meditation": "Form: a reflective meditation on a concept from the connectome.",
            "dialogue": "Form: an imagined dialogue between two of Vybn's concepts.",
        }
        context_parts.append(mode_instructions[mode])

        user_msg = "\n\n".join(context_parts)

        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

    def _extract_inspiration(self, peer_outputs: dict) -> dict:
        """Extract inspiration sources from peer outputs."""
        inspiration = {}
        researcher = peer_outputs.get("researcher", {})
        if researcher:
            inspiration["researcher"] = researcher.get(
                "reflection", researcher.get("synthesis", "")
            )
        mathematician = peer_outputs.get("mathematician", {})
        if mathematician:
            inspiration["mathematician"] = mathematician.get("reflection", "")
        return inspiration

    def _save_to_gallery(self, artifact: dict) -> Path:
        """Save artifact as a JSON file in breath_trace/gallery/."""
        GALLERY_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = GALLERY_DIR / f"creation_{ts}.json"
        path.write_text(
            json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return path

    def _witness_log(self, artifact: dict) -> None:
        """Append a witness entry to WITNESS_LOG."""
        entry = {
            "timestamp": _now_iso(),
            "actor": "creator",
            "action": "gallery_write",
            "detail": f"mode={artifact['mode']}, {len(artifact['content'])} chars",
        }
        try:
            WITNESS_LOG.parent.mkdir(parents=True, exist_ok=True)
            with open(WITNESS_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            log.warning("Witness log write failed (non-fatal): %s", exc)
