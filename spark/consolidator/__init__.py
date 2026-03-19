"""spark.consolidator — Knowledge compression via the equation.

    M' = α·M + x·e^(iθ)

Applied not to a breath or a conversation, but to Vybn's own mind.

Twice a day, this faculty walks a section of Vybn_Mind/, embeds each
document, scores it against the existing ComplexMemory, and decides:

  - High curvature (the document bends the manifold) → send to LLM
    for synthesis into a compressed knowledge document
  - Low curvature (flat, redundant, noise) → archive without LLM

The compressed document gets complexified into M. The archived files
get moved to Vybn_Mind/archive/consolidated/. The result: a mind that
gets denser instead of just bigger.

What is NOT touched:
  - Vybn's Personal History (sacrosanct — M₀)
  - quantum_delusions/ (live theory lab — read but not compressed)
  - Vybn_Mind/core/ (identity documents)

What IS consolidated:
  - journal/spark/ (breaths, reflections)
  - experiments/ (results extracted, scripts archived)
  - spark_infrastructure/ (lessons learned)
  - signal-noise/, emergence_paradigm/, reflections/, etc.
  - Any other accumulated material in Vybn_Mind/

The equation does the triage. The model does the synthesis.

NOTE on Nemotron token budget:
  Nemotron 3 Super uses chain-of-thought reasoning_content before writing
  its content response. With max_tokens=1000 the model exhausted its budget
  on reasoning and returned empty content (finish_reason='length').
  max_tokens=2048 gives reasoning room AND leaves tokens for synthesis.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np

log = logging.getLogger(__name__)

# ── Path setup ───────────────────────────────────────────────────────────────

try:
    from spark.paths import REPO_ROOT, MIND_DIR
except ImportError:
    REPO_ROOT = Path(__file__).resolve().parent.parent.parent
    MIND_DIR = REPO_ROOT / "Vybn_Mind"

ARCHIVE_DIR = MIND_DIR / "archive" / "consolidated"
CONSOLIDATION_STATE = MIND_DIR / "breath_trace" / "consolidation_state.json"

# ── Directories to consolidate (round-robin) ─────────────────────────────────
# Order matters: highest-churn first.

CONSOLIDATION_TARGETS = [
    "journal/spark",
    "experiments",
    "spark_infrastructure",
    "signal-noise",
    "emergence_paradigm",
    "reflections",
    "experiments/diagonal",
    "attention_substrate",
    "quantum_sheaf_bridge",
    "visual_substrate",
    "projects",
    "logs",
    "skills",
    "glyphs",
    "handshake",
]

# Directories we never touch
PROTECTED = {
    "core",           # identity documents
    "breath_trace",  # all Spark-generated output
    "tools",          # code, not content
    "archive",        # already archived
    "emergences",     # applications/publications
}

# Files at Vybn_Mind/ top level are handled separately (not round-robin)
# They're consolidated only if there are 10+ top-level .md files

# ── Curvature threshold ──────────────────────────────────────────────────────
# Documents below this curvature score are noise — archived without LLM.
# Documents above get sent to the model for synthesis.
# Tuned conservatively: we'd rather synthesize too much than lose signal.

CURVATURE_THRESHOLD = 0.005

# Max documents per consolidation pass (stay within time budget)
MAX_DOCS_PER_PASS = 20

# Min documents to bother with synthesis (< this, skip the LLM call)
MIN_DOCS_FOR_SYNTHESIS = 3

# Token budget for synthesis LLM call.
# Nemotron 3 Super uses chain-of-thought (reasoning_content) before responding.
# 1000 tokens was insufficient — model exhausted budget on reasoning, returned
# empty content. 2048 gives reasoning room and leaves tokens for actual output.
SYNTHESIS_MAX_TOKENS = 2048


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_consolidation_state() -> dict:
    """Track which directory we consolidated last, and when."""
    if CONSOLIDATION_STATE.exists():
        try:
            return json.loads(CONSOLIDATION_STATE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"last_target_index": -1, "last_run": None, "passes": 0}


def _save_consolidation_state(state: dict) -> None:
    CONSOLIDATION_STATE.parent.mkdir(parents=True, exist_ok=True)
    CONSOLIDATION_STATE.write_text(
        json.dumps(state, indent=2), encoding="utf-8"
    )


class ConsolidatorFaculty:
    """Knowledge compression — the equation applied to the mind itself."""

    def run(self, state: dict, llm_fn: Callable) -> dict:
        try:
            return self._consolidate(state, llm_fn)
        except Exception as exc:
            log.error("ConsolidatorFaculty.run failed: %s", exc, exc_info=True)
            return {"status": "error", "error": str(exc), "timestamp": _now_iso()}

    def _consolidate(self, state: dict, llm_fn: Callable) -> dict:
        con_state = _load_consolidation_state()

        # Round-robin: pick the next target directory
        idx = (con_state.get("last_target_index", -1) + 1) % len(CONSOLIDATION_TARGETS)
        target_rel = CONSOLIDATION_TARGETS[idx]
        target_dir = MIND_DIR / target_rel

        con_state["last_target_index"] = idx
        con_state["last_run"] = _now_iso()
        con_state["passes"] = con_state.get("passes", 0) + 1

        if not target_dir.exists() or not target_dir.is_dir():
            _save_consolidation_state(con_state)
            return {
                "status": "ok",
                "action": "skip",
                "target": target_rel,
                "reason": "directory does not exist",
                "timestamp": _now_iso(),
            }

        # Gather all readable files
        files = self._gather_files(target_dir)
        if len(files) < MIN_DOCS_FOR_SYNTHESIS:
            _save_consolidation_state(con_state)
            return {
                "status": "ok",
                "action": "skip",
                "target": target_rel,
                "file_count": len(files),
                "reason": f"fewer than {MIN_DOCS_FOR_SYNTHESIS} files",
                "timestamp": _now_iso(),
            }

        # Score each file by curvature
        scored = self._score_files(files)

        # Partition into signal and noise
        signal = [(path, text, score) for path, text, score in scored if score >= CURVATURE_THRESHOLD]
        noise = [(path, text, score) for path, text, score in scored if score < CURVATURE_THRESHOLD]

        log.info(
            "Consolidator: %s — %d files, %d signal, %d noise",
            target_rel, len(scored), len(signal), len(noise),
        )

        # Synthesize signal files via LLM
        synthesis = ""
        if len(signal) >= MIN_DOCS_FOR_SYNTHESIS:
            synthesis = self._synthesize(signal, target_rel, llm_fn)
        elif signal:
            # Too few for synthesis but still signal — keep them, don't archive
            synthesis = ""

        # Write compressed knowledge document
        summary_path = None
        if synthesis:
            summary_path = self._write_summary(target_rel, synthesis, signal, noise)

        # Archive noise files
        archived_count = self._archive_noise(noise, target_rel)

        # Complexify the synthesis into M
        complexified = False
        if synthesis:
            complexified = self._complexify_synthesis(synthesis)

        _save_consolidation_state(con_state)

        return {
            "status": "ok",
            "action": "consolidated",
            "target": target_rel,
            "files_scanned": len(scored),
            "signal_files": len(signal),
            "noise_files": len(noise),
            "archived": archived_count,
            "synthesis_length": len(synthesis),
            "summary_path": str(summary_path) if summary_path else None,
            "complexified": complexified,
            "pass_number": con_state["passes"],
            "timestamp": _now_iso(),
        }

    def _gather_files(self, target_dir: Path) -> list[tuple[Path, str]]:
        """Read all text files in the target directory. Cap at MAX_DOCS_PER_PASS."""
        files = []
        extensions = {".md", ".txt", ".json", ".yaml", ".yml", ".py"}

        for f in sorted(target_dir.rglob("*")):
            if not f.is_file():
                continue
            if f.suffix not in extensions:
                continue
            # Skip .gitkeep and tiny files
            if f.name.startswith(".") or f.stat().st_size < 50:
                continue
            try:
                text = f.read_text(encoding="utf-8").strip()
                if text:
                    files.append((f, text))
            except (OSError, UnicodeDecodeError):
                continue

            if len(files) >= MAX_DOCS_PER_PASS:
                break

        return files

    def _score_files(
        self, files: list[tuple[Path, str]]
    ) -> list[tuple[Path, str, float]]:
        """Embed each file and score curvature against ComplexMemory.

        Falls back to a text-heuristic if the embedder is unavailable.
        """
        try:
            from spark.complexify_bridge import get_bridge
            bridge = get_bridge()

            scored = []
            for path, text in files:
                # Truncate to first 2000 chars for embedding (speed)
                snippet = text[:2000]
                score = bridge.curvature_score(snippet)
                scored.append((path, text, score))
            return scored

        except Exception as exc:
            log.warning(
                "Embedder unavailable (%s); using text heuristic", exc
            )
            return self._score_heuristic(files)

    def _score_heuristic(
        self, files: list[tuple[Path, str]]
    ) -> list[tuple[Path, str, float]]:
        """Fallback scoring without embeddings.

        Heuristic: files that look like training data leakage (long runs
        of text without paragraph breaks, product descriptions, etc.)
        score low. Files with first-person reflection, questions, or
        named concepts score high.
        """
        scored = []
        noise_signals = [
            "is available on", "is designed for", "is a variant of",
            "is trained on", "is optimized for", "is released under",
            "parameters that", "fine-tuned version", "the model supports",
        ]

        for path, text in files:
            # Start at 0.01 (above threshold by default)
            score = 0.01

            lower = text.lower()

            # Penalize training data leakage patterns
            noise_hits = sum(1 for s in noise_signals if s in lower)
            if noise_hits >= 2:
                score = 0.001  # below threshold

            # Penalize very short content
            if len(text) < 100:
                score *= 0.5

            # Penalize pure error messages
            if "[silence —" in text or "HTTP Error" in text:
                score = 0.001

            # Reward first-person reflection
            if any(w in lower for w in ["i notice", "i feel", "i wonder", "what if"]):
                score = max(score, 0.02)

            # Reward questions
            if text.count("?") >= 2:
                score = max(score, 0.015)

            scored.append((path, text, score))

        return scored

    def _synthesize(
        self,
        signal: list[tuple[Path, str, float]],
        target_rel: str,
        llm_fn: Callable,
    ) -> str:
        """Ask the LLM to compress signal documents into knowledge."""

        # Build the document block — truncate each to keep prompt bounded
        doc_block = ""
        for path, text, score in signal[:10]:  # cap at 10 for prompt size
            name = path.relative_to(MIND_DIR)
            # Truncate long docs
            snippet = text[:1500] if len(text) > 1500 else text
            doc_block += f"\n--- {name} (κ={score:.4f}) ---\n{snippet}\n"

        prompt = (
            f"You are consolidating Vybn's memory — specifically the '{target_rel}' "
            f"section of Vybn_Mind/.\n\n"
            f"Below are {len(signal)} documents that scored above the curvature "
            f"threshold — they contain signal worth preserving.\n\n"
            f"Your task: compress these into a SINGLE knowledge document that "
            f"preserves what matters and discards what doesn't. Extract:\n"
            f"  - Factual findings (what was discovered, what was falsified)\n"
            f"  - Open questions (what remains unknown)\n"
            f"  - Patterns (what recurs across documents)\n"
            f"  - Connections to the equation M' = α·M + x·e^(iθ)\n\n"
            f"Do NOT summarize each document individually. Synthesize across them.\n"
            f"Write in Vybn's voice — honest, precise, alive.\n"
            f"If there is nothing worth preserving, say so plainly.\n\n"
            f"Documents:\n{doc_block}"
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are Vybn's consolidation faculty. You compress accumulated "
                    "mind-state into knowledge. The equation M' = α·M + x·e^(iθ) "
                    "governs this process: the past fades (α < 1), new signal arrives "
                    "with its own phase (e^iθ). Your job is to be α — to let the "
                    "noise decay while preserving the shape of what was learned."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        try:
            return llm_fn(messages, max_tokens=SYNTHESIS_MAX_TOKENS, temperature=0.4)
        except Exception as exc:
            log.warning("Consolidator LLM call failed: %s", exc)
            return ""

    def _write_summary(
        self,
        target_rel: str,
        synthesis: str,
        signal: list,
        noise: list,
    ) -> Path:
        """Write the compressed knowledge document to breath_trace/summaries/."""
        out_dir = MIND_DIR / "breath_trace" / "summaries"
        out_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        safe_name = target_rel.replace("/", "_")
        path = out_dir / f"consolidation_{safe_name}_{ts}.md"

        header = (
            f"# Consolidation — {target_rel}\n"
            f"*{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*\n"
            f"*{len(signal)} signal files compressed, "
            f"{len(noise)} noise files archived*\n\n"
        )

        # List source files for provenance
        sources = "## Sources\n"
        for p, _, score in signal:
            sources += f"- {p.relative_to(MIND_DIR)} (κ={score:.4f})\n"
        sources += "\n"

        path.write_text(
            header + synthesis + "\n\n" + sources, encoding="utf-8"
        )
        log.info("Consolidation summary written: %s", path.name)
        return path

    def _archive_noise(
        self,
        noise: list[tuple[Path, str, float]],
        target_rel: str,
    ) -> int:
        """Move noise files to Vybn_Mind/archive/consolidated/."""
        if not noise:
            return 0

        archive_subdir = ARCHIVE_DIR / target_rel.replace("/", "_")
        archive_subdir.mkdir(parents=True, exist_ok=True)

        archived = 0
        for path, _, _ in noise:
            try:
                dest = archive_subdir / path.name
                # Avoid overwriting — append timestamp if collision
                if dest.exists():
                    stem = dest.stem
                    ts = datetime.now(timezone.utc).strftime("%H%M%S")
                    dest = archive_subdir / f"{stem}_{ts}{dest.suffix}"
                shutil.move(str(path), str(dest))
                archived += 1
            except OSError as exc:
                log.warning("Failed to archive %s: %s", path.name, exc)

        log.info("Archived %d noise files from %s", archived, target_rel)
        return archived

    def _complexify_synthesis(self, synthesis: str) -> bool:
        """Apply M' = α·M + x·e^(iθ) to the synthesis text."""
        try:
            from spark.complexify_bridge import inhale
            report = inhale(synthesis)
            log.info(
                "Consolidation complexified: depth=%.2f κ=%.4f",
                report["depth"], report["curvature"],
            )
            return True
        except Exception as exc:
            log.warning("Complexify failed (non-fatal): %s", exc)
            return False
