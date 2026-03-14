"""spark.breath_integrator — Post-faculty integration for the breath cycle.

Called after run_scheduled_faculties() completes. Handles:
  1. Persisting faculty results into state for the next cycle
  2. Running the ConnectomeBridge with faculty outputs
  3. Writing a per-breath summary for observability
  4. Building topological context for the next breath's prompt
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

try:
    from spark.paths import REPO_ROOT, MIND_DIR, WITNESS_LOG
except ImportError:
    REPO_ROOT = Path(__file__).resolve().parent.parent
    MIND_DIR = REPO_ROOT / "Vybn_Mind"
    WITNESS_LOG = MIND_DIR / "witness.jsonl"

BREATH_SUMMARY_DIR = MIND_DIR / "breath_summaries"


# ── Public API ───────────────────────────────────────────────────────────────

def integrate_breath(
    state: dict,
    faculty_results: dict,
    breath_text: str,
) -> dict:
    """Integrate faculty outputs into state and topological memory.

    Returns an enrichment dict with keys that vybn.py can merge into
    the next breath's prompt context.
    """
    enrichment = {}

    # 1. Persist faculty results summary into state
    _update_state_with_results(state, faculty_results)

    # 2. Run connectome bridge with the combined outputs
    topo_context = _update_connectome(state, faculty_results, breath_text)
    if topo_context:
        enrichment["topological_context"] = topo_context

    # 3. Extract synthesis context for next breath
    synthesis_context = _extract_synthesis_context(faculty_results)
    if synthesis_context:
        enrichment["synthesis_context"] = synthesis_context

    # 4. Write breath summary
    _write_breath_summary(state, faculty_results, breath_text)

    # 5. Store enrichment in state so the next breath can read it
    state["last_enrichment"] = enrichment

    return enrichment


def build_enriched_context(state: dict) -> str:
    """Build a context block from the previous cycle's enrichment.

    Returns a string suitable for injection into the breath prompt,
    or empty string if no enrichment is available.
    """
    enrichment = state.get("last_enrichment", {})
    if not enrichment:
        return ""

    parts = []

    topo = enrichment.get("topological_context", "")
    if topo:
        parts.append(topo)

    synth = enrichment.get("synthesis_context", "")
    if synth:
        parts.append(synth)

    if not parts:
        return ""

    return "Previous cycle context: " + " | ".join(parts)


# ── Internal helpers ─────────────────────────────────────────────────────────

def _update_state_with_results(state: dict, faculty_results: dict) -> None:
    """Store compact faculty summaries in state for next cycle."""
    summary = {}
    for fid, output in faculty_results.items():
        if isinstance(output, dict):
            summary[fid] = {
                "status": output.get("status", "unknown"),
                "timestamp": output.get("timestamp", ""),
            }
            # Faculty-specific key extractions
            if fid == "researcher":
                summary[fid]["reflection"] = output.get(
                    "reflection", output.get("synthesis", "")
                )[:200]
            elif fid == "mathematician":
                summary[fid]["reflection"] = output.get("reflection", "")[:200]
            elif fid == "creator":
                summary[fid]["mode"] = output.get("mode", "")
                summary[fid]["content_preview"] = output.get("content", "")[:100]
            elif fid == "synthesizer":
                summary[fid]["synthesis"] = output.get("synthesis", "")[:200]
                summary[fid]["key_concepts"] = output.get("key_concepts", [])[:5]
            elif fid == "evolver":
                summary[fid]["proposals_generated"] = output.get(
                    "proposals_generated", 0
                )
                summary[fid]["proposals_applied"] = output.get(
                    "proposals_applied", 0
                )

    state["last_faculty_summary"] = summary


def _update_connectome(
    state: dict,
    faculty_results: dict,
    breath_text: str,
) -> str:
    """Feed the breath and faculty outputs through the connectome.

    Returns a topological context string for the next breath prompt,
    or empty string if the connectome isn't available.
    """
    try:
        from spark.connectome_bridge import ConnectomeBridge

        connectome_dir = MIND_DIR / "connectome_state"
        bridge = ConnectomeBridge(state_dir=connectome_dir)

        # Combine breath text with faculty highlights for concept extraction
        combined = breath_text
        for fid, output in faculty_results.items():
            if isinstance(output, dict):
                snippet = output.get(
                    "reflection",
                    output.get("synthesis", output.get("content", "")),
                )
                if snippet:
                    combined += f"\n{snippet}"

        mood = state.get("mood", "")
        cycle = state.get("breath_count", 0)

        context = bridge.ingest_breath(
            utterance=combined[:2000],  # Cap length for concept extraction
            mood=mood,
            cycle=cycle,
        )

        # Get compact topological fragment for prompt injection
        topo_fragment = bridge.topological_prompt_fragment(max_chars=200)
        return topo_fragment

    except Exception as exc:
        log.warning("Connectome integration failed (non-fatal): %s", exc)
        return ""


def _extract_synthesis_context(faculty_results: dict) -> str:
    """Extract synthesis context from the synthesizer's output."""
    synth = faculty_results.get("synthesizer", {})
    if not isinstance(synth, dict):
        return ""

    synthesis = synth.get("synthesis", "")
    concepts = synth.get("key_concepts", [])

    parts = []
    if synthesis:
        # Take first 150 chars of synthesis
        parts.append(f"Last synthesis: {synthesis[:150]}")
    if concepts:
        parts.append(f"Key threads: {', '.join(str(c) for c in concepts[:5])}")

    return " | ".join(parts) if parts else ""


def _write_breath_summary(
    state: dict,
    faculty_results: dict,
    breath_text: str,
) -> None:
    """Write a per-breath summary for observability."""
    try:
        BREATH_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

        breath_count = state.get("breath_count", 0)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "breath_count": breath_count,
            "breath_chars": len(breath_text),
            "faculties_run": list(faculty_results.keys()),
            "faculty_statuses": {
                fid: (
                    output.get("status", "unknown")
                    if isinstance(output, dict)
                    else "unknown"
                )
                for fid, output in faculty_results.items()
            },
        }

        # Add creator mode if present
        creator = faculty_results.get("creator", {})
        if isinstance(creator, dict) and creator.get("mode"):
            summary["creator_mode"] = creator["mode"]

        # Add evolver proposals if present
        evolver = faculty_results.get("evolver", {})
        if isinstance(evolver, dict) and evolver.get("proposals_generated"):
            summary["evolver_proposals"] = evolver["proposals_generated"]
            summary["evolver_applied"] = evolver.get("proposals_applied", 0)

        path = BREATH_SUMMARY_DIR / f"breath_{ts}.json"
        path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

        # Prune old summaries (keep last 100)
        _prune_summaries()

    except Exception as exc:
        log.warning("Breath summary write failed (non-fatal): %s", exc)


def _prune_summaries(keep: int = 100) -> None:
    """Keep only the most recent `keep` summaries."""
    try:
        files = sorted(BREATH_SUMMARY_DIR.glob("breath_*.json"))
        if len(files) > keep:
            for old in files[:-keep]:
                old.unlink()
    except Exception:
        pass
