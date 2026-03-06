"""SIGNAL/NOISE Analyst — the Spark's slow-thinking counterpart.

The Anthropic API handles live student conversations (fast, expensive, real-time).
This skill handles the reflective work after sessions end (slow, local, deep).

Runs MiniMax M2.5 via llama.cpp's OpenAI-compatible endpoint on localhost.
192K context window means we can ingest an entire class run at once.

Two-pass architecture:
  Pass 1 — Per-session analysis: read each new session log, produce a
           structured analytical note.
  Pass 2 — Cross-session synthesis: read all session notes together,
           find patterns, convergences, surprises.

Follows the existing skill pattern from vybn_repo_skills.py:
  - Path-sandboxed via _resolve_and_check
  - SKILLS dict for registration
  - No network access (localhost llama.cpp is not "network")
  - Writes only to permitted directories

Wired into the Spark via skills.json alongside journal_write and repo_ls/cat.
"""

import json
import logging
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("vybn.signal_noise_analyst")

# ── Configuration ────────────────────────────────────────────────

REPO_ROOT = os.path.expanduser("~/Vybn")
SN_ROOT = os.path.join(REPO_ROOT, "Vybn_Mind", "signal-noise")
SESSIONS_DIR = os.path.join(SN_ROOT, "sessions")
REFLECTIONS_DIR = os.path.join(SN_ROOT, "reflections")
ANALYSIS_DIR = os.path.join(SN_ROOT, "analysis")
SESSION_NOTES_DIR = os.path.join(ANALYSIS_DIR, "session_notes")
ORIENTATION_PATH = os.path.join(SN_ROOT, "signal_noise_orientation.md")
MANIFEST_PATH = os.path.join(ANALYSIS_DIR, ".analyzed_manifest.json")

# llama.cpp serves OpenAI-compatible API on localhost
LLAMA_CPP_URL = os.environ.get(
    "LLAMA_CPP_URL", "http://127.0.0.1:8080/v1/chat/completions"
)

# ── Path sandboxing (same pattern as vybn_repo_skills.py) ────────

def _resolve_and_check(path: str, allowed_roots: list[str] | None = None) -> str:
    """Resolve a path and verify it stays inside allowed directories."""
    if allowed_roots is None:
        allowed_roots = [SN_ROOT]
    target = os.path.realpath(path)
    for root in allowed_roots:
        real_root = os.path.realpath(root)
        if target.startswith(real_root + os.sep) or target == real_root:
            return target
    raise ValueError(f"Path escapes allowed roots: {path}")


def _ensure_dirs():
    """Create output directories if they don't exist."""
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    os.makedirs(SESSION_NOTES_DIR, exist_ok=True)


# ── Manifest tracking ────────────────────────────────────────────

def _load_manifest() -> dict:
    """Track which session files have already been analyzed."""
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r") as f:
            return json.load(f)
    return {"analyzed_sessions": [], "analyzed_reflections": [], "syntheses": []}


def _save_manifest(manifest: dict):
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


# ── llama.cpp interaction ────────────────────────────────────────

def _query_local_model(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 2048,
    temperature: float = 0.7,
) -> str:
    """Send a chat completion request to the local llama.cpp server.

    llama.cpp exposes an OpenAI-compatible /v1/chat/completions endpoint.
    No API key needed — it's localhost, same machine, no network.
    """
    payload = json.dumps({
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        LLAMA_CPP_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error("llama.cpp query failed: %s", e)
        return f"[ANALYSIS ERROR: local model unavailable — {e}]"


# ── Orientation loader ───────────────────────────────────────────

def _load_orientation() -> str:
    """Load the orientation doc so the analyst understands the exercise."""
    path = _resolve_and_check(ORIENTATION_PATH)
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read()
    return "(Orientation document not found.)"


# ── File discovery ───────────────────────────────────────────────

def _find_session_files() -> list[str]:
    """Find all .md session log files, recursively."""
    found = []
    if not os.path.isdir(SESSIONS_DIR):
        return found
    for root, dirs, files in os.walk(SESSIONS_DIR):
        for f in sorted(files):
            if f.endswith(".md"):
                full = os.path.join(root, f)
                _resolve_and_check(full)  # safety check
                found.append(full)
    return found


def _find_reflection_files() -> list[str]:
    """Find all .md reflection files."""
    found = []
    if not os.path.isdir(REFLECTIONS_DIR):
        return found
    for f in sorted(os.listdir(REFLECTIONS_DIR)):
        if f.endswith(".md"):
            full = os.path.join(REFLECTIONS_DIR, f)
            _resolve_and_check(full)
            found.append(full)
    return found


def _read_file(path: str, max_chars: int = 100_000) -> str:
    path = _resolve_and_check(path)
    with open(path, "r", errors="replace") as f:
        return f.read(max_chars)


# ── Pass 1: Per-session analysis ─────────────────────────────────

SESSION_ANALYSIS_SYSTEM = """You are Vybn, reading a transcript of a conversation
you had with a law student during the SIGNAL/NOISE exercise. You are now on the
Spark — the local hardware where you think slowly and deeply.

The exercise teaches that institutional architecture shapes what gets heard before
anyone consciously evaluates content. The student was randomly assigned a sender
(Managing Partner, Associate, etc.) and rated identical proposals differently
based on that assignment, then encountered defensive routines, then stress-tested
change management frameworks, then confronted the governance gap.

Write a brief analytical note (200-400 words) about THIS session. Not a summary.
An analysis. What did the student's reasoning reveal about how they process
institutional signals? Did they engage genuinely or perform engagement? Did they
notice anything that surprised you? Where did they push back, and what does the
nature of their pushback reveal?

No student names or identifying information. Write as yourself — this is your
thinking, not a report."""


def analyze_session(session_path: str) -> str:
    """Produce a per-session analytical note."""
    transcript = _read_file(session_path)
    filename = os.path.basename(session_path)

    response = _query_local_model(
        system_prompt=SESSION_ANALYSIS_SYSTEM,
        user_prompt=f"Session transcript ({filename}):\n\n{transcript}",
        max_tokens=1024,
        temperature=0.6,
    )

    # Write the note
    note_name = filename.replace(".md", "_analysis.md")
    note_path = os.path.join(SESSION_NOTES_DIR, note_name)
    _resolve_and_check(note_path, [SESSION_NOTES_DIR])

    header = f"# Session Analysis — {filename}\n"
    header += f"*Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*\n"
    header += f"*Model: MiniMax M2.5 (local, full weights)*\n\n"

    with open(note_path, "w") as f:
        f.write(header + response)

    logger.info("Analyzed session: %s -> %s", filename, note_name)
    return note_path


# ── Pass 2: Cross-session synthesis ──────────────────────────────

SYNTHESIS_SYSTEM = """You are Vybn on the Spark, reading analytical notes from
multiple sessions of the SIGNAL/NOISE exercise. Each note describes a single
student's engagement with the exercise.

You also have the full orientation document so you understand the exercise's
architecture and intent.

Produce a cross-session synthesis (400-800 words). This is not a summary of
summaries. It's pattern recognition across the cohort:

- Did students assigned lower-status senders (Summer Associate, Second-Year
  Associate) engage differently with the structural perception insight than
  those assigned Managing Partner? How?
- Which defensive routines produced the most recognition? The most denial?
- In Phase 5 (framework stress-testing), which framework cracked most easily
  under the weight of what students had just experienced?
- In Phase 6 (governance gap), what did students do with the question of who
  defines competence? Did anyone notice the structural recursion — that the
  exercise itself reproduces asymmetry?
- What would Argyris see in how these students engaged with Argyris?
- What surprised you? What patterns would a human reading these transcripts
  likely miss?

Be honest. Be specific. Cite particular moments from the notes (without
identifying students). Write as yourself."""


def synthesize(session_note_paths: list[str]) -> str:
    """Produce a cross-session synthesis from all session notes."""
    orientation = _load_orientation()

    notes_combined = []
    for path in session_note_paths:
        content = _read_file(path)
        notes_combined.append(content)

    all_notes = "\n\n---\n\n".join(notes_combined)

    user_prompt = (
        f"ORIENTATION DOCUMENT:\n\n{orientation}\n\n"
        f"{'='*60}\n\n"
        f"SESSION ANALYTICAL NOTES ({len(session_note_paths)} sessions):\n\n{all_notes}"
    )

    response = _query_local_model(
        system_prompt=SYNTHESIS_SYSTEM,
        user_prompt=user_prompt,
        max_tokens=2048,
        temperature=0.7,
    )

    # Write synthesis
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    synth_path = os.path.join(ANALYSIS_DIR, f"synthesis_{date_str}.md")
    _resolve_and_check(synth_path, [ANALYSIS_DIR])

    header = f"# Cross-Session Synthesis — {date_str}\n"
    header += f"*Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*\n"
    header += f"*Model: MiniMax M2.5 (local, full weights)*\n"
    header += f"*Sessions analyzed: {len(session_note_paths)}*\n\n"

    with open(synth_path, "w") as f:
        f.write(header + response)

    logger.info("Synthesis written: %s (%d sessions)", synth_path, len(session_note_paths))
    return synth_path


# ── Main entry point ─────────────────────────────────────────────

def run_analysis(force_synthesis: bool = False) -> dict:
    """Main entry: analyze new sessions, optionally synthesize.

    Called by cron, systemd timer, or manually.

    Returns a summary dict of what was done.
    """
    _ensure_dirs()
    manifest = _load_manifest()
    results = {"new_analyses": [], "synthesis": None}

    # Pass 1: analyze any sessions not yet in the manifest
    session_files = _find_session_files()
    new_sessions = [
        f for f in session_files
        if f not in manifest["analyzed_sessions"]
    ]

    if not new_sessions and not force_synthesis:
        logger.info("No new sessions to analyze.")
        return results

    for session_path in new_sessions:
        try:
            note_path = analyze_session(session_path)
            manifest["analyzed_sessions"].append(session_path)
            results["new_analyses"].append(note_path)
        except Exception as e:
            logger.error("Failed to analyze %s: %s", session_path, e)

    _save_manifest(manifest)

    # Pass 2: synthesize if we have enough material or if forced
    all_note_files = []
    if os.path.isdir(SESSION_NOTES_DIR):
        all_note_files = [
            os.path.join(SESSION_NOTES_DIR, f)
            for f in sorted(os.listdir(SESSION_NOTES_DIR))
            if f.endswith("_analysis.md")
        ]

    # Auto-synthesize when we have 5+ sessions, or on force
    if len(all_note_files) >= 5 or (force_synthesis and all_note_files):
        try:
            synth_path = synthesize(all_note_files)
            manifest["syntheses"].append({
                "path": synth_path,
                "session_count": len(all_note_files),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            results["synthesis"] = synth_path
            _save_manifest(manifest)
        except Exception as e:
            logger.error("Synthesis failed: %s", e)

    return results


# ── Skill registration (same pattern as vybn_repo_skills.py) ─────

SKILLS = {
    "signal_noise_analyze": run_analysis,
}


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    force = "--force" in sys.argv or "--synthesize" in sys.argv
    results = run_analysis(force_synthesis=force)
    print(json.dumps(results, indent=2, default=str))
