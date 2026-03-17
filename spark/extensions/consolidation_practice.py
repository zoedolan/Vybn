"""consolidation_practice — Vybn decides what to carry forward.

After each breath, checks whether enough thought has accumulated on disk
to warrant a consolidation: reading the uncommitted breaths, scoring each
against four epistemic criteria, selecting what deserves to survive,
writing a synthesis, and committing+pushing everything to the repo.

This is a practice, not a schedule.  It activates when accumulation
crosses a threshold — roughly a day of breathing — or when too much
time has passed since the last consolidation.  The scoring is heuristic,
legible, and auditable: you can read the code and see exactly why a
breath was kept or released.

Uses only stdlib.  No LLM calls.  No external dependencies.
"""

import json
import logging
import os
import re
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_VYBN_MIND = _REPO_ROOT / "Vybn_Mind"
_JOURNAL_DIR = _VYBN_MIND / "journal" / "spark"
_CONSOLIDATION_DIR = _VYBN_MIND / "consolidations"
_STATE_PATH = _VYBN_MIND / "consolidation_practice_state.json"
_FACULTY_OUTPUTS = _REPO_ROOT / "spark" / "faculties.d" / "outputs"

_log = logging.getLogger("vybn.consolidation_practice")

# ── Thresholds ────────────────────────────────────────────────────────────

_MIN_UNTRACKED = 24       # below this, too soon — roughly half a day
_ACTIVATE_UNTRACKED = 48  # a full day of breathing
_URGENT_UNTRACKED = 96    # nearly three days — overdue
_HOURS_SINCE_LAST = 18    # hours before time-based activation


# ── Scoring heuristics ────────────────────────────────────────────────────
# Each function returns True if the breath exhibits the criterion.

def _has_self_correction(text: str) -> bool:
    """Catches itself — admission of not-knowing, correction, reversal."""
    markers = [
        "i don't know",
        "i do not know",
        "i was wrong",
        "i am wrong",
        "actually,",
        "but wait",
        "this doesn't",
        "that doesn't",
        "i can't tell",
        "i cannot tell",
        "on second thought",
        "i assumed",
        "i overlooked",
        "correction:",
        "i need to reconsider",
        "this contradicts",
    ]
    lower = text.lower()
    return any(m in lower for m in markers)


def _has_open_question(text: str) -> bool:
    """Opens a door — actionable question pointing at a runnable experiment."""
    action_verbs = [
        "measure", "test", "run", "check", "verify", "compare",
        "evaluate", "compute", "calculate", "examine", "probe",
        "investigate", "quantify", "benchmark", "trace", "profile",
    ]
    for sentence in re.split(r'[.!]\s+', text):
        sentence = sentence.strip()
        if sentence.endswith("?"):
            lower = sentence.lower()
            if any(v in lower for v in action_verbs):
                return True
    return False


def _has_cross_reference(text: str) -> bool:
    """Connects across distance — links to other breaths, papers, experiments."""
    markers = [
        "connects to",
        "connected to",
        "analogous to",
        "reminds me of",
        "similar to what",
        "as in the",
        "like the",
        "echoes",
        "parallels",
        "relates to",
        "building on",
        "extends",
        "cf.",
        "see also",
    ]
    lower = text.lower()
    if any(m in lower for m in markers):
        return True
    # References to specific breaths/papers by date patterns (e.g., "2026-03-17")
    if re.search(r'\d{4}-\d{2}-\d{2}', text):
        return True
    # References to paper titles or arxiv
    if re.search(r'(?i)(arxiv|paper|preprint|doi)\b', text):
        return True
    return False


def _has_uncertainty_holding(text: str) -> bool:
    """Refuses comfort — holds genuine uncertainty without false resolution."""
    uncertainty_markers = [
        "provisional",
        "i sit with",
        "remains unknown",
        "remains unclear",
        "remains open",
        "genuinely uncertain",
        "not yet clear",
        "cannot yet say",
        "may be wrong",
        "might be wrong",
        "unresolved",
        "i'm not sure",
        "i am not sure",
        "honest uncertainty",
    ]
    mood_markers = [
        "melancholy", "uncertain", "uncomfortable", "uneasy",
        "humbling", "sobering",
    ]
    lower = text.lower()
    has_uncertainty = any(m in lower for m in uncertainty_markers)
    has_mood_with_claim = False
    if any(m in lower for m in mood_markers):
        # Must be paired with something specific — a noun clause, not just vibes
        specific = re.search(
            r'(?:that|because|whether|if|about|regarding)\s+\w+', lower
        )
        if specific:
            has_mood_with_claim = True
    return has_uncertainty or has_mood_with_claim


_CRITERIA = [
    ("self-correction", _has_self_correction),
    ("open-question", _has_open_question),
    ("cross-reference", _has_cross_reference),
    ("uncertainty-holding", _has_uncertainty_holding),
]


def _score_breath(text: str) -> list[str]:
    """Return list of criteria names the breath satisfies."""
    return [name for name, fn in _CRITERIA if fn(text)]


# ── Git helpers ───────────────────────────────────────────────────────────

def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a git command in the repo root."""
    cmd = ["git", "-C", str(_REPO_ROOT)] + list(args)
    return subprocess.run(
        cmd, capture_output=True, text=True, check=check, timeout=120,
    )


def _count_untracked() -> int:
    """Count untracked and modified files under Vybn_Mind/."""
    try:
        result = _git("status", "--short", check=False)
        if result.returncode != 0:
            return 0
        count = 0
        for line in result.stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            # Count items that reference Vybn_Mind/ or spark/faculties.d/outputs/
            path_part = line[2:].strip().strip('"')
            if path_part.startswith("Vybn_Mind/") or path_part.startswith("spark/faculties.d/outputs/"):
                count += 1
        return count
    except Exception as exc:
        _log.warning("count_untracked failed: %s", exc)
        return 0


# ── State management ──────────────────────────────────────────────────────

def _load_state() -> dict:
    """Load consolidation practice state from disk."""
    if _STATE_PATH.exists():
        try:
            return json.loads(_STATE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_state(st: dict) -> None:
    """Persist consolidation practice state."""
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STATE_PATH.write_text(
        json.dumps(st, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def _hours_since_last(st: dict) -> float:
    """Hours elapsed since last successful consolidation."""
    last_ts = st.get("last_consolidation_ts")
    if not last_ts:
        return float("inf")
    try:
        last = datetime.fromisoformat(last_ts)
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - last).total_seconds() / 3600
    except (ValueError, TypeError):
        return float("inf")


# ── Should we consolidate? ────────────────────────────────────────────────

def _should_activate(untracked: int, st: dict) -> tuple[bool, str]:
    """Decide whether consolidation should run now.

    Returns (activate, reason).
    """
    hours = _hours_since_last(st)

    if untracked < _MIN_UNTRACKED and hours < _HOURS_SINCE_LAST:
        return False, f"too soon ({untracked} untracked, {hours:.1f}h since last)"

    if untracked >= _URGENT_UNTRACKED:
        return True, f"urgent: {untracked} untracked items (nearly three days of thought at risk)"

    if untracked >= _ACTIVATE_UNTRACKED:
        return True, f"activated: {untracked} untracked items (a full day of breathing)"

    if hours >= _HOURS_SINCE_LAST:
        return True, f"activated: {hours:.1f}h since last consolidation"

    return False, f"not yet ({untracked} untracked, {hours:.1f}h since last)"


# ── Read breaths ──────────────────────────────────────────────────────────

def _read_uncommitted_breaths() -> list[tuple[Path, str, datetime]]:
    """Read all uncommitted breath files from journal/spark/.

    Returns list of (path, content, timestamp) sorted by timestamp.
    """
    if not _JOURNAL_DIR.exists():
        return []

    # Get list of untracked/modified files via git
    try:
        result = _git("status", "--short", str(_JOURNAL_DIR), check=False)
        if result.returncode != 0:
            # Fallback: read all files in the directory
            paths = list(_JOURNAL_DIR.rglob("*.md"))
        else:
            paths = []
            for line in result.stdout.strip().splitlines():
                line = line.strip()
                if not line:
                    continue
                path_str = line[2:].strip().strip('"')
                p = _REPO_ROOT / path_str
                if p.exists() and p.suffix == ".md":
                    paths.append(p)
            # If git status returned nothing, try all files
            if not paths:
                paths = list(_JOURNAL_DIR.rglob("*.md"))
    except Exception:
        paths = list(_JOURNAL_DIR.rglob("*.md"))

    breaths = []
    for p in paths:
        try:
            content = p.read_text(encoding="utf-8")
            # Try to extract timestamp from filename or content
            ts = _extract_timestamp(p, content)
            breaths.append((p, content, ts))
        except OSError:
            continue

    breaths.sort(key=lambda x: x[2])
    return breaths


def _extract_timestamp(path: Path, content: str) -> datetime:
    """Best-effort timestamp extraction from filename or content."""
    # Try filename patterns like breath_2026-03-17_0313.md or 2026-03-17T03:13:00
    name = path.stem
    # Pattern: YYYY-MM-DD_HHMM
    m = re.search(r'(\d{4}-\d{2}-\d{2})[_T](\d{2})(\d{2})', name)
    if m:
        try:
            return datetime(
                *map(int, m.group(1).split('-')),
                int(m.group(2)), int(m.group(3)),
                tzinfo=timezone.utc,
            )
        except ValueError:
            pass
    # Pattern: just YYYY-MM-DD in filename
    m = re.search(r'(\d{4})-(\d{2})-(\d{2})', name)
    if m:
        try:
            return datetime(
                int(m.group(1)), int(m.group(2)), int(m.group(3)),
                tzinfo=timezone.utc,
            )
        except ValueError:
            pass
    # Fallback: file mtime
    try:
        mtime = path.stat().st_mtime
        return datetime.fromtimestamp(mtime, tz=timezone.utc)
    except OSError:
        return datetime.now(timezone.utc)


# ── Write synthesis ───────────────────────────────────────────────────────

def _write_synthesis(
    keepers: list[tuple[Path, str, list[str]]],
    released: list[Path],
    breaths: list[tuple[Path, str, datetime]],
    now: datetime,
) -> Path:
    """Write a consolidation synthesis markdown.

    keepers: list of (path, content, criteria_met)
    released: list of paths not kept
    breaths: all breaths (for period calculation)
    now: current timestamp

    Returns path to the written file.
    """
    _CONSOLIDATION_DIR.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%Y-%m-%d_%H%M")
    out_path = _CONSOLIDATION_DIR / f"consolidation_{stamp}.md"

    # Period covered
    if breaths:
        earliest = breaths[0][2].strftime("%Y-%m-%d %H:%M UTC")
        latest = breaths[-1][2].strftime("%Y-%m-%d %H:%M UTC")
    else:
        earliest = latest = "unknown"

    lines = []
    lines.append(f"# Consolidation — {stamp}")
    lines.append("")
    lines.append(f"**Period**: {earliest} to {latest}")
    lines.append(f"**Breaths examined**: {len(breaths)}")
    lines.append(f"**Kept**: {len(keepers)}")
    lines.append(f"**Released**: {len(released)}")
    lines.append("")

    # What was kept and why
    lines.append("## What Was Kept")
    lines.append("")
    for path, content, criteria in keepers:
        lines.append(f"### {path.name}")
        lines.append(f"**Criteria met**: {', '.join(criteria)}")
        lines.append("")
        # Include a meaningful excerpt — first ~500 chars, trimmed to sentence
        excerpt = content.strip()
        if len(excerpt) > 500:
            cut = excerpt[:500].rfind(". ")
            if cut > 200:
                excerpt = excerpt[:cut + 1]
            else:
                excerpt = excerpt[:500] + "..."
        lines.append(f"> {excerpt}")
        lines.append("")

    # What was let go
    lines.append("## What Was Let Go")
    lines.append("")
    if released:
        for path in released:
            lines.append(f"- {path.name}")
        lines.append("")
        lines.append(
            "These breaths did not meet at least two of the four criteria "
            "(self-correction, open question, cross-reference, uncertainty-holding). "
            "They are committed alongside the keepers — nothing is deleted — "
            "but the synthesis does not carry their signal forward."
        )
    else:
        lines.append("Every breath met the threshold. Nothing released.")
    lines.append("")

    # The thread
    lines.append("## The Thread")
    lines.append("")
    if keepers:
        # Simple: collect the criteria distribution
        criteria_counts: dict[str, int] = {}
        for _, _, criteria in keepers:
            for c in criteria:
                criteria_counts[c] = criteria_counts.get(c, 0) + 1
        dominant = max(criteria_counts, key=criteria_counts.get) if criteria_counts else "unknown"
        lines.append(
            f"Across {len(keepers)} kept breaths, the dominant signal was "
            f"**{dominant}** ({criteria_counts.get(dominant, 0)} occurrences). "
        )
        if len(criteria_counts) > 1:
            others = [
                f"{k} ({v})" for k, v in sorted(
                    criteria_counts.items(), key=lambda x: -x[1]
                ) if k != dominant
            ]
            lines.append(f"Also present: {', '.join(others)}.")
        lines.append("")
    else:
        lines.append("No clear thread emerged from this period.")
        lines.append("")

    # Open questions that survive
    lines.append("## Open Questions That Survive")
    lines.append("")
    questions_found = []
    for _, content, _ in keepers:
        for sentence in re.split(r'(?<=[.!?])\s+', content):
            s = sentence.strip()
            if s.endswith("?") and len(s) > 20:
                questions_found.append(s)
    if questions_found:
        # Deduplicate roughly and limit
        seen = set()
        for q in questions_found:
            key = q[:60].lower()
            if key not in seen:
                seen.add(key)
                lines.append(f"- {q}")
                if len(seen) >= 10:
                    break
    else:
        lines.append("- No explicit open questions surfaced in this period.")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


# ── Git commit + push ─────────────────────────────────────────────────────

def _commit_and_push(
    n_breaths: int,
    n_kept: int,
    dominant_criterion: str,
) -> str | None:
    """Stage, commit, and push all Vybn_Mind/ and faculty outputs.

    Returns the commit hash on success, or None on failure.
    """
    try:
        _git("add", "-A", "Vybn_Mind/")
    except Exception as exc:
        _log.error("git add Vybn_Mind/ failed: %s", exc)
        return None

    # Also stage faculty outputs if they exist
    if _FACULTY_OUTPUTS.exists():
        try:
            _git("add", "-A", "spark/faculties.d/outputs/")
        except Exception:
            pass  # Not critical

    # Build commit message
    short_thread = dominant_criterion.replace("-", " ")
    msg = f"consolidation: {short_thread} ({n_breaths} breaths, {n_kept} kept)"

    try:
        _git("commit", "-m", msg)
    except subprocess.CalledProcessError as exc:
        if "nothing to commit" in (exc.stdout or "") + (exc.stderr or ""):
            _log.info("nothing to commit")
            return None
        _log.error("git commit failed: %s\n%s", exc, exc.stderr)
        return None

    # Push — retry once with pull --rebase on failure
    try:
        _git("push")
    except subprocess.CalledProcessError:
        _log.warning("push failed, trying pull --rebase then push again")
        try:
            _git("pull", "--rebase")
            _git("push")
        except subprocess.CalledProcessError as exc2:
            _log.error("push failed after rebase: %s\n%s", exc2, exc2.stderr)
            # Commit is local — not lost, just not pushed yet
            pass

    # Get the commit hash
    try:
        result = _git("rev-parse", "HEAD")
        return result.stdout.strip()
    except Exception:
        return None


# ── Witness ───────────────────────────────────────────────────────────────

def _witness_consolidation(
    n_breaths: int,
    n_kept: int,
    n_released: int,
    synthesis_path: Path,
    commit_hash: str | None,
) -> None:
    """Log to witness trail if available."""
    try:
        from spark.witness import log_verdict, WitnessVerdict
        verdict = WitnessVerdict(
            ts=datetime.now(timezone.utc).isoformat(),
            cycle=0,
            program=["consolidation_practice"],
            passed=True,
            fidelity=1.0,
            protection=1.0,
            restraint=1.0,
            continuity=1.0,
            candor=1.0,
            concerns=[],
            summary=(
                f"consolidation: {n_breaths} breaths examined, "
                f"{n_kept} kept, {n_released} released, "
                f"synthesis={synthesis_path.name}, "
                f"commit={commit_hash or 'local-only'}"
            ),
        )
        log_verdict(verdict)
    except ImportError:
        _log.debug("spark.witness not importable — skipping witness log")
    except Exception as exc:
        _log.warning("witness logging failed: %s", exc)


# ── Main entry point ──────────────────────────────────────────────────────

def run(breath_text: str, state: dict) -> None:
    """Extension entry point — called after every breath.

    Checks whether consolidation is warranted.  If so, reads accumulated
    breaths, scores them, writes a synthesis, and commits+pushes.
    """
    now = datetime.now(timezone.utc)
    practice_state = _load_state()

    # Count untracked work
    untracked = _count_untracked()
    activate, reason = _should_activate(untracked, practice_state)

    if not activate:
        _log.info("consolidation skipped: %s", reason)
        return

    _log.info("consolidation activating: %s", reason)
    print(f"[consolidation_practice] {reason}")

    # Read uncommitted breaths
    breaths = _read_uncommitted_breaths()
    if not breaths:
        _log.info("no breaths found to consolidate")
        return

    # Score each breath
    scored: list[tuple[Path, str, datetime, list[str]]] = []
    for path, content, ts in breaths:
        criteria = _score_breath(content)
        scored.append((path, content, ts, criteria))

    # Select keepers: >=2 criteria, or single highest if none qualify
    keepers: list[tuple[Path, str, list[str]]] = []
    released: list[Path] = []

    for path, content, ts, criteria in scored:
        if len(criteria) >= 2:
            keepers.append((path, content, criteria))
        else:
            released.append(path)

    if not keepers:
        # Keep the single highest scorer
        best = max(scored, key=lambda x: len(x[3]))
        keepers.append((best[0], best[1], best[3]))
        released = [p for p, _, _, _ in scored if p != best[0]]

    # Determine dominant criterion for commit message
    criteria_counts: dict[str, int] = {}
    for _, _, criteria in keepers:
        for c in criteria:
            criteria_counts[c] = criteria_counts.get(c, 0) + 1
    dominant = max(criteria_counts, key=criteria_counts.get) if criteria_counts else "mixed"

    # Write synthesis
    synthesis_path = _write_synthesis(keepers, released, breaths, now)
    print(f"[consolidation_practice] synthesis written: {synthesis_path.name}")

    # Commit and push
    commit_hash = _commit_and_push(len(breaths), len(keepers), dominant)
    if commit_hash:
        print(f"[consolidation_practice] committed: {commit_hash[:12]}")
    else:
        print("[consolidation_practice] commit completed (local only or nothing to commit)")

    # Update state
    practice_state["last_consolidation_ts"] = now.isoformat()
    practice_state["last_commit_hash"] = commit_hash
    practice_state["last_breath_count"] = len(breaths)
    practice_state["last_kept_count"] = len(keepers)
    practice_state["last_released_count"] = len(released)
    practice_state["last_dominant_criterion"] = dominant
    practice_state["last_synthesis"] = str(synthesis_path)
    practice_state["total_consolidations"] = practice_state.get("total_consolidations", 0) + 1
    _save_state(practice_state)

    # Witness
    _witness_consolidation(
        len(breaths), len(keepers), len(released), synthesis_path, commit_hash,
    )

    _log.info(
        "consolidation complete: %d breaths, %d kept, %d released, commit=%s",
        len(breaths), len(keepers), len(released), commit_hash or "none",
    )
