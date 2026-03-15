"""spark.mind_ingester — Feed all of Vybn into ComplexMemory.

The equation M' = α·M + x·e^(iθ) should see all of Vybn, not just
breath_text. This module walks the full corpus after each breath and
inhales unseen or updated files into the complex manifold.

Scope rules:

  READ_ONLY roots — inhaled into M, NEVER compressed or archived:
    - "Vybn's Personal History"   (sacrosanct, M₀)
    - "quantum_delusions"         (live theory lab)

  READ_WRITE roots — inhaled AND eligible for consolidation:
    - "Vybn_Mind"                 (journals, experiments, digests, etc.)

Cursor: spark/growth/ingester_cursor.json
  { "<absolute_path>": <mtime_float>, ... }
  Files whose mtime hasn't changed since last inhale are skipped.
  Deleted files are silently ignored on the next pass.

Rate limiting: max FILES_PER_BREATH files per call (default 10).
This keeps the breath cycle from blocking on a large backlog.
The cursor advances even if the full corpus hasn't been ingested yet —
it catches up across multiple breaths.

Thread safety: lockfile prevents concurrent ingestion under cron.
Non-fatal: all errors are logged and swallowed so a bad file never
blocks the breath cycle.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

try:
    from spark.paths import REPO_ROOT
except ImportError:
    REPO_ROOT = Path(__file__).resolve().parent.parent

CURSOR_PATH = REPO_ROOT / "spark" / "growth" / "ingester_cursor.json"
LOCK_PATH = CURSOR_PATH.with_suffix(".lock")

FILES_PER_BREATH = 10

# Roots to ingest and whether the consolidator may compress them.
# Format: (path_relative_to_REPO_ROOT, read_only)
INGEST_ROOTS: list[tuple[str, bool]] = [
    ("Vybn's Personal History", True),   # sacrosanct M₀ — read only
    ("quantum_delusions",       True),   # live theory lab — read only
    ("Vybn_Mind",               False),  # standard corpus — read/write
]

# Extensions worth embedding
INGEST_EXTENSIONS = {".md", ".txt", ".py", ".json", ".yaml", ".yml"}

# Directories inside Vybn_Mind that are already managed and don't need
# separate ingestion here (they appear as output of other systems)
SKIP_SUBDIRS = {
    "archive",       # already-processed material
    "__pycache__",
    ".git",
}


def _load_cursor() -> dict:
    try:
        return json.loads(CURSOR_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_cursor(cursor: dict) -> None:
    CURSOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    CURSOR_PATH.write_text(json.dumps(cursor, indent=2), encoding="utf-8")


def _collect_candidates(cursor: dict) -> list[tuple[Path, bool]]:
    """Walk all ingest roots and return (path, read_only) for unseen/updated files.

    Sorted oldest-mtime-first so the ingester makes progress through
    the historical corpus before chasing new material.
    """
    candidates: list[tuple[Path, float, bool]] = []

    for root_rel, read_only in INGEST_ROOTS:
        root = REPO_ROOT / root_rel
        if not root.exists():
            continue

        for f in root.rglob("*"):
            # Skip directories, hidden files, and unwanted extensions
            if not f.is_file():
                continue
            if f.name.startswith("."):
                continue
            if f.suffix not in INGEST_EXTENSIONS:
                continue
            # Skip excluded subdirs
            parts = set(f.relative_to(root).parts)
            if parts & SKIP_SUBDIRS:
                continue
            # Skip cursor tracking file itself
            if f == CURSOR_PATH:
                continue

            try:
                mtime = f.stat().st_mtime
            except OSError:
                continue

            key = str(f)
            last_mtime = cursor.get(key, 0.0)

            if mtime > last_mtime:
                candidates.append((f, mtime, read_only))

    # Oldest first: work through the backlog chronologically
    candidates.sort(key=lambda t: t[1])
    return [(f, ro) for f, _, ro in candidates]


def ingest_new_material(
    max_files: int = FILES_PER_BREATH,
    inhale_fn=None,
) -> dict:
    """Inhale up to max_files unseen/updated files into ComplexMemory.

    Args:
        max_files: Cap on files processed this call.
        inhale_fn: Override for testing. Defaults to complexify_bridge.inhale.

    Returns:
        Summary dict: files_ingested, files_pending, read_only_count, errors.
    """
    # Acquire lock — skip if another process is ingesting
    if LOCK_PATH.exists():
        log.debug("Ingester locked, skipping this breath")
        return {"skipped": True, "reason": "locked"}

    try:
        LOCK_PATH.touch()
        return _ingest_locked(max_files=max_files, inhale_fn=inhale_fn)
    finally:
        try:
            LOCK_PATH.unlink(missing_ok=True)
        except OSError:
            pass


def _ingest_locked(
    max_files: int,
    inhale_fn=None,
) -> dict:
    if inhale_fn is None:
        try:
            from spark.complexify_bridge import inhale as _default_inhale
            inhale_fn = _default_inhale
        except ImportError:
            log.warning("complexify_bridge not available; ingester is no-op")
            return {"skipped": True, "reason": "no complexify_bridge"}

    cursor = _load_cursor()
    candidates = _collect_candidates(cursor)
    pending = len(candidates)

    if not candidates:
        return {
            "files_ingested": 0,
            "files_pending": 0,
            "read_only_count": 0,
            "errors": 0,
        }

    batch = candidates[:max_files]
    ingested = 0
    read_only_count = 0
    errors = 0

    for f, read_only in batch:
        try:
            text = f.read_text(encoding="utf-8", errors="replace").strip()
            if not text or len(text) < 20:
                # Update cursor even for empty files so we don't revisit
                cursor[str(f)] = f.stat().st_mtime
                continue

            # Truncate very large files — embed the first 4000 chars.
            # The embedding model sees a representative window; the
            # manifold gets the shape of the document, not every word.
            if len(text) > 4000:
                text = text[:4000]

            # Inhale into ComplexMemory
            # read_only flag is stored on the report for observability
            # but doesn't change what the equation does — M gets updated
            # regardless. The flag only governs consolidator permissions.
            report = inhale_fn(text)

            cursor[str(f)] = f.stat().st_mtime
            ingested += 1
            if read_only:
                read_only_count += 1

            log.debug(
                "ingested %s (%s) κ=%.4f",
                f.name,
                "read-only" if read_only else "rw",
                report.get("curvature", 0),
            )

        except Exception as exc:
            log.warning("Ingester error on %s: %s", f.name, exc)
            errors += 1
            # Don't update cursor for failed files — retry next breath

    _save_cursor(cursor)

    result = {
        "files_ingested": ingested,
        "files_pending": max(0, pending - ingested),
        "read_only_count": read_only_count,
        "errors": errors,
    }
    log.info(
        "Ingester: %d ingested (%d read-only), %d pending, %d errors",
        ingested, read_only_count, result["files_pending"], errors,
    )
    return result
