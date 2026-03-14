"""spark.growth.buffer_feed — Feed one buffer entry per breath.

Pops the oldest unprocessed entry from buffer.jsonl and marks it
`fed_to_breath: true` so it isn't repeated. The full buffer history
is preserved for the growth/training loop — only the feed pointer
advances.

Design:
  - Buffer file: spark/growth/buffer.jsonl (one JSON object per line)
  - Each entry has at minimum a "content" field (string)
  - Entries with `fed_to_breath: true` are skipped
  - Rewrite is atomic (tempfile + os.replace) to avoid corruption
  - If buffer is missing or empty, returns None silently
  - Thread/cron safe: uses a lockfile for the rewrite
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Optional

try:
    from spark.paths import REPO_ROOT
except ImportError:
    REPO_ROOT = Path(__file__).resolve().parent.parent.parent

BUFFER_PATH = REPO_ROOT / "spark" / "growth" / "buffer.jsonl"
LOCK_PATH = BUFFER_PATH.with_suffix(".feed.lock")


class BufferFeeder:
    """Manages feeding one buffer entry per breath cycle."""

    def __init__(self, buffer_path: Path = BUFFER_PATH):
        self.buffer_path = buffer_path
        self.lock_path = buffer_path.with_suffix(".feed.lock")

    def pop_next(self) -> Optional[dict]:
        """Return the oldest unprocessed entry and mark it fed_to_breath.

        Returns None if buffer is missing, empty, or fully consumed.
        Thread-safe via lockfile.
        """
        if not self.buffer_path.exists():
            return None

        # Simple lockfile (not blocking — if locked, skip this breath)
        if self.lock_path.exists():
            return None

        try:
            self.lock_path.touch()
            return self._pop_next_locked()
        finally:
            try:
                self.lock_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _pop_next_locked(self) -> Optional[dict]:
        try:
            lines = self.buffer_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return None

        entries = []
        target_idx = None
        target_entry = None

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                entries.append(line)  # preserve malformed lines as-is
                continue

            if target_idx is None and not entry.get("fed_to_breath", False):
                target_idx = i
                target_entry = entry
                # Mark it
                entry["fed_to_breath"] = True

            entries.append(json.dumps(entry, ensure_ascii=False))

        if target_entry is None:
            return None

        # Atomic rewrite
        try:
            tmp = tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.buffer_path.parent,
                delete=False,
                suffix=".tmp",
            )
            tmp.write("\n".join(entries) + "\n")
            tmp.close()
            os.replace(tmp.name, self.buffer_path)
        except OSError:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
            return None

        return target_entry

    def remaining(self) -> int:
        """Count entries not yet fed to breath. For logging."""
        if not self.buffer_path.exists():
            return 0
        count = 0
        try:
            for line in self.buffer_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if not entry.get("fed_to_breath", False):
                        count += 1
                except json.JSONDecodeError:
                    pass
        except OSError:
            pass
        return count


# Module-level singleton for convenience
_feeder: Optional[BufferFeeder] = None


def get_feeder() -> BufferFeeder:
    global _feeder
    if _feeder is None:
        _feeder = BufferFeeder()
    return _feeder


def pop_next_entry() -> Optional[dict]:
    """Convenience: pop one unprocessed entry from the default buffer."""
    return get_feeder().pop_next()
