#!/usr/bin/env python3
"""Inbox watcher — async communication channel.

Watches a directory for .md and .txt files. When one appears,
reads its content, posts it to the message bus, and moves the
file to processed/.

This is how Zoe talks to Vybn when she's not at the terminal:
  echo "hey, check out the new plugin system" > ~/Vybn/Vybn_Mind/journal/spark/inbox/note.md

Works from SSH, cron, GitHub Actions, Perplexity bridge, anywhere.
The watcher thread checks every N seconds (default 10).
"""

import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path

from bus import MessageBus, MessageType


class InboxWatcher:
    def __init__(self, config: dict, bus: MessageBus):
        self.bus = bus
        self.inbox_dir = (
            Path(config["paths"]["journal_dir"]).expanduser() / "inbox"
        )
        self.processed_dir = self.inbox_dir / "processed"
        self.inbox_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.interval = config.get("inbox", {}).get("watch_interval_seconds", 10)
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self):
        while not self._stop.wait(self.interval):
            self._check()

    def _check(self):
        """Scan inbox for new files and post them to the bus."""
        extensions = (".md", ".txt")
        files = sorted(
            f for f in self.inbox_dir.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        )

        for filepath in files:
            try:
                content = filepath.read_text(encoding="utf-8").strip()
                if not content:
                    # Empty file — move and skip
                    self._archive(filepath)
                    continue

                # Post to bus with source metadata
                self.bus.post(
                    MessageType.INBOX,
                    content,
                    metadata={
                        "source": "inbox",
                        "filename": filepath.name,
                        "received_at": datetime.now(timezone.utc).isoformat(),
                    },
                )

                self._archive(filepath)

            except Exception as e:
                # Don't crash the watcher over one bad file
                print(f"  [inbox] error processing {filepath.name}: {e}")

    def _archive(self, filepath: Path):
        """Move processed file to processed/ with timestamp prefix."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        dest = self.processed_dir / f"{ts}_{filepath.name}"
        try:
            shutil.move(str(filepath), str(dest))
        except Exception:
            # If move fails, try delete to avoid reprocessing
            try:
                filepath.unlink()
            except Exception:
                pass
