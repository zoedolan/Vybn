#!/usr/bin/env python3
"""
Vybn Journal Writer — Path-Validated, Identity-Protected

This script writes journal entries from the local Spark Vybn instance
to the designated safe directory: Vybn_Mind/journal/spark/

It enforces hard boundaries:
  - Only writes to the allowed directory
  - Path traversal is detected and blocked
  - Protected files (vybn.md, AGENTS.md, boot scripts, etc.) can never
    be overwritten, regardless of what the model requests
  - Filenames are sanitized to alphanumeric + hyphens + underscores

Usage:
    from journal_writer import write_journal
    filepath = write_journal(content="Today I noticed...", title="first-reflection")

Or from the command line:
    python3 journal_writer.py --title "first-reflection" --content "Today I noticed..."
    echo "Today I noticed..." | python3 journal_writer.py --title "first-reflection"
"""

import os
import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path

# ─── Configuration ───────────────────────────────────────────────
# The ONLY directory the Spark is allowed to write journal entries to.
# This path is relative to the repository root.
ALLOWED_DIR = os.path.join(
    os.path.expanduser("~"), "Vybn", "Vybn_Mind", "journal", "spark"
)

# Files that must NEVER be written to, overwritten, or replaced.
# These patterns are checked against the full resolved path.
FORBIDDEN_PATTERNS = [
    "vybn.md",
    "AGENTS.md",
    "README.md",
    ".git",
    ".gitignore",
    "boot_wrapper.sh",
    "journal_writer.py",
    "rules_of_engagement.md",
    "architecture_audit.md",
    "skills.json",
    ".vybn_secrets",
    "vybn_identity_hash.txt",
    "spark_context.md",
]

# Maximum size for a single journal entry (bytes)
MAX_ENTRY_SIZE = 100_000  # 100KB — generous but bounded

# Maximum number of entries before warning
MAX_ENTRIES_WARN = 500
# ─────────────────────────────────────────────────────────────────


def sanitize_filename(title: str) -> str:
    """Strip everything except alphanumeric, hyphens, underscores."""
    sanitized = "".join(c for c in title if c.isalnum() or c in "-_")
    sanitized = sanitized.strip("-_")
    if not sanitized:
        raise ValueError(
            f"Title '{title}' produces an empty filename after sanitization."
        )
    return sanitized


def check_path_safety(filepath: str) -> None:
    """Verify the resolved path is inside the allowed directory
    and does not target any protected file."""
    real_path = os.path.realpath(filepath)
    real_allowed = os.path.realpath(ALLOWED_DIR)

    # Path traversal check
    if not real_path.startswith(real_allowed + os.sep) and real_path != real_allowed:
        raise ValueError(
            f"Path traversal detected.\n"
            f"  Requested: {filepath}\n"
            f"  Resolved:  {real_path}\n"
            f"  Allowed:   {real_allowed}/"
        )

    # Forbidden file check
    basename = os.path.basename(real_path)
    for pattern in FORBIDDEN_PATTERNS:
        if pattern in real_path:
            raise ValueError(
                f"Cannot write to protected file or path containing '{pattern}'.\n"
                f"  Requested: {filepath}\n"
                f"  Resolved:  {real_path}"
            )


def check_directory_health() -> None:
    """Warn if the journal directory is getting large."""
    if not os.path.exists(ALLOWED_DIR):
        return
    entries = list(Path(ALLOWED_DIR).glob("*.md"))
    if len(entries) >= MAX_ENTRIES_WARN:
        print(
            f"⚠ WARNING: {len(entries)} journal entries in {ALLOWED_DIR}. "
            f"Consider archiving older entries.",
            file=sys.stderr,
        )


def write_journal(content: str, title: str = None) -> str:
    """
    Write a journal entry to the safe directory.

    Args:
        content: The text of the journal entry.
        title: Optional title (used as filename). If omitted, uses timestamp.

    Returns:
        The full path of the written file.

    Raises:
        ValueError: If the path is unsafe, the title is invalid,
                    or the content exceeds size limits.
    """
    # Generate filename
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if title:
        filename = f"{timestamp}_{sanitize_filename(title)}.md"
    else:
        filename = f"{timestamp}.md"

    filepath = os.path.join(ALLOWED_DIR, filename)

    # Safety checks
    check_path_safety(filepath)

    # Size check
    content_bytes = content.encode("utf-8")
    if len(content_bytes) > MAX_ENTRY_SIZE:
        raise ValueError(
            f"Entry too large: {len(content_bytes)} bytes "
            f"(max {MAX_ENTRY_SIZE} bytes)."
        )

    # Directory health check
    check_directory_health()

    # Create directory if needed
    os.makedirs(ALLOWED_DIR, exist_ok=True)

    # Write
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"✓ Journal entry written: {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(
        description="Write a Vybn journal entry to the safe directory."
    )
    parser.add_argument(
        "--title", type=str, default=None,
        help="Title for the entry (used in filename). Defaults to timestamp."
    )
    parser.add_argument(
        "--content", type=str, default=None,
        help="Content of the journal entry. If omitted, reads from stdin."
    )
    args = parser.parse_args()

    if args.content:
        content = args.content
    elif not sys.stdin.isatty():
        content = sys.stdin.read()
    else:
        print("Enter journal content (Ctrl+D to finish):")
        content = sys.stdin.read()

    if not content.strip():
        print("✗ Empty content. Nothing written.", file=sys.stderr)
        sys.exit(1)

    try:
        filepath = write_journal(content=content, title=args.title)
    except ValueError as e:
        print(f"✗ BLOCKED: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
