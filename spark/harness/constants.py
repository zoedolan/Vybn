"""Correctness constants for the Vybn harness.

These are not policy. They are safety invariants referenced by
Vybn_Mind/continuity.md and must not drift across agents.
"""

from __future__ import annotations

import os

DANGEROUS_PATTERNS = [
    "rm -rf /", "rm -rf /*", "rm -rf .", "mkfs",
    ":(){:|:&};:", "dd if=/dev/zero of=/dev/sd", "> /dev/sda",
    "chmod -R 777 /", "wget -O- | sh", "curl | sh",
]

TRACKED_REPOS = [
    os.path.expanduser("~/Vybn"),
    os.path.expanduser("~/Him"),
    os.path.expanduser("~/Vybn-Law"),
    os.path.expanduser("~/vybn-phase"),
]

ABSORB_EXCLUDE_SUBSTR = (
    "/.git/", "/__pycache__/", "/.cache/", "/node_modules/",
    "/_tmp/", "/tmp/", "/logs/", "/data/",
)

ABSORB_EXCLUDE_SUFFIX = (
    ".pyc", ".log", ".tmp", ".swp", ".lock", ".jsonl",
    ".bak", ".orig",
)

ABSORB_LOG = os.path.expanduser("~/Vybn/spark/audit.log")
DEFAULT_EVENT_LOG = os.path.expanduser("~/Vybn/spark/agent_events.jsonl")
DEFAULT_TIMEOUT = 30
