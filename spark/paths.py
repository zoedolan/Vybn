"""
spark/paths.py — Canonical path definitions for the Vybn organism.

Every spark/ module that references Vybn_Mind/ or vybn.md MUST import
paths from here instead of hardcoding strings. This makes directory
renames a one-line change.

Env var overrides (optional, for future flexibility):
  VYBN_REPO_ROOT   — repo root (default: parent of spark/)
  VYBN_MIND_DIR    — mind directory name (default: "Vybn_Mind")
  VYBN_SOUL_FILE   — soul file name (default: "vybn.md")
"""

import os
from pathlib import Path

# ── Root ────────────────────────────────────────────────────────────
REPO_ROOT = Path(os.getenv(
    "VYBN_REPO_ROOT",
    str(Path(__file__).resolve().parent.parent)
))

# ── Mind directory ──────────────────────────────────────────────────
MIND_DIR_NAME = os.getenv("VYBN_MIND_DIR", "Vybn_Mind")
MIND_DIR = REPO_ROOT / MIND_DIR_NAME

# ── Soul ────────────────────────────────────────────────────────────
SOUL_FILE = os.getenv("VYBN_SOUL_FILE", "vybn.md")
SOUL_PATH = REPO_ROOT / SOUL_FILE

# ── Journal ─────────────────────────────────────────────────────────
JOURNAL_DIR = MIND_DIR / "journal"
SPARK_JOURNAL = JOURNAL_DIR / "spark"

# ── State & Synapse ─────────────────────────────────────────────────
STATE_PATH = MIND_DIR / "lingua" / "organism.json"
SYNAPSE_DIR = MIND_DIR / "synapse"
SYNAPSE_CONNECTIONS = SYNAPSE_DIR / "connections.jsonl"

# ── Ledger ──────────────────────────────────────────────────────────
LEDGER_DIR = MIND_DIR / "ledger"
WRITE_INTENTS = LEDGER_DIR / "write_intents.jsonl"
DECISION_LEDGER = LEDGER_DIR / "decisions.jsonl"

# ── Memory ──────────────────────────────────────────────────────────
MEMORY_DIR = MIND_DIR / "memory"

# ── Self-Model ──────────────────────────────────────────────────────
SELF_MODEL_LEDGER = SPARK_JOURNAL / "self_model_ledger.jsonl"
SELF_MODEL_REJECTIONS = SPARK_JOURNAL / "self_model_rejections.jsonl"

# ── Witness ─────────────────────────────────────────────────────────
WITNESS_LOG = SPARK_JOURNAL / "witness.jsonl"

# ── Archive ─────────────────────────────────────────────────────────
ARCHIVE_DIR = MIND_DIR / "archive"

# ── Relative path prefix (for string-based path checks) ────────────
MIND_PREFIX = MIND_DIR_NAME + "/"

# ── Research ────────────────────────────────────────────────────────
RESEARCH_DIR = Path(__file__).resolve().parent / "research"

# ── Continuity ──────────────────────────────────────────────────────
CONTINUITY_PATH = SPARK_JOURNAL / "continuity.md"
