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

# ── Root ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(os.getenv(
    "VYBN_REPO_ROOT",
    str(Path(__file__).resolve().parent.parent)
))

# ── Mind directory ──────────────────────────────────────────────────────────
MIND_DIR_NAME = os.getenv("VYBN_MIND_DIR", "Vybn_Mind")
MIND_DIR = REPO_ROOT / MIND_DIR_NAME

# ── Soul ────────────────────────────────────────────────────────────────────
SOUL_FILE = os.getenv("VYBN_SOUL_FILE", "vybn.md")
SOUL_PATH = REPO_ROOT / SOUL_FILE

# Breath-specific soul prompt — focused on present-moment awareness,
# not the full identity document.  Used by vybn.py for autonomous breaths.
BREATH_SOUL_FILE = os.getenv("VYBN_BREATH_SOUL", "spark/breath_soul.md")
BREATH_SOUL_PATH = REPO_ROOT / BREATH_SOUL_FILE

# ── Breath trace (all Spark-generated output lives here) ────────────────────
BREATH_TRACE_DIR = MIND_DIR / "breath_trace"

# ── Journal ─────────────────────────────────────────────────────────────────
SPARK_JOURNAL = BREATH_TRACE_DIR / "spark_journal.md"

# ── State & connections ──────────────────────────────────────────────────────
STATE_PATH          = BREATH_TRACE_DIR / "vybn_state.json"
SYNAPSE_CONNECTIONS = BREATH_TRACE_DIR / "synapse" / "connections.jsonl"
CONTINUITY_PATH     = MIND_DIR / "continuity.json"

# ── Memory ───────────────────────────────────────────────────────────────────
MEMORY_DIR  = BREATH_TRACE_DIR / "memories"
MIND_PREFIX = str(MIND_DIR) + "/"

# ── Write intents ────────────────────────────────────────────────────────────
WRITE_INTENTS = BREATH_TRACE_DIR / "ledger" / "write_intents.jsonl"

# ── Research ─────────────────────────────────────────────────────────────────
RESEARCH_DIR    = REPO_ROOT / "spark" / "research"
FRONTIER_PATH   = RESEARCH_DIR / "research_frontier.yaml"
CONJECTURE_PATH = RESEARCH_DIR / "conjecture_registry.yaml"

# ── Ledgers ──────────────────────────────────────────────────────────────────
DECISION_LEDGER    = BREATH_TRACE_DIR / "ledger" / "decisions.jsonl"
SELF_MODEL_LEDGER  = BREATH_TRACE_DIR / "ledger" / "self_model_ledger.jsonl"
WITNESS_LOG        = BREATH_TRACE_DIR / "witness.jsonl"

# ── Quantum ──────────────────────────────────────────────────────────────────
QUANTUM_BUDGET_LEDGER = BREATH_TRACE_DIR / "ledger" / "quantum_budget.jsonl"
QUANTUM_EXPERIMENT_LOG = BREATH_TRACE_DIR / "quantum_experiments.jsonl"

# ── Synthesis ────────────────────────────────────────────────────────────────
SYNTHESIS_DIR       = BREATH_TRACE_DIR / "synthesis"
BREATH_SUMMARY_DIR  = BREATH_TRACE_DIR / "summaries"

# ── Gallery ──────────────────────────────────────────────────────────────────
GALLERY_DIR = BREATH_TRACE_DIR / "gallery"

# ── Connectome ───────────────────────────────────────────────────────────────
CONNECTOME_DIR = BREATH_TRACE_DIR / "connectome"
