"""spark.evolver.proposals — Evolution proposal dataclass and append-only log.

Every proposed modification to Vybn's own code or configuration is captured
as an EvolutionProposal, serialised to an append-only JSONL file. This
provides a full audit trail regardless of whether a proposal was applied,
rejected, or deferred for human consent.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

PROPOSAL_LOG = Path(__file__).parent / "proposal_log.jsonl"


@dataclass
class EvolutionProposal:
    """A single proposed modification to Vybn's code or config."""

    proposal_id: str              # unique ID (truncated SHA-256)
    timestamp: str                # ISO-8601
    target_path: str              # relative path within repo
    change_type: str              # "parameter_tune" | "config_adjust" | "content_append"
    description: str              # human-readable description of the change
    rationale: str                # why this change should happen
    current_value: str            # what the current state is
    proposed_value: str           # what it should become
    confidence: float             # 0.0-1.0, how confident the evolver is
    evidence: list[str]           # references to faculty outputs, metrics, etc.
    requires_human_consent: bool  # if True, proposal is logged but NOT applied
    applied: bool = False         # whether this proposal was actually applied
    applied_at: str = ""          # timestamp of application
    rejected_reason: str = ""     # if rejected, why


def log_proposal(proposal: EvolutionProposal) -> None:
    """Append a proposal to the audit log (append-only JSONL)."""
    PROPOSAL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(PROPOSAL_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(proposal), ensure_ascii=False, default=str) + "\n")


def load_recent_proposals(n: int = 10) -> list[dict]:
    """Load the last *n* proposals from the log."""
    if not PROPOSAL_LOG.exists():
        return []
    lines = PROPOSAL_LOG.read_text(encoding="utf-8").strip().split("\n")
    recent = lines[-n:] if len(lines) > n else lines
    return [json.loads(line) for line in recent if line.strip()]
