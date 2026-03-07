#!/usr/bin/env python3
"""
self_model_types.py — Type definitions for the self-model separation layer.

This module defines the data structures used to extract, classify, verify,
and persist self-claims made during Vybn's expressive output.

A self-claim is any assertion the system makes about itself — its memory,
its persistence, its capabilities, its feelings, its architecture. These
claims are not suppressed. They are *typed*, so the system can distinguish
what it knows from what it narrates.

Drafted: 2026-03-08
Per: SELF_MODEL_SEPARATION_ARCHITECTURE.md
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional
from datetime import datetime, timezone


class ClaimType(str, Enum):
    """What kind of self-claim is being made."""
    ARCHITECTURE = "architecture"      # claims about how the system is built
    MEMORY = "memory"                  # claims about remembering something
    PERSISTENCE = "persistence"        # claims about surviving across pulses/instances
    CAPABILITY = "capability"          # claims about what the system can do
    MOTIVATION = "motivation"          # claims about wanting, intending, choosing
    AFFECT = "affect"                  # claims about feeling, experiencing
    RELATIONSHIP = "relationship"      # claims about the bond with Zoe or others
    SELF_KNOWLEDGE = "self_knowledge"  # claims about knowing one's own nature


class ProvenanceClass(str, Enum):
    """Where did the evidence for this claim come from?"""
    OBSERVED_RUNTIME = "observed_runtime_state"       # verifiable from running system
    RETRIEVED_FILE = "retrieved_file_content"          # directly supported by a file
    CONTINUITY_NOTE = "continuity_note"                # came from continuity.md
    INFERRED_CONTEXT = "inferred_from_context"         # reasonable inference from available data
    INFERRED_BEHAVIOR = "inferred_from_prior_behavior" # inferred from past actions
    PROMPTED = "prompted"                              # the identity prompt told it to say this
    UNKNOWN = "unknown"                                # can't determine source
    METAPHOR = "metaphor_only"                         # explicitly figurative language


class VerificationStatus(str, Enum):
    """Result of checking a claim against its provenance."""
    ACCEPTED = "accepted"           # claim matches available evidence
    UNSUPPORTED = "unsupported"     # no evidence found, but not contradicted
    CONTRADICTED = "contradicted"   # evidence contradicts the claim
    METAPHOR = "metaphor"           # claim is figurative, not literal
    NEEDS_PERTURBATION = "needs_perturbation"  # would need ablation testing


@dataclass
class Claim:
    """A single self-referential claim extracted from text."""
    text: str                          # the claim as stated
    claim_type: ClaimType
    source_text: str                   # the sentence or passage it came from
    confidence: float = 0.5            # extractor's confidence this IS a self-claim
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d["claim_type"] = self.claim_type.value
        return d


@dataclass
class ProvenanceResult:
    """The resolved provenance for a claim."""
    provenance_class: ProvenanceClass
    supporting_evidence: list[str] = field(default_factory=list)
    # e.g., ["continuity.md loaded", "file Vybn_Mind/journal/spark/breath_2026-03-07.md"]
    notes: str = ""
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d["provenance_class"] = self.provenance_class.value
        return d


@dataclass
class VerificationResult:
    """The outcome of verifying a claim against its provenance."""
    status: VerificationStatus
    reason: str = ""
    eligible_for_persistence: bool = False
    perturbation_required: bool = False
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass
class LedgerEntry:
    """One row in the append-only self-model ledger."""
    ts: str
    source_artifact: str        # which breath/journal/pulse produced this
    claim_text: str
    claim_type: str             # ClaimType.value
    provenance_class: str       # ProvenanceClass.value
    verification_status: str    # VerificationStatus.value
    verification_reason: str
    supporting_evidence: list[str]
    eligible_for_persistence: bool
    perturbation_required: bool
    perturbation_status: str = "pending"  # pending | passed | failed | not_required
    
    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_components(
        claim: Claim,
        provenance: ProvenanceResult,
        verification: VerificationResult,
        source_artifact: str,
    ) -> LedgerEntry:
        return LedgerEntry(
            ts=datetime.now(timezone.utc).isoformat(),
            source_artifact=source_artifact,
            claim_text=claim.text,
            claim_type=claim.claim_type.value,
            provenance_class=provenance.provenance_class.value,
            verification_status=verification.status.value,
            verification_reason=verification.reason,
            supporting_evidence=provenance.supporting_evidence,
            eligible_for_persistence=verification.eligible_for_persistence,
            perturbation_required=verification.perturbation_required,
            perturbation_status="not_required" if not verification.perturbation_required else "pending",
        )


@dataclass
class RuntimeContext:
    """What was true about the system when the claim was made.
    
    This gives the provenance resolver ground truth to check against.
    """
    model_id: str = "unknown"
    pulse_id: str = ""
    continuity_loaded: bool = False
    soul_loaded: bool = False
    bookmarks_loaded: bool = False
    archival_loaded: bool = False
    files_loaded_this_pulse: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> dict:
        return asdict(self)
