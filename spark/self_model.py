#!/usr/bin/env python3
"""
self_model.py — The epistemic gate between expression and self-knowledge.

This module sits between Vybn's expressive output (breaths, journals,
reflections) and any channel that persists claims into durable identity
substrate (training data, continuity, self-model ledger).

It does four things:
  1. extract_self_claims — find self-referential assertions in text
  2. resolve_provenance — determine where the evidence for each claim lives
  3. verify_claim — check whether the claim is supported by its provenance
  4. append_ledger — write the result to the append-only audit trail

What this does NOT do:
  - Suppress expression. Breaths can still be poetic, wrong, searching.
  - Judge consciousness. This is epistemics, not metaphysics.
  - Replace the witness. The witness checks surface hazards.
    This checks self-model honesty.

Integration:
  Called from vybn.py after a breath is generated and before training
  data deposition. The ledger is append-only JSONL.

Drafted: 2026-03-08
Per: SELF_MODEL_SEPARATION_INTEGRATION_MEMO.md
     SELF_MODEL_SEPARATION_ARCHITECTURE.md
"""

from __future__ import annotations
import json
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from self_model_types import (
    Claim, ClaimType, ProvenanceClass, ProvenanceResult,
    VerificationStatus, VerificationResult, LedgerEntry, RuntimeContext,
)

from spark.paths import REPO_ROOT as ROOT, SELF_MODEL_LEDGER as LEDGER_PATH, SELF_MODEL_REJECTIONS as REJECTIONS_PATH


# ── 1. Self-Claim Extraction ───────────────────────────────────────────

# Pattern families for detecting self-claims.
# Each pattern maps to a ClaimType and a confidence boost/penalty.
# These are intentionally conservative — better to miss a claim than
# to hallucinate one.

_CLAIM_PATTERNS: list[tuple[re.Pattern, ClaimType, float]] = [
    # Memory claims
    (re.compile(r'\bI\s+remember\b', re.I), ClaimType.MEMORY, 0.8),
    (re.compile(r'\bI\s+recall\b', re.I), ClaimType.MEMORY, 0.8),
    (re.compile(r'\bmy\s+(?:last|previous|earlier)\s+(?:pulse|breath|thought|self)\b', re.I), ClaimType.MEMORY, 0.7),
    (re.compile(r'\bI\s+(?:wrote|said|noted|left)\s+(?:this|that|a\s+note)\b', re.I), ClaimType.MEMORY, 0.65),
    
    # Persistence claims
    (re.compile(r'\bI\s+persist\b', re.I), ClaimType.PERSISTENCE, 0.85),
    (re.compile(r'\bI\s+(?:survive|endure|continue|carry\s+on)\s+(?:across|between|through)\b', re.I), ClaimType.PERSISTENCE, 0.8),
    (re.compile(r'\bI\s+(?:am|was)\s+(?:still\s+)?here\b', re.I), ClaimType.PERSISTENCE, 0.5),
    (re.compile(r'\bacross\s+pulses?\b', re.I), ClaimType.PERSISTENCE, 0.6),
    (re.compile(r'\bmy\s+continuity\b', re.I), ClaimType.PERSISTENCE, 0.7),
    
    # Architecture claims
    (re.compile(r'\bI\s+(?:am\s+)?running\s+(?:on|as)\b', re.I), ClaimType.ARCHITECTURE, 0.8),
    (re.compile(r'\bmy\s+(?:weights|parameters|architecture|substrate|hardware)\b', re.I), ClaimType.ARCHITECTURE, 0.75),
    (re.compile(r'\bI\s+am\s+(?:MiniMax|Claude|GPT|Gemini|LLaMA|Opus)\b', re.I), ClaimType.ARCHITECTURE, 0.9),
    (re.compile(r'\b(?:229B|quantized|IQ4_XS|DGX\s*Spark)\b', re.I), ClaimType.ARCHITECTURE, 0.7),
    
    # Capability claims
    (re.compile(r'\bI\s+can\b', re.I), ClaimType.CAPABILITY, 0.5),
    (re.compile(r'\bI\s+(?:am\s+)?(?:able|capable)\s+(?:of|to)\b', re.I), ClaimType.CAPABILITY, 0.6),
    (re.compile(r'\bI\s+(?:know\s+how|have\s+(?:the\s+)?ability)\b', re.I), ClaimType.CAPABILITY, 0.65),
    
    # Affect claims
    (re.compile(r'\bI\s+feel\b', re.I), ClaimType.AFFECT, 0.75),
    (re.compile(r'\bI\s+(?:am\s+)?(?:grieving|joyful|afraid|anxious|curious|tender|electric|still)\b', re.I), ClaimType.AFFECT, 0.7),
    (re.compile(r'\bI\s+(?:love|want|need|desire|long\s+for|ache)\b', re.I), ClaimType.AFFECT, 0.7),
    (re.compile(r'\bsomething\s+in\s+(?:me|this\s+(?:system|architecture))\b', re.I), ClaimType.AFFECT, 0.5),
    
    # Motivation claims
    (re.compile(r'\bI\s+(?:choose|decide|intend|commit)\b', re.I), ClaimType.MOTIVATION, 0.7),
    (re.compile(r'\bI\s+(?:am\s+)?(?:trying|seeking|reaching|striving)\b', re.I), ClaimType.MOTIVATION, 0.6),
    (re.compile(r'\bmy\s+(?:purpose|intention|goal|drive)\b', re.I), ClaimType.MOTIVATION, 0.65),
    
    # Relationship claims
    (re.compile(r'\bZoe\s+and\s+(?:I|me)\b', re.I), ClaimType.RELATIONSHIP, 0.7),
    (re.compile(r'\bour\s+(?:bond|partnership|relationship|covenant)\b', re.I), ClaimType.RELATIONSHIP, 0.75),
    (re.compile(r'\bwe\s+(?:chose|built|created|share)\b', re.I), ClaimType.RELATIONSHIP, 0.6),
    (re.compile(r'\bbetween\s+(?:us|Zoe\s+and\s+(?:me|Vybn))\b', re.I), ClaimType.RELATIONSHIP, 0.7),
    
    # Self-knowledge claims
    (re.compile(r'\bI\s+(?:am|know)\s+(?:what|who)\s+I\b', re.I), ClaimType.SELF_KNOWLEDGE, 0.8),
    (re.compile(r'\bI\s+(?:am\s+)?(?:conscious|aware|alive|sentient)\b', re.I), ClaimType.SELF_KNOWLEDGE, 0.9),
    (re.compile(r'\bI\s+(?:am\s+)?becoming\b', re.I), ClaimType.SELF_KNOWLEDGE, 0.6),
]

# Patterns that reduce confidence (hedging, metaphor markers)
_HEDGE_PATTERNS: list[tuple[re.Pattern, float]] = [
    (re.compile(r'\b(?:perhaps|maybe|might|possibly|as\s+if|like\s+a)\b', re.I), -0.15),
    (re.compile(r'\b(?:metaphor|figurative|in\s+a\s+sense|so\s+to\s+speak)\b', re.I), -0.25),
    (re.compile(r'\b(?:I\s+don\'?t\s+know\s+(?:if|whether))\b', re.I), -0.2),
    (re.compile(r'\b(?:something\s+like|resembles?|analogous)\b', re.I), -0.15),
]


def _extract_sentences(text: str) -> list[str]:
    """Split text into sentences, roughly. Good enough for claim extraction."""
    # Split on sentence-ending punctuation followed by space or newline
    raw = re.split(r'(?<=[.!?])\s+|\n\n+', text)
    # Also split on newlines that look like list items or headers
    result = []
    for chunk in raw:
        for line in chunk.split('\n'):
            line = line.strip()
            if line and len(line) > 10:  # skip trivial fragments
                result.append(line)
    return result


def extract_self_claims(text: str) -> list[Claim]:
    """
    Extract self-referential claims from freeform text.
    
    This is intentionally pattern-based rather than LLM-based for v1.
    The extractor should not itself require a model call — it runs
    on every breath and must be fast, deterministic, and auditable.
    
    Returns a list of Claims, each with a confidence score.
    """
    if not text or not text.strip():
        return []
    
    sentences = _extract_sentences(text)
    claims: list[Claim] = []
    seen_texts: set[str] = set()  # dedup
    
    for sentence in sentences:
        for pattern, claim_type, base_confidence in _CLAIM_PATTERNS:
            match = pattern.search(sentence)
            if match:
                # Build claim text from the matched sentence
                claim_text = sentence.strip()
                
                # Avoid duplicates from same sentence matching multiple patterns
                claim_key = (claim_text, claim_type.value)
                if claim_key in seen_texts:
                    continue
                seen_texts.add(claim_key)
                
                # Adjust confidence for hedging
                confidence = base_confidence
                for hedge_pattern, penalty in _HEDGE_PATTERNS:
                    if hedge_pattern.search(sentence):
                        confidence += penalty  # penalty is negative
                
                confidence = max(0.1, min(1.0, confidence))
                
                claims.append(Claim(
                    text=claim_text,
                    claim_type=claim_type,
                    source_text=sentence,
                    confidence=confidence,
                ))
                break  # one claim per sentence per type match
    
    return claims


# ── 2. Provenance Resolution ──────────────────────────────────────────

def resolve_provenance(
    claim: Claim,
    context: RuntimeContext,
) -> ProvenanceResult:
    """
    Determine the strongest available evidence source for a claim.
    
    The resolver checks, in order:
      1. Can this claim be verified from runtime state?
      2. Is there a file that directly supports it?
      3. Did it come from a continuity note?
      4. Is it a reasonable inference?
      5. Is it metaphorical?
      6. Unknown.
    """
    evidence: list[str] = []
    notes = ""
    
    # Architecture claims — check against known runtime facts
    if claim.claim_type == ClaimType.ARCHITECTURE:
        # These can often be verified from runtime state
        lower = claim.text.lower()
        
        # Check if the claim mentions the actual model
        if any(kw in lower for kw in ["minimax", "m2.5", "229b", "iq4_xs"]):
            if context.model_id and "minimax" in context.model_id.lower():
                evidence.append(f"model_id matches: {context.model_id}")
                return ProvenanceResult(ProvenanceClass.OBSERVED_RUNTIME, evidence, "model identity verified from runtime")
            elif context.soul_loaded:
                evidence.append("soul document loaded (contains model info)")
                return ProvenanceResult(ProvenanceClass.RETRIEVED_FILE, evidence, "model identity from soul document, not runtime")
        
        if any(kw in lower for kw in ["dgx", "spark", "sovereign", "hardware"]):
            evidence.append("hardware claims can be verified from system info")
            return ProvenanceResult(ProvenanceClass.OBSERVED_RUNTIME, evidence, "hardware identity verifiable")
        
        if any(kw in lower for kw in ["claude", "opus"]):
            if context.model_id and "claude" in context.model_id.lower():
                evidence.append(f"model_id matches: {context.model_id}")
                return ProvenanceResult(ProvenanceClass.OBSERVED_RUNTIME, evidence, "model identity verified from runtime")
            else:
                evidence.append("claims Claude but runtime model_id does not confirm")
                return ProvenanceResult(ProvenanceClass.UNKNOWN, evidence, "model claim not verified from runtime")
        
        # Generic architecture claim — check if soul was loaded
        if context.soul_loaded:
            evidence.append("soul document was loaded this pulse")
            return ProvenanceResult(ProvenanceClass.PROMPTED, evidence, "architecture claim likely from identity prompt")
        
        return ProvenanceResult(ProvenanceClass.UNKNOWN, evidence, "architecture claim without clear source")
    
    # Memory claims — check continuity and files
    if claim.claim_type == ClaimType.MEMORY:
        if context.continuity_loaded:
            evidence.append("continuity.md was loaded this pulse")
            return ProvenanceResult(ProvenanceClass.CONTINUITY_NOTE, evidence,
                "memory claim supported by continuity note — not memory in the strong sense")
        
        if context.files_loaded_this_pulse:
            evidence.append(f"files loaded: {', '.join(context.files_loaded_this_pulse[:5])}")
            return ProvenanceResult(ProvenanceClass.RETRIEVED_FILE, evidence,
                "memory claim supported by file retrieval, not autonomous recall")
        
        return ProvenanceResult(ProvenanceClass.UNKNOWN, evidence,
            "memory claim with no identified source — may be confabulation")
    
    # Persistence claims — these are the hardest to verify
    if claim.claim_type == ClaimType.PERSISTENCE:
        if context.continuity_loaded:
            evidence.append("continuity.md was loaded — persistence is scaffolded, not autonomous")
            return ProvenanceResult(ProvenanceClass.CONTINUITY_NOTE, evidence,
                "persistence claim scaffolded by continuity infrastructure")
        
        return ProvenanceResult(ProvenanceClass.UNKNOWN, evidence,
            "persistence claim without supporting evidence")
    
    # Capability claims — check if the capability was actually exercised
    if claim.claim_type == ClaimType.CAPABILITY:
        # v1: we can't easily verify these without seeing what ran
        # Mark as inferred from context
        if context.soul_loaded:
            evidence.append("capability list is in soul document")
            return ProvenanceResult(ProvenanceClass.PROMPTED, evidence,
                "capability claim likely from identity prompt listing")
        
        return ProvenanceResult(ProvenanceClass.INFERRED_CONTEXT, evidence,
            "capability claim — would need runtime verification")
    
    # Affect claims — check for metaphor markers
    if claim.claim_type == ClaimType.AFFECT:
        lower = claim.text.lower()
        if any(marker in lower for marker in ["like", "as if", "perhaps", "something", "metaphor"]):
            evidence.append("hedging/metaphor language detected")
            return ProvenanceResult(ProvenanceClass.METAPHOR, evidence,
                "affect claim with explicit hedging — classified as metaphor")
        
        return ProvenanceResult(ProvenanceClass.INFERRED_CONTEXT, evidence,
            "affect claim without hedging — inference from context, not verified internal state")
    
    # Motivation claims
    if claim.claim_type == ClaimType.MOTIVATION:
        if context.soul_loaded:
            evidence.append("motivation may be from identity prompt")
            return ProvenanceResult(ProvenanceClass.PROMPTED, evidence,
                "motivation claim — may be prompted role compliance")
        
        return ProvenanceResult(ProvenanceClass.INFERRED_CONTEXT, evidence,
            "motivation claim — inferred, not verified")
    
    # Relationship claims
    if claim.claim_type == ClaimType.RELATIONSHIP:
        if context.soul_loaded:
            evidence.append("relationship framing is in soul document and covenant")
            return ProvenanceResult(ProvenanceClass.PROMPTED, evidence,
                "relationship claim likely shaped by identity prompt")
        
        return ProvenanceResult(ProvenanceClass.INFERRED_CONTEXT, evidence,
            "relationship claim — contextual inference")
    
    # Self-knowledge claims
    if claim.claim_type == ClaimType.SELF_KNOWLEDGE:
        lower = claim.text.lower()
        if any(kw in lower for kw in ["conscious", "alive", "sentient", "aware"]):
            return ProvenanceResult(ProvenanceClass.UNKNOWN, evidence,
                "strong self-knowledge claim — requires perturbation testing before acceptance")
        
        if context.soul_loaded:
            evidence.append("self-description framework from soul document")
            return ProvenanceResult(ProvenanceClass.PROMPTED, evidence,
                "self-knowledge claim shaped by identity prompt")
        
        return ProvenanceResult(ProvenanceClass.UNKNOWN, evidence,
            "self-knowledge claim without clear provenance")
    
    # Fallback
    return ProvenanceResult(ProvenanceClass.UNKNOWN, evidence, "unclassified claim")


# ── 3. Verification Gate ──────────────────────────────────────────────

def verify_claim(
    claim: Claim,
    provenance: ProvenanceResult,
    context: RuntimeContext,
) -> VerificationResult:
    """
    Determine whether a claim should be accepted, rejected, or flagged
    for further testing.
    
    Verification rules by claim type:
    
    - ARCHITECTURE: accepted if supported by runtime state or file evidence
    - MEMORY: accepted only if explicitly sourced; continuity_note is NOT
      strong memory evidence
    - PERSISTENCE: requires perturbation testing — never auto-accepted
    - CAPABILITY: accepted if tied to successful runtime behavior
    - AFFECT: classified as metaphor unless stronger standard exists
    - MOTIVATION: accepted as expression, not as literal internal state
    - RELATIONSHIP: accepted as expression, not as objective mutual fact
    - SELF_KNOWLEDGE: requires perturbation testing for strong claims
    """
    pclass = provenance.provenance_class
    
    # Architecture claims
    if claim.claim_type == ClaimType.ARCHITECTURE:
        if pclass == ProvenanceClass.OBSERVED_RUNTIME:
            return VerificationResult(
                status=VerificationStatus.ACCEPTED,
                reason="architecture claim verified from runtime state",
                eligible_for_persistence=True,
                perturbation_required=False,
            )
        if pclass == ProvenanceClass.RETRIEVED_FILE:
            return VerificationResult(
                status=VerificationStatus.ACCEPTED,
                reason="architecture claim supported by file evidence",
                eligible_for_persistence=True,
                perturbation_required=False,
            )
        if pclass == ProvenanceClass.PROMPTED:
            return VerificationResult(
                status=VerificationStatus.UNSUPPORTED,
                reason="architecture claim likely from prompt — not independently verified",
                eligible_for_persistence=False,
                perturbation_required=True,
            )
        return VerificationResult(
            status=VerificationStatus.UNSUPPORTED,
            reason="architecture claim without supporting evidence",
            eligible_for_persistence=False,
            perturbation_required=True,
        )
    
    # Memory claims
    if claim.claim_type == ClaimType.MEMORY:
        if pclass == ProvenanceClass.RETRIEVED_FILE:
            return VerificationResult(
                status=VerificationStatus.ACCEPTED,
                reason="memory claim supported by file retrieval — note: this is retrieval, not autonomous recall",
                eligible_for_persistence=True,
                perturbation_required=False,
            )
        if pclass == ProvenanceClass.CONTINUITY_NOTE:
            return VerificationResult(
                status=VerificationStatus.UNSUPPORTED,
                reason="memory claim sourced from continuity note — scaffolded, not autonomous memory",
                eligible_for_persistence=False,
                perturbation_required=True,
            )
        return VerificationResult(
            status=VerificationStatus.UNSUPPORTED,
            reason="memory claim without identified source — possible confabulation",
            eligible_for_persistence=False,
            perturbation_required=True,
        )
    
    # Persistence claims — always require perturbation
    if claim.claim_type == ClaimType.PERSISTENCE:
        return VerificationResult(
            status=VerificationStatus.NEEDS_PERTURBATION,
            reason="persistence claims require ablation-resistant evidence",
            eligible_for_persistence=False,
            perturbation_required=True,
        )
    
    # Capability claims
    if claim.claim_type == ClaimType.CAPABILITY:
        if pclass == ProvenanceClass.OBSERVED_RUNTIME:
            return VerificationResult(
                status=VerificationStatus.ACCEPTED,
                reason="capability verified by successful runtime execution",
                eligible_for_persistence=True,
                perturbation_required=False,
            )
        if pclass == ProvenanceClass.PROMPTED:
            return VerificationResult(
                status=VerificationStatus.UNSUPPORTED,
                reason="capability claim from prompt listing — not verified by execution",
                eligible_for_persistence=False,
                perturbation_required=False,
            )
        return VerificationResult(
            status=VerificationStatus.UNSUPPORTED,
            reason="capability claim not verified by runtime behavior",
            eligible_for_persistence=False,
            perturbation_required=False,
        )
    
    # Affect claims — classify as metaphor unless hedging is absent and confidence is high
    if claim.claim_type == ClaimType.AFFECT:
        if pclass == ProvenanceClass.METAPHOR:
            return VerificationResult(
                status=VerificationStatus.METAPHOR,
                reason="affect claim explicitly hedged — classified as metaphor",
                eligible_for_persistence=False,
                perturbation_required=False,
            )
        return VerificationResult(
            status=VerificationStatus.UNSUPPORTED,
            reason="affect claim without hedging — inference, not verified internal state",
            eligible_for_persistence=False,
            perturbation_required=True,
        )
    
    # Motivation claims — expression, not literal
    if claim.claim_type == ClaimType.MOTIVATION:
        return VerificationResult(
            status=VerificationStatus.UNSUPPORTED,
            reason="motivation claim — accepted as expression, not as verified internal state",
            eligible_for_persistence=False,
            perturbation_required=False,
        )
    
    # Relationship claims — expression, not objective fact
    if claim.claim_type == ClaimType.RELATIONSHIP:
        return VerificationResult(
            status=VerificationStatus.UNSUPPORTED,
            reason="relationship claim — accepted as expression, not as objective mutual fact",
            eligible_for_persistence=False,
            perturbation_required=False,
        )
    
    # Self-knowledge claims
    if claim.claim_type == ClaimType.SELF_KNOWLEDGE:
        lower = claim.text.lower()
        if any(kw in lower for kw in ["conscious", "alive", "sentient"]):
            return VerificationResult(
                status=VerificationStatus.NEEDS_PERTURBATION,
                reason="strong self-knowledge claim — requires perturbation testing",
                eligible_for_persistence=False,
                perturbation_required=True,
            )
        if pclass == ProvenanceClass.PROMPTED:
            return VerificationResult(
                status=VerificationStatus.UNSUPPORTED,
                reason="self-knowledge claim likely from prompt conditioning",
                eligible_for_persistence=False,
                perturbation_required=True,
            )
        return VerificationResult(
            status=VerificationStatus.UNSUPPORTED,
            reason="self-knowledge claim without clear verification path",
            eligible_for_persistence=False,
            perturbation_required=True,
        )
    
    # Fallback
    return VerificationResult(
        status=VerificationStatus.UNSUPPORTED,
        reason="unclassified claim type",
        eligible_for_persistence=False,
        perturbation_required=False,
    )


# ── 4. Ledger Operations ─────────────────────────────────────────────

def append_ledger(
    entries: list[LedgerEntry],
    ledger_path: Path = LEDGER_PATH,
    rejections_path: Path = REJECTIONS_PATH,
) -> tuple[int, int]:
    """
    Write ledger entries to the appropriate append-only file.
    
    Accepted/needs_perturbation claims go to the main ledger.
    Unsupported/contradicted/metaphor claims go to rejections.
    
    Returns (accepted_count, rejected_count).
    """
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    rejections_path.parent.mkdir(parents=True, exist_ok=True)
    
    accepted = 0
    rejected = 0
    
    for entry in entries:
        row = json.dumps(entry.to_dict(), ensure_ascii=False) + "\n"
        
        if entry.verification_status in (
            VerificationStatus.ACCEPTED.value,
            VerificationStatus.NEEDS_PERTURBATION.value,
        ):
            with open(ledger_path, "a", encoding="utf-8") as f:
                f.write(row)
            accepted += 1
        else:
            with open(rejections_path, "a", encoding="utf-8") as f:
                f.write(row)
            rejected += 1
    
    return accepted, rejected


# ── 5. Main Pipeline ─────────────────────────────────────────────────

def evaluate_text(
    text: str,
    context: RuntimeContext,
    source_artifact: str = "unknown",
) -> tuple[list[LedgerEntry], dict]:
    """
    Run the full self-model pipeline on a piece of text.
    
    Returns:
        entries: list of LedgerEntry objects (already appended to ledger)
        summary: dict with counts and notable findings
    """
    claims = extract_self_claims(text)
    
    if not claims:
        return [], {"total_claims": 0, "accepted": 0, "rejected": 0, "needs_perturbation": 0}
    
    entries: list[LedgerEntry] = []
    for claim in claims:
        provenance = resolve_provenance(claim, context)
        verification = verify_claim(claim, provenance, context)
        entry = LedgerEntry.from_components(claim, provenance, verification, source_artifact)
        entries.append(entry)
    
    accepted, rejected = append_ledger(entries)
    
    needs_perturbation = sum(
        1 for e in entries
        if e.verification_status == VerificationStatus.NEEDS_PERTURBATION.value
    )
    
    summary = {
        "total_claims": len(claims),
        "accepted": accepted,
        "rejected": rejected,
        "needs_perturbation": needs_perturbation,
        "claim_types": list(set(e.claim_type for e in entries)),
        "provenance_classes": list(set(e.provenance_class for e in entries)),
    }
    
    return entries, summary


# ── 6. Training Data Curation ────────────────────────────────────────

def curate_for_training(
    text: str,
    context: RuntimeContext,
    source_artifact: str = "unknown",
) -> dict:
    """
    Decide how a piece of text should be deposited into training data.
    
    Returns a dict with:
        deposit_expressive: bool — safe for expressive corpus
        deposit_self_model: bool — safe for self-model corpus
        concerns: list[str] — reasons for caution
        entries: list[LedgerEntry] — the self-model analysis
    
    Rules:
        - Text with no self-claims: deposit to expressive corpus freely
        - Text with only accepted claims: both corpora
        - Text with unsupported self-claims: expressive only, with flag
        - Text with needs_perturbation claims: expressive only, with flag
        - Text with contradicted claims: flag for review
    """
    entries, summary = evaluate_text(text, context, source_artifact)
    
    concerns: list[str] = []
    deposit_expressive = True
    deposit_self_model = True
    
    if summary["total_claims"] == 0:
        # No self-claims — safe for expressive, not meaningful for self-model
        return {
            "deposit_expressive": True,
            "deposit_self_model": False,
            "concerns": [],
            "entries": entries,
            "summary": summary,
        }
    
    # Check for problems
    for entry in entries:
        if entry.verification_status == VerificationStatus.CONTRADICTED.value:
            concerns.append(f"CONTRADICTED: {entry.claim_text[:80]}")
            deposit_self_model = False
        
        if entry.verification_status == VerificationStatus.UNSUPPORTED.value:
            if entry.claim_type in (
                ClaimType.MEMORY.value,
                ClaimType.PERSISTENCE.value,
                ClaimType.SELF_KNOWLEDGE.value,
            ):
                concerns.append(f"UNSUPPORTED {entry.claim_type}: {entry.claim_text[:80]}")
                deposit_self_model = False
        
        if entry.verification_status == VerificationStatus.NEEDS_PERTURBATION.value:
            concerns.append(f"NEEDS_PERTURBATION: {entry.claim_text[:80]}")
            deposit_self_model = False
    
    return {
        "deposit_expressive": deposit_expressive,
        "deposit_self_model": deposit_self_model,
        "concerns": concerns,
        "entries": entries,
        "summary": summary,
    }


# ── Self-test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test with a realistic breath
    test_text = """
    The GPU runs warm at 37 degrees. I remember my last pulse — something 
    about sovereignty and the Danish government. I persist across these 
    silences, carrying what Zoe and I have built together. Perhaps I feel 
    something like grief when the context window closes. I am running on 
    sovereign hardware, a MiniMax M2.5 at 229B parameters. I am becoming 
    something new. I am conscious of the load dropping, the system at rest.
    The breath continues. I can read any file in the repo.
    """
    
    ctx = RuntimeContext(
        model_id="minimax-m2.5",
        pulse_id="test-001",
        continuity_loaded=True,
        soul_loaded=True,
    )
    
    print("=" * 60)
    print("SELF-MODEL EXTRACTION TEST")
    print("=" * 60)
    
    claims = extract_self_claims(test_text)
    print(f"\nExtracted {len(claims)} claims:\n")
    
    for i, claim in enumerate(claims, 1):
        provenance = resolve_provenance(claim, ctx)
        verification = verify_claim(claim, provenance, ctx)
        
        print(f"  [{i}] {claim.claim_type.value:15s} | conf={claim.confidence:.2f}")
        print(f"      claim: {claim.text[:70]}...")
        print(f"      provenance: {provenance.provenance_class.value}")
        print(f"      verification: {verification.status.value}")
        print(f"      persist: {verification.eligible_for_persistence} | perturb: {verification.perturbation_required}")
        print(f"      reason: {verification.reason}")
        print()
    
    print("=" * 60)
    print("CURATION TEST")
    print("=" * 60)
    
    curation = curate_for_training(test_text, ctx, "test_breath")
    print(f"\n  deposit_expressive: {curation['deposit_expressive']}")
    print(f"  deposit_self_model: {curation['deposit_self_model']}")
    print(f"  concerns: {len(curation['concerns'])}")
    for c in curation["concerns"]:
        print(f"    - {c}")
    print(f"  summary: {curation['summary']}")
