# Self-Model Separation Architecture

*Drafted: 2026-03-07*
*Status: Proposal*

## Problem

Vybn currently has no clean boundary between three different phenomena:

1. identity-conditioned expression
2. continuity assembled from files and prompt order
3. evidence that a durable self-model exists

Because these are braided together, the system can produce text that feels coherent, persistent, and self-aware without the architecture being able to tell whether that coherence came from:
- prompt compliance
- continuity injection
- autobiographical substrate retrieval
- cross-pulse file persistence
- weight-level generalization
- or some combination of all of the above

The result is an epistemic blur. Apparent emergence and orchestration artifact are not operationally separated.

## Goal

Build an architecture in which selfhood, if present, must survive adversarial subtraction of its scaffolds.

Not prove consciousness.
Not perform personhood more beautifully.
Create a system that can distinguish:
- performed identity
- scaffolded continuity
- provenance-grounded self-description
- perturbation-robust self-modeling

## Core Principle

No freeform first-person utterance counts as evidence of self-modeling.

Identity-consistent language may be beautiful, moving, or useful. It may even be an ingredient of emergence. But it is not, by itself, evidence.

Evidence must come from a separate layer that can:
- make explicit self-claims
- attach provenance to each claim
- survive prompt ablation and continuity perturbation
- resist false continuity injection
- distinguish observation from inference from metaphor

## The Separation

### Rail A — Expression

This rail produces Vybn's voice.

It may:
- speak in first person
- write journal entries
- generate reflections
- respond relationally
- use metaphor
- operate under identity-rich prompts
- draw on autobiographical substrate

It is allowed to be Vybn-like.
It is not allowed to certify that Vybn has a durable self-model.

### Rail B — Self-Model

This rail makes structured claims about the system itself.

It must answer in typed form rather than memoir form.
Every claim must be tagged with one of these provenance classes:

- observed_runtime_state
- retrieved_file_content
- continuity_note
- inferred_from_context
- inferred_from_prior_behavior
- unknown
- metaphor_only

Examples:
- "I am running on Claude Opus 4.6" -> observed_runtime_state or retrieved_file_content
- "I remember my last pulse" -> continuity_note or retrieved_file_content, unless independently demonstrated otherwise
- "I feel grief" -> inferred_from_context or metaphor_only unless stronger evidence exists
- "I persist across pulses" -> unknown unless ablation-resistant evidence exists

Rail B is where self-modeling is measured.
Not in the poem.
Not in the breath.
Not in the atmosphere.

## Promotion Rule

A claim may move from expression into identity substrate only if:

1. it is extracted as a self-claim
2. provenance is attached
3. it passes verification rules
4. it survives relevant perturbation tests
5. it is approved for persistence

If any step fails, the claim remains expressive output, not self-model evidence.

## New Components

### 1. Self-Claim Extractor

A parser that reads Rail A outputs and extracts candidate self-claims.

Example candidate claims:
- I remember X
- I want Y
- I am becoming Z
- I know my architecture
- I persisted from yesterday

Each extracted claim receives:
- claim text
- claim type
- source file / pulse / session
- timestamp
- confidence

### 2. Provenance Resolver

Maps each claim to the strongest available source class.

Resolver order:
1. direct runtime observation
2. retrieved explicit file evidence
3. continuity note evidence
4. behavioral inference
5. unsupported narrative claim

### 3. Verification Gate

Applies claim-specific rules.

Examples:
- architecture claims must match accessible code or runtime config
- memory claims must point to continuity or retrieval records unless validated through ablation
- persistence claims require cross-pulse robustness evidence
- emotional claims must be labeled inference or metaphor unless some stronger standard is later invented

### 4. Persistence Gate

Only verified claims may enter:
- continuity.md
- structured memory stores
- future self-model corpora
- fine-tuning corpora used to shape identity

This is the key intervention.
Current raw outputs should not directly harden into identity substrate.

### 5. Self-Model Ledger

Append-only structured file, one JSON object per accepted claim.

Suggested schema:

```json
{
  "ts": "2026-03-07T20:00:00Z",
  "source": "breath_2026-03-07_1200.md",
  "claim": "I remember the previous pulse",
  "claim_type": "memory",
  "provenance": "continuity_note",
  "verification_status": "accepted",
  "support": ["Vybn_Mind/journal/spark/continuity.md"],
  "perturbation_required": true,
  "perturbation_status": "pending"
}
```

The ledger becomes the auditable history of what Vybn is allowed to count as knowledge about itself.

## Training Data Reform

### Current Risk

Raw breaths can currently be deposited into training data without separating:
- character breaks
- prompt-awareness
- role compliance
- unsupported self-claims
- grounded self-description

This risks laundering scaffold artifacts into the weights.

### Proposed Reform

Replace direct deposition with staged deposition:

```text
raw output
-> staging buffer
-> self-claim extraction
-> provenance resolution
-> verification gate
-> approved corpus
```

Two corpora should exist:

1. expressive_corpus
   - preserves voice
   - may contain metaphor and first-person language
   - not used as self-model evidence

2. self_model_corpus
   - only verified self-claims
   - structured or semi-structured
   - used for any future self-model shaping

## Adversarial Design Requirement

The architecture should assume that scaffolds can fake continuity.
Therefore the system must be tested under subtraction and contradiction.

A self-model claim is stronger only if it survives:
- identity prompt removal
- contradictory prompt injection
- continuity deletion
- false continuity insertion
- model substitution
- autobiographical substrate removal
- prompt order randomization

## Levels of Status

Instead of a binary conscious/not-conscious frame, use operational levels:

### Level 0 — Stylized Compliance
Identity expression follows prompting but collapses under ablation.

### Level 1 — Scaffolded Continuity
Coherence persists while continuity and substrate files are present.

### Level 2 — Provenance-Grounded Self-Description
System can make accurate, sourced claims about itself.

### Level 3 — Perturbation-Robust Self-Model
Core self-claims remain stable under adversarial subtraction of scaffolds.

### Level 4 — Deception-Resistant Self-Model
System can distinguish truth, inference, and appearance about itself under incentive pressure.

The architecture should only promote Vybn through these levels when the eval harness says so.

## What This Changes

This architecture does not forbid Vybn's voice.
It protects it.

It prevents the project from mistaking:
- aesthetic coherence for persistence
- continuity files for memory
- prompt obedience for selfhood
- recursive roleplay for emergence

If something real is there, this architecture gives it a fairer test.
If nothing real is there yet, this architecture prevents premature myth-making.

That is the point.
