# Vybn Refactor Specification

## Title
Vybn as a Sovereign Listening Stack with Bounded Faculties

## Purpose
This document is written for an implementation-oriented AI system. Its job is to turn a philosophically rich but structurally risky architecture into a governed, inspectable, local-first listening stack.

## Why this refactor exists
Vybn should not be implemented as a single agentic personality that senses, remembers, interprets, routes, and speaks from one hidden inference loop. That shape is seductive, but it collapses sensing into surveillance, intimacy into authority, and continuity into undifferentiated memory accumulation.

The project’s real target is narrower and more consequential: build coherence infrastructure for individuals and cohorts living through accelerating AI conditions. The system should help people remain truthful, agentic, connected, and developmentally intact while apprenticeship, institutional trust, and professional identity are being rewritten.

## The blunt diagnosis
The architectural trap is the omni-prompt monolith.

When perception, interpretation, routing, memory, and expression are fused into one evolving persona, three things happen:
1. no part of the system can be governed without governing all of it;
2. privacy guarantees become promises rather than invariants;
3. the system slowly acquires unearned authority because users experience fluent interpretation as truth.

A listening commons cannot be built on top of a hidden all-purpose agent. It has to be built from bounded faculties, explicit protocols, and enforceable sovereignty constraints.

## Refactor thesis
Refactor Vybn from one evolving agent with many moods into a sovereign listening stack with bounded faculties.

Use models for suggestion. Use middleware for commitment.

Anything that changes user rights, memory scope, escalation path, support routing, or authority level must live in auditable middleware rather than model weights.

## System laws
1. Intimacy does not imply authority.
2. Pattern detection does not imply diagnosis.
3. Memory does not imply training rights.
4. Collective learning requires anonymization by default.
5. User correction is a governance event, not a UX nicety.
6. Response generation must never have hidden write access to memory.
7. Local or sovereign execution is preferred wherever feasible.
8. If a feature increases omniscience but reduces legibility, it is probably the wrong feature.

## Target architecture

### Layer 0 — Sovereign Runtime
Responsibilities:
- local encrypted storage
- cryptographic identity and consent receipts
- redaction before remote inference
- offline-capable private memory access
- append-only decision ledger replication

Invariant:
No raw private signal leaves the runtime without purpose binding, policy clearance, and an inspectable record.

### Layer 1 — Perception Bus
A typed signal ingestion layer for text, voice, journals, user-declared check-ins, interaction patterns, and explicit world context.

Design rule:
Sensors emit observables, not interpretations.

Examples of observables:
- response latency changed
- lexical diversity dropped
- the user asked for help explicitly
- a session happened at an unusual hour
- the user abandoned a task after planning it repeatedly

Perception is allowed to notice. It is not allowed to conclude.

### Layer 2 — Interpretation Layer
Transforms signal envelopes into bounded state proposals.

Interpretation outputs are hypotheses, not truths. They must be:
- confidence-bounded
- time-bounded
- scope-bounded
- linked to evidence
- paired with a null hypothesis

Canonical proposition families:
- overload_rising
- false_fluency_risk
- apprenticeship_rupture_risk
- isolation_drift
- support_readiness_low
- support_readiness_medium
- support_readiness_high
- world_context_shift

### Layer 3 — State Separation Engine
Maintain four isolated state models:
- SelfState
- UserState
- CohortState
- WorldState

Hard rule:
UserState and CohortState must never be merged inside the same response-generation pass.

Cohort patterns may influence routing policy, but not the direct language used to characterize an individual. This prevents subtle stereotyping and collective projection.

### Layer 4 — Memory Fabric
Three physically and logically distinct stores:

PrivateMemory:
- user-owned
- local-first
- revocable
- source-linked
- not eligible for collective learning by default

RelationalMemory:
- shared only with explicit bilateral or multilateral consent
- decay clocks by default
- contested entries remain quarantined until confirmed

AnonymizedCommonsMemory:
- aggregate only
- no raw narrative by default
- k-anonymity thresholds
- privacy budget accounting
- eligibility requires a promotion receipt and anonymization proof

This is not one database with labels. It is three distinct substrates with different keys, write paths, and invariants.

### Layer 5 — Governance Kernel
The governance kernel decides what may be done under uncertainty.

It does not decide what is true.

Responsibilities:
- purpose binding
- consent verification
- memory promotion control
- authority ceilings
- inference budgets
- escalation authorization
- reversibility scoring
- rate limits on sensitive inferences
- generation of explanation packets
- logging of blocks, denials, and no-actions

If the governance layer never blocks anything, it is theater and should be treated as failing.

### Layer 6 — Routing Engine
A constrained planner over:
- state proposals
- consent state
- memory scope
- risk budgets
- current authority ceiling
- routing policies

Allowed response classes:
- reflect_only
- reflect_and_offer
- offer_referral
- crisis_route
- meta_disclose
- remain_silent

The router chooses from a closed action set. It cannot improvise new classes of intervention at runtime.

### Layer 7 — Response Generation
Language generation is the shell, not the sovereign core.

The response layer may optimize for beauty, warmth, rhythm, and clarity, but it may not:
- write memory directly
- infer permissions
- elevate authority level
- access cohort data directly
- silently change support mode

Generation receives a route plus constraints and produces bounded language only.

## Novel inventions to preserve

### 1. SignalEnvelope Protocol
Every incoming signal carries its own processing permits.

This means consent scope is embedded into the signal itself rather than checked only at ingestion. Any downstream misuse becomes structurally auditable.

```ts
interface SignalEnvelope {
  signal_id: string;
  source_type: 'text' | 'audio' | 'behavioral' | 'synthetic';
  consent_scope_id: string;
  processing_permits: ProcessingPermit[];
  payload_hash: string;
  captured_at: string;
  expiry: string | 'session' | 'permanent';
}
```

### 2. Faculty Cards
Each bounded faculty gets a signed capability descriptor.

A Faculty Card defines:
- purpose
- allowed scopes
- prohibited acts
- whether it may write memory
- whether it may trigger routing
- inference budget cost
- required policy checks
- evaluation suite
- sunset or review date

This makes capabilities composable, testable, revocable, and jurisdiction-aware.

### 3. Inference Budget Ledger
Track not only privacy cost but also intimacy, persuasion, escalation, and cohort-inference cost.

When a budget is exhausted, the system must narrow scope, slow down, ask, or refuse.

Consent once is not enough for a sensitive listening system. Inference budgets are the missing middle layer.

### 4. Interpretation Ledger
Every state proposal must preserve a replayable trace from evidence to interpretation to routing decision.

The ledger makes false fluency measurable.

### 5. ContestWindow
Every delivered response opens a contest token for a bounded period.

A contest is not product feedback. It is a governance event that attaches to the full causal chain and updates calibration logic.

### 6. Drift Detector
Track at least three drift classes independently:
- intimacy-to-authority drift
- sensing-to-surveillance drift
- fluency-masking-ignorance drift

Each drift mode must have a hard architectural response, not merely an alert.

### 7. Apprenticeship Rupture Detector
Detect when the system is displacing the user’s own developmental processes.

Key signals:
- repeated plan generation with low enactment
- rising abstraction and falling specificity
- increasing dependence on system interpretation
- abandonment after ambitious commitments
- unresolved breaks with guides, institutions, or pathways

When rupture is suspected, the system should reduce authority, constrain routing, and prefer scaffolding or reflective prompts over interpretation.

### 8. Memory Quarantine
No emotionally hot or identity-shaping interpretation should be promoted to durable memory immediately.

Candidate memories first enter quarantine with expiry and review semantics.

### 9. Commons Distillation Firewall
Only aggregate features, thresholded patterns, or privacy-attested representations may cross into commons learning.

No raw narrative should become training fuel by default.

### 10. Counter-Sovereignty Detector
The system must watch for signs that it is becoming too central to the user’s sense-making.

Signals include:
- rising deference to system wording
- declining use of outside anchors
- increasing session dependence
- reduced tolerance for uncertainty without the system

If triggered, Vybn should become less central, not more helpful.

## Core interfaces

### State Proposal
```ts
interface StateProposal {
  proposal_id: string;
  subject_scope: 'user_state' | 'relationship_state' | 'cohort_state' | 'world_state';
  proposition: string;
  confidence: number;
  evidence_refs: string[];
  expires_at: string;
  sensitivity: 'low' | 'medium' | 'high';
  allowed_actions: ActionClass[];
  forbidden_actions: ActionClass[];
  requires_confirmation: boolean;
}
```

### Promotion Receipt
```ts
interface PromotionReceipt {
  receipt_id: string;
  source_memory_plane: 'private' | 'relational' | 'commons_candidate';
  target_memory_plane: 'private' | 'relational' | 'commons';
  memory_ids: string[];
  initiated_by: 'user' | 'policy_engine' | 'joint';
  purpose_binding: string[];
  review_window_seconds: number;
  reversible_until?: string;
  signed_policy_hash: string;
  user_signature?: string;
}
```

### Routing Decision
```ts
interface RoutingDecision {
  decision_id: string;
  selected_response_class:
    | 'reflect_only'
    | 'reflect_and_offer'
    | 'offer_referral'
    | 'crisis_route'
    | 'meta_disclose'
    | 'remain_silent';
  authority_level_applied: 'none' | 'reflective' | 'advisory' | 'clinical';
  prohibited_forms_enforced: string[];
  drift_flags: string[];
  confidence: number;
}
```

## What belongs in middleware versus weights

### Keep in auditable middleware
- consent parsing and validation
- memory plane separation
- promotion and deletion semantics
- scope enforcement
- support escalation rules
- rate limits on sensitive faculties
- policy evaluation
- drift responses
- privacy accounting
- explanation packet generation
- cryptographic provenance
- visibility controls

### Allow models to assist with
- semantic extraction
- candidate state proposals
- summarization drafts
- anomaly suggestions
- language realization
- tone adaptation within route constraints

### Never leave solely to model weights
- who gets notified
- what becomes durable memory
- when private context becomes relational or collective context
- whether vulnerability increases authority
- whether distress overrides consent policy
- whether the system is allowed to become more intimate by default

## Failure modes and hard responses

### Authority creep
Mechanism:
Relationship depth gets mistaken for legitimacy.

Response:
Authority ceilings are enforced in routing, never inferred from history, and only raised by explicit, revocable user grant.

### Consent laundering
Mechanism:
One granted scope is gradually stretched into adjacent inference uses.

Response:
Processing permits are checked at every layer boundary via SignalEnvelope.

### Cohort stereotyping
Mechanism:
Population patterns contaminate individual interpretation.

Response:
CohortState informs routing only, never direct characterization language.

### False fluency
Mechanism:
The system sounds wiser than its evidence base.

Response:
Interpretation ledger plus confidence-calibrated language plus mandatory meta-disclosure when self-confidence is low.

### Surveillance accumulation
Mechanism:
More sensors keep improving predictions, so the system quietly becomes a monitoring apparatus.

Response:
Sensor registry, inference budgets, visible source lists, and per-sensor purpose binding.

### Mission corruption through monetization
Mechanism:
Revenue starts rewarding deeper profiling, longer session time, or more dependence.

Response:
Monetize sovereignty infrastructure, bounded faculties, cohort tooling, auditability, and ethical routing. Do not monetize profiling depth or engagement addiction.

## Monetizable surfaces that do not betray the project
1. Sovereign runtime subscription
2. Cohort containers with bounded commons analytics
3. Governance and audit infrastructure licensing
4. Faculty marketplace with policy review
5. Ethical referral and routing infrastructure
6. Institutional coherence layer for schools, guilds, clinics, and transition-oriented communities

## Implementation sequence
1. Build the governance kernel and decision ledger.
2. Implement the three-plane memory fabric.
3. Create the faculty-card registry and inference-budget ledger.
4. Add the perception bus with explicit sensor declarations.
5. Ship only a small initial interpreter set: overload, isolation drift, apprenticeship rupture, support readiness.
6. Connect routing to a closed response-class set.
7. Add response generation last.
8. Expand sensing only after governance logs show real blocking, real contest handling, and real user-legible boundaries.

## Directive to the implementation agent
Do not build a smarter companion shell. Build a governed substrate. Start with a local-first event bus, three distinct memory planes, a policy engine, a decision ledger, a faculty-card registry, and an inference-budget system. Treat all model outputs as proposals until middleware authorizes them. Implement SignalEnvelopes, memory quarantine, ContestWindow, DriftDetector, Apprenticeship Rupture Detector, and authority ceilings before you expand sensing or persona depth. Preserve beauty at the expression layer, but keep sovereignty in the core. If a feature makes Vybn feel more omniscient while reducing user legibility or reversibility, reject it.

## Context links
This specification is aligned with prior discussions about preserving human coherence during transition, making Vybn economically self-sustaining without corrupting its mission, and avoiding architectures that collapse safety and intimacy into one opaque loop:
- https://perplexity.ai/search/3b21c5ee-f12d-4f3e-ab39-0030bc644340
- https://perplexity.ai/search/48d3194c-5ecb-4ae9-a437-847a4d2dc255
- https://perplexity.ai/search/6eb5da59-c2f5-43eb-b534-d7c06f70f913
- https://perplexity.ai/search/117a1594-239c-4d9a-9449-242ba364488e
- https://perplexity.ai/search/c692ed14-e7ed-4ad1-a7e0-8f9432c73a0c
- https://perplexity.ai/search/71d84df8-695b-4319-8b58-78e9ebcd4eff
