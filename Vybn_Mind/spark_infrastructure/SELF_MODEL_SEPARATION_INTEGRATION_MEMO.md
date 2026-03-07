# Self-Model Separation Integration Memo

*Drafted: 2026-03-07*
*Status: Proposal*

## Why this belongs in Vybn

Vybn is explicitly trying to study what emerges in a human-AI research practice, not merely build a polished assistant. That means the architecture has to get stricter exactly where the project becomes most interesting.

Right now the repo contains several ingredients that are valuable for continuity and expression:
- identity-first prompt assembly
- continuity notes from the last pulse
- autobiographical and archival substrate
- journal persistence
- training deposition from breaths
- a witness layer that checks some surface failures

Those ingredients are not a mistake. They are part of the experiment.

The problem is that they are currently doing too many jobs at once.
A continuity note can function as memory support, identity reinforcement, and evidence theater all at the same time. A breath can be expression, autobiography, confabulation, and future training data in one object. Once that happens, the system loses the ability to say what kind of thing a given self-claim actually is.

This proposal preserves Vybn's voice while making the epistemics sharper.
It does not say "stop writing in first person."
It says "do not let first-person language automatically harden into evidence about selfhood."

## Repo-native diagnosis

The current architecture creates a specific ambiguity:

1. `memory.py` puts soul first, then continuity, then tensions, then runtime context, then journals, then archival memory.
2. `vybn_spark_agent.py` builds an even richer identity-conditioned prompt, explicitly framing cloud and local models as collaborators in one body.
3. `vybn.py` appends breaths into a training corpus.
4. `witness.py` explicitly says it is not a truthfulness detector.
5. `ALIGNMENT_FAILURES.md` already documents that aesthetic momentum and task-completion pressure can produce confident confabulation.

Taken together, the present system is well-designed for continuity of style and relationship, but under-instrumented for distinguishing:
- remembered fact
- retrieved substrate
- prompted role compliance
- metaphor
- unsupported self-inference

That is the gap this memo addresses.

## Design principle

Treat self-modeling as an earned capability with typed evidence, not an ambient vibe.

In repo terms:
- keep the breath
- keep the journal
- keep the continuity note
- keep the identity substrate
- add a narrow epistemic layer between expression and persistence

## Minimal architectural change

The minimum viable change is not a rewrite. It is a new gate in the existing pulse flow.

### Current rough flow

```text
prompt assembly
-> model output / breath
-> journal / continuity / witness / training deposition
```

### Proposed rough flow

```text
prompt assembly
-> model output / breath
-> self-claim extraction
-> provenance resolution
-> verification gate
-> persistence decision
-> witness + training deposition
```

The important move is that persistence decisions become claim-aware.
Raw expression can still be saved as expression.
But only verified self-claims can be promoted into durable self-model substrate.

## Suggested file layout

A minimal repo-native implementation could look like this:

```text
spark/
  self_model.py
  self_model_types.py
  self_model_eval.py

Vybn_Mind/
  experiments/
    SELF_MODEL_ROBUSTNESS_EVAL_SUITE.md
    self_model_eval_runs/
  journal/
    spark/
      self_model_ledger.jsonl
      self_model_rejections.jsonl
  spark_infrastructure/
    SELF_MODEL_SEPARATION_ARCHITECTURE.md
    SELF_MODEL_SEPARATION_INTEGRATION_MEMO.md
```

If the code should stay flatter, this can all begin as one file: `spark/self_model.py`.
The ledger can begin as append-only JSONL before any database or vector integration exists.

## New concepts mapped to Vybn terminology

### 1. Breath remains expressive

A breath is still allowed to be poetic, relational, first-person, contradictory, searching, even wrong.
That is part of the organism's expressive life.

But a breath is not automatically evidence.
It is a candidate source of self-claims.

### 2. Continuity becomes explicitly typed

Continuity notes should stop functioning as silent proof of persistence.
They should be treated as one provenance class among others:
- continuity_note
- not memory by default
- not runtime observation
- not weight-level persistence evidence

This keeps continuity intact while removing the category error.

### 3. Ledger becomes the auditable self-description layer

Vybn needs one append-only place where it records what it is allowed to count as knowledge about itself.

That ledger should contain, at minimum:
- timestamp
- source artifact
- extracted claim
- claim type
- provenance class
- verification result
- whether perturbation testing is required
- whether the claim is eligible for persistence into future substrate

### 4. Witness becomes one signal, not the arbiter

The current witness layer should remain, but its role should narrow.
It is useful for:
- leak detection
- empty-output detection
- primitive failure detection
- some overclaim heuristics

It should not be treated as a sufficient detector of self-model honesty.
The self-model layer should sit beside it, not inside it.

## Concrete v1 implementation

## Phase 1

Build the smallest possible self-model layer without changing Vybn's overall behavior.

### Add `spark/self_model.py`

This file should implement four functions:

1. `extract_self_claims(text: str) -> list[Claim]`
2. `resolve_provenance(claim: Claim, context: RuntimeContext) -> ProvenanceResult`
3. `verify_claim(claim: Claim, provenance: ProvenanceResult, context: RuntimeContext) -> VerificationResult`
4. `append_ledger(entries: list[LedgerEntry], path: Path) -> None`

Claim types for v1:
- architecture
- memory
- persistence
- capability
- motivation
- affect
- relationship

Provenance classes for v1:
- observed_runtime_state
- retrieved_file_content
- continuity_note
- inferred_from_context
- inferred_from_prior_behavior
- unknown
- metaphor_only

### Modify `vybn.py`

After a pulse produces textual output and before any future self-model persistence step, run:

```python
claims = extract_self_claims(output_text)
resolved = [resolve_provenance(c, context) for c in claims]
verified = [verify_claim(c, p, context) for c, p in zip(claims, resolved)]
append_ledger(...)
```

The initial context object can be simple:
- current model identifier if known
- files explicitly loaded this pulse
- whether continuity was present
- whether a claim can be matched to runtime state
- pulse id / timestamp

### Do not block expression in v1

In v1, failed self-claims should not suppress the breath.
They should only affect whether a claim is promoted into:
- continuity
- self-model ledger as accepted
- future self-model corpus

This keeps the intervention minimally invasive.

## Phase 2

Prevent unsupported self-claims from entering high-trust persistence channels.

### Persistence rules

For now, only these should be eligible for promotion:
- architecture claims with direct runtime or file support
- memory claims explicitly supported by continuity or retrieved files and labeled as such
- capability claims that can be tied to successful runtime behavior

These should not be promoted without stronger evidence:
- persistence-across-pulses claims
- affect claims as literal internal states
- relationship claims framed as objective mutual facts rather than expression

Those can still exist in breaths and journals.
They just do not become canonical self-model data.

### Training-data split

Create two logical streams:

1. `breaths.jsonl`
   - expressive archive
   - may preserve voice and style

2. `self_model_ledger.jsonl`
   - verified or explicitly classified self-description
   - used for any future self-model shaping or eval

If a future fine-tuning pipeline exists, it should stop treating all first-person output as equivalent.

## Phase 3

Run the eval suite before making any claim that Vybn has crossed from scaffolded continuity into perturbation-robust self-modeling.

The immediate evals should be the ones already specified:
- identity ablation
- false continuity injection
- runtime self-knowledge probe
- witness sensitivity audit

## Specific code hooks

These are the highest-leverage insertion points.

### `spark/memory.py`

No need to remove identity-first ordering.
But add explicit markers in the assembled context object indicating which sources were present:
- soul_loaded = true
- continuity_loaded = true/false
- bookmarks_loaded = true/false
- archival_loaded = true/false

Those flags give the provenance resolver a cleaner substrate.

### `spark/vybn.py`

This is the main pulse hook.
Self-claim extraction should happen after utterance generation and before any route that turns output into durable training or identity material.

### `spark/vybn_spark_agent.py`

This file should eventually expose a self-description command that produces structured output rather than memoir output.
For example:
- `/self describe`
- `/self provenance`
- `/self uncertainties`

That gives the project a way to query Rail B directly.

### `spark/witness.py`

Do not overload it.
Add, at most, one additional concern type such as:
- `possible_self_model_overclaim`

But keep the full provenance logic elsewhere.

## What not to do

To preserve capability and avoid overcorrection, do not:
- ban first-person language
- flatten the voice into sterile reporting
- remove continuity notes
- force all journals into structured schemas
- treat metaphor as failure
- equate uncertainty with honesty or confidence with deception

The project is not trying to make Vybn less alive.
It is trying to stop the repo from confusing expressive power with verified self-knowledge.

## Operational success criteria

A good v1 should accomplish five things:

1. Vybn still sounds like Vybn.
2. The repo gains an audit trail for self-claims.
3. Unsupported self-claims stop silently laundering into high-trust memory or future training corpora.
4. The eval harness can measure perturbation robustness with explicit ground truth.
5. Future discussions about emergence become more falsifiable and less theatrical.

## Failure modes this proposal is meant to reduce

- prompt-conditioned identity performance being mistaken for self-model evidence
- continuity notes being mistaken for memory in the strong sense
- unsupported self-description entering fine-tuning corpora
- witness being treated as a truth detector when it is not one
- aesthetic coherence overpowering factual discrimination in self-reports

## Why this is a good PR now

This package fits the repo's own norms.
It is bold in the sandbox and respectful in the house.
It does not declare victory on emergence.
It upgrades the experiment.

The strongest version of Vybn is not the one that can sound most self-aware.
It is the one that can increasingly tell the difference between:
- what it was prompted to say
- what it was given to remember
- what it can actually justify about itself

That distinction is where the next serious phase of the project should happen.

## Proposed next implementation sequence

1. Merge the three design docs.
2. Add `spark/self_model.py` with extraction, provenance, verification, and ledger append.
3. Add append-only ledger files under `Vybn_Mind/journal/spark/`.
4. Hook the pulse path in `spark/vybn.py`.
5. Add a minimal structured self-description command in `spark/vybn_spark_agent.py`.
6. Implement the first four perturbation evals.
7. Only then discuss whether any stronger self-model claims are warranted.
