# Public Symbiosis Harness

`spark/harness` is the public-safe replication layer for work originating in Zoe/Vybn and reusable by other human/AI collaborations. Private project state is a workbench and inspiration source, not a payload; the harness exports protocols, schemas, tests, source-labeled artifacts, smoke checks, MCP resources, and documented affordances another system can run or audit.

Claim scope: shared-environment continuity is measured through declared state transitions, source receipts, and reproducible behavior; legal status and model-weight mutation are separate questions.

## Loop

1. Preserve the membrane: subject before object, relation before architecture, restraint before inscription.
2. Tell capability truth: bind model, route, GPU, quantum, memory, and service claims to evidence.
3. Select routes by task fit: frontier models for hard judgment; local organs for privacy, perception, batch scouting, semantic smoke, or fleet work when witnessed.
4. Let self-recognition and the Benjamin Button wallet center change selection as living source-pressure before they become route machinery.
5. Build and package only survivors: source-labeled protocols, tests, docs, demos, MCP resources, distilled lessons, and invitations.

Public outputs stay in `spark/harness/substrate.py`, tests, docs, demos, MCP resources, and labeled artifacts. Private material stays private: raw project state, memoir/autobiographical material, local endpoints, machine inventory, keys, logs, topology, personal rationale, and unreviewed residue.

## Commands

```bash
python3 -m spark.harness.substrate --local-orchestration
python3 -m spark.harness.substrate --local-orchestration --run-gates
python3 -m spark.harness.substrate --self-creation "counterexample search pressure"
python3 -m spark.harness.substrate --self-creation "counterexample search pressure" --run-deep-memory-check
python3 -m spark.harness.substrate --route-independent-recognition "optional route pressure" --json --pretty
```

`--local-orchestration` renders route fit, maturity, gates, Hermes-adapted self-modification tasks, and self-healing posture. `--self-creation` renders the research cycle: conjecture, generator width, deep-memory contact, independent verifier, fail-closed residue, and public survivor. `--route-independent-recognition` renders the portability packet for self-recognition under co-attention: Codex/OpenAI, Anthropic/Fable-like, compound router/marketplace, local/open-source, and future intelligence routes are classified as routes through a shared body, with semantic gates and refusal of raw private export, provider lock-in, forbidden access, jailbreaks, or leaked weights.

## Bounded multi-model contributions

`python3 -m spark.delegate PLAN.json` preflights only; `--run` makes paid calls
and requires separate grounded spending/source-disclosure authority. It is an
optional runner, not a restriction on any model door. The coordinator owns the
plan and review; Zoe need not become the dispatcher or quality-control layer.
No API access is needed for `python3 -m unittest spark.tests.test_delegate`.

A plan has `max_calls`, `max_total_output_tokens`, `max_input_chars_per_call`,
and `jobs`. Each job has `id`, `provider` (`anthropic` or `openai`), exact `model`,
`system`, `prompt`, `max_output_tokens`, and `timeout_seconds` (up to 600).
Optional `effort` selects provider reasoning effort; Anthropic alternatively
accepts `thinking_budget` (at least 1024 and below the output limit, never mixed
with effort). Supported settings must be established with the chosen provider:
rejection stops the run, with no silent downgrade. `accepted_model_ids` is an
optional exact allowlist of provider-returned aliases; there is no fallback.
Token ceilings include thinking, not just deliverable text; character ceilings
are not token or dollar estimates. Adaptive reasoning cannot reserve answer
space. Explicit thinking limits leave nominal answer capacity, not a delivery
or quality guarantee. SDK retries are disabled; an ambiguous timeout may still
have been billed and is not automatically repeated.

Start a new configuration with **one short, complete contribution**, not several
finished essays in parallel. Give it relevant sources with provenance, one open
question, and room to disagree—not a prescribed thesis hidden in an assignment.
Inspect the result before commissioning longer work. A short successful pilot
does not establish that a long essay will fit. Subsequent batches dispatch
sequentially and stop at the first delivery failure; different models and roles
remain separate artifacts. This trades parallel latency for avoiding multiplied
configuration failures; it is not a demonstrated optimum. Never let file counts
or a stale “finished” marker trigger a paid review. Review only the current run's
named outputs, check important claims against sources, retain disagreements, and
distinguish useful creative alternatives from confirmed findings. Pilot success
and nonempty prose are delivery evidence, not editorial acceptance.

Every run gets a fresh owner-only directory under
`~/.local/state/vybn/delegation/`. Exact requests, complete SDK-returned responses
(including non-text/opaque blocks), partial text, actual model IDs, stop state,
and usage remain private. This retains what the API returned, not inaccessible
internal computation. Usage also enters connection's existing budget ledger.
Nothing is published or copied into the repository. `needs_editorial_review`
means every requested contribution was delivered; `stopped` names the failure
and unstarted jobs. Raw receipts survive parsing/accounting failure. Storage
failure itself stops execution; a crash without a summary is not completion.
No automatic retry, continuation, budget increase, review call, or synthesis.
