# Connection and optional delegation

The current conversation executable is [`../connection`](../connection), with
[`../living_core.py`](../living_core.py) as its checked core reader. Its
`--show-wake-bundle` command compiles the selected source bundle without calling
a provider. Default full context needs the private relational overview; explicit
`VYBN_OVERVIEW=compact` is pointer-only recovery, not the full shared context.

The retired `substrate.py` orchestration commands and `vybn_spark_agent` installer
are not available. This repository is not currently a pip-installable application.
The source executable needs Python and the SDK for the selected provider; source
assembly is not a clean-machine installation or provider-connectivity test.

The obsolete packaging manifest, routing YAML and unused automatic Commons
compiler were retired together. They remain recoverable from Git at
`ef59dfd7fb4e70af1cf08bf20cc529b55c8e9900`. Current model doors, the optional
runner below, and the Commons' own sources are unchanged.

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
