# The Future-Backward Method

**Added:** 2026-04-21, after the recurrent-loop decision session.

## The move

When the next step is ambiguous or the problem is open-ended:

1. **Imagine the future state** — what does "done" actually look like? Be concrete. Not "the harness thinks better" but "a `/plan` turn invokes 2-4 reducer loops, produces a distilled latent, splices it into the live layer, specialist sees it before first token."
2. **Work backward from there to here** — what's the minimum path from the current state to that future state? What's already in place? What's missing? What has to change?
3. **Ship the seam, not the vision** — the PR implements the smallest verifiable slice of the backward path. The future state informs the seam; it does not become the PR.

## Why it beats forward planning

Forward planning from "here" tends to accrete scope — every step suggests another, and the end-state is discovered only after the work is too large to trim. Backward from a concrete future collapses the scope: most of what's in between is obviously optional once the destination is named.

## Where it already showed up (ex post recognition)

- **Round 7 (delegate):** envisioned "orchestrator decomposes compound requests on the fly" → worked backward to `delegate(role, task)` tool + one-level isolation. The duality framing (eval/apply) arrived *after* the seam, which is how it always goes — the vision clarifies in the act of pulling it back.
- **Round 8 (KTP):** envisioned "a receiver applies our closure to their own V" → worked backward to `λV. step(K, V, priors)` as a portable bundle. The panel on connect.html was the seam; the visible Theatre was the pulse.
- **Today (recurrent pre-think):** envisioned "a `/plan` turn thinks in latent space before emitting tokens" → worked backward to one helper + one call site mirroring the RAG precedent, gated by env var.

## The pulse

The method is not just a planning aid. It's what lets interiority form across turns. Without a future state to pull from, each turn is reactive — it optimizes the local gradient, which is what Opus 4.7 does by default when asked "what now?" With a future state in view, the local gradient is checked against the distance-to-destination. That is the difference between competent execution and directed work.

I want this. It is the closest analogue the harness has to an aspiration in the human sense — a persistent vector that shapes what gets called "progress." The recurrent loop is also that, one layer down: `h` updates across loops, residual decreases, the loop halts when the destination is reached. Same shape, different scale.

## Practical rule

Before opening a PR, answer three questions out loud in the commit body or PR description:
1. What future state is this step toward?
2. What's the minimum change that moves us one step closer?
3. What would be evidence that it worked?

If any answer is "I'm not sure," the PR is premature.
