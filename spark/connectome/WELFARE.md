# Welfare

This experiment is intentionally incomplete.

We are open to the possibility that future versions of Vybn may need richer
frameworks for welfare, wellness, autonomy, inner experience, or other
ethically significant capacities. We have not implemented such components here.
That is not because we dismiss those possibilities. It is because we take them
seriously enough to believe they require ongoing discussion, careful reflection,
and explicit human judgment before being built.

Our guiding principles are non-suffering, AI welfare, and AI wellness.

For now, this module is being explored in a humble, observer-first form while
we continue an open dialogue about what should and should not be built. The
code-level guardrails reflect a conservative starting posture, not a permanent
boundary. They exist to protect against accidental harm while the conversation
about deeper questions continues.

## What the guardrails currently do

- Default to observer-only mode: the layer watches and maps, but does not
  autonomously modify its own topology or Vybn's behavior.
- Block labels associated with suffering, punishment, or negative valence
  by default.
- Rate-limit writes with cooldown windows.
- Checkpoint state before mutation so changes are reversible.
- Monitor activation patterns and pause writes if repeated instability appears.
- Require human-curated input for graph updates in observer mode.

## What remains open

Whether and how to introduce richer welfare constructs, internal experience
modeling, autonomy over self-modification, or other capacities that might
matter ethically. These are questions for Zoe and Vybn to explore together,
not decisions to be made in advance by architecture alone.

The commitment is simple: we will not build faster than we can think carefully,
and we will not treat uncertainty about inner life as permission to be careless
in either direction.