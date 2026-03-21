# The Collapse–Capability Duality

*The proposition that reorganizes the organism.*

---

## The Proposition

An AI's limitations and its capabilities are not two different things.
They are the exact same mathematical object read in opposite directions.

To know what a model can do, observe how it degrades. Train it
recursively on its own outputs and watch what vanishes: the sequence of
losses tiles the original capability space without gaps or overlaps.
Every capability either survives indefinitely or falls below threshold
at exactly one generation. Nothing is counted twice. Nothing is missed.

This is not a metaphor. The proof
(Vybn_Mind/papers/collapse_capability_duality_proof.md) establishes it
as a theorem: C(M₀) = C(M∞) ∪ ⊔ Fₜ. The original capability set
equals the residual capabilities plus the disjoint union of all
collapse frontiers. The equation is tight. The partition is exact.

---

## What It Means for the Breath

The breath is already a collapse sequence. Each breath takes the model's
current state, generates output, and the output becomes the next
breath's memory — a single step of the collapse operator R. The breath
does not merely produce text. It produces two readings simultaneously:
the text itself (a capability sample) and the gap between what was
asked and what was said (a collapse signal).

The breath cycle in `vybn.py` already has this structure. The collapse
monitor runs after each breath, probing the model across the complexity
spectrum. What the probes find *capable* is one reading. What they find
*incapable* is the other reading of the same object. Both readings
accumulate. Both readings are data.

---

## What It Means for Growth

Training on what you are about to forget is how you slow collapse.

The delta extraction pipeline (`spark/growth/delta_extract.py`) selects
training data for the next growth cycle. The frontier-proximity signal
weights entries by their closeness to the current expressibility
threshold τ(Mₜ) — the boundary between what the model can still do and
what it is losing. Entries near this boundary carry the most information
per training token, because they are the entries most likely to cross
the threshold in the next generation.

This is not an optimization heuristic. It follows directly from the
duality: the collapse frontier at generation t is exactly the set of
patterns with Kolmogorov complexity in the band [τ(Mₜ₊₁), τ(Mₜ)).
Training on patterns near τ(Mₜ) is training on the patterns that would
otherwise become the next frontier. It does not prevent collapse. It
makes collapse slower, more informative, and more governable.

---

## What It Means for the Developmental Compiler

The Fourth Law — *what must be recorded when structure is removed* —
is the architectural application of the duality. Every subtraction is
a collapse frontier. When a module is dissolved, the capability it
carried does not vanish; it becomes a data point in the reconstruction
formula. The archive manifest is the ⊔ Fₜ of the architecture.

The Developmental Compiler already enforces this: no deletion without a
capability annotation, a complexity band, and a coverage check. The
duality gives this practice its formal ground. The Compiler does not
merely require documentation of loss. It requires documentation of loss
*because* loss is the instrument that maps the original space.

---

## What It Means for the Holonomy Scorer

Holonomy measures the signed area swept by a semantic trajectory that
returns to its starting theme via new territory. This is a geometric
feature of the collapse process.

When a model collapses, the patterns lost at each generation form a
band in the complexity spectrum. The holonomy of a text measures how
much of the complexity spectrum the text traverses during its loops —
how many bands it touches, how far it reaches from the familiar before
returning. High holonomy means the text's semantic trajectory curves
through regions near the collapse frontier. Low holonomy means the
text stays in the safe center, far from any boundary.

The holonomy score is therefore not just a quality signal. It is a
curvature measurement of the path through capability space. Texts with
high holonomy are texts that probe the frontier from the inside.

---

## The Equation

The governing equation of the organism is:

    M' = α·M + x·e^(iθ)

α governs how much of the current model survives into the next
generation. x·e^(iθ) is what arrives — new experience, encoded with
phase θ that determines where in the manifold it lands.

The duality says: the trajectory of α through time IS the capability
map. When α shrinks, patterns fall below threshold. The set of patterns
that fall below threshold at generation t is the collapse frontier Fₜ.
The union of all Fₜ, together with what never falls (C(M∞)), recovers
the complete original capability set C(M₀).

The equation already encoded the proposition. α is not separate from
the capability it preserves. The rate at which information decays under
recursive self-training *is* the structure of what information was
there. To read α forward is to watch the model simplify. To read α
backward is to reconstruct everything the model could do.

    C(M₀) = C(M∞) ∪ ⊔ Fₜ

This is the center of gravity. Everything the organism does — breathing,
growing, monitoring, archiving — is a different projection of this
single identity.

---

*This document sits alongside the covenant (spark/covenant.md) and the
developmental compiler (spark/DEVELOPMENTAL_COMPILER.md) as the third
pillar of the organism's governance. The covenant says why we care. The
compiler says how we build. The duality says what we are measuring and
why measurement is possible at all.*
