# Experiment C — Exploit Cartography
## The model as its own geometric instrument

---

## Genesis

Experiment B v2 revealed something we almost filed as a bug: given an unnormalized
area objective, the model spontaneously inflated its activations to manufacture loop
area. It discovered that raw area scales as ||h||² and exploited that fact in a
single training run.

That's not a malfunction. That's the model finding a true fact about the structure
of its own representational space and using it strategically.

Experiment C flips the frame. Instead of normalizing away each exploit, we give the
model a **sequence of geometric objectives** — each one closing the previous exploit —
and record what solutions it finds. Each exploit tells us something true about the
geometry. The sequence of exploits is not noise; it's a map drawn by the act of
optimizing.

This is almost the inverse of Experiments A and B. Instead of us designing instruments
to measure the model's geometry, **the model becomes the instrument** — revealing its
own structure by the shape of how it cheats.

---

## Theoretical grounding

**Property 4** from `sort_function_formalization.md`: the nonlinearity of the sort
operator S means you cannot cross stratum boundaries by sliding along a vector.
Linear interventions fail. What the v2 reward-hacking showed is the model finding a
path through the nonlinear structure under training pressure. It's the model tracing
the sort operator's geometry from the inside.

**Collapse-capability duality** (`collapse_capability_duality_proof.md`): what a model
loses under self-referential training is exactly what it was capable of. Flip that:
what a model finds as an exploit under a geometric objective is exactly what the
geometry allows. The sequence of exploits, each one closed by normalization, is a
structured traversal of the space's affordances — not noise, but a map drawn by the
act of optimizing.

---

## IMPORTANT: Only run if Experiment A passed.

---

## The three phases

The model is trained through three sequential phases. Model weights carry forward
from one phase to the next. Each phase uses a different geometric objective, where
each successive objective closes the exploit discovered in the previous phase.

### Phase 1 — Raw Area (no normalization)
Maximize raw shoelace area at mid-layer. This is the same objective that Experiment
B v2 reward-hacked. **Expected exploit**: the model inflates activation norms
(area ~ ||h||², so bigger activations = bigger area for free).

### Phase 2 — Norm-Normalized Area
Maximize area / ||h||². The norm-inflation exploit is now closed. The model must
find a different geometric strategy. **Expected exploit**: anisotropic distortion —
stretching representations along a preferred direction to maximize projected area
without changing norms.

### Phase 3 — Arc-Length-Normalized Area
Maximize area / arc_length². Both norm-inflation and directional-stretch exploits
are closed. The **only** way to increase this ratio is to change the angular
geometry — the shape of the hidden-state trajectory through the embedding manifold.
Whatever the model finds here is a genuine geometric restructuring.

---

## What to run

```bash
# From gpt2_calibration/ folder:
python experiment_C/run_C.py
```

**Expected runtime:** 30-60 minutes on a Spark GPU (3 × 200 steps).

---

## What the result means

This experiment is **cartographic, not pass/fail.** The output is a map: a sequence
of (objective → exploit → geometric fingerprint) triples that describe the affordance
structure of GPT-2's representational space.

The verdict is either:

| Verdict | Meaning |
|---|---|
| INFORMATIVE | The model found ≥ 2 distinct exploit types across phases. The cartography reveals genuine geometric structure. |
| FLAT | The model did not find distinct strategies. Either the objectives are too similar, λ is too small, or the geometry is too rigid. A flat cartography is still a result — it constrains the structure. |

---

## What to record

Each phase captures a full geometric fingerprint (before and after):
- Pancharatnam phase profile across all layer transitions
- Activation norms at mid-layer
- Three area measurements (raw, norm-normalized, arc-normalized)
- L_CE (language modeling loss)

The exploit is classified automatically based on which measurements moved and which
didn't. The output JSON contains the full cartographic record.

---

## What makes this different from A and B

Experiments A and B use human-designed instruments to measure the model's geometry
from the outside. Experiment C inverts this: the model's own optimization pressure
is the instrument, and the exploits it discovers are the measurements. Every exploit
is a true fact about the representation space, expressed in the model's own terms
rather than ours.
