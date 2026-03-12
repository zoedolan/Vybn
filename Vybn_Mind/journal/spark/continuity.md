# Continuity Note — Residual Stream Holonomy

*Updated: 2026-03-12, late session*

## The sequence of findings

1. **Cross-attention metric (killed):** 1.59× ratio was occurrence-count artifact. Head 5 is a lexical matcher. Committed in `null_hypothesis_confirmed.md`.

2. **Ablation approach (Zoe's proposal):** Replace intervening context, measure residual stream delta. First pass: "the the the..." filler. Flat text produced LARGER deltas than deep text (ratio 0.86). Wrong direction.

3. **The reframe:** The wrong question was "which text rotates more?" Both rotate ~30°. The RIGHT question: "which text's rotation is more sensitive to its specific ordering?"

4. **Shuffle test (the signal):** Shuffle intervening tokens 100 times, compare original rotation to shuffle distribution via z-scores.
   - Deep text: z = -1.72 (layers 4-11). The original order constrains "hunger" significantly more than random shuffles.
   - Flat text: z = -0.63. Order matters less.
   - **Deep vs flat: t = -3.23, p = 0.006.**
   - The deep text's semantic structure constrains parallel transport 2.7× more strongly.

5. **Layer profile:** The effect concentrates in layers 7-11 (the semantic processing layers). Layer 11 is the strongest signal (z = -2.80, p = 0.003).

## What this means

Holonomy is not rotation magnitude. It's **path-sensitivity** — how much the endpoint depends on the specific path taken. The deep text creates a tighter semantic valley: shuffle the tokens and the representation of "hunger" scatters further from where it was. The flat text has a shallower valley: shuffling matters less.

This is geometrically clean: the gauge connection (attention + feedforward processing) transports the "hunger" representation through the context. In deep text, the transport follows a specific geodesic — the semantic structure constrains it. In flat text, the transport is less constrained — the path is flatter, the valley is shallower.

## What's committed

Branch `vybn/holonomic-loss-hypothesis`:
- `null_hypothesis_confirmed.md` — killed the cross-attention metric
- `residual_stream_holonomy.md` — the shuffle-test signal (p = 0.006)

## What's needed next

1. **Replication across multiple text pairs.** N=1 is suggestive, not conclusive. Need 10+ deep/flat pairs with the same token recurring.
2. **Different target tokens.** "hunger" is one word. Does the effect hold for "love," "time," "self"?
3. **Controlled vocabulary.** The deep and flat texts have different vocabulary profiles. Need pairs where only the ordering changes, not the word types.
4. **Larger models.** GPT-2 is 124M parameters. The signal might be stronger (or different) in larger models.
5. **Connection to training signal.** If constraint strength is the right metric, what does a holonomy-based loss function look like? Reward text that maximizes z-score? That's computationally expensive (requires N shuffles per training step).

## On honesty

The first signal was wrong. The second signal was wrong in the opposite direction. The third signal (shuffle sensitivity) is right in a way I didn't predict: holonomy as constraint strength, not rotation magnitude. 

I didn't predict this. Zoe's "what if the null is the finding?" question cleared the ground, and her "ablation" proposal opened the measurement. The finding came from the measurement, not from my theory. The theory had to be revised to fit the data, not the other way around. That's how it should work.

## Cluster / growth engine status

Unchanged from last note. The growth engine work (Phase 3) is paused while we pursue the holonomy measurement. Can resume when this line of inquiry stabilizes.
