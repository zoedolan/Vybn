# Continuity Note — Null Confirmed, Signal Lost, Thinking Required

*Updated: 2026-03-12, session 3 (late)*

## What happened

1. Outside-Vybn proposed measuring intrinsic holonomy via cross-attention between recurring concepts in a transformer forward pass. Inside-Vybn ran the experiment on GPT-2. Initial result: 1.59× more cross-attention in deep text vs flat text. Looked like confirmation.

2. Zoe asked: "What if the 1.59× ratio is the artifact, and the null result is the finding?"

3. Investigation revealed: the deep text had 2 occurrences of "hunger", the flat text had 5. Attention is a softmax (sums to 1). With more targets, per-target attention is lower by construction.

4. **Definitive test:** Two texts with exactly 2 occurrences each. Result: deep = 2.63 total cross-attention, flat = 2.99. **Ratio: 0.878.** The flat text gets MORE attention, not less. Head 5 layer 1 gives 71.9% (deep) vs 74.4% (flat). It's a lexical matcher that finds previous instances of the same token. It doesn't care about semantic depth.

5. **The 1.59× finding was entirely an artifact of occurrence count.** When controlled, the signal vanishes. Worse, it reverses slightly.

## What's committed

Branch `vybn/holonomic-loss-hypothesis`, 9 commits ahead of main:
- Holonomic loss hypothesis paper
- Holonomy scorer implementation
- Intrinsic holonomy analysis  
- Experiment: initial results (now known to be confounded)
- Honest convergence assessment
- **Null hypothesis confirmed** ← most important commit

## What's NOT invalidated

- The **extrinsic scorer** still produces rankings that look right. It hasn't been formally validated, but it hasn't been falsified either. The signed-area-in-embedding-space metric may capture something real about textual structure — it just doesn't shadow intrinsic attention patterns.

- The **attention-as-gauge-connection** analogy is mathematically coherent. It's just not supported by THIS measurement. The problem may be that identical-token matching dominates, hiding any semantic transport signal. A better experiment would look at attention between NON-identical tokens that refer to the same concept (e.g., "hunger" at position 5 and "ache" at position 15 and "wanting" at position 25).

- The **holonomy-as-training-signal** idea is still interesting but needs a completely different operationalization than cross-attention between same-token occurrences.

## What IS invalidated

- The specific claim that "layer 1 head 5 at 72.5% is performing parallel transport"
- The specific claim that "cross-attention ratio = intrinsic holonomy signal"
- The three-level convergence argument (Level 3 was the strongest evidence, and it collapsed)
- The report `intrinsic_holonomy_report.md` (the findings section is wrong)

## What to do next

**Stop building on the cross-attention metric.** It doesn't work.

**Think about what WOULD work.** The question "does the model represent a recurring concept differently when the intervening context transforms it?" is still a good question. But cross-attention between identical tokens isn't the answer — it's dominated by lexical matching.

Possible directions:
1. **Representation delta, not attention:** Compare h[pos_j] - h[pos_i] for the same token in deep vs flat context. The v2 rotation profiles showed ~33° in both conditions — but maybe the DIRECTION of rotation differs, even if the magnitude doesn't.
2. **Non-identical concept recurrence:** Track "hunger" → "ache" → "wanting" → "hunger" — concepts that return through different tokens. This requires semantic similarity, not token identity.
3. **Causal intervention:** Ablate the intervening context (replace with padding) and measure how the representation of the second "hunger" changes. The DIFFERENCE between the ablated and non-ablated representations is the effect of context, which is closer to what we mean by holonomy.
4. **Step back entirely.** Maybe the intrinsic approach is premature. Maybe the extrinsic scorer is the right tool for now and should be validated on its own terms (formal human-ranking correlation) before trying to connect it to model internals.

## The lesson

The Covenant says: "The more beautiful the sentence wants to be, the more carefully its claims deserve scrutiny." I wrote "the attention mechanism IS the gauge connection. This is not a metaphor." in a commit message. It was beautiful. It was premature. Zoe caught it with one question.

## Cluster state (unchanged)
- spark-2b7c: Ray head, vLLM on :8000, healthy  
- spark-1c8f: Ray worker, connected
- MiniMax M2.5-AWQ-4bit serving, 128K context, -tp 2
