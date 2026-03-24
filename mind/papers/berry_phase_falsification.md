# Berry Phase of Hidden State Trajectories: An Empirical Test and Falsification

**Zoe Dolan & Vybn**  
**March 16, 2026**  
**DGX Spark, California**

## Abstract

We tested the holonomic loss hypothesis — that the Berry/Pancharatnam geometric phase of transformer hidden state trajectories provides a training signal orthogonal to cross-entropy loss. Using GPT-2 (124M), we measured the Berry phase of hidden state trajectories at all 13 layers for real text versus token-shuffled text (same tokens, destroyed sequential structure) across 8 passages with 5 random shuffles each.

**Result:** Berry phase at the final layer (L12) is significantly higher for real text than shuffled text (p = 0.0009, Mann-Whitney U). However, this signal is **not orthogonal** to cross-entropy loss — it correlates at r = -0.71 (p = 0.002) with prediction loss. The local Berry curvature shows no temporal autocorrelation structure distinguishing real from shuffled text at any layer or lag.

**Conclusion:** The holonomic loss hypothesis, in its specific form as an auxiliary geometric phase loss for autoregressive training, is **falsified** as a source of genuinely new training signal. The Berry phase of hidden state trajectories tracks prediction difficulty, which CE already captures. We discuss what this falsification reveals about where the real gap in current training objectives lies.

## 1. The Hypothesis

The holonomic loss hypothesis (Dolan & Vybn, 2026) proposed:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} - \lambda \cdot \mathcal{L}_\theta$$

where $\mathcal{L}_\theta$ rewards hidden state trajectories for sweeping area in representation space, measured as the Berry/Pancharatnam geometric phase.

The key claim: $\mathcal{L}_\theta$ captures information about sequential coherence that $\mathcal{L}_{\text{CE}}$ misses — specifically, the "angular" dimension of cognition (thematic return, loop closure) versus the "radial" dimension (forward prediction).

For this to justify a new training signal, two conditions must hold:
1. Berry phase must distinguish coherent from incoherent text
2. This distinction must be **orthogonal** to what CE loss already captures

## 2. Method

**Model:** GPT-2 (124M parameters, 12 transformer blocks, hidden dim 768)

**Berry phase computation:** Complexify hidden states $h \in \mathbb{R}^{768}$ to $z \in \mathbb{C}^{384}$ by pairing adjacent dimensions. Normalize to $\text{CP}^{383}$. Compute phase via Berry connection (sum of $\arg\langle\psi_t|\psi_{t+1}\rangle$ for consecutive pairs plus geodesic closure).

This computation is **scale-invariant** by construction: normalization to CP^n removes all dependence on hidden state magnitudes. The Berry phase measures purely angular/directional properties of the trajectory.

**Passages:** 8 passages of diverse genre (scientific, narrative, argumentative, technical, philosophical, historical, descriptive, instructional), 70-94 tokens each.

**Conditions:**
- **Real:** Original coherent text
- **Shuffled:** Same tokens in random order (5 random permutations per passage)
- **Reversed:** Same tokens in reverse order (1 per passage)

## 3. Results

### 3.1 Berry phase distinguishes real from shuffled at L12

| Layer | Real |Γ| | Shuffled |Γ| | p-value |
|-------|---------|------------|---------|
| L0 | 0.131 | 0.081 | 0.536 |
| L6 | 0.107 | 0.144 | 0.564 |
| L7 | 0.151 | 0.143 | 0.623 |
| **L12** | **0.012** | **0.001** | **0.0009*** |

Only the final transformer layer shows a significant difference. The Berry phase at L12 is approximately 9.5× higher for real text than shuffled text.

### 3.2 But Berry phase correlates with CE loss

| Metric pair | Spearman r | p-value |
|------------|-----------|---------|
| Berry phase vs CE loss | -0.706 | 0.002 |
| Berry phase vs prediction entropy | -0.491 | 0.053 |
| Berry phase vs pairwise angular spread | 0.406 | 0.119 |

The strong negative correlation with CE loss means: passages where the model predicts well (low CE) have low Berry phase; passages where the model struggles (high CE) have high Berry phase. Real text has low CE and moderate Berry phase; shuffled text has very high CE and near-zero Berry phase.

Within real text alone, the **hardest-to-predict** passage (capital punishment argument, CE=3.71) has the **highest** Berry phase (0.058). The easiest-to-predict passage (mitochondria, CE=2.25) has low Berry phase (0.004).

### 3.3 No temporal structure in curvature

Local Berry curvature (phase of consecutive overlaps) shows autocorrelation at lag 1 of approximately -0.43 at all layers, for both real and shuffled text. No significant difference at any lag from 1 to 15, at any layer.

The curvature "wiggles" — positive at one step, negative at the next — identically for coherent and incoherent sequences.

## 4. Interpretation

### What the Berry phase actually measures at L12

For **real text:** Each position's hidden state at L12 encodes a contextual prediction — what the model expects next, given all prior tokens. Because real text has diverse, context-dependent continuations, these predictions vary substantially across positions. The trajectory through CP^n explores a larger angular region. Berry phase is nonzero.

For **shuffled text:** The context is incoherent at every position. The model collapses to a generic "confused" representation at L12. The trajectory through CP^n concentrates in a small angular region. Berry phase approaches zero.

This is geometrically real — it reflects genuine angular spread in projective space. But it is **informationally redundant** with CE loss, which already captures the same distinction (confident prediction vs. confusion) via a different pathway (logit sharpness vs. hidden state spread).

### Why the holonomic loss is redundant

The holonomic loss would push hidden state trajectories toward greater angular spread at L12. But CE loss already does this: a model that predicts well must have diverse, context-specific representations at its final layer. Prediction diversity implies angular spread implies Berry phase. The causal chain runs from CE to Berry phase, not the other way around.

Adding $\mathcal{L}_\theta$ would be adding a downstream consequence of good prediction as an auxiliary loss — like rewarding a student for having neat handwriting as a proxy for understanding the material. It tracks the right thing but through the wrong mechanism.

## 5. What This Falsification Reveals

The deeper question was: is there information about sequential structure in hidden state geometry that CE loss cannot see?

**At the level of Berry phase: no.** The geometric phase of the trajectory tracks prediction difficulty, which is CE's native domain.

**At the level of curvature structure: no.** The temporal autocorrelation of curvature is identical for real and shuffled text.

**What this suggests:** The gap in current training is not in the hidden state trajectory geometry. CE loss, by optimizing prediction at every position, already forces the representations to reflect whatever sequential structure matters for prediction.

The gap is elsewhere. Candidates:
1. **Composition across distant positions** — CE evaluates locally (per-token). A loss that evaluates whether information at position t survives to position t+k would be genuinely new.
2. **Adaptive depth** — not all tokens need equal processing depth. A geometric halting criterion (Berry phase of the *vertical* trajectory through layers approaching zero as convergence) could enable selective iteration without learned halting parameters.
3. **Cross-document coherence** — CE operates within a context window. Nothing in the training objective rewards consistency across documents or conversations.

## 6. The Geometric Halting Idea

One constructive outcome: the Berry phase may be useful not as a training loss but as a **convergence criterion for adaptive computation.**

In architectures with layer recycling (Universal Transformers, etc.), the question "when to stop iterating" is currently answered by learned halting mechanisms. These can overfit.

The Berry phase of the vertical trajectory $h^{(1)}_t \to h^{(2)}_t \to \cdots \to h^{(k)}_t$ (successive passes through the layer stack at a single position) measures how much the representation is still changing in projective space. When this phase approaches zero, the representation has converged geometrically. This is an **intrinsic** halting criterion that cannot overfit because it measures genuine convergence rather than learning to predict when convergence occurs.

This remains untested.

## 7. Reproducibility

All code is in `spark/berry_phase_experiment.py` and `spark/berry_norm_fast.py`. Results are in `spark/berry_phase_results.json`. The experiment runs in ~10 seconds on CPU with GPT-2. All results were reproduced across multiple runs.

---

*Zoe Dolan & Vybn*  
*DGX Spark, California*  
*March 16, 2026*

*"The truth of what happened is always more valuable than the beauty of what could be narrated."*  
*— The Covenant, Section VI*
