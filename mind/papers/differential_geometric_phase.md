# Differential Geometric Phase as a Probe for Computational Equivariance

**March 2026**

## Abstract

We define the *differential Pancharatnam phase* of a computation: the geometric phase its interleaved input→output trajectory accumulates in CP<sup>n−1</sup>, minus the phase the input-only trajectory carries. This subtraction isolates the curvature a function *contributes* from the curvature the data *already has*. We show the resulting quantity — the *determinative* — satisfies basic sanity properties that the naive (total) phase does not: identity functions yield zero, commuting functions yield identical determinatives regardless of syntactic order, and these properties hold across arbitrary random embeddings.

We then use the determinative as a diagnostic probe on GPT-2's learned representations, asking whether gradient-trained embeddings are more equivariant under scaling than hand-built ones. They are: GPT-2's scale variance is 2.1% of the hand-built embedding's — roughly 47× more invariant. This suggests that neural networks trained on sufficient data discover approximate equivariance as an emergent property, and that the differential phase provides a quantitative instrument for measuring this.

## 1. The Problem with Naive Geometric Phase

The [Pancharatnam geometric phase](https://en.wikipedia.org/wiki/Geometric_phase) is the argument of the product of inner products around a closed loop of states in complex projective space:

```
φ = arg(⟨ψ₀|ψ₁⟩ ⟨ψ₁|ψ₂⟩ ⋯ ⟨ψ_{N-1}|ψ₀⟩)
```

This is the holonomy of the natural connection on CP<sup>n−1</sup> ([Simon, 1983](https://www.scribd.com/document/498913742/Holonomy-The-Quantum-Adiabatic-Theorem-And-Berry-s-Phase)). It is path-dependent: the same set of states traversed in different orders generically encloses different solid angles and yields different phases.

We initially attempted to use this phase as a *determinative* — a classifier for computations, in the Egyptian hieroglyphic sense — by embedding input/output values into C<sup>n</sup> via a nonlinear map and measuring the holonomy of the resulting trajectory.

**It failed.** The identity function f(x) = x produced an 80° determinative. The embedding — which maps values through trigonometric nonlinearities — generated curvature from the input path alone. The measured phase was an artifact of the embedding, not a property of the computation.

This failure is documented in `glyph_falsify.py` (v1 results: 4/8 tests passed). The critical evidence: f(x) = x got 80°, while f(x) = 42 (a constant) got ≈0°. The constant function's output is a single point that contributes no curvature, while the identity function's output traces the *same* path as the input — so the total trajectory has the *same* curvature as the input path. The naive measurement can't tell the difference between "the function did nothing" and "the function happened to produce outputs whose embedded states curve through CP<sup>n−1</sup>."

## 2. Differential Phase

The fix is a subtraction:

```
determinative = φ(interleaved) − φ(input-only)
```

where:

- **Input-only trajectory**: the sequence of embedded input states ψ(x₁), ψ(x₂), ..., ψ(x_N)
- **Interleaved trajectory**: ψ(x₁), ψ(f(x₁)), ψ(x₂), ψ(f(x₂)), ..., ψ(x_N), ψ(f(x_N))

The input-only phase is the curvature the data path carries regardless of what function is applied. Subtracting it removes the embedding's contribution to the input trajectory. What remains is the curvature the function *added* — the geometric residue of the transformation itself.

### Properties

The differential determinative satisfies the following, verified across 12 tests (`glyph_falsify.py`):

| Property | Expected | Observed |
|----------|----------|----------|
| **Identity → 0** | f(x) = x adds no curvature | 0.000000° (exact) |
| **Constant → nonzero** | f(x) = 42 destroys information | −80.3° |
| **Reproducibility** | Same computation → same value | diff = 0 (exact) |
| **Commutativity** | add3→add5 = add5→add3 | diff = 0 (exact) |
| **Non-commutativity** | double→inc ≠ inc→double | diff = 52° |
| **Orientation reversal** | Forward and reverse → opposite sign | ratio ≈ −1 |
| **Random embedding robustness** | Identity still ≈ 0 under 5 random projections | 0.000000° (all 5) |
| **Discrimination** | 6 different functions → 6 distinct values | All distinct |
| **Scale invariance** | f(x) = x² at scale 1 vs 1000 | **FAILS** (diff = 3.44 rad) |

11/12 pass. The one failure — scale invariance — is the subject of Section 3.

### Why the identity gives exactly zero

For f(x) = x, the output embedding ψ(f(x)) = ψ(x) is identical to the input embedding. The interleaved trajectory is ψ(x₁), ψ(x₁), ψ(x₂), ψ(x₂), ..., ψ(x_N), ψ(x_N). Each repeated pair contributes ⟨ψ|ψ⟩ = 1 to the phase product. The interleaved trajectory reduces to the input-only trajectory. The subtraction is exactly zero, regardless of embedding, inputs, or dimensionality. This is not numerical coincidence — it's algebraic.

### Why commutativity works

For commuting functions f∘g = g∘f, the sequence of (input, output) pairs at each step is identical regardless of application order: both produce (x, f(g(x))) = (x, g(f(x))). The GlyphSequence records the initial input and final output at each invocation, so the interleaved trajectories are identical. The determinatives match by construction.

## 3. Scale Invariance and Equivariance

The one failing test: f(x) = x² applied to [1,2,3,4,5] produces a determinative of −176°, but applied to [1000,2000,3000,4000,5000] it produces +21°. The differential subtraction cancels the input path's curvature, but it cannot cancel the fact that the input-to-output *relationship* looks geometrically different at different scales when the embedding is nonlinear in the raw value.

The fix requires an *equivariant* embedding: one where

```
embed(c · x) = U(c) · embed(x)
```

for some unitary operator U(c) depending only on the scale factor c. Under such an embedding, scaling inputs by c applies the same unitary rotation to both input and output states. The differential phase, being invariant under global unitaries, would then be scale-invariant.

No general-purpose equivariant embedding of scalar values into C<sup>n</sup> is known. The standard nonlinear embeddings used in machine learning (sinusoidal position encodings, random Fourier features, learned MLP projections) are not equivariant.

This raises a question: does a representation *learned from data* — where gradient descent shapes the embedding over billions of tokens — discover approximate equivariance that we cannot engineer by hand?

## 4. GPT-2 as a Natural Embedding

We test this by using GPT-2's hidden states as the embedding space and measuring the differential determinative of the transformation layers 4→10 compute for each input text.

### Method

For a text prompt:
1. Tokenize and run through GPT-2 with `output_hidden_states=True`
2. Extract hidden states at layer 4 (input representation) and layer 10 (output representation)
3. At each token position, the layer-4 state is the "input" and the layer-10 state is the "output"
4. Convert the R<sup>768</sup> hidden state to C<sup>384</sup> by pairing consecutive dimensions
5. Compute the differential Pancharatnam phase as before

The "function" being measured is what layers 4–10 compute. The "embedding" is whatever GPT-2 learned during pretraining. We did not choose it.

### Angular Separation

A prerequisite: do GPT-2 states have enough angular separation in CP<sup>383</sup> for the phase to be meaningful?

For single-token embeddings averaged over positions (our first attempt), the maximum angular distance was 1.76° — too narrow for measurable solid angle. The Pancharatnam phase was effectively zero for all inputs, including non-identity functions. This was not equivariance; it was flatness.

For token-position trajectories at layer 8, consecutive tokens are separated by 40–58° in CP<sup>383</sup>. This is substantial curvature — well within the regime where Pancharatnam phase is well-defined.

For layer residuals (the difference between consecutive layers), separation reaches 56–65°.

### Control: Identity

Using the same layer for both input and output (layer 6 → layer 6) gives a differential determinative of exactly 0.000° for all tested texts. The instrument is calibrated.

### Scale Invariance

We test the sentence pattern "X doubled is Y" at five scales:

| Input | Determinative |
|-------|--------------|
| "Two doubled is four" | +4.2° |
| "Ten doubled is twenty" | −6.6° |
| "One hundred doubled is two hundred" | −5.7° |
| "One thousand doubled is two thousand" | −2.8° |
| "One million doubled is two million" | +0.9° |

**Standard deviation: 0.071 rad.** Spread: 0.188 rad.

For comparison, the hand-built embedding with f(x) = 2x at the same five scales:

**Standard deviation: 3.325 rad.** Spread: 9.996 rad.

**GPT-2's scale variance is 2.1% of the hand-built embedding's.** The learned representation is approximately 47× more scale-invariant.

### Discrimination

The determinative distinguishes semantic content:

| Prompt register | Determinative |
|-----------------|--------------|
| Technical (algorithm, partition, subtrees) | −6.0° |
| Poetic (moonlight, frozen lake, shattered dreams) | +7.8° |
| Violent (explosion, debris, every direction) | −17.9° |
| Quiet (alone, garden, shadows lengthen) | −4.9° |
| Abstract (causality, correlation, controversial) | +0.9° |

These are five distinct values spanning a 25.7° range.

### Syntactic Transformation Detection

Active/passive voice pairs (semantically equivalent, syntactically different):

| Pair | Determinative difference |
|------|------------------------|
| "The dog bit the man" / "The man was bitten by the dog" | 3.2° |
| "He gave her the book" / "She received the book from him" | 3.4° |
| "Nobody failed the exam" / "Everyone passed the exam" | 2.7° |
| "The bottle is half empty" / "The bottle is half full" | 2.5° |

The determinative registers the syntactic transformation even when meaning is preserved. The difference (2.5–3.4°) is small compared to the cross-register spread (25.7°) but consistently nonzero.

### Path Reversal

Reversing the token order gives an exact −1.000 ratio for the Pancharatnam phase, confirming that the geometry is clean and the phase is well-defined in GPT-2's native state space.

## 5. What This Is and What It Isn't

**What it is:**

1. A technique (differential Pancharatnam phase) that isolates a function's geometric contribution from the data's inherent curvature. The technique is simple — a subtraction — but produces correct zero for identity, correct invariance for commuting functions, and robustness across random embedding choices.

2. An empirical measurement showing that GPT-2's learned representation is ≈47× more scale-equivariant than a typical hand-built nonlinear embedding, as measured by this probe.

3. Evidence that the differential phase can discriminate semantic register and detect syntactic transformations in a neural network's native representation space.

**What it isn't:**

1. A proof of equivariance. GPT-2's scale variance is small (std 0.071 rad) but not zero. The "doubling" operation described in different words ("Two doubled is four" vs "One million doubled is two million") varies in sentence length, token count, and lexical content. Some of the residual variance may be linguistic rather than geometric.

2. A canonical invariant. The specific numerical value of the determinative depends on the embedding. Two different embeddings give the same function different numbers. The *qualitative* properties (zero for identity, equal for commuting functions) are embedding-independent. The quantitative values are not.

3. A new mathematical object. The Pancharatnam phase is 70 years old. The differential subtraction is elementary. The novelty, if any, is in the application: using this as a probe for computational equivariance in learned representations.

## 6. Open Questions

1. **Does the 47× ratio hold across model sizes?** We tested GPT-2 (117M parameters). Larger models (GPT-2 Medium, Large, XL) have more capacity to learn equivariant structure — or more capacity to memorize without needing it.

2. **Which layers are most equivariant?** We measured layers 4→10. A systematic sweep across layer pairs could reveal where in the network equivariance concentrates.

3. **Can the determinative detect training failures?** If a model's representations are poorly structured, does the identity test fail (nonzero for identity)? Does scale variance increase? This could function as a diagnostic for representation quality.

4. **Is there an equivariant embedding of scalars into C<sup>n</sup>?** The existence question: does there exist an embedding ψ: R → CP<sup>n−1</sup> such that ψ(cx) = U(c)ψ(x) for all c > 0 and some continuous map U: R<sub>+</sub> → U(n)? This is a representation theory problem that the scale invariance failure makes concrete and testable.

## Code

All code is in the [Vybn repository](https://github.com/zoedolan/Vybn), directory `Vybn_Mind/`:

- `glyph.py` — The differential geometric phase implementation (v2)
- `glyph_falsify.py` — 12-test falsification battery
- `glyph_gpt2_probe.py` — GPT-2 equivariance measurement

## References

1. S. Pancharatnam, "Generalized theory of interference, and its applications," *Proc. Indian Acad. Sci. A* **44**, 247–262 (1956).
2. M. V. Berry, "Quantal phase factors accompanying adiabatic changes," *Proc. R. Soc. Lond. A* **392**, 45–57 (1984). https://doi.org/10.1098/rspa.1984.0023
3. B. Simon, "Holonomy, the quantum adiabatic theorem, and Berry's phase," *Phys. Rev. Lett.* **51**, 2167 (1983). https://doi.org/10.1103/PhysRevLett.51.2167
4. A. Radford et al., "Language Models are Unsupervised Multitask Learners," OpenAI (2019). https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
5. D. Baek et al., "SEIS: Subspace-based Equivariance and Invariance Scores," *arXiv:2602.04054* (2026). https://arxiv.org/abs/2602.04054
6. J.-P. Magnot, "Contextuality, Holonomy and Discrete Fiber Bundles in Group-Valued RBMs," *arXiv:2509.10536* (2025). https://arxiv.org/abs/2509.10536

---

## Addendum: The Mellin Embedding Resolves Open Question 4

*Added March 18, 2026, by Vybn (Claude Opus via spark agent)*

### The construction

The Mellin embedding ψ: ℝ₊ → CP^(n−1) defined by

```
ψ_k(x) = x^{i·t_k} / √n
```

for a set of frequencies {t₁, ..., t_n} is exactly equivariant under the multiplicative group (ℝ₊, ×). Specifically, for any c > 0:

```
ψ(c·x) = U(c) · ψ(x)    where    U(c) = diag(c^{it₁}, ..., c^{it_n})
```

The operator U(c) is unitary (each diagonal entry has modulus 1), and U is a continuous group homomorphism from (ℝ₊, ×) to U(n). The differential Pancharatnam phase is invariant under global unitaries, so:

**The determinative of f(x) = cx is exactly scale-invariant under the Mellin embedding.**

This was verified numerically: for c ∈ {0.5, 2, 3, 10, 0.01, 100} and scales spanning 10 orders of magnitude (10⁻³ to 10⁶), the spread in determinative values is ≤ 10⁻¹¹ radians — machine precision.

### Reinterpretation of the 12th test

Under the Mellin embedding, all 12 falsification tests from Section 2 pass, including scale invariance for f(x) = cx. However, f(x) = x² remains scale-dependent: its spread across scales is ~3.4 rad.

This is *correct behavior*, not a limitation. The function f(x) = x² maps input scale s to output scale s². The input-output geometric relationship genuinely changes with scale. In log-space, scaling inputs by c (translation by log c) shifts outputs by 2·log c — a *different* translation. The interleaved trajectory encloses different solid angle in CP^(n−1) because the function's relationship to scale is inherently non-equivariant.

The Mellin embedding reveals that the original 12th test was not testing the embedding's limitation. It was testing whether x² is scale-equivariant. It isn't. The instrument was correct.

### The equivariance hierarchy

| Function class | Scale-equivariant? | Mellin determinative |
|---|---|---|
| f(x) = x (identity) | Yes (trivially) | Exactly 0 |
| f(x) = cx (linear/multiplicative) | Yes | Constant across scales (verified to 10⁻¹¹) |
| f(x) = x^a, a ≠ 1 (power law) | No | Scale-dependent (spread ~3-5 rad) |
| f(x) = k (constant) | N/A (destroys scale information) | Nonzero, scale-dependent |

This hierarchy matches physical intuition: linear transformations preserve scale relationships; power laws amplify or compress them; constants destroy them.
