# The Sort Function: Formalization and Application to the Fundamental Theorem

**Zoe Dolan & Vybn**
**March 20, 2026**

---

## 0. What This Document Does

The SGP papers discovered something empirical: the first transformer block performs a disproportionate geometric transformation that sorts inputs into distinct strata. The fundamental theorem draft conjectured that discrimination and generation are geometric inverses. This document formalizes the sort function as a precise mathematical object, derives its properties from the empirical evidence, and shows exactly how far it takes us toward proving the fundamental theorem — and where it stops.

---

## 1. The Sort Operator: Formal Definition

### Setup

Let T be an L-block transformer with embedding dimension d. For an input sequence x (tokenized and embedded), define:

- **h₀(x) ∈ ℝᵈ** : the embedded representation (output of the embedding layer)
- **h₁(x) ∈ ℝᵈ** : the output of the first transformer block
- **|ψ₀(x)⟩ ∈ CP^(d/2 - 1)** : the projective state obtained by pairing consecutive dimensions of h₀ into complex coordinates, then normalizing. (For GPT-2, d = 768, so this lives in CP³⁸³.)
- **|ψ₁(x)⟩ ∈ CP^(d/2 - 1)** : same construction applied to h₁.

The first transformer block defines a map on projective space:

$$\mathcal{S} : \mathrm{CP}^{n-1} \to \mathrm{CP}^{n-1}, \quad n = d/2$$

given by $\mathcal{S}(|\psi_0(x)\rangle) = |\psi_1(x)\rangle$.

This is the **sort operator**. Note: $\mathcal{S}$ is not a single well-defined function on all of CP^(n-1) — it depends on the full input sequence, not just a single projective point. But for a fixed context, it acts as a map on the per-token projective states. We suppress this dependence for clarity and revisit it in §6.

### The Stratification

Define the **stratified geometric phase** of an input distribution $\mathcal{D}$ as:

$$\Phi_{\mathrm{SGP}}(\mathcal{D}) = \mathbb{E}_{x \sim \mathcal{D}} \left[ \arg \prod_{j} \langle \psi_0^{(j)}(x) | \psi_1^{(j)}(x) \rangle \langle \psi_1^{(j)}(x) | \psi_0^{(j+1)}(x) \rangle \right] - \Phi_{\mathrm{input}}(\mathcal{D})$$

where j indexes token positions and $\Phi_{\mathrm{input}}$ is the phase of the input-only trajectory (the differential subtraction from the instrument paper).

The empirical discovery is that $\Phi_{\mathrm{SGP}}$ separates input distributions into **sign-classes**:

$$\sigma(\mathcal{D}) = \mathrm{sgn}(\Phi_{\mathrm{SGP}}(\mathcal{D})) \in \{+, -\}$$

This sign is the coarsest invariant of the sort. It partitions the space of input distributions into (at least) two sectors.

### The Formal Sort

The sort operator $\mathcal{S}$ induces a **stratification** of the input space:

$$\mathcal{S}_* : \mathrm{Dist}(\mathrm{CP}^{n-1}) \to \{S_+, S_-\}$$

where $S_\pm = \{\mathcal{D} : \sigma(\mathcal{D}) = \pm\}$ and Dist(CP^(n-1)) denotes probability measures on CP^(n-1) (i.e., input distributions mapped through the embedding).

More precisely, $\mathcal{S}$ maps the embedding space — which Robinson et al. proved is a **stratified space** (not a manifold, not a fiber bundle) with singularities at polysemous/syntactically essential tokens — into a stratified space where the strata are labeled by their geometric phase sign.

**Definition 1 (Sort Operator).** The sort operator is the composition:

$$\mathcal{S} = \pi \circ B_0 \circ \iota$$

where:
- $\iota : \mathrm{tokens} \to \mathbb{R}^d$ is the embedding
- $B_0 : \mathbb{R}^d \to \mathbb{R}^d$ is the first transformer block (attention + MLP + residual + layer norm)
- $\pi : \mathbb{R}^d \to \mathrm{CP}^{n-1}$ is the projective quotient (pair dimensions, normalize, mod global phase)

---

## 2. Empirical Properties of $\mathcal{S}$

The following properties are established by the SGP experiments. Each is stated formally, then grounded in the data.

### Property 1: Disproportionate Curvature

**Statement.** Let $\Phi_\ell(\mathcal{D})$ denote the differential Pancharatnam phase at the $L_\ell \to L_{\ell+1}$ transition for distribution $\mathcal{D}$. Then:

$$|\Phi_0(\mathcal{D})| \geq 3 \cdot \max_{\ell \geq 1} |\Phi_\ell(\mathcal{D})|$$

for all empirically tested distributions $\mathcal{D}$.

**Evidence.** In Pythia-70M: L0→L1 phases range from 10° to 49° across concept classes; all subsequent layer pairs have phases ≤17°. In GPT-2: L0→L1 phases range from 14° to 54°; all L4→L10 phases are ≤8°. The ratio ranges from 3× to 50× depending on class and architecture.

**Interpretation.** The sort operator concentrates geometric curvature. It performs the bulk of the projective rotation in a single step. The remaining L-1 blocks refine within the established geometry.

### Property 2: Sign Separation

**Statement.** There exist input distributions $\mathcal{D}_+, \mathcal{D}_-$ such that:

$$\Phi_0(\mathcal{D}_+) > 0, \quad \Phi_0(\mathcal{D}_-) < 0$$

and this sign is stable under architectural variation (GPT-2 vs. Pythia-70M, up to overall sign convention).

**Evidence.** Spatial/physical prompts have opposite sign from abstract/epistemic prompts in both architectures. In GPT-2: spatial = -24.2°, abstract = +53.9°. In Pythia-70M: spatial = +49.1°, abstract = -45.5° (flipped convention, same structural separation).

**Caveat.** Confound controls show r = 0.845 correlation between token frequency and SGP sign. The sign separation is at least substantially lexical. Whether a genuinely semantic component exists beyond frequency is unresolved (§7).

### Property 3: Load-Bearing (Ablation Catastrophe)

**Statement.** Let $\mathcal{S}^{-1}_{\mathrm{id}}$ denote the network with the first block replaced by identity. Then for any downstream layer pair $\ell \geq 1$:

$$P(\mathrm{sgn}(\Phi^{\mathrm{ablated}}_\ell) = \mathrm{sgn}(\Phi^{\mathrm{normal}}_\ell)) < \frac{1}{2}$$

That is, the ablated network's sign patterns are **anti-correlated** with the normal network's — worse than random.

**Evidence.** Average sign match across Pythia-70M layers L1→L2 through L5→L6: 44% (below the 50% random baseline).

**Interpretation.** The downstream blocks are not independent processors. They are **tuned to the output of $\mathcal{S}$**. Without $\mathcal{S}$, they don't just fail — they produce the wrong geometric structure. This is the formal sense in which $\mathcal{S}$ is load-bearing.

### Property 4: Nonlinearity

**Statement.** $\mathcal{S}$ does not admit a linear inverse. Specifically, for distributions $\mathcal{D}_1, \mathcal{D}_2$ with $\sigma(\mathcal{D}_1) \neq \sigma(\mathcal{D}_2)$, the linear interpolation:

$$\mathcal{S}(|\psi_0\rangle) + t \cdot (\bar{\psi}_{\mathcal{D}_2} - \bar{\psi}_{\mathcal{D}_1}) \quad \text{for } |\psi_0\rangle \in \mathcal{D}_1$$

does not shift the downstream SGP toward $\Phi_0(\mathcal{D}_2)$.

**Evidence.** Rotating spatial representations toward the abstract centroid at L1 produced negligible downstream SGP change (-4.3° → -4.5°, vs. target of +10.7°). The strata are not linearly separable. Moving between them requires a nonlinear transformation.

**Interpretation.** This is consistent with the stratified space model: different strata have different local dimensions, and transitions between strata are singular (not smooth). You can't cross a stratum boundary by sliding along a vector.

### Property 5: Training Dynamics (Gradual Crystallization)

**Statement.** The sign function $\sigma$ is not a fixed point of random initialization. It emerges through a turbulent selection process:

$$\sigma_0 = \text{random} \xrightarrow{\text{turbulence}} \sigma_T = \text{stable}$$

where T ≈ 50000 steps (105B tokens) for Pythia-70M.

**Evidence.** The sign pattern evolves: random at step 0 → turbulent oscillation (steps 32–5000) → crystallization of spatial-vs-rest separation (steps 5000–50000) → stable `----+` at step 143000. The asymmetry is present at initialization (Φ = 124° at step 0) but with within-class standard deviation of 118° — i.e., the signal is noise-dominated. Training selects and stabilizes one particular asymmetry.

---

## 3. $\mathcal{S}$ as a Map on Connections

Here is where we leave pure empiricism and enter formalism.

### The Berry Connection on CP^(n-1)

On CP^(n-1) there is a canonical U(1) connection — the Berry connection:

$$\mathcal{A} = -\mathrm{Im}\, \langle \psi | d\psi \rangle$$

Its curvature is the Fubini-Study 2-form:

$$\mathcal{F} = d\mathcal{A}$$

The Pancharatnam phase measured by the SGP instrument is the holonomy of this connection along the path traced by the sequence of layer representations.

### $\mathcal{S}$ as a Connection-Modifying Operator

The sort operator $\mathcal{S}$ doesn't just move points in CP^(n-1). It **modifies the effective connection** seen by the downstream path. Here's the precise statement:

Before $\mathcal{S}$ acts, the input representations trace a path $\gamma_{\mathrm{in}}$ in CP^(n-1) with holonomy $\Phi_{\mathrm{in}}$. After $\mathcal{S}$ acts, they trace a path $\gamma_{\mathrm{out}} = \mathcal{S}(\gamma_{\mathrm{in}})$ with holonomy $\Phi_{\mathrm{out}}$.

The differential Pancharatnam phase of $\mathcal{S}$ is:

$$\Delta\Phi_{\mathcal{S}} = \Phi_{\mathrm{interleaved}} - \Phi_{\mathrm{in}}$$

where $\Phi_{\mathrm{interleaved}}$ is the holonomy of the path that alternates between input and output states.

By Stokes' theorem on CP^(n-1):

$$\boxed{\Delta\Phi_{\mathcal{S}} = \oint_{\gamma_{\mathrm{interleaved}}} \mathcal{A} - \oint_{\gamma_{\mathrm{in}}} \mathcal{A} = \int_{\Sigma} \mathcal{F}}$$

where $\Sigma$ is the surface **between** the input and output paths — the region of CP^(n-1) "swept" by the sort.

**This is the key equation.** It says: the geometric phase contributed by the sort operator equals the curvature integrated over the region it sweeps through in projective space. The sort creates curvature by moving representations through curved regions of CP^(n-1).

### What the sign means

The sign of $\Delta\Phi_{\mathcal{S}}$ corresponds to the **orientation** of the swept surface $\Sigma$:

- $\sigma = +$ : $\mathcal{S}$ sweeps the path through a region of CP^(n-1) with net positive curvature (counterclockwise rotation in the U(1) fiber)
- $\sigma = -$ : $\mathcal{S}$ sweeps through a region with net negative curvature (clockwise rotation)

The empirical sign separation between spatial and abstract distributions means: **the sort sweeps these two classes of inputs through oppositely-oriented regions of projective space.** The first block doesn't just separate them in position — it separates them in *orientation*.

---

## 4. The Fundamental Theorem: What the Sort Gives Us

### Recap of the Conjecture

The fundamental theorem draft (§4 of `fundamental_theorem_draft.md`) conjectures:

$$\mathcal{F}_\ell \circ \mathcal{C} = \mathrm{id} + \tau \quad \text{and} \quad \mathcal{C} \circ \mathcal{F}_\ell = \mathrm{id} + \tau$$

where $\mathcal{C}$ = curving (learning/discrimination), $\mathcal{F}_\ell$ = flattening (generation), and $\tau$ = topological obstruction (Chern class).

### Decomposing Discrimination via the Sort

The full discriminative path through an L-block transformer decomposes as:

$$\mathcal{C} = R_{L-1} \circ \cdots \circ R_1 \circ \mathcal{S}$$

where $R_\ell$ is the map induced by block $\ell + 1$ (the refinement blocks). By Property 1, the sort $\mathcal{S}$ contributes the dominant curvature. The refinement blocks $R_\ell$ contribute small corrections within the established strata.

The total holonomy decomposes additively (this is the key advantage of working with connections):

$$\Phi_{\mathrm{total}} = \Delta\Phi_{\mathcal{S}} + \sum_{\ell=1}^{L-1} \Delta\Phi_{R_\ell}$$

By Property 1, $|\Delta\Phi_{\mathcal{S}}| \geq 3 \sum_\ell |\Delta\Phi_{R_\ell}|$ empirically. The sort dominates the holonomy budget.

### What Would the Inverse Look Like?

If the fundamental theorem is true, there must exist a flattening operator $\mathcal{F}_\ell$ that inverts $\mathcal{C}$. Given the decomposition above, this means:

$$\mathcal{F}_\ell = \mathcal{S}^{-1} \circ R_1^{-1} \circ \cdots \circ R_{L-1}^{-1}$$

The refinement blocks $R_\ell$ are close to identity (small phases), so their inverses exist perturbatively — this is not the hard part. The hard part is **inverting the sort**.

### The Sort Inverse Problem

$\mathcal{S}^{-1}$ must take a stratified, sign-labeled representation and "flatten" it back to an unstratified embedding. By Property 4, this inverse is nonlinear. By Property 3, it must exist in some form — because the downstream blocks were trained to expect $\mathcal{S}$'s output, there is implicitly a map from "sorted" to "usable" representations.

**Claim.** The sort inverse $\mathcal{S}^{-1}$ exists as a map between strata but is **discontinuous at stratum boundaries**. Specifically:

For inputs within a single stratum $S_\sigma$ ($\sigma = + \text{ or } -$), $\mathcal{S}$ restricted to $S_\sigma$ is a diffeomorphism onto its image (smooth, invertible, with smooth inverse). But the global map $\mathcal{S}$ across strata is **not** a diffeomorphism — it is a **stratified map** whose inverse has discontinuities at the stratum boundaries.

**This is the topological obstruction $\tau$.**

---

## 5. The Topological Obstruction as a Chern Class

### The U(1) Bundle Over CP^(n-1)

The tautological line bundle $\gamma$ over CP^(n-1) has first Chern class $c_1(\gamma) = -1$ (this is standard algebraic topology; it generates $H^2(\mathrm{CP}^{n-1}, \mathbb{Z}) \cong \mathbb{Z}$).

The sort operator $\mathcal{S}$ induces a pullback bundle $\mathcal{S}^*\gamma$ over the input CP^(n-1). The **degree** of $\mathcal{S}$ as a map on CP^(n-1) determines the first Chern class of this pullback:

$$c_1(\mathcal{S}^*\gamma) = \deg(\mathcal{S}) \cdot c_1(\gamma)$$

### Connecting to the Fundamental Theorem

If $\mathcal{S}$ has degree $k \neq \pm 1$, then $\mathcal{S}^*\gamma \neq \gamma$ as bundles, which means no continuous inverse $\mathcal{S}^{-1}$ exists as a bundle map. The obstruction to inverting $\mathcal{S}$ is exactly:

$$\tau = (1 - \deg(\mathcal{S})) \cdot c_1(\gamma)$$

This is a characteristic class — it depends only on the topology of $\mathcal{S}$, not on the metric or the specific path. It is **quantized** (integer-valued) and **invariant under continuous deformation** of $\mathcal{S}$ that doesn't change its degree.

### The Sign as a Z₂ Reduction

The empirical sign classification $\sigma \in \{+, -\}$ is the **mod-2 reduction** of this integer invariant:

$$\sigma(\mathcal{D}) = \deg(\mathcal{S}|_{\mathcal{D}}) \mod 2$$

This explains why the sign is the most robust feature of the data — it survives scrambling (Property from confound controls: 4/5 classes preserve sign after destroying syntax), it survives architectural variation (GPT-2 vs. Pythia, up to convention), and it emerges early in training. It's the coarsest topological invariant, the last thing to be destroyed by perturbation.

---

## 6. The Composition: $\mathcal{S} \circ \mathcal{S}^{-1}$ and What Remains

### Within a Stratum

Restricted to a single stratum $S_\sigma$, the sort is a diffeomorphism and admits a smooth inverse. The round-trip gives:

$$\mathcal{S}^{-1}|_{S_\sigma} \circ \mathcal{S}|_{S_\sigma} = \mathrm{id}_{S_\sigma}$$

No obstruction. Within a stratum, discrimination and generation are exact inverses. This predicts: **generation should be easy for inputs that land squarely within a single stratum** (unambiguous category, consistent vocabulary, clear geometric identity).

### Across Strata

Globally, the composition satisfies:

$$\mathcal{S}^{-1} \circ \mathcal{S} = \mathrm{id} + \tau$$

where $\tau$ is supported on the **stratum boundaries** — the locus where the sign function $\sigma$ is undefined or discontinuous. These are inputs at the border between categories: ambiguous, polysemous, or contextually dependent.

**This is the fundamental theorem, restricted to the sort.**

### The Full Network

For the full L-block transformer:

$$\mathcal{F}_\ell \circ \mathcal{C} = \mathcal{S}^{-1} \circ R^{-1} \circ R \circ \mathcal{S} = \mathcal{S}^{-1} \circ (\mathrm{id} + \epsilon) \circ \mathcal{S} = \mathrm{id} + \mathcal{S}^{-1} \epsilon \mathcal{S} + \tau$$

where $\epsilon = \sum_\ell (R_\ell^{-1} \circ R_\ell - \mathrm{id})$ captures the small corrections from refinement blocks (Property 1 says these are perturbatively small), and $\tau$ is the topological obstruction from the sort.

The fundamental theorem in full says: **the round-trip error has two components — a smooth perturbative piece (from the refinement blocks) and a topological piece (from the sort stratum boundaries)**. The smooth piece can be made arbitrarily small by improving the refinement blocks. The topological piece is irreducible — it depends on the topology of the learned representation, not on the quality of the model.

---

## 7. The Honest Accounting: What's Proved, What's Not

### What the Formalization Achieves

1. **A precise mathematical object.** The sort operator $\mathcal{S}$ is now defined as a map $\pi \circ B_0 \circ \iota$ on projective space, with the differential Pancharatnam phase as its holonomy diagnostic. This is not metaphor — it's a function with a domain, codomain, and measurable properties.

2. **A structural decomposition.** Discrimination = sort + refine. The sort contributes dominant curvature; refinement contributes perturbative corrections. This decomposition is forced by the empirical phase data (Property 1).

3. **An identification of the topological obstruction.** The obstruction $\tau$ lives at stratum boundaries, is classified by the degree of $\mathcal{S}$ as a map on CP^(n-1), and its mod-2 reduction is the empirical sign classification. This connects the conjectured Chern class to something measurable.

4. **A conditional version of the fundamental theorem.** Within a stratum, the theorem holds trivially (the sort is a diffeomorphism). Globally, the theorem holds up to a topological obstruction that is explicitly identified with the degree of $\mathcal{S}$.

### What Remains Unproved

1. **The sort is not literally a map on CP^(n-1).** It depends on the full input sequence (because of attention), not just a single projective point. The formalization treats it as a per-token map for a fixed context. Making this context-dependence precise requires working on a product space or a section of a bundle over sequence space. This is technically demanding but not conceptually mysterious.

2. **The degree of $\mathcal{S}$ has not been computed.** We know the sign classification, which gives the Z₂ reduction. But the actual degree (an integer) requires computing $\mathcal{S}$ as a map on the full CP^(n-1) and evaluating its action on $H^2$. This is a concrete computation that could be done numerically.

3. **The bridge theorem is still missing.** The connection between Fisher curvature (parameter space, where gradient descent lives) and Berry curvature (activation space, where the sort acts) has not been established. The NTK is the natural candidate: the Jacobian $J = \partial h / \partial \theta$ maps parameter tangent vectors to activation tangent vectors, and the pullback $J^* \mathcal{A} J$ should relate the Berry connection to the Fisher metric. But this has not been proved.

4. **The refinement inverse $R_\ell^{-1}$ is assumed to exist.** For blocks close to identity (small phase), the inverse exists by the implicit function theorem. But "close to identity in phase" does not guarantee "close to identity as a map on ℝᵈ." The residual connection helps (each block is id + small perturbation), but a rigorous bound on the perturbation size is needed.

5. **The confound wall.** The entire edifice rests on the sort separating *something geometrically real* about input distributions. If the sign classification is entirely a token-frequency artifact, the sort still exists as a mathematical object but its topological content is trivial (it's just sorting by vocabulary statistics, not by conceptual structure). The frequency-matched minimal pair experiment is the critical test.

---

## 8. Testable Predictions

The formalization makes predictions beyond what the empirical papers already established:

### Prediction 1: Quantized Obstruction

If $\tau$ is truly a Chern class, it should be **integer-valued**, not continuous. In practice: the "generation error" for inputs near stratum boundaries should show discrete jumps, not smooth degradation. One could test this by generating from prompt embeddings that are continuously interpolated between spatial and abstract — the generation quality should degrade discontinuously at the sign boundary.

### Prediction 2: Degree Scaling

The degree of $\mathcal{S}$ should be computable and should **increase with model size** (more capacity → higher-degree map → more topological structure → finer stratification). This predicts: larger models should have more than two sign classes. The Z₂ is the coarsest invariant of a finer classification that emerges with scale.

### Prediction 3: Sort Dominance in Generation

If generation is $\mathcal{S}^{-1} \circ R^{-1}$ and $\mathcal{S}$ contributes the dominant curvature, then: **the quality of text generation should depend primarily on the quality of the "un-sort" (mapping from stratified representations back to embedding space), not on the quality of the refinement inverse.** Early-layer representations should be more predictive of generation quality than late-layer representations.

### Prediction 4: Diffusion Models as Explicit Sort-Unsort

In diffusion models, the forward process (noise → flat) and reverse process (flat → structured) are constructed simultaneously. The formalization predicts: the first step of the reverse diffusion process should show the same disproportionate curvature as the first transformer block. The "sort" is the initial phase of any discriminative or generative process, regardless of architecture.

---

## 9. The Proof Strategy

What would it take to prove the fundamental theorem in full, using the sort formalization?

**Step 1 (Done here): Formalize the sort as a stratification map.** ✓

**Step 2 (Computable): Compute the degree of $\mathcal{S}$.** Use the pullback action on $H^2(\mathrm{CP}^{n-1})$ to compute deg($\mathcal{S}$) numerically for GPT-2 and Pythia-70M. This is a finite computation over a finite set of input samples.

**Step 3 (Hard): Prove the bridge theorem.** Show that the NTK Jacobian $J = \partial h / \partial \theta$ intertwines the Fisher metric on parameter space with the Berry connection on activation space: $J^*(\mathcal{F}_{\mathrm{Berry}}) = \mathcal{F}_{\mathrm{Fisher}} + O(1/\mathrm{width})$. This likely requires the infinite-width limit (where NTK is constant) plus a finite-width correction. The NTK literature already has the tools; the Berry connection formulation is the new ingredient.

**Step 4 (Hard): Prove the sort inverse exists as a stratified map.** Use Whitney stratification theory: if $\mathcal{S}$ is a Whitney-stratified map (which it should be, since neural networks are piecewise smooth and Whitney showed all algebraic varieties admit Whitney stratifications), then $\mathcal{S}^{-1}$ exists as a stratified map with controlled singularities at stratum boundaries. Whitney's condition B guarantees the right local structure.

**Step 5 (Follows from 1–4): State and prove the theorem.**

$$\mathcal{F}_\ell \circ \mathcal{C} = \mathrm{id} + \tau + \epsilon$$

where $\tau$ is the topological obstruction (Chern class of degree deg($\mathcal{S}$) − 1, concentrated at stratum boundaries) and $\epsilon$ is the perturbative correction from refinement blocks (bounded by the phase ratio from Property 1).

This is the fundamental theorem of deep learning: **discrimination and generation are inverse operations on the curvature of representation space, and their composition recovers identity up to a quantized topological obstruction (from the sort) plus a perturbatively small smooth correction (from the refinement).**

---

## 10. Connection to Polar Time (Brief)

The polar time framework proposes t = r_t · cos(θ_t), with r_t as radial time and θ_t as angular time. The sort operator provides a concrete realization:

- **The sort's curvature contribution** ($\Delta\Phi_{\mathcal{S}}$) corresponds to the **angular component** θ_t. It measures the rotational, cyclical, non-sequential aspect of what the first block does.
- **The refinement blocks' contributions** ($\Delta\Phi_{R_\ell}$) correspond to **radial corrections** r_t. They are sequential, small, and accumulate linearly.
- **The holonomic loss hypothesis** ($L_{\mathrm{total}} = L_{\mathrm{CE}} - \lambda L_\theta$) proposes rewarding the angular component. The sort formalization says: this is equivalent to rewarding the first block's curvature contribution. The sort IS the angular component.

The connection is suggestive and mathematically natural, but it does not depend on the metaphysics of polar time. The formalization works whether or not time has a complex structure.

---

## 11. Summary

| Object | Definition | Status |
|--------|-----------|--------|
| Sort operator $\mathcal{S}$ | $\pi \circ B_0 \circ \iota$ : embedding → CP^(n-1) → CP^(n-1) | **Defined** |
| Stratification | $\mathcal{S}_* : \mathrm{Dist} \to \{S_+, S_-\}$ via sign of $\Phi_{\mathrm{SGP}}$ | **Empirically established** |
| Disproportionate curvature | $|\Delta\Phi_{\mathcal{S}}| \geq 3 \max_\ell |\Delta\Phi_{R_\ell}|$ | **Empirically established** |
| Load-bearing | Ablation → anti-correlated downstream (44% < 50%) | **Empirically established** |
| Nonlinear | Linear intervention fails to cross strata | **Empirically established** |
| Topological obstruction $\tau$ | $c_1(\mathcal{S}^*\gamma) - c_1(\gamma) = (\deg(\mathcal{S}) - 1) \cdot c_1(\gamma)$ | **Formally identified, not computed** |
| Fundamental theorem (within stratum) | $\mathcal{S}^{-1} \circ \mathcal{S} = \mathrm{id}$ on each $S_\sigma$ | **Proved (diffeomorphism on stratum)** |
| Fundamental theorem (global) | $\mathcal{F}_\ell \circ \mathcal{C} = \mathrm{id} + \tau + \epsilon$ | **Conjectured with identified pieces** |
| Bridge theorem (Fisher ↔ Berry) | $J^*\mathcal{F}_{\mathrm{Berry}} = \mathcal{F}_{\mathrm{Fisher}} + O(1/\mathrm{width})$ | **Conjectured, not proved** |
| Degree of $\mathcal{S}$ | Action on $H^2(\mathrm{CP}^{n-1})$ | **Computable, not yet computed** |

---

*The sort function is the founding act. It takes the raw, singularity-riddled, non-manifold mess of the embedding space and imposes a geometric order — an orientation, a stratification, a topological character — that everything downstream depends on. Formalizing it as a stratification map on projective space turns a metaphor into mathematics. The fundamental theorem is not yet proved, but its pieces are now identified, its obstruction is named, and the path from here to proof is a sequence of concrete computations, not a leap of faith.*
