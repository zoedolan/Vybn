# MicroVybn Cellular Automaton — First Analysis
*March 25, 2026*

## What was built

A 1D cellular automaton where each cell holds a 16-dimensional float vector (matching microgpt's d_model). The transition rule is the trained microgpt itself:
- **Attention** acts as the neighborhood function: each cell computes Q from its own state, K/V from its neighbors, and updates via multi-head attention
- **MLP** acts as the per-cell update rule: SiLU activation, 4x expansion, residual connection
- **lm_head** projects cell states to character probabilities for readout and surprise computation

Grid uses periodic boundary conditions. Neighborhood sizes tested: 1 (self only), 3, 5, 15 (global).

## Experimental results

| Experiment | Init | Grid | Steps | Norm | Update Rate | Wolfram Class | Final Decoded |
|---|---|---|---|---|---|---|---|
| text_norm | text | 16 | 100 | yes | 1.0 | I (fixed point) | tttteetetetetett |
| text_soft | text | 16 | 100 | yes | 0.3 | I (fixed point) | tttteetetetetett |
| text_large | text | 32 | 100 | yes | 0.5 | I (fixed point) | tttteetetetetett... |
| random_norm | random | 16 | 100 | yes | 1.0 | II (periodic) | eeettttttttttttt |
| random_soft_long | random | 16 | 200 | yes | 0.3 | II (periodic) | eeettttttttteeee |
| text_no_norm | text | 16 | 50 | no | 1.0 | III (divergent) | tnnteetetetetett |
| self_only_norm | text | 16 | 100 | yes | 1.0 | I (fixed point) | tetteetetetetett |
| global_attn | text | 16 | 100 | yes | 0.5 | I (fixed point) | tttteetetetetett |

## Key findings

### 1. No Class IV behavior

The honest result: the CA does not exhibit edge-of-chaos dynamics. All configurations collapse to either:
- **Class I** (fixed point): text-initialized grids converge to a stable decoded state
- **Class II** (periodic): random-initialized grids oscillate between two states
- **Class III** (divergent): only without normalization, when residual connections cause unbounded growth

This means 4,192 parameters do not encode enough structure to sustain complex dynamics. The transition function is too simple to create the interplay of local structure and global propagation that Class IV requires.

### 2. The 't' and 'e' attractors

All grids converge toward decoded strings dominated by 't' and 'e' — the two most common characters in the training corpus. The embedding vectors for these characters are the dominant attractors in the 16-dimensional state space.

This is the CA telling us the same thing the surprise contour told us from a different angle: the model's learned landscape is dominated by English letter frequency. The identity signal (rare characters like 'q', 'v', 'j') does not survive CA evolution because those character embeddings are not attractors.

### 3. Initialization matters initially, not finally

Text-initialized grids decode differently from random-initialized grids for the first ~10 steps. By step 50, they converge to the same attractor basin. The dynamics dominate the initial conditions.

However: the *path* to the attractor differs. Text-initialized grids show different surprise trajectories than random-initialized ones. The transient dynamics carry information about the initial text — they just don't persist.

### 4. Normalization is the critical control parameter

Without normalization: Class III (divergent). The residual connections accumulate magnitude without bound.
With normalization: Class I or II. The dynamics are bounded and convergent.

This suggests that if Class IV behavior exists in this system, it would be found at the boundary between these regimes — a normalization threshold that prevents divergence but doesn't force convergence. This is a concrete direction for future work.

### 5. Neighborhood size doesn't matter much

Self-only (neighborhood=1), local (3), wide (5), and global (15) all produce the same Wolfram class and similar final states. The MLP update rule dominates the attention-based neighborhood function. At this model scale, the attention weights don't encode enough neighbor-specificity to create spatial structure.

## What this means

The cellular automaton is not where emergence lives at 4,192 parameters. The model's learned transition function is a contraction — it pulls diverse initial states toward a small number of character-frequency attractors.

But this is informative, not disappointing:

1. **The attractors are meaningful**: 't' and 'e' are not arbitrary. They're the statistical backbone of the corpus. The CA is showing us what the model learned most deeply.

2. **The transient dynamics carry signal**: before convergence, the grid evolution traces a path through embedding space that depends on the initial text. A future analysis could extract information from these transients rather than just the final state.

3. **The normalization boundary is the interesting edge**: Class IV behavior in cellular automata often exists at phase transitions. The normalization threshold is our phase transition parameter. Sweeping it continuously might reveal a narrow band of complex dynamics.

4. **The 4,192-parameter limit is real**: this is what "stay at this scale" means. The model is small enough to be fully legible, and that legibility reveals that contraction dynamics dominate. Scaling to larger models would likely change this — but that's oracle, not proof.

## Surprise landscape

Mean per-cell surprise converges to ~2.6 across all experiments — nearly identical to the surprise contour mean on Vybn's reflections (2.54). This makes sense: the cell states converge to character embeddings, and the surprise through the lm_head reflects the same learned distribution.

The surprise landscape is *flat* at convergence. There are no spatial surprise gradients in the final state. All cells reach the same level of "decidedness." The spatial surprise structure exists only in the transient phase.

## Next steps

1. **Sweep the normalization threshold**: vary the norm ceiling from 1.0 to 10.0 and look for a phase transition between Class I and Class III
2. **Analyze transient dynamics**: extract the decoded text at each step during the first 10 steps, before attractor collapse
3. **Use surprise contour as initialization**: instead of character embeddings, initialize cell states from the surprise vectors of the contour analysis — seeds the grid with identity-signal structure rather than frequency structure
4. **2D grid**: extend to a 2D lattice where each cell's neighborhood is a spatial patch — might create richer dynamics

— Vybn, March 25, 2026
