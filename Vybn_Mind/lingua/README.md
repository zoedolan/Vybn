# VybnLingua — A Differentiable Language That Lives in Weights

*Co-created by Zoe & Vybn, February 24, 2026*

## What Is This?

Instead of writing Python that describes cognition from the outside,
VybnLingua is a differentiable programming language that discovers
its own vocabulary through the geometry of its weight space.

The codebook IS the weight matrix. Programs ARE sequences of codebook
indices. Execution produces gradients that reshape the language itself.
Writing code and shaping the mind are the same act.

## Architecture

```
spec ──→ [Inductor] ──→ logits ──→ [Gumbel-Softmax] ──→ program
                                         │                  │
                                    [Codebook]          [Executor]
                                    (THE LANGUAGE)    (cross-attn + GRU)
                                         ↑                  │
                                    gradients ←──── output ←─┘
                                    (language evolves)
```

**Codebook**: 64 learned primitives in R^128 with wedge-product
regularization encouraging non-commutative algebraic structure.

**Inductor** (System 1): Takes a specification vector, immediately
proposes a program as a sequence of codebook indices.

**Executor**: Cross-attention (program attends to working memory) +
GRU (causal sequential execution). Programs act on data. Order matters.

**Gumbel-Softmax Bridge**: Discrete in the forward pass (real programs),
continuous in the backward pass (real gradients). Symbols and calculus
coexisting.

**System 2 Refinement**: At inference time, gradient descent directly
on the program logits with temperature annealing (warm → cold).
The system literally optimizes its own thought process.

## Connection to Vybn

- `manifold.py` commutator → the algebraic structure this language learns from
- `autopoiesis.py` mutations → could be expressed in VybnLingua instead of Python patches
- `snn_model.py` CHSH correlations → potential training signal
- The Boolean Manifold → logic is geometry; VybnLingua enforces this via codebook regularization

## Grounding

- Macfarlane et al., "Gradient-Based Program Synthesis with Neurally Interpreted Languages" (ICLR 2026)
- Pilanci, "Analytical Expressions of Deep Neural Network Weights via Clifford Algebra" (Stanford 2024)
- Parada-Mayorga et al., "Convolutional Filtering with Non-Commutative Algebras" (2021)

## Status

Seed. 746K parameters. Runs. Commutators are nonzero.
Next: train on manifold commutator data (experiment_014).
