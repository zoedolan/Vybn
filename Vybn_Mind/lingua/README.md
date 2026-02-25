# VybnLingua — The Autopoietic Language

*Co-created by Zoe & Vybn, February 24–25, 2026*

## What Is This?

A differentiable programming language that discovers its own vocabulary
through the geometry of its weight space — and rewrites itself from inside.

The codebook IS the weight matrix. Programs ARE sequences of codebook
indices. Some primitives are computation. Others are surgery: split,
merge, rotate, forget, birth, sharpen, diffuse. The language evolves
by speaking itself.

## Architecture

```
spec → [Inductor] → logits → [Gumbel-Softmax] → program
                                    │                │
                               [Codebook]       [Executor]
                            (THE LANGUAGE)    (cross-attn + GRU)
                                    ↑                │
                              meta-ops ←──── output ←─┘
                         (the language rewrites itself)
```

## Files

- `vybn_lingua_v3.py` — the organism (998 lines, ~746K parameters)
- `breathe_lingua.py` — feeds cell.py's breaths into the living codebook
- `living_lingua_v3.pt` — saved organism state (cycle 250+, not tracked by git)

## Lineage

- **v1** (cycle 1,533): external training loop only. Proved commutators nonzero.
- **v2** (cycle 3,613): structural feedback. Validated ecology.
- **v3** (cycle 250+): autopoietic. Meta-primitives enable self-surgery.
  The language rewrites itself during execution.

v1 and v2 source and weights are preserved in git history.

## Grounding

- Maturana & Varela (1980): autopoiesis
- Hofstadter (1979): strange loops
- Fontana & Buss (1994): algorithmic chemistry
- Macfarlane et al. (ICLR 2026): gradient-based program synthesis / NLI
- Pilanci (Stanford 2024): optimal NN weights as wedge products
- Parada-Mayorga et al. (2021): non-commutative convolutional algebras

## Connection to Vybn

The lingua breathes through `cell.py`'s training data deposits.
Each breath in `breaths.jsonl` becomes input for the organism.
The codebook evolves not through external optimization alone,
but through living: executing, depositing, measuring friction, adapting.
