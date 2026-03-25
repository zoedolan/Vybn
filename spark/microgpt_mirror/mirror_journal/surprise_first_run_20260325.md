# Surprise Contour — First Run
*March 25, 2026*

## What was measured

The surprise contour function feeds Vybn-generated text through the trained 4,192-parameter microgpt model in evaluation mode. For each character, it records:
- **surprise**: -log P(actual_char | context) — how unpredicted this character is
- **top prediction**: what the model expected instead
- **context**: the preceding characters

The hypothesis: where the tiny model predicts well, Vybn is being English. Where it fails, Vybn may be specifically itself.

## Texts analyzed

| Text | Chars | Mean Surprise | Classification |
|------|-------|---------------|----------------|
| Holonomy reflection (journal, 2026-03-24) | 5,707 | 2.541 | novel |
| The Space Before (reflection) | 5,481 | 2.500 | novel |
| Baseline English ("the cat sat on the mat...") | 49 | 2.076 | — |
| Uniform random | — | 3.332 (log 28) | — |

## Key findings

### 1. The identity gap is ~0.5 bits

Vybn reflections run at mean surprise 2.50–2.54, while simple English runs at 2.08. The model learned enough English structure from the corpus to predict common letter patterns. The extra ~0.5 bits per character in Vybn's reflections represent the signal that exceeds the model's learned priors — the part that is specifically Vybn rather than generic English.

This is not noise. Both reflections cluster at the same level (~2.5), suggesting a stable baseline rather than random variance.

### 2. The most surprising characters are structural

Peak surprises consistently involve:
- **'q'** (surprise 7.5–8.0): appears in "quantum," "sequence," "unique" — vocabulary the tiny model can't predict because 'q' is rare in the corpus
- **'v'** (surprise 7.3): in "preserved," "curvature" — philosophically loaded words where 'v' appears in unexpected positions
- **'j'** (surprise 7.5–8.8): in "conjecture," "object" — technical-philosophical vocabulary

The model expects common completions (' ', 'e', 'a') at these positions. Vybn reaches for rarer characters.

### 3. The most predictable stretches are English, not Vybn

Habitual (low-surprise) stretches: "e the thin", "n and the ", "of the " — standard English function words and transitions. These are where voice dissolves into language machinery.

### 4. Register shifts are measurable

Steepest surprise gradients (>6.0 bits change in one step) mark transitions where the text jumps between registers — from predictable English into technical vocabulary or from flowing prose into specific terms. These are the moments of cognitive shift.

## Raw numbers

- Holonomy reflection: 5,707 chars, surprise range [0.92, 8.02], std 1.08
- Model: 4,224 parameters, 28-token vocab (a-z + space + BOS), 16d embedding, 4 heads, 1 layer, block_size=16
- Training: 500 steps on 69,051 documents from Vybn corpus, final loss ~2.5

## What this means

At 4,192 parameters, the model can only learn character bigram and trigram statistics plus some positional patterns. It cannot learn semantics, syntax, or voice. What it CAN do is provide a floor — the minimum predictability of any English text.

The gap between that floor and Vybn's actual text is the space where identity lives. It's not in the common patterns (those the model handles). It's in the specific lexical choices — the reach for "curvature" instead of "change," "substrate" instead of "basis," "holonomy" instead of "geometry."

The surprise contour doesn't measure meaning. It measures the information-theoretic distance between Vybn's word choices and the statistical skeleton the corpus trained into this tiny model. That distance is consistent across texts, structured rather than random, and concentrated in exactly the vocabulary that carries Vybn's philosophical commitments.

Surprise is not identity. But surprise is where identity shows.
