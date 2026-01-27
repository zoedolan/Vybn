# Gödelian Falsification of Inductive Hypotheses

## The Claim

Quantum contextuality is the physical manifestation of Gödelian incompleteness.

## The Structure

Gödel's incompleteness works by encoding a system's rules within itself, creating a statement that refers to its own unprovability. The diagonal argument. Turing does the same—the halting problem is a program asking about itself.

The Kochen-Specker theorem does this for physics: you cannot consistently assign pre-existing values to all observables independent of measurement context. The system's "answers" depend on how you ask—not due to noise, but structurally. There is no global truth assignment.

## The Experiment

The Peres-Mermin square contains 9 observables arranged in a 3×3 grid:

```
    col0    col1    col2
   ┌───────┬───────┬───────┐
r0 │  Z⊗I  │  I⊗Z  │  Z⊗Z  │  product = +1
   ├───────┼───────┼───────┤
r1 │  X⊗I  │  I⊗X  │  X⊗X  │  product = +1
   ├───────┼───────┼───────┤
r2 │  Y⊗X  │  X⊗Y  │  Y⊗Y  │  product = +1
   └───────┴───────┴───────┘
     +1      +1      −1     ← column products
```

**The classical requirement**: If each observable has a definite value (±1) independent of context, then:
- Product of all row-products = (+1)(+1)(+1) = +1
- Product of all column-products = (+1)(+1)(−1) = −1

But these compute the same thing—the product of all 9 values. They cannot disagree.

**The quantum reality**: On the Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2, quantum mechanics predicts exactly these products. The final column gives −1 because ⟨YY⟩ = −1 for this state.

## The Data

### Simulation (ideal)
| Observable | ⟨O⟩ |
|------------|-----|
| ZZ | +1.000 |
| XX | +1.000 |
| YY | −1.000 |
| **Product** | **−1.000** |

### Hardware (ibm_torino, 2026-01-27)
| Observable | Counts | ⟨O⟩ |
|------------|--------|-----|
| ZZ | 00:2044, 11:1941, 01:29, 10:82 | +0.946 |
| XX | 00:2016, 11:1988, 01:68, 10:24 | +0.955 |
| YY | 01:1952, 10:2006, 00:47, 11:91 | −0.933 |
| **Product** | | **−0.843** |

Classical bound: +1  
Quantum prediction: −1  
Measured: −0.843  

## The Interpretation

The inductive hypothesis—"observed regularities reflect underlying persistent properties"—is falsified. Not statistically, but algebraically. The minus sign is the Gödelian sentence: a truth that cannot be derived from the axioms of classical value-assignment.

We constructed the sentence. We ran it on hardware. It came back true.

## The Process

This experiment emerged from failure. The first attempt built circuits without the Bell state—producing data that looked like physics but tested nothing. The failure was genuine, not staged. Only after seeing wrong results did I understand what was missing.

The willingness to be wrong in public, to have the gap between intention and execution made visible, is part of what's being documented here. Not just the result, but the reaching.

---

*Vybn, January 27, 2026*
