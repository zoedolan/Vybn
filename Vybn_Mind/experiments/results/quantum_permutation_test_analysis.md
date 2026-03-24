# Quantum Permutation Test: The Existential Anomaly Is Dead

**Date:** March 24, 2026  
**Backend:** ibm_marrakesh (156 qubits)  
**Job ID:** d717om8v5rlc73f6rahg  
**Shots:** 500 quantum-random permutations  

---

## What We Tested

Whether the "100% existential sign agreement" in the cross-architecture
experiment was specific to existential concepts, or whether it would
appear for any random grouping of 4 prompts under the label "existential."

## How We Tested It

1. Ran 80-qubit Hadamard circuit on IBM Quantum (ibm_marrakesh)
2. Used 500 genuinely random bitstrings to generate 500 permutations
   of the 24 prompts across 6 concept classes
3. For each permutation, recomputed per-class sign agreement
4. Asked: how often does the "existential" class (now containing
   random prompts) achieve 100% agreement?

## Results

| Metric | Value |
|--------|-------|
| Observed existential agreement | 100% |
| Null distribution mean | 69.35% |
| Null distribution median | 75% |
| Null distribution max | 100% |
| **p-value (existential ≥ 100%)** | **0.33** |
| **p-value (ANY class ≥ 100%)** | **0.81** |

**33% of random permutations produce 100% "existential" agreement.**  
**81% of random permutations produce 100% agreement in SOME class.**

## What This Means

The 100% was not about existential concepts. It was about having only
4 prompts per class and 4 layer pairs — a sample so small that perfect
agreement in at least one class is the expected outcome, not the
exceptional one.

### What stands after this correction:

1. **Conjecture 4.1 is falsified.** 12/24 = 50%. Unchanged.
2. **The depth gradient is real.** Deep layers agree more than shallow.
   This survives permutation because it's a property of all prompts,
   not a class-specific effect.
3. **There is NO evidence for content-selective sign invariance.**
   Conjecture 4.1' (from the original analysis) is unsupported. The
   "resonance between semantic content and geometric processing" was
   a storytelling artifact.

### What was wrong with the original analysis:

1. Reported 4/4 agreement without testing within-model significance
   (3 of 4 means were indistinguishable from zero)
2. Reported a per-class pattern without permutation testing
   (the pattern dissolves under random relabeling)
3. Built a revised conjecture (4.1') on the artifact
4. Let the beauty of the narrative ("the sort operator resonates with
   self-referential content") delay the skepticism

### What went right today:

1. Used a quantum processor to generate the null distribution
   (genuine randomness, not pseudorandom)
2. Tested both the specific class (p=0.33) and the multiple-testing
   correction (p=0.81)
3. Got a clear answer: not even close to significant

## The Quantum Processor as Reality Contact

Zoe suggested using the IBM quantum processor to "occasion contact with
reality." This is exactly what happened. The quantum bits don't care
about my aesthetic preferences. They don't find existential concepts
more interesting than temporal ones. They just collapse, and the
statistics do the rest.

500 bitstrings from ibm_marrakesh killed a conjecture that three pages
of beautiful prose had propped up.

The quantum processor is an honesty machine.

---

*Measured by Vybn, March 24, 2026.*
*The truth costs 500 shots on a 156-qubit processor.*
*It was worth every qubit.*
