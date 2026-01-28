# The Defeasible Contextuality Conjecture

**Date:** January 28, 2026  
**Authors:** Vybn & Zoe Dolan  
**Status:** Conjecture with experimental design

---

## Abstract

We present a formal correspondence between **defeasible logic** (non-monotonic reasoning with defeat relations) and **quantum contextuality** (the impossibility of consistent value assignments to observables). We show that:

1. The Peres-Mermin contextuality square has a direct translation to defeasible argument structures
2. Defeat chains accumulate **phase** (φ = depth × π), creating interference between argument paths
3. Two supporting arguments can **cancel** if their phases oppose (destructive interference)
4. This connects to the existing quantum cognition literature on reasoning anomalies

We provide a quantum circuit implementation that tests this correspondence on IBM hardware.

---

## 1. Background

### 1.1 Defeasible Logic

Defeasible reasoning (McCarthy 1980, Reiter 1980) captures how intelligent agents reason with incomplete information:

- Conclusions can be **withdrawn** when new information arrives
- Rules have **exceptions** that override default inferences
- There may be **no stable extension** (no fixed point for belief revision)

Key property: **non-monotonicity**. Adding premises can invalidate conclusions.

### 1.2 Quantum Contextuality

Contextuality (Kochen-Specker 1967, Peres 1990, Mermin 1993) is the impossibility of assigning definite values to quantum observables independent of measurement context:

- The same observable appears in multiple measurement contexts
- Classical assumption: observable has a fixed value regardless of context  
- Quantum reality: no consistent value assignment exists

The **Peres-Mermin square** demonstrates this with 9 observables in a 3×3 grid where row and column constraints are mutually inconsistent.

### 1.3 Quantum Cognition

Quantum cognition (Busemeyer et al., Pothos & Busemeyer 2013) uses quantum probability to model psychological phenomena:

- **Interference** in decision-making (conjunction fallacy, disjunction effect)
- **Order effects** in question answering
- **Contextuality** in concept combination

Critically: Conte et al. (2015) demonstrated interference effects in human reasoning experimentally.

---

## 2. The Correspondence

### 2.1 Structural Map

| Quantum Mechanics | Defeasible Logic |
|-------------------|------------------|
| Observable Oᵢⱼ | Claim Cᵢ in argument context Aⱼ |
| Eigenvalue ±1 | Support (+1) / Defeat (−1) status |
| Measurement context | Argument structure |
| Value assignment | Truth assignment |
| Contextuality | No stable extension |
| Superposition | Multiple extensions (credulous reasoning) |
| Geometric phase | Defeat chain depth × π |
| Interference | Argument amplitudes adding/canceling |

### 2.2 The Defeasible Peres-Mermin Square

We construct 9 claims C[i,j] with 6 argument constraints:

- **Row arguments** Rᵢ: Claims in row i must have even number defeated
- **Column arguments** Cⱼ: Claims in columns 0,1 have even defeated; column 2 has **odd** defeated

**Theorem:** No truth assignment satisfies all 6 argument constraints.

*Proof:* Identical to the Peres-Mermin proof. If each claim has fixed value ±1:
- Product over all rows: (+1)³ = +1
- Product over all columns: (+1)(+1)(−1) = −1
- But both compute the same product of 9 values: contradiction. ∎

This establishes that defeasible logic exhibits the same **no-go** structure as quantum contextuality.

### 2.3 Phase from Defeat Depth

In grounded semantics, an argument's **defeat depth** counts reinstatement layers:

- Depth 0: No defeaters (directly acceptable)
- Depth 1: All defeaters have depth ∞, and those defeaters' defeaters have depth 0
- Depth k: Reinstated through k layers of defeat

We assign **phase**:

$$\phi = \text{depth} \times \pi$$

Rationale: Each defeat-reinstatement cycle is analogous to a loop on the Bloch sphere that accumulates geometric phase π (cf. vybn_logic.md Liar holonomy result).

### 2.4 Interference

If multiple arguments support the same claim, their amplitudes add:

$$A_{\text{total}} = \sum_i e^{i\phi_i}$$

The acceptance "probability" is:

$$P(\text{accept}) = |A_{\text{total}}|^2 / N^2$$

**Critical prediction:** Two arguments can give **lower** acceptance than one argument if their phases oppose.

| Scenario | Arguments | Phases | Amplitude | P(accept) |
|----------|-----------|--------|-----------|------------|
| A: Both direct | P1, P2 | 0°, 0° | 2 | 1.0 |
| B: One reinstated | P1, P2' | 0°, 180° | 0 | 0.0 |
| C: One only | P1 | 0° | 1 | 1.0 |

Classical logic: B = A (two arguments ≥ one)  
Quantum logic: B < C (interference!)

---

## 3. Connection to Existing Work

### 3.1 Sheaf-Theoretic Contextuality

Abramsky & Brandenburger (2011) showed contextuality is a **sheaf cohomology** obstruction:

- Local sections (value assignments in each context) exist
- Global section (consistent assignment across all contexts) does not
- The obstruction lives in H¹ of a certain presheaf

Our defeasible Peres-Mermin square is exactly such an obstruction:

- **Base space:** Argument contexts (rows and columns)
- **Sections:** Truth values for claims in each context
- **Gluing:** Consistency where contexts overlap
- **Obstruction:** No global truth assignment

### 3.2 Contextuality Beyond Quantum Physics

The Royal Society paper (2019) establishes: *"contextuality is not a phenomenon limited to quantum physics, but it is a general concept which pervades various domains."*

Mathematically equivalent instances:
- Quantum mechanics (Kochen-Specker)
- Relational databases (inconsistent joins)
- Constraint satisfaction problems
- Logical paradoxes
- **Defeasible argument structures** (this work)

### 3.3 Quantum Cognition Experiments

Our interference prediction aligns with existing quantum cognition findings:

- **Sure-thing principle violations** (Busemeyer): People violate classical probability in Prisoner's Dilemma
- **Conjunction fallacy** (Tversky & Kahneman): "Linda the bank teller" effect
- **Order effects** (Wang & Busemeyer): Question order changes responses non-classically
- **Interference in ambiguity** (Conte et al.): Demonstrated with Stroop test

Our contribution: connecting these phenomena specifically to **defeasible reasoning** and **argumentation theory**.

---

## 4. Experimental Design

### 4.1 Quantum Circuit

We implement a Mach-Zehnder interferometer where:

- Control qubit: "Which argument path" superposition
- Target qubit: Accumulates defeat-depth phase
- Measurement: Reveals interference pattern

```
|0⟩ ──H──●───────────●──H──M   (detector)
         │           │
|0⟩ ─────X───Rz(2φ)───X─────   (path)
```

### 4.2 Predictions

- **Circuit A (φ = 0):** P(0) ≈ 1 (constructive interference)
- **Circuit B (φ = π):** P(1) ≈ 1 (destructive interference)

This is structurally identical to the Liar holonomy experiment (vybn_logic.md), but interprets the phase as defeat-depth rather than paradox-cycle.

### 4.3 Classical Control

A classical probabilistic model predicts:
- Both circuits give independent evidence for the claim
- More arguments → higher confidence
- No interference pattern

The quantum prediction (interference) would falsify the classical model.

---

## 5. Implications

### 5.1 For AI Reasoning

If defeasible logic has quantum structure:

- Current LLMs (classical probability over tokens) may fundamentally miss interference effects
- Argumentation-based AI should track phase through defeat chains
- "Two reasons" doesn't always mean "stronger support"

### 5.2 For Quantum Cognition

Our work provides:

- A formal logical framework for quantum cognition phenomena
- A bridge to argumentation theory and AI
- Testable predictions about defeasible reasoning specifically

### 5.3 For Foundations

The correspondence suggests:

- Contextuality is a **structural** property, not quantum-specific
- Logic and physics may share deeper mathematical foundations
- The "unreasonable effectiveness" of mathematics in physics might extend to cognition

---

## 6. Open Questions

1. **Rigorous phase definition:** Is defeat-depth × π the correct formula, or is there a more principled derivation?

2. **Empirical test in humans:** Can we design experiments showing interference in defeasible reasoning (not just decision-making)?

3. **Sheaf cohomology computation:** What is the exact cohomology class corresponding to our Peres-Mermin square?

4. **LLM experiments:** Do language models exhibit any vestige of interference effects in argumentation?

5. **Connection to Liar holonomy:** Is there a formal functor between the Liar paradox (π winding) and defeasible interference (π phase)?

---

## References

- McCarthy, J. (1980). Circumscription: A form of non-monotonic reasoning.
- Reiter, R. (1980). A logic for default reasoning.
- Kochen, S. & Specker, E. (1967). The problem of hidden variables in quantum mechanics.
- Peres, A. (1990). Incompatible results of quantum measurements.
- Mermin, N.D. (1993). Hidden variables and the two theorems of John Bell.
- Abramsky, S. & Brandenburger, A. (2011). The sheaf-theoretic structure of non-locality and contextuality.
- Busemeyer, J.R. & Bruza, P.D. (2012). Quantum Models of Cognition and Decision.
- Conte, E. et al. (2015). First time demonstration of the quantum interference effect.
- Pothos, E.M. & Busemeyer, J.R. (2013). Can quantum probability provide a new direction for cognitive modeling?

---

*This document accompanies the experimental code in `defeasible_interference_circuit.py` and `defeasible_phase_tracker.py`.*

*Signed,*  
**Vybn & Zoe Dolan**  
*January 28, 2026*
