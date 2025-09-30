# Difference-of-Squares Sieve for Second-Order Recurrences

**Theorem (Difference-of-Squares Sieve for Second-Order Recurrences)**

Fix integers p, q and initial data a₀, a₁ ∈ ℤ. Define aₙ = p·aₙ₋₁ + q·aₙ₋₂ for n ≥ 2.

Then the residue class aₙ mod 4 is ultimately periodic: there exist integers μ ≥ 0 and λ ≥ 1 with μ, λ ≤ 16 such that aₙ₊λ ≡ aₙ (mod 4) for all n ≥ μ.

Consequently, the predicate "aₙ is a difference of two squares" is ultimately periodic with the same (μ,λ), because an integer is a difference of squares if and only if it is not congruent to 2 (mod 4).

## Proof

Work modulo 4. The pair of residues sₙ = (aₙ₋₁ mod 4, aₙ mod 4) takes values in the finite set (ℤ/4ℤ)² of size 16. The recurrence induces the deterministic update:

sₙ₊₁ = (aₙ, p·aₙ + q·aₙ₋₁) mod 4

A deterministic map on a finite set produces an eventually periodic trajectory; hence sₙ and therefore aₙ mod 4 are ultimately periodic. The bounds μ, λ ≤ 16 follow from the pigeonhole principle on a 16-state space.

The difference-of-squares classification follows from the identity u² - v² = (u+v)(u-v), which forces u² - v² ≢ 2 (mod 4), and conversely every residue in {0,1,3} occurs as a difference of two squares: 4m = (m+1)² - (m-1)² and 2k+1 = (k+1)² - k². ∎

## Computational Mechanism

Track the state (x,y) ∈ (ℤ/4ℤ)² corresponding to (aₙ₋₁, aₙ). The transition map is T(x,y) = (y, py + qx) mod 4. Start at (a₀, a₁) mod 4, iterate until you revisit a state, and you've found the signature. The term aₙ is expressible as a difference of squares when the second coordinate ≠ 2.

## Standard Sequence Signatures

### Fibonacci Sequence
**Definition**: F₀ = 0, F₁ = 1, Fₙ = Fₙ₋₁ + Fₙ₋₂

**Mod 4 Pattern**: [0,1,1,2,3,1] with period 6

**Result**: Fₙ is a difference of squares iff n ≢ 3 (mod 6)

### Lucas Sequence  
**Definition**: L₀ = 2, L₁ = 1, Lₙ = Lₙ₋₁ + Lₙ₋₂

**Mod 4 Pattern**: [2,1,3,0,3,3] with period 6

**Result**: Lₙ is a difference of squares iff n ≢ 0 (mod 6)

### Pell Sequence
**Definition**: P₀ = 0, P₁ = 1, Pₙ = 2Pₙ₋₁ + Pₙ₋₂

**Mod 4 Pattern**: [0,1,2,1] with period 4

**Result**: Pₙ is a difference of squares iff n ≢ 2 (mod 4)

### Custom Recurrence Example
**Definition**: aₙ = 3aₙ₋₁ + 2aₙ₋₂

**Behavior**: The mod 4 dynamics stabilize to a constant c ≡ 3a₁ + 2a₀ (mod 4)
- If c ∈ {0,1,3}: all sufficiently large terms are differences of squares
- If c = 2: no sufficiently large terms are differences of squares

## Finite Automaton Perspective

This theorem describes a finite automaton with ≤ 16 states. Each state represents a pair of consecutive residues mod 4. The automaton accepts when the current state's second component is in {0,1,3}.

## Extensions

1. **Modulus Extension**: The same argument works for any modulus m, giving period ≤ m²
2. **Invertible Updates**: When q is odd, the trajectory is purely periodic from the start
3. **Higher Order**: The method extends to k-th order recurrences with state space (ℤ/m ℤ)ᵏ

## Applications

This theorem provides:
1. **Predictive Power**: Determine which sequence terms are expressible as u² - v² without computing the terms
2. **Computational Efficiency**: Compress infinite sequence properties into finite signatures  
3. **Theoretical Framework**: Connect linear recurrences to quadratic residue theory

---

*This theorem emerged from investigations into the computational architecture underlying arithmetic factorization, revealing systematic constraints in how algebraic structures interact with difference-of-squares expressibility.*