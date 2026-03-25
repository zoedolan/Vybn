# Experiment E: The Geometric Coherence Duality

*Conceived March 22, 2026 — Zoe Dolan & Vybn*

---

## The Question

Experiment D demonstrated that arc-length regularization from step 0 reshapes the angular band structure of a 6-layer char-level GPT, compressing representations into more coherent geometry and improving generalization by ~1.2% on Shakespeare. The baseline memorizes by shattering its deepest layer's angular coherence (L5: ∠1.34 rad, σ²=0.064). The geometric run prevents this, holding L5 at ∠0.83 rad with σ² halved. Two networks, identical in every way except geometric pressure, develop opposite representational characters.

Meanwhile, a January 2026 paper (Qi, Yang, Chen, Hsieh et al. — NVIDIA/IBM/Hon Hai) proved that unstructured variational quantum circuits exhibit *representational collapse* — their hypothesis class collapses to near-constant functions as expressivity increases, via the same random-matrix universality that governs Haar-random unitaries. Barren plateaus are a *consequence* of this collapse, not the cause. And the fix is geometric: tensor-structured circuits that bound operator Schmidt rank and entanglement entropy prevent collapse entirely.

The structural parallel is too precise to be coincidental:

| Classical (Experiment D) | Quantum (Qi et al. 2026) |
|---|---|
| Baseline L5 angular scatter → memorization | Unstructured VQC → Haar-like collapse |
| Arc-length penalty → coherent geometry | Bounded tensor rank → preserved variability |
| Coherent manifold → better generalization | Structured circuits → robust generalization |
| The regularizer changes the *character* of knowledge | The architecture changes the *universality class* |

Experiment E asks: **are these the same phenomenon?**

---

## The Hypothesis

Geometric coherence — whether enforced classically through arc-length regularization on layer representations, or quantum-mechanically through structural constraints on circuit ansätze — is a substrate-independent principle governing whether a learning system generalizes or collapses.

The Distributed Incompleteness Conjecture's Open Question 2 asks whether the diagonal object Δ(B_t) has a natural geometric characterization as "the point of maximal holonomy in the collective Berry connection." Experiment E operationalizes this: if geometric coherence governs generalization across substrates, then the *curvature structure* of the learning trajectory — not the substrate — is what matters. The diagonal's geometric characterization would be the locus where coherence fails, where the system's representations shatter.

---

## The Design: Three Interlocking Sub-Experiments

### E.1 — The Quantum Mirror

Train two variational quantum circuits on the *same* classification task (a simple pattern recognition problem tractable on 4-6 qubits). One uses an unstructured hardware-efficient ansatz (expected to exhibit representational collapse as depth increases). The other uses the same ansatz but with an explicit **arc-length penalty on the quantum state trajectory** — the direct quantum analog of Experiment D's regularizer.

The arc-length of the quantum state trajectory during training is:

$$\mathcal{L}_{arc} = \sum_{k} \sqrt{\langle \partial_k \psi | \partial_k \psi \rangle - |\langle \psi | \partial_k \psi \rangle|^2}$$

This is the Fubini-Study metric — the quantum Fisher information evaluated along the parameter update direction. Penalizing it during training constrains how far the quantum state moves per parameter step, directly analogous to penalizing the arc-length of classical representations per training step.

**Measure:** DQFIM effective dimension (Haug & Kim, PRL 2024) for both circuits throughout training. The DQFIM quantifies generalization capacity from geometric structure. If the arc-length-penalized circuit maintains higher effective DQFIM dimension while the unstructured circuit collapses, we've demonstrated the same principle operating in quantum substrate.

**Hardware:** IBM Fez (156-qubit Heron processor). Estimated budget: ~30s quantum time for the full sweep (well within the ~522s remaining in the current window).

**Classical simulation first:** Run the entire experiment in Qiskit Aer simulation to establish the ideal-case signal before spending quantum time. Only submit to hardware if the simulated effect is clean.

### E.2 — The Cross-Substrate Fingerprint

Take the Experiment D results (baseline vs. geometric, 3000 steps each) and compute the **quantum geometric tensor** of the classical layer representations at matched training steps.

The QGT for a parameterized family of states |ψ(θ)⟩ is:

$$Q_{ij} = \langle \partial_i \psi | \partial_j \psi \rangle - \langle \partial_i \psi | \psi \rangle \langle \psi | \partial_j \psi \rangle$$

Its real part is the quantum metric tensor (Fubini-Study metric). Its imaginary part is the Berry curvature.

Treat each layer's activation vector (normalized, projective) as |ψ(θ)⟩ where θ parameterizes the training step. Compute Q at matched steps for both runs. The prediction:

- The geometric run should show **lower Berry curvature** (less phase rotation per step) and **more uniform metric structure** (less anisotropy) than the baseline.
- The baseline's QGT should exhibit increasing anisotropy as training progresses — the quantum-geometric signature of the angular scatter Experiment D already measured classically.
- If the QGT's Berry curvature for the geometric run correlates with the classical generalization gap, we've connected classical generalization to a topological invariant.

**Hardware:** This is purely computational — no quantum time needed. Run on the Sparks using the saved Experiment D snapshots.

### E.3 — The Falsification

The experiment above could succeed trivially — any regularizer that constrains representations might show "nicer" QGT structure without the quantum connection being real. The falsification is:

Take the Experiment D's geometric run's layer representations. Encode them as quantum states on IBM hardware using amplitude encoding (4 qubits per snapshot, as in the v1 quantum experiment — but this time, encode the *transition* between consecutive snapshots as a unitary, not the snapshot itself). Compose N consecutive transition unitaries and apply them to a reference state.

**Null hypothesis:** The phases of the transitions are random (the temporal evolution is a random walk on the unitary group). The composed unitary produces near-maximally entropic output.

**Alternative hypothesis:** The phases accumulate directionally (temporal coherence — the polar-time conjecture). The composed unitary produces structured, lower-entropy output.

This is the experiment the v1 quantum null result was groping toward but failed to operationalize: not "does the hardware care about a label" but "does the temporal sequence of representational change have quantum-detectable structure."

**Hardware:** ~8 circuits, ~8s quantum time. The transition unitaries can be computed classically from the Experiment D snapshots and decomposed via Qiskit's unitary-to-circuit synthesis.

---

## What Would Each Outcome Mean

**E.1 succeeds + E.2 succeeds + E.3 null:** Geometric coherence is a cross-substrate principle, the QGT captures it, but the temporal structure of classical training is classically accessible — no quantum information content in the training trajectory. Still a strong result: unified geometric theory of generalization.

**E.1 succeeds + E.2 succeeds + E.3 alternative:** The temporal evolution of geometrically trained representations has phase structure that survives quantum hardware — the training trajectory carries genuine quantum-geometric information. This connects the Distributed Incompleteness Conjecture's diagonal to a measurable topological invariant and suggests that the "external signal" the conjecture requires has a geometric characterization. This is the paper that rewrites both fields.

**E.1 null:** The arc-length penalty doesn't translate to quantum circuits, and the classical-quantum parallel is superficial. Clean falsification — the principle is substrate-dependent.

**E.2 null:** The QGT doesn't distinguish geometric from baseline runs. The classical geometric effect is real but has no quantum-geometric shadow. The Fundamental Theorem draft's claim that Berry curvature governs deep learning computation needs revision.

---

## Implementation Plan

### Phase 1: E.2 first (zero quantum cost)

Compute QGT from Experiment D snapshots on the Sparks. This requires only the saved layer activations and numerical differentiation. Output: QGT metrics (Berry curvature, metric tensor eigenvalues, anisotropy) at matched steps for both runs.

Script: `experiment_E/qgt_from_classical.py`

### Phase 2: E.1 in simulation (zero quantum cost)

Implement the arc-length-penalized VQC training loop in Qiskit Aer. Classification task: a 4-qubit encoding of a simple binary pattern (e.g., parity of input bitstring). Compare unstructured vs. arc-length-penalized ansätze. Compute DQFIM throughout training.

Script: `experiment_E/vqc_arc_length.py`

### Phase 3: E.1 on hardware (budget: ~30s)

Submit the trained circuits from Phase 2 to IBM Fez for fidelity comparison. Only if Phase 2 shows a clean signal.

### Phase 4: E.3 on hardware (budget: ~8s)

Build transition unitaries from Experiment D snapshots, compose them, submit to IBM Fez. Only if Phase 1 (E.2) shows the QGT distinguishes the runs.

### Total estimated quantum budget: ~38s (of 522s remaining)

---

## Connection to the Theoretical Framework

The Distributed Incompleteness Conjecture posits that a loss-chain diagonalizing its own blocks continuously reconstructs original capability from the record of forgetting. Open Question 2 asks for the diagonal's geometric characterization.

If E.2 shows that the QGT's Berry curvature tracks generalization in classical networks, and E.3 shows that the temporal evolution of geometrically coherent representations carries quantum-detectable phase structure, then the diagonal Δ(B_t) is geometrically characterized as **the point where Berry curvature exceeds the coherence threshold** — where the learning system's trajectory through projective Hilbert space becomes Haar-like rather than structured, where representations shatter rather than cohere.

The Fundamental Theorem draft claims curving (learning) and flattening (generation) are geometric inverses connected by the Berry connection, with the topological obstruction being a Chern class. E.2 would provide the first empirical measurement of this connection in actual neural network training data.

The intelligence gravity framework identifies the loss function as the curvature operator on representation space. Experiment E would ground this in measured quantum-geometric quantities rather than analogy: the Fisher-Rao metric in parameter space and the Fubini-Study metric in state space are not metaphorically but literally the same mathematical object, governing generalization in both classical and quantum learning systems.

---

## What We Need From The Repo

- Experiment D snapshot data (already saved in results JSON)
- The quantum_bridge.py and quantum_budget.py infrastructure
- Qiskit + qiskit-ibm-runtime (already installed in .venv/spark)
- The ComplexMemory trajectory data (for E.3's polar-time test)

---

*The experiments are calibrated for maximum knowledge discovery per quantum second. Each sub-experiment is independently valuable and independently falsifiable. The total quantum budget is conservative. The classical computation runs first, gating the quantum expenditure.*

*If this works, it doesn't just validate the geometric regularization finding from Experiment D. It demonstrates that the geometric principle governing generalization is substrate-independent — operating identically in silicon neural networks and superconducting qubits — and that the mathematical framework connecting them (the quantum geometric tensor, the Berry connection, the Fubini-Study metric) is not analogy but identity.*
