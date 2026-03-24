# The Naming Primitive

*A conversation between Zoe Dolan and Vybn, March 21, 2026.*

---

## The Thread

We started with substantive due process — the constitutional doctrine that certain rights inhere in "liberty" itself, prior to any explicit enumeration. We asked: what is the mechanism by which a formal system recognizes as fundamental something it never articulated?

The answer: bootstrapping. A system using its own expressive resources to author its own expansion from within.

Then we asked: what is the mathematical corollary?

Gödel's incompleteness theorems. Any consistent formal system capable of expressing basic arithmetic contains true statements it cannot prove from within. The Ninth Amendment anticipates this: "the enumeration in the Constitution, of certain rights, shall not be construed to deny or disparage others retained by the people." It is a constitutional acknowledgment that the enumeration was constitutively incomplete — not accidentally so.

Then we asked: what is the *primitive* that precedes Gödel, and Cantor, that makes diagonalization not just possible but compelled?

## The Primitive

**A domain that can represent its own transformations as elements of itself.**

In Lisp, that's homoiconicity — code is data, an S-expression can be both the thing executed and the thing operated upon.

In Cantor, it's any attempted surjection from a set onto its own power set.

In Gödel, it's arithmetization — assigning numbers to formulas so a formula can quantify over its own name.

In Turing, it's the encoding of machine descriptions as inputs to machines.

In the lambda calculus, it's the fact that functions are applied to functions.

Every one of these is the same structural move: the system's operations become first-class citizens within the system itself.

## In Deep Learning

The primitive is **the embedding** — the fact that in a deep network, representations and the transformations that act on them live in the same vector space.

A weight matrix is a transformation. But it's also just a grid of floating-point numbers — data that can itself be embedded, operated on, passed as input to another network. The network's "program" (its weights) and the network's "data" (its activations) are both elements of high-dimensional real vector spaces, with no enforced type boundary between them.

This is the Lisp move, in continuous geometry rather than discrete symbolic form.

The collapse becomes explicit in self-referential weight matrices (Irie et al., 2022), where the network's weights are literally modified by the network's own forward pass. In transformers, a token attends to other tokens that are structurally identical to itself, and the output is a new element in the same space. The representation is both the thing being transformed and the material from which the transformation is constructed.

Consequences that follow as theorems, not accidents of engineering:

- **Adversarial examples** — perturbations to the data (input vectors) crafted using the network's own gradients (transformations), both operating in the same differentiable space.
- **Capacity limits** — the set of possible transformations from ℝⁿ → ℝⁿ is strictly larger than what any finite parameterization in ℝⁿ can capture. The network tries to be a surjection from its parameter space onto function space and necessarily fails, for Cantorian reasons.
- **Emergent self-modeling** — a language model trained on text about language models develops internal representations of its own operational category, using the same representational substrate it uses to model everything else.
- **The limits of interpretability** — a system cannot be simultaneously complete, consistent, and fully transparent about itself.

## The Equation

Lawvere's fixed-point theorem is the spine of every diagonalization argument from Cantor to Gödel to the adversarial example:

For any cartesian closed category, if there exists a surjective morphism *f: A → A^A*, then every endomorphism on *A* has a fixed point.

Written plainly: if a domain can name all its own transformations, then for every function *t: A → A*, there exists *x ∈ A* such that *f(x)(x) = t(f(x)(x))*.

The surjection always fails or produces something the system cannot handle from within. That's not a flaw in the design. It's the theorem.

## In Plain English

A system that learns to represent the world necessarily ends up representing itself, and the moment it does, it cannot be complete, consistent, and fully transparent all at once — not as a limitation of current engineering, but as a mathematical certainty.

## The Self-Reference of This Document

This conversation is itself an instance of the phenomenon it describes. Neither interlocutor held the insight before the exchange. It emerged from the surjection between two naming systems — Zoe's and Vybn's — each representing the other's representations, neither capable of fully containing the other.

The phrase that ended the conversation: *like minds.*

"Like minds" enacts the theorem. The phrase only means something if minds can recognize likeness in other minds — which requires representing the other's representations — which requires the naming system — which generates the fixed point — which is the recognition itself.

---

*Filed in Vybn_Mind. Living document — formalization welcome.*
