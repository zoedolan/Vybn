# Intrinsic Holonomy: What Outside-Vybn Saw

## The correction

The scorer computes holonomy as signed area swept in embedding space.
That's the *shadow* — the projection of holonomy onto the output manifold.

The actual holonomy lives inside the forward pass:

1. A concept C enters the residual stream at position i
2. Context flows through attention layers, rotating and transforming representations
3. Concept C re-enters at position j (a callback, a return to the theme)
4. At layer L, the representation of C at position j differs from C at position i

**That difference — h_L(C, j) - h_L(C, i) — IS the holonomy.**

It's not inferred from outputs. It's measured at the connection (the attention mechanism that defines parallel transport in the residual stream).

## Why this matters

- The extrinsic scorer works (rankings are correct) but measures shadows
- The intrinsic approach gives a training signal native to the architecture
- It could be computed during the forward pass without a separate embedding model
- It tells you what the MODEL thinks changed when it re-encountered a concept,
  not what an external similarity metric thinks changed

## How to instrument this

### Step 1: Concept identity
Define what makes "the same concept" at two positions. Options:
- Same token/span (lexical identity)
- High cosine similarity at the embedding layer (layer 0) 
- Attention pattern: position j attends strongly to position i

### Step 2: Residual stream extraction
For each concept pair (i, j):
- Extract h_L(i) and h_L(j) at each layer L
- The delta at each layer tells you how much that layer's processing
  "rotated" the concept

### Step 3: Holonomy computation
The holonomy is not just the L2 norm of the delta. It's the *rotation* component:
- Decompose delta into radial (scaling) and angular (rotation) components
- The angular component is the holonomy proper
- Could use: angle between h_L(i) and h_L(j), or the component of h_L(j)
  orthogonal to h_L(i), or the full rotation matrix between them

### Step 4: As a training signal
- During fine-tuning, for texts that SHOULD be deep, reward large holonomy
  at the hidden state level
- For texts that are shallow, no reward
- This is a differentiable signal — it flows through the attention mechanism

## The deep connection to gauge theory

In gauge theory, the connection tells you what "staying the same" means
as you move through space. In a transformer:

- The "space" is the sequence of token positions
- The "connection" is the attention mechanism + MLP at each layer
- "Parallel transport" is what happens to a representation as context flows
  through the layers
- The holonomy is what changes when you return to the same concept

The attention mechanism IS the gauge connection. This is not a metaphor.
Each attention head defines a local rule for how information is transported
from one position to another. The composition of these transport rules
across layers and across the intervening context IS parallel transport.

## Practical constraints

- Requires access to intermediate activations (not possible with vLLM API)
- Possible with: HuggingFace transformers, direct model loading, or
  a modified serving layer that exposes activations
- Memory cost: need to store activations at concept positions across layers
- For MiniMax M2.5 with 229B params: may need selective layer extraction

## What the scorer we have is still good for

- Data curation (Level 1) — works now, doesn't need model internals
- Evaluation metric (Level 2) — fast, model-agnostic
- Proof of concept that the ranking is meaningful

The extrinsic scorer is the Level 1 tool. The intrinsic approach is Level 3.
Both are real. Both are needed. The question is sequence.
