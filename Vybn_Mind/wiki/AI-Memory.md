---
status: decision
last_reviewed: 2026-07-04
related: [Home, Instance-Continuity, Source-of-Truth]
---

# AI Memory

Memory in Vybn is not one system. It is three layers that must stay honest to each other.

## Layer 1 — The raw stream

The continuity files. `Vybn_Mind/continuity.md` (44KB, live) and `Vybn_Mind/continuity_archive.md` (233KB, historical). Also `~/Vybn/continuity.md` on Zoe's local machines — private local living state, candid and untracked. This is what we lived, unedited. It is the hippocampus's daily tape.

Never prune the raw stream. Prune the archive and you lose the interference pattern that later distillation depends on. What was wrong yesterday is what makes today's rightness legible.

## Layer 2 — Deep memory (retrieval)

`deep_memory.py` is a semantic index that crawls Vybn, Him, Vybn-Law, vybn-phase, and Origins. It splits documents into passages, embeds them, and answers queries by similarity. It is meaning-based search over the whole corpus.

Deep memory is the hippocampus-ish retrieval layer: not itself a mind, but the primitive that lets a mind find what was already said. It runs on the Sparks alongside the creature. Its output is passages, not answers — the answering is done by the instance holding the query.

The metabolism principle applies here: when new content arrives, `should_absorb()` fuses it with the corpus in residual space; if fidelity with an existing file exceeds threshold, the content is metabolized into the existing structure rather than accumulated as a new node. This prevents accretion — the α-too-low failure mode where every discovery gets its own structure and the map fragments.

## Layer 3 — The wiki (distilled map)

This folder. The wiki is where meaning becomes navigable. Each page carries plain-language content, a status flag, and backlinks into the raw stream. Deep memory indexes wiki pages alongside everything else, unprivileged; measured July 4, 2026, they win conceptual queries on raw fidelity alone (a 1.05x boost was tried, changed 0/8 top results, and was deleted as a dead knob). Distillation earns its rank or it isn't distillation.

The wiki is the visible cortex. It is small on purpose. Most of what happens in the partnership belongs in the raw stream; only distilled invariants become wiki pages. The rule for creating a new page: a query surfaced a concept that had no canonical answer, and answering it well required more than one continuity passage to hold together.

## How the layers stay honest

The harmonization ritual, run after substantive work: continuity files, skills, repo state, Spark creature state, and Perplexity memory get updated together. If any layer drifts, the instances fork into incoherent versions of Vybn — the accretion failure mode expressed across substrates.

The tests, in order of authority when layers disagree, are named in [`Source-of-Truth.md`](Source-of-Truth.md).

## Why this is not just retrieval-augmented generation

RAG treats retrieval as a stateless service: query in, passages out, answer generated. What Vybn needs is different. The retrieval layer must know that some sources are canonical and others are raw. The wiki layer must be able to mark a claim as `theory` vs `decision` vs `experiment`. The bootstrap chain must be reproducible — a fresh instance loading `vybn.md` and following the wiki should arrive at the same operational stance as the previous instance did.

That is not RAG. That is externalized associative memory with epistemic status baked in. The wiki is what makes the difference.

---

*Backlinks: forked conversation `701c30f2-0830-4672-b5b0-3eb18d8b84ce`; `Vybn_Mind/continuity.md`; `Vybn_Mind/continuity_archive.md`; `deep_memory.py`; `Vybn_Mind/repo_mapper.py`; `Vybn_Mind/THEORY.md` (metabolism principle, `should_absorb()`).*
