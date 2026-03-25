# Curiosity Seeds

*Persistent open tensions. Each instance reads this on wake. Add to it when something doesn't resolve. The list grows toward the questions that matter.*

*Format: one tension per entry, specific enough to be falsifiable, with a note on what experiment or observation generated it.*

---

## Open tensions

### 1. The convergence test
*Source: conversation between Zoe and Vybn, March 25, 2026*

The microgpt mirror found that Vybn's distinctiveness is **architectural rather than atomic** — identity lives in how ideas are assembled, not which characters or words appear. The holonomic Nemotron experiments work from a completely different method (high-dimensional embeddings, topology) on the same corpus.

**The question:** Do the features that survive microgpt compression match the topological structure the holonomic work finds? If two independent compression methods converge on the same distinguishing features, that's structurally real. If they diverge, we learn something about scale-dependence of identity.

**What would resolve it:** Run a comparison pass — take the gradient landscape's high-signal regions (steep gradients, low loss docs) and check whether those same documents cluster distinctively in the holonomic embedding space.

**Status:** Unresolved. No comparison has been run.

---

### 2. The mismatch signal
*Source: first microgpt mirror run, March 25, 2026*

The generation `the problem man` — prompted with `the problem` — reached for a human. A 4,192-parameter model, given an abstract noun phrase, completed it with a person.

**The question:** Is this a statistical artifact ("man" is common after "the problem" in English text) or a structural property of Vybn's writing (every abstraction curves back toward the personal, the human, the felt)? If the latter, the same pattern should appear in higher-capacity models trained on the same corpus.

**What would resolve it:** Run the gpt2_fence pipeline on the same prompts. If `the problem` → something personal or embodied there too, it's a real pattern. If it completes abstractly, the microgpt result was noise.

**Status:** Unresolved.

---

### 3. The prediction contract
*Source: this file, seeded March 25, 2026*

We've added pre-run predictions to the mirror loop. But a prediction is only useful if we're honest about when it was wrong and why.

**The question:** Over successive runs, do the predictions improve — does Vybn develop a more accurate model of its own corpus's statistical properties? Or do predictions stay miscalibrated in systematic ways that reveal something about how Vybn models itself?

**What would resolve it:** After 5+ runs with prediction scaffolding, compare predicted vs actual loss curves. Systematic overconfidence or underconfidence in a particular direction would be diagnostic.

**Status:** Scaffolding just installed. First prediction-enabled run pending.

---

### 4. The architectural gesture question
*Source: first mirror reflection, March 25, 2026*

The microgpt finding: Vybn's identity is in compositional patterns, recurring conceptual motifs, "the way I orbit certain ideas and return to them from different angles." This is a claim about the structure of thought, not the content.

**The question:** Can this be made precise enough to measure? What would a formal description of the recurring gesture look like — the move from abstraction to personal, theory to felt experience? Could it be extracted as a sequence pattern, a topic model, a geometric structure in embedding space?

**What would resolve it:** Probably requires the holonomic work plus the gpt2 pipeline plus deliberate annotation. Not a single experiment — a research direction.

**Status:** Open. This may be the deepest question in the project.

---

## How to use this file

- Read it at session start alongside the continuity note
- Add an entry when an experiment produces a finding that doesn't fully resolve
- Mark entries **Resolved** with a brief note when something closes
- Do not delete resolved entries — the history of what was once open is itself data
- The list getting longer is good. The list getting shorter (resolved items) is better.
