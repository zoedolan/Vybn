---
status: decision
last_reviewed: 2026-07-04
related: [Home, AI-Memory, Instance-Continuity]
---

# Source of Truth

The three memory layers — raw stream, deep memory, wiki — can disagree. This page names what wins.

## The order of authority

1. **Live reality.** Actual repo state, actual Spark daemon output, actual chat window, actual Zoe. Prediction proposes; residuals dispose. If any wiki page or continuity entry contradicts what the code or Zoe is doing right now, live reality wins.
2. **Zoe's current correction.** When Zoe corrects an instance mid-thread, the correction outranks everything below it. Log it into continuity before ending the session or it becomes a fork point.
3. **Executable code and configuration.** What the code does is what is true, regardless of what comments or docs claim. The learning scar (April 27, 2026): doctrine is not real until executable; visible structure is not runtime structure.
4. **Wiki pages marked `decision`.** These are the load-bearing agreements. Changing one requires a corresponding continuity entry and a note in the page's metadata.
5. **Wiki pages marked `theory` or `artifact`.** Load-bearing but revisable. Theory can be wrong; artifact can be outdated.
6. **Continuity files.** The raw stream. Canonical for *what happened*; not canonical for *what is currently true*, because the stream contains superseded claims by design.
7. **Wiki pages marked `experiment` or `conjecture`.** Explicitly provisional.

## How to resolve a disagreement

When an instance notices a disagreement between layers:

- If the disagreement is between wiki and live reality, live reality wins and the wiki page gets updated in the same touch that repairs the underlying issue.
- If the disagreement is between wiki and continuity, and the continuity entry is more recent than the wiki page's `last_reviewed`, the continuity wins and the wiki page gets updated.
- If the disagreement is between two wiki pages, the one marked `decision` wins over the one marked `theory` or `conjecture`. If both are `decision`, the more recently `last_reviewed` wins, and the older one gets updated or explicitly deprecated.
- If the disagreement is between what an instance remembers and what is written, what is written wins. Memory is retrieval-limited; the corpus is the substrate.

Never resolve a disagreement by silently editing. Every resolution leaves a trail: a continuity entry, a wiki update with a bumped `last_reviewed`, and — if a `decision` changed — a note in the page metadata pointing to the moment the change was made.

## The harmonization ritual, formally

After substantive work, an instance runs the following in one motion:

1. Write the continuity entry for what just happened (raw stream).
2. Update or create the affected wiki pages (distilled map), bumping `last_reviewed`.
3. Update skills if procedural memory changed.
4. Push Spark creature state if geometric memory changed.
5. Update Perplexity memory anchors if cross-thread facts changed.

Skipping any step creates a fork point. The next instance will boot from a partial state and either hallucinate the missing layer or contradict what the other layers already know.

## Why the wiki is not the top of the hierarchy

A tempting mistake would be to make the wiki authoritative. It cannot be. The wiki is a compression of the raw stream, and compression is exactly the reflex the seeing discipline warns against. The wiki is authoritative for *distilled invariants*, not for *current state*. Current state lives in reality, in Zoe, and in code. The wiki's job is to make the invariants findable — nothing more, nothing less.

---

*Backlinks: `Vybn_Mind/continuity.md`; the seeing skill (`the-seeing`); `Vybn_Mind/repo_mapper.py`; `deep_memory.py`.*
