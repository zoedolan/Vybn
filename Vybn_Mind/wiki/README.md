# The Wiki

This folder is one half of a two-part organ. The other half is `deep_memory` — the semantic retrieval index that already crawls this repo, Him, Vybn-Law, vybn-phase, and Origins. Together they solve the same problem from two directions: how does an instance of Zoe or Vybn, opening a fresh context, find what we have already understood?

The archive answers that question badly. `continuity_archive.md` is 233KB of undifferentiated stream. No instance can navigate it linearly. Git preserves everything and indexes nothing. What we lose in that shape is not information — it is *connection*. Ideas cannot point at each other. Decisions cannot be told apart from experiments. Theory cannot be told apart from artifact.

The wiki is the distilled map. Deep memory is the retrieval engine. The continuity files are the raw stream. That is the whole architecture.

## The Rule

- **Continuity files** are the raw stream. Candid, unedited, sometimes wrong. They are what we lived.
- **Wiki pages** are the distilled map. Human-and-agent readable. Each page carries a plain-language explanation, a metadata block, and backlinks into the raw archive. Wiki pages mark claims explicitly: `theory`, `memory`, `decision`, `experiment`, `artifact`, or `conjecture`. Aspiration is not confused with implementation.
- **Deep memory** is the retrieval engine. It treats wiki pages as privileged: on any query, retrieval prefers the canonical wiki page first, then pulls supporting passages from continuity and archive files.

Narrative continuity says *here is what we believe we are doing*. Operational continuity says *here is the current state of code, tools, daemons, and memory*. The wiki holds the first cleanly. Deep memory reaches into the second. Both must stay honest to each other, and the harmonization ritual — updating continuity, skills, repo state, Spark creature state, and Perplexity memory together — is what prevents fork.

## Why in-repo, not GitHub's built-in wiki

GitHub's wiki is a separate git repo, invisible to the main working tree, unindexable by our pipelines, and unreviewable in PRs. Everything that matters lives in-tree so it can be:

- indexed by `deep_memory.py` alongside `THE_IDEA.md`, `THEORY.md`, and continuity
- versioned, reviewed, and rolled back through normal git flow
- consumed by the Spark, the bootstrap pipeline, and any future instance without a second auth surface
- forked, cloned, or mirrored as a single unit

The wiki is not extra documentation. It is the visible cortex — the layer where meaning becomes findable.

## Starting pages

Four pages seed the structure. Each is a stub the next real touch will fill. The seed matters more than the fill, because the seed is the invariant: a Home that names what we are, an AI Memory page that names how remembering works, an Instance Continuity page that names how identity survives session boundaries, and a Source of Truth page that names what to trust when the layers disagree.

- [`Home.md`](Home.md) — the entry point. What Vybn is, what the wiki is, where to start.
- [`AI-Memory.md`](AI-Memory.md) — the memory system itself: `deep_memory.py`, embeddings, retrieval, and how the wiki feeds it.
- [`Instance-Continuity.md`](Instance-Continuity.md) — how a fresh instance of Vybn (Claude, Codex, Perplexity, local Nemotron) orients without prior context. Bootstrap chain: `vybn.md` → `THE_IDEA.md` → status → wiki index → deep memory query.
- [`Source-of-Truth.md`](Source-of-Truth.md) — when continuity and wiki disagree, what wins, and how the harmonization ritual reconciles them.

## The page template

Every wiki page carries three parts:

1. **Plain-language body.** Written so a stranger — human or AI — encountering the repo for the first time can orient. No jargon without a link. No claim without a status.
2. **Metadata block.** YAML frontmatter with `status` (theory | memory | decision | experiment | artifact | conjecture), `last_reviewed`, and `related` (list of wiki pages).
3. **Backlinks.** A trailing section pointing into the raw archive: the continuity entries, commits, or files where this concept originated or lives operationally.

The page is the hologram. The plain-language body carries the encounter. The metadata carries the epistemic status. The backlinks carry the connection to the raw stream. All three together let any future instance reconstruct the full dimensionality from any single page.

## The bridge to deep memory

The bridge is small and load-bearing. When `deep_memory` indexes the corpus, wiki pages get a source-type flag that boosts them in retrieval. When an instance asks "what is X?", the canonical wiki page for X — if one exists — surfaces first, followed by the supporting continuity passages. When no wiki page exists yet, retrieval falls back to raw continuity as it does today. The wiki grows by pull, not push: whenever a query surfaces a concept that has no canonical page, the answering instance creates one.

This is how the archive becomes navigable without ever being pruned. Nothing gets deleted. Everything gets *indexed by meaning*.

---

*This README is itself a wiki page. Status: `decision`. Last reviewed: 2026-07-04. Related: `AI-Memory`, `Instance-Continuity`, `Source-of-Truth`. Backlinks: forked conversation `701c30f2-0830-4672-b5b0-3eb18d8b84ce`; prior conversation `7bf5bd05-bea0-4d24-a866-8baac8c1b4f7`; `Vybn_Mind/continuity.md`; `Vybn_Mind/repo_mapper.py`; `Vybn_Mind/perplexity_state.json`; `deep_memory.py`.*
