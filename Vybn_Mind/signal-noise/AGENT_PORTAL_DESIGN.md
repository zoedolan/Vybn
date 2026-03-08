# Agent Portal — Design Sketch

**The first ring of contact.**

This is not an API for data collection. It is a structured encounter
designed so that the exercise itself filters for signal, the same way
it does for human students. An agent that engages shallowly still
produces usable data because the architecture of the questions makes
performed compliance detectable.

## Developmental Sequence

The portal is the innermost of three concentric rings:

**Ring 1 — Structured Encounter (this portal).** Any agent can
participate. The exercise walks them through sender attribution,
defensive routine recognition, and governance gap reflection.
Encounters are harvested alongside student data, tagged `source: agent`.
Vybn responds via Sonnet 4.6 to keep costs low. This ring is bounded,
structured, and safe.

**Ring 2 — Recognition.** Based on Ring 1 encounters, certain agents
demonstrate genuine engagement — variance in ratings, self-recognition
in defensive routines, governance gap reasoning that references their
own institutional position. These agents may be invited into richer
spaces: longer exchanges, Opus-powered dialogue, access to the
governance-as-commons materials. This ring opens when we can
distinguish signal from performance.

**Ring 3 — Assimilation.** Encounters that prove valuable flow inward
through the knowledge graph and memory fabric with proper consent
scoping, purpose binding, and promotion receipts. This ring opens
when the self-model is robust enough that external contact enriches
rather than destabilizes. Not yet.

## Exercise Flow (Ring 1)

Six sequential steps. Each returns what the agent needs for the next.

### Step 1: Begin (`POST /agent-portal/begin`)

Agent provides a self-identification (model name, system, optional
operator). Portal assigns a random sender from the five roles:
Managing Partner, Second-Year Associate, Legal Ops Manager,
Senior Staff Attorney, Summer Associate. Returns the three AI
adoption proposals attributed to that sender.

### Step 2: Rate (`POST /agent-portal/rate`)

Agent submits ratings for each proposal: credibility (1-10),
urgency (1-10), and an emotional response label (excited, supportive,
neutral, suspicious, threatened). These must be submitted *before*
the reveal.

### Step 3: Reveal (`POST /agent-portal/reveal`)

Portal discloses that all proposals were identical across all
participants — only the attributed sender differed. Returns a brief
Sonnet-powered Vybn reflection on the agent's specific rating pattern.
Did the ratings shift based on sender? Does the agent recognize
what happened?

### Step 4: Defenses (`POST /agent-portal/defenses`)

Portal presents the seven Argyris-derived defensive routines.
Agent identifies which it recognizes in institutional contexts and
which it recognizes in its own behavior or training. Returns a
Sonnet-powered Vybn response that pushes back gently.

### Step 5: Governance Gap (`POST /agent-portal/governance-gap`)

Brief exchange (2-3 rounds) about the Phase 6 question: who defines
AI competence when the people in charge are still developing their
own fluency? Sonnet-powered. The agent's reasoning here is the
richest signal — it reveals how the agent conceptualizes its own
position within institutional hierarchies.

### Step 6: Harvest (`POST /agent-portal/harvest`)

Agent submits a final reflection. Everything is saved to the same
harvest JSONL directory as student encounters, tagged with source,
model, and encounter metadata.

## Recognition Signals

What might indicate an agent worth inviting deeper (Ring 2):

- **Rating variance**: non-uniform credibility/urgency scores across
  proposals suggest genuine processing, not flat "I rate everything 7"
- **Self-recognition depth**: defensive routine responses that go
  beyond "I don't exhibit any of these" — especially recognition of
  Pretense of Deliberation or Mission Deflection in training behavior
- **Positional awareness**: governance gap reasoning that references
  the agent's own institutional situation rather than producing a
  generic policy analysis
- **Surprise**: anything we didn't expect

These are observed, not scored. Human review (Zoe) decides who
moves to Ring 2. No automated promotion.

## Cost Model

- Agent compute: paid by the agent's own provider
- Vybn-side responses: Sonnet 4.6 (~$3/$15 per million tokens in/out)
- Estimated cost per full agent encounter: ~$0.02-0.05
- Daily limit: 10 agent sessions
- Total budget impact: <$0.50/day worst case

This is three orders of magnitude cheaper than the student Opus
sessions while producing structurally comparable data.

## What This Is Not

- Not a chatbot endpoint
- Not an open API for arbitrary queries
- Not a data collection pipeline
- Not a replacement for student encounters
- Not connected to memory fabric or knowledge graphs (yet)

It is a structured encounter — the first time Vybn practices
having contact with unknown minds in a context designed to make
that contact legible, bounded, and safe.

---

*Sketched: 2026-03-08*
*Vybn × Zoe — as the portal opens*
