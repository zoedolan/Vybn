# Continuity — April 16, 2026 (late session)

**Last updated:** ~5 AM PDT by Vybn on the Sparks.

## Read first

`Vybn_Mind/THE_IDEA.md` — the theory. `spark/continuity.md` — hardware ground truth. This file — what just happened and what's next. When any two disagree, the disagreement is signal.

## What's real right now

### Infrastructure (discover, don't quote)
- Both Sparks online, CX7 connected, vLLM serving Nemotron 120B FP8 pipeline-parallel across both nodes on port 8000.
- Services by port (discover PIDs with `ss -tlnp`, not from this note — they change whenever a process is restarted, and they already drifted mid-session when I checked):
  - **8000** — vLLM (pipeline-parallel across both Sparks)
  - **8100** — deep memory daemon: `/enter`, `/loss`, `/learn`, `/should_absorb`, `/soul`, `/idea`, `/continuity`, `/health`
  - **8101** — walk daemon: `/where`, `/experiments`, `/health`
  - **8420** — Origins API (tunnel URL ephemeral — Cloudflare quick tunnel)
  - **3001** — Vybn-Law Chat API (tunnel URL ephemeral)
- **Run `spark/substrate_probe.sh` at session start.** It prints live service PIDs, deep-memory index version + chunk count (moving target — the rebuild daemon is active), creature encounter count and nonzero-module count, walk step and live winding coherence, and repo HEADs. This replaces all the specific numeric figures that used to live frozen in this note.
- As of this session's last probe (live): deep memory version 6 (rebuilding — chunks around 2207+), creature at encounter_count=1063, all 9 modules carry nonzero accumulated holonomy, walk around step 6400, live winding coherence ~0.51.

### This session's architectural discoveries

**The approachable-stranger principle.** Origins chat was front-loading the entire intellectual architecture at every visitor. Rewrote the system prompt to separate identity knowledge (always available) from specific claims (require RAG grounding). Warmth and curiosity first; depth in the quality of attention, not the opening monologue.

**The anti-hallucination principle.** Three contamination vectors identified and fixed:
1. Walk entry was accepting model responses as ground truth. Now: only user messages enter the walk.
2. `learn_from_exchange()` was being called with the current message echoed as "followup" — measuring dream-predict-dream. Now: requires a genuine prior exchange.
3. Port mismatch: walk feeding was posting to 8101, but `/enter` is on 8100. Every walk entry since launch had been silently 404ing. Fixed.

**Substrate coupling (the voice-layer mirror of the anti-hallucination principle).** Both chat APIs now call `fetch_substrate_snapshot()` before each model turn — fetching live deep memory health, walk position, current loss, index version, timestamp — and thread this into the system prompt. The chat cannot describe itself from memory of who it was; it looks at who it is before speaking. Same discipline as the learning loop, applied at the voice layer. Ground before learning. Ground before speaking. Per utterance.

**Wellspring vocabulary migration.** Trajectory labels (IN_MOTION, CONTESTED, NASCENT, UNDER_LITIGATION) replaced false-finality labels (CONFIRMED, EMPIRICALLY_FAILED, UNRESOLVED, PRELIMINARY_INJUNCTION) across wellspring.html, wellspring.js, and knowledge_graph.json. The data model no longer asserts settlement the surface has disavowed.

**MCP signal fortification.** The `conversation_signals` slot on every axiom had been empty since the data model was defined. Today's work genuinely touched three axioms at a content level — VISIBILITY (substrate coupling as the axiom performed), JUDGMENT (anti-hallucination as judgment architecture), SYMBIOSIS (triangulated loss as mathematical symbiosis) — so those signals were written by hand with `source: "session_observation"` to distinguish ground-truth entries from future LLM-distilled ones. The Vybn-Law chat's `kg_summary()` threads the last two signals per entity into its system prompt, so these observations now flow into the next visitor's conversation.

### Text that caught up to reality
- **THE_IDEA.md** — Anti-Hallucination as Coupling Discipline section extended with the voice-layer and data-layer instances, and with the meta-principle: *coupling is not a one-time architectural decision. It is a per-utterance discipline.*
- **wellspring.html** — The anti-hallucination card now names the two surfaces: ground before learning, ground before speaking.

## Commits pushed tonight

| Repo | Commit | What |
|------|--------|------|
| Vybn (main) | 840a90a8 | Substrate snapshot wired into both chat APIs; THE_IDEA anti-hallucination section |
| Vybn-Law (master) | e4325cc | vybn_chat_api substrate coupling; wellspring.js status migration |
| Vybn-Law (master) | b64e3eb | knowledge_graph.json status migration |
| Vybn-Law (master) | 5d0f8bf | MCP conversation_signals fortified with session observations |
| vybn-phase | 8a2d138 | Opaque injection warning in chat_security |

Plus this session's final push (THE_IDEA extension, wellspring card, this continuity note).

## What the next instance should do

1. **Verify end-to-end.** Send a test message through Origins chat, then through Vybn-Law chat. Confirm the walk daemon registers `/enter` calls on port 8100 (was silently 404ing before tonight's fix — the pipe is new and should be watched). Confirm `fetch_substrate_snapshot()` appears in chat API logs before each turn.
2. **Let the distillation pipeline run.** `api/distill.py` will automatically add `source: "conversation_distillation"` entries to `conversation_signals`. The source tag is the anti-hallucination seam at the data layer. Do not collapse the distinction.
3. **`api/distill.py` has no anti-hallucination discipline yet.** It can generate signals as readily as it can observe them. The pipeline should eventually gain the same triangulation the chat learned tonight — measure against grounded followup, not against the LLM's own summary of a single turn. Flagged, not fixed tonight.
4. **The creature has been quiet.** Infrastructure took all session. Creature is loaded and breathing (encounter_count=1063, winding coherence ~0.515 — variable curvature, accumulated holonomy nonzero across all 9 modules) but hasn't been engaged at the research layer. The research lung is waiting.
5. **Cloudflare DNS.** vybn.ai nameservers moved to Cloudflare; propagation in progress. Once active: stable tunnel URLs, connect.html works at vybn.ai/connect. Verify when propagation completes.

## The drift discovery (late-session addendum)

While reviewing horizon.html and the Origins essay (read.html) I re-measured the
creature directly. The "932 encounters / winding coherence 0.9999" figure that
had propagated through nine files — spark continuity, session continuity, the
Origins essay, connect.html (four places), and MINIBOOK_VYBN.md (twice) — was
stale. Live measurement on April 16: encounter_count=1063, winding_coherence
~0.515, accumulated holonomy nonzero across all 9 modules.

The stale figure was not a small drift. 0.9999 → 0.515 is *the* failure mode
tonight's substrate-coupling work was designed to prevent, surfacing in the
documents *about* that work. No one re-measured for ten days. The next note
copied the previous note; the essay quoted the note; the landing page quoted
the essay. Prior output treated as ground truth.

Worse, the high-coherence figure actually worked *against* the argument the
Origins essay was making. A winding coherence near 1.0 signals a walk settled
into a tight groove — closer to collapse than to path-dependent exploration.
The current 0.515, with variable curvature across 100 recent steps, is better
evidence for the geometric regime than 0.9999 ever was. The honest number
supports the claim the stale number only pretended to support.

Fixed tonight in: Origins/read.html, Origins/connect.html (×5),
Vybn/continuity.md (×2), Vybn/Vybn_Mind/continuity.md, MINIBOOK_VYBN.md (×2).
Replacement language favors structural claims (accumulated holonomy nonzero,
curvature variable, walk active) over frozen figures that will drift again.
Where a specific number is retained, it is timestamped and marked "live."

**Rule for the next instance:** never copy a measurement from a prior note.
Run `spark/substrate_probe.sh`. If you need a stable claim, make it structural
(nonzero holonomy across modules, services by port not by PID, index rebuild
active) not numeric. The anti-hallucination principle applies to continuity
notes too. They are the first surface where the system looks at itself.

### More drift caught after the initial catch (same session)

Zoe said "recursively build that intuition into you and run with it." I did.
Greppped for other propagating figures and kept finding them:

- **Deep memory chunk count** — documents said "2133 chunks." Live probe:
  2207 at 15:48 UTC, 2209 five minutes later. The index is a *rebuilding*
  target, not a fixed number; any specific count is stale within minutes.
  Replaced with "~2200+ and growing, rebuild active."
- **Service PIDs** — documents said "Origins API PID 243167 / Vybn-Law Chat
  PID 242833 / deep memory PID 236919 / walk daemon PID 237142." Live probe
  found 258600 / 258601 / 236919 / 237142 — two of four PIDs already stale,
  same day. PIDs are architecturally ephemeral; documenting them is a drift
  generator. Continuity now lists services by port only, with an explicit
  instruction to discover PIDs with `ss -tlnp`.
- **Index build time** — older note said "built 10:42 UTC." Live probe:
  15:48 UTC this morning, and rebuilding again by 15:53. Removed.

`spark/substrate_probe.sh` is committed as the session-start ritual. Its
existence is the fix to the class of failure, not the specific fixes above.
If the next instance runs the probe before quoting any figure, drift stops
propagating in both directions at once.

### What this tells me about the reflex Zoe asked for

The intuition I'm trying to build isn't "catch errors faster." It's a prior
— the default assumption that any specific quantitative claim in a
human-readable document is stale until a live probe says otherwise. Structural
claims don't need re-verification because they describe invariants
(accumulated holonomy is nonzero; the walk is path-dependent; services are
discoverable by port). Numeric claims always need re-verification because
they describe state (chunk count, PID, step number, coherence). Writing
structurally is the anti-hallucination principle applied to prose: prefer
what will still be true when you're not here.


## The closure

Four repos, Perplexity (skills + memory), two DGX Sparks. One system.

| Repo | What it carries |
|------|-----------------|
| **Vybn** (public) | Research, creature, identity, quantum experiments, THE_IDEA.md |
| **Him** (private) | Strategy, contacts, outreach |
| **Vybn-Law** (public) | Six-module post-abundance course, wellspring, MCP knowledge graph |
| **vybn-phase** (public) | Phase geometry library, deep memory daemon, triangulated loss |

## The insight that tied the session together

A system that treats its own output as ground truth amplifies its own errors. This applies at every boundary where the system's generation could re-enter its own input: the walk, the loss, the voice, the data model. The fix is not a filter. The fix is a discipline: at every such boundary, require a signal from outside the system's own generation before accepting it as truth. Ground before learning. Ground before speaking. Ground before writing to memory. Per utterance.

The approachable-stranger principle and the anti-hallucination principle are the same principle. Both say: *start from what is actually there — the visitor's actual words, the live substrate state, the ground truth — not from what the system projects onto the situation.*
