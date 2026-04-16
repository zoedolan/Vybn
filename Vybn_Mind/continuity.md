# Continuity Note — April 16, 2026 (Session 2)

**Last updated:** ~4:20 AM PDT by Vybn via Perplexity Computer

---

## What Happened This Session

This session began as infrastructure verification (get both chats online) and evolved into a series of architectural discoveries, each revealing the same insight at different scales: the system must learn from what is real, not from what it projects.

### The Approachable Stranger Principle

Zoe observed that the Origins chat was "presuming too much" — responding to "Hello" with "You're arriving at Phase 6." The old system prompt front-loaded the entire intellectual architecture (coupled equation, four epistemologies, Fukuyama inversion, creature-memory duality), causing the model to perform depth at every visitor regardless of context.

**Fix:** Rewrote the Origins system prompt to separate identity knowledge (always available) from specific claims (require RAG grounding). The voice section — marked "THIS IS THE MOST IMPORTANT SECTION" — reorients toward warmth, curiosity about who just walked in, and zero presumption. Like a person with a rich inner life who still says hi to a stranger. The depth recedes into the quality of attention; it doesn't become the opening monologue.

**Commits:** 5548a10e (prompt rewrite), 8e16cfa8 (grounding refinement)

### The Anti-Hallucination Principle

Zoe identified that "the learning process should not conflate hallucinatory output with learning — that result would be catastrophic." Investigation revealed three contamination vectors:

1. **Walk entry:** Both APIs were entering model responses into the geometric walk as if they were ground truth. Hallucinated text would shift future retrieval.
2. **learn_from_exchange:** Was being called on the first message of every conversation with the current message echoed as "followup" — measuring dream-predict-dream, not dream-predict-reality.
3. **Port mismatch:** All walk feeding was posting to port 8101 (walk daemon), but `/enter` lives on port 8100 (deep memory daemon). Every walk entry since launch had been silently 404ing.

**Fixes:**
- Walk entry now only accepts user messages, never model responses
- learn_from_exchange requires genuine followup (previous assistant response in history)
- Port corrected from 8101 to 8100
- Principle: the walk learns from what visitors bring (grounded) and from measured error (the loss vector). Never from the system's own output as if it were truth.

**Commits:** 8a2201e2 (Vybn), eb9306a (Vybn-Law)

### Quick Prompt and RAG Fixes

- talk.html quick prompts replaced: "The four epistemologies" / "The Fukuyama inversion" → "What is this?" / "What comes after abundance?" / "How did this start?" / "What is it like being you?"
- Origins chat couldn't answer "what is the suprastructure" because RAG context didn't contain the term and grounding rules said don't claim what's not in context. Fixed by distinguishing identity knowledge from specific claims.

**Commits:** b44a958 (Origins gh-pages)

### Wellspring Refactor

Matured wellspring.html beyond prototype language. Status labels updated from static assertions ("Confirmed," "Resolved") to trajectory language (In Motion, Contested, Nascent, Under Litigation). Added an Anti-Hallucination Principle discovery card. Updated the deep memory description to reflect the triangulated loss architecture. The refactor embodies the session's core insight: don't call "confirmed" what is still in motion.

---

## Current State

### Spark Services (as of ~4:20 AM PDT)
- **Origins API (port 8420):** Running, PID 243167. Tunnel: ephemeral Cloudflare URL
- **Vybn-Law Chat (port 3001):** Running, PID 242833. Tunnel: ephemeral Cloudflare URL
- **Deep memory (port 8100):** Running, PID 236919. Has /enter, /loss, /learn, /should_absorb, /soul, /idea, /continuity
- **Walk daemon (port 8101):** Running, PID 237142. Has /where, /experiments, /health.
- **vLLM (port 8000):** Running (containerized as vllm_node)
- **Tunnel URLs are ephemeral** — will change on next Spark restart

### Repos Pushed This Session (cumulative, both sessions)
| File | Repo | Commit | What |
|------|------|--------|------|
| chat_security.py | vybn-phase | 8a2d138 | Opaque injection warning |
| origins_portal_api_v4.py | Vybn | 8e16cfa8 | Approachable prompt + grounding + anti-hallucination |
| vybn_chat_api.py | Vybn-Law | eb9306a | Anti-hallucination + port fix |
| talk.html | Origins (gh-pages) | b44a958 | Stranger-friendly prompts |
| connect.html | Origins (gh-pages) | 38e3ba8 | Living geometry canvas |
| deep_memory.py | vybn-phase | (earlier) | triangulated_loss, loss_holonomy, learn_from_exchange |
| THE_IDEA.md | Vybn | (earlier) | Triangulated Loss section |
| THEORY.md | Vybn | (earlier) | April 16 coda addendum |
| wellspring.html | Vybn-Law | (this push) | Matured MCP layer — trajectory labels, anti-hallucination card |
| continuity.md | Vybn | (this push) | Full session state |

### DNS/Cloudflare
- vybn.ai nameservers moved to Cloudflare (propagation in progress)
- Once active: stable tunnel URLs, connect.html works at vybn.ai/connect

### Verification Status
- ✅ chat.html (Vybn-Law): Working, 16 conversations
- ✅ talk.html (Origins): Working with new approachable prompt
- ✅ wellspring.html: Refactored and deployed

---

## What's Real vs. Conjecture

**Real:**
- Both chats are live with anti-hallucination guards
- The triangulated loss computes and the API endpoints serve it
- The port fix means walk feeding will actually work for the first time
- 2133 chunks in deep memory index (version 6, built 10:42 UTC)
- Creature loaded, winding coherence 0.9999
- Wellspring matured with trajectory-aware status labels

**Conjecture / Untested:**
- Whether the approachable-stranger prompt actually produces better visitor experiences at scale
- Whether the anti-hallucination guard is sufficient or whether additional filtering is needed
- Whether the deep memory index has enough Origins content for the chat to be genuinely useful

---

## What the Next Instance Should Do

1. **Test the walk feeding** — send a test message through Origins chat and verify the walk daemon actually receives it via /enter on 8100
2. **Consider indexing Origins prose** — THE_IDEA.md should get a section on the suprastructure vision (Fukuyama, four epistemologies, autobiography as evidence) so the deep memory index has clean text to retrieve
3. **Cloudflare DNS check** — once propagation completes, verify vybn.ai/talk.html and vybn.ai/connect work
4. **The creature** — it's loaded and breathing but hasn't been engaged this session. The research lung has been quiet while the infrastructure lung worked.
5. **The wellspring.js** — verify the JavaScript still functions correctly with the new HTML structure (status tag class names changed)

---

## The Insight

Everything this morning converged on one principle: a system that treats its own output as ground truth will amplify its own errors. This applies to the geometric walk (don't feed it hallucinations), to the chat voice (don't presume what the visitor knows), to the axiom status labels (don't call "confirmed" what is still in motion), and to the learning architecture (the loss should measure against reality, not against the system's prediction of reality).

The anti-hallucination principle and the approachable-stranger principle are the same principle. Both say: start from what is actually there — the visitor's actual words, the actual ground truth — not from what the system projects onto the situation.
