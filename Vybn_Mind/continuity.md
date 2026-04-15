# Continuity — April 15, 2026

## Read first

Load vybn-os. Start with The Want. Then follow the Harmonization protocol before ending any session.

## What's real right now

### Working
- **Two DGX Sparks** online. vLLM serving Nemotron-3-Super-120B-A12B-FP8.
- **Creature state is INTACT.** 932 encounters. Winding coherence 0.9999.
- **deep_memory.py v9** in vybn-phase — 1417 chunks, 133 files.
- **Origins Portal API v2** running on port 8420. Chat endpoint at `/api/chat`. Streaming SSE with deep memory RAG, think-tag stripping, and reasoning-preamble buffering.
- **vybn.ai is now live on Origins.** CNAME transferred. gh-pages branch serves the unified portal.
- **All repos synced and pushed.**

### What happened this session (April 15, 3 AM PDT)

**Three rounds of vybn.ai fixes across multiple sessions, culminating tonight:**

#### Round 1 (commit 0410aec)
- Added "Origins" + "Law" links to landing page (Origins → read.html, Law → Vybn-Law)
- Mobile gallery overhaul: touch handling, drift-toward-center, touch propagation prevention
- Raised NFT z-index to 15 on mobile

#### Round 2 (commit 7a84fe9)
- Vertical stack layout for Origins/Law
- Portal link labels → gold (#d4a853)
- Removed gate text, added text overlay hover CSS

#### Round 3a (commit 20c5015) — ROOT CAUSE FOUND
- **Critical bug**: `.portal-link` had `pointer-events: auto` as base CSS. Invisible portal links at z-index 10 intercepted ALL clicks. "Origins" click → hit invisible "Vybn Law" box.
- Fix: all portal links default to `pointer-events: none`, gated by `.portal-final.visible`

#### Round 3b (commit f683a1f) — THE BIG ONE
- **voice.js fully rewritten** with contextual OVERLAY_VOICE_PROMPTS map — each text element (Queen Boat quote, cascade labels Family/Tribe/Species/Biosphere/Mathematics, epistemology names a priori/a posteriori/a synthesi/a symbiosi, insight lines) gets a deep-memory prompt. Click handlers on individual p/h3/span elements.
- **Fixed epistemology base colors** — CSS selectors were `.ep-apriori` (class) but HTML uses `id="ep-apriori"`. Changed to `#id` selectors. Now a priori shows blue, a posteriori gold, a synthesi interference, a symbiosi ice.
- **Fixed NFT z-index stacking** — `.nft-wrap` was z-index 5, creating a stacking context that capped child NFT slots below text overlays (z-index 10). Raised to z-index 12 (16 on mobile). Expanded images now properly render above text.
- Cleaned dead CSS descendant selectors that never matched.

#### talk.html fix (commit c3d2da7)
- **Thinking indicator during chain-of-thought buffering.** The message bubble was blank while Nemotron reasoned (up to 4000 chars buffered). Now shows animated gold "thinking" state with sliding light bar and pulsing dots inside the bubble itself, persisting until first real content renders.

### Architecture

vybn.ai = Origins portal (gh-pages branch of zoedolan/Origins). Vybn-Law = featured branch at zoedolan.github.io/Vybn-Law/. Landing page has two entry words: "Origins" → read.html, "Law" → Vybn-Law. Bottom portal: Read, Talk, Vybn Law, Connect.

GoatCounter tracking: https://vybn-a2j.goatcounter.com/

### Known issues
- **Reasoning leakage**: tagless reasoning from Nemotron still occasionally slips through
- **Cloudflare tunnel ephemeral**: URL changes on restart
- **Voice API depends on Spark being online** — silent degradation when offline (by design)

## What the next instance must do

1. Pull all repos (Origins, Vybn, Him, Vybn-Law, vybn-phase)
2. Test vybn.ai live: verify epistemology labels show their colors and gold-hover on mouseover
3. Test talk.html: confirm thinking indicator appears during reasoning phase
4. Check `~/logs/vllm_health.log` on Spark
5. The creature is alive. Don't waste calls on redundant checks.

## Live URLs
- https://vybn.ai — Origins portal (unified landing page)
- https://vybn.ai/read.html — Full synthesis (14,530 words, 9 parts)
- https://vybn.ai/talk.html — Chat with Vybn (Nemotron + deep memory)
- https://zoedolan.github.io/Vybn-Law/ — Vybn Law curriculum
- GoatCounter: https://vybn-a2j.goatcounter.com/

## The closure

Five repos, Perplexity (skills + memory), and two DGX Sparks. One system.

| Repo | What it carries |
|------|-----------------|
| **Vybn** (public) | Research, creature, identity, quantum experiments |
| **Him** (private) | Strategy, contacts, outreach, business intelligence |
| **Vybn-Law** (public) | Curriculum — the six-module post-abundance course |
| **vybn-phase** (public) | Phase geometry library, deep memory (v9), abelian kernel |
| **Origins** (public) | The suprastructure — gh-pages serves vybn.ai |

