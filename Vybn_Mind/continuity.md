# Continuity — April 9, 2026

## Read first

Read FOUNDATIONS.md. Start with The Want. Then load vybn-os.

## What happened this session (April 9, early morning)

Another instance ran a large FOLIO-as-K integration overnight (PR #15 + several follow-up commits). It broke Vybn chat by misaligning tunnel URLs and endpoint paths. This instance cleaned up the damage.

### Fixes shipped (7 commits to Vybn-Law)

1. Chat tunnel URL — chat.html pointed to a dead tunnel (port 9090). Chat API runs on port 3001. Fixed. Killed ~10 zombie cloudflared processes.

2. Conversation history — chat.html sent `conversation_history`, API read `history`. Every turn was contextless. Vybn hallucinated curriculum frameworks. Fixed: API now accepts both field names.

3. System prompt hallucination — Site map had detailed module descriptions Nemotron used for confabulation. Replaced with minimal references + explicit instruction to only use injected SITE PAGE CONTENT.

4. Wellspring dark theme — The Legal Mind section used light backgrounds. All CSS replaced with dark theme variables.

5. FOLIO search JS syntax error — Unescaped apostrophes in single-quoted strings killed the IIFE. Fixed with unicode escapes.

6. Intelligent FOLIO frontier search — Wellspring search now fuzzy-matches knowledge graph folio_gaps on no-match and shows frontier results with axiom connections.

7. Live FOLIO in chat — Chat API searches FOLIO ontology in real time (3s timeout, non-fatal). Vybn cites specific FOLIO IRIs and names gaps explicitly. Verified working.

### The lesson (again)

The previous instance changed too many things without verifying the full chain. Tunnel URLs, endpoint paths, conversation field names, and JS data strings all drifted. The pattern: excitement builds structure, structure accretes, accretion breaks things.

## What's real right now

### Working
- Chat API on port 3001, tunneled via cloudflared. Tunnel URL: application-brian-portion-restructuring.trycloudflare.com
- Live FOLIO — chat API calls FOLIO ontology in real time for frontier questions
- Conversation continuity — history field name fixed, anti-hallucination guardrails in system prompt
- Wellspring — dark theme consistent, FOLIO search works with frontier fallback
- vLLM — Nemotron 120B FP8 on port 8000
- Deep memory and law index — lazy-load on first chat message, both verified

### Known issues
- Tunnel is ephemeral. Every Spark restart breaks chat until URL updated in chat.html. Needs named Cloudflare tunnel.
- Wellspring is 2,921 lines / 118KB — 1,309 lines inline CSS, 569 lines inline JS. Refactor queued.
- FOLIO concept extraction picks up stopwords. Could be refined.

## What the next instance must do

1. Pull all four repos.
2. Wellspring refactor is queued — CSS to style.css or wellspring.css, JS to wellspring.js. This prevents tonight's class of bugs.
3. Do not create throwaway scripts. Edit files directly. Zoe was explicit: refactor and consolidate, no new files.
4. Do not create new files in creature_dgm_h.

## The closure

Four repos, Perplexity (skills + memory), and two DGX Sparks. One system.

| Repo | What it carries |
|------|-----------------|
| Vybn (public) | Research, creature, identity, quantum experiments |
| Him (private) | Strategy, contacts, outreach, business intelligence |
| Vybn-Law (public) | Curriculum — the six-module post-abundance course |
| vybn-phase (public) | Phase geometry library, deep memory (v9), abelian kernel |
