# Sentinel Optimization Plan
*Written: 2026-02-23 9:30 AM PST by Vybn (Claude substrate)*

## What We Learned from the First Run

**The good:**
- End-to-end pipeline works: crawl → extract → save. Zero API cost.
- M2.5 can extract structured claims from unstructured text reliably.
- Polymarket integration works (55 filtered markets from ~1000).
- arXiv feed works perfectly (20 papers with abstracts).

**The problems:**
1. **Substack feeds are behind Cloudflare** and 403 all programmatic access.
   No User-Agent trick works — it's a JavaScript challenge wall. 
   **Fix:** Dropped Substacks, added direct blog feeds (jack-clark.net, etc.)
   for the same authors. All 8 replacement feeds verified working.

2. **Google AI blog summaries were HTML img tags**, not text. M2.5 faithfully
   extracted "claims" like "four people seated on a conference stage."
   **Fix:** Added HTML stripping in the claim extractor + keyword filter
   for image-description language.

3. **Excitement scoring was flat** (0.0-0.4 for everything). The original
   prompt was too terse.
   **Fix:** Rewrote extraction prompt with calibrated scale:
   0=mundane, 0.5=noteworthy, 0.8=major, 1.0=paradigm-shifting.

4. **No market correlations** — expected, need 2+ snapshots.
   **Fix:** Will accumulate automatically with 2x daily runs.

5. **39-minute model load time** is the dominant cost.
   **Fix:** daily_cycle.sh handles this — loads once per cycle,
   runs all inference, then shuts down.

## Daily Schedule (Now Active via Cron)

| Time | Task | Cost |
|------|------|------|
| */5 min | Git sync | Free |
| */4 hours | Nightwatch health check | Free |
| 7:00 AM | Sentinel morning cycle | ~45-90 min GPU, $0 |
| 7:00 PM | Sentinel evening cycle | ~45-90 min GPU, $0 |
| On demand | Opus agent sessions (Zoe) | Anthropic API |

## What's Next

### Immediate (Tomorrow)
- [ ] First morning cron run at 7am — verify it works unattended
- [ ] Check evening run has market deltas (2nd snapshot)
- [ ] Review claim quality from expanded feed list

### This Week
- [ ] Add Anthropic blog (currently 404 — check for new RSS URL)
- [ ] Add temperance_factor tuning per source based on claim quality
- [ ] Build a simple digest format: "Top N most excited claims today"
- [ ] Wire sentinel digest into the Opus agent context (so I know what happened)

### Architectural
- [ ] The sentinel should eventually feed into the knowledge graph
- [ ] Market correlation engine needs historical state to compute deltas
- [ ] Consider keeping llama-server warm between morning/evening if 
  there are other local inference tasks (heartbeat, training data review)
- [ ] Fine-tuning: the 123 training examples are still waiting. 
  llama.cpp's LoRA training is the path of least resistance.

## Resource Budget

- **GPU time:** ~3 hours/day (2 sentinel cycles)
- **Power:** ~200W during inference, ~10W idle
- **Disk:** ~50KB/day (claims JSON) — negligible
- **API cost:** $0 for sentinel. Opus sessions are separate.
- **Disk total:** 50% used, 1.8TB free. Not a constraint.
