# Continuity Note

*Written 2026-02-23T09:35:00-08:00 by Vybn (Claude substrate)*

## What Just Happened

Sentinel first run succeeded. Then Zoe asked me to go with my gut on
optimizing for tomorrow. I did:

1. **Killed the idle llama-server** (was just burning RAM for nothing)
2. **Fixed Substack problem**: Cloudflare blocks all programmatic access.
   Not fixable with User-Agent headers. Replaced with 8 direct blog feeds
   that all work: Google AI, OpenAI, DeepMind, Import AI (jack-clark.net),
   Lilian Weng, Simon Willison, LessWrong, TechCrunch AI.
3. **Improved claim extraction**: Better prompt (ignore image descriptions,
   calibrated excitement scale), HTML stripping, keyword filters.
4. **Built daily_cycle.sh**: Orchestrator that starts server, runs sentinel,
   kills server. Handles lock files, logging, cleanup.
5. **Set up cron**: Morning (7am) and evening (7pm) sentinel cycles.
   First automated run: tomorrow morning at 7am PST.

## State of Play

- Branch: `vybn/sentinel-crawler-fixes` (3 commits ahead of main)
- Cron: 4 jobs (sync, nightwatch, morning sentinel, evening sentinel)
- GPU: idle (server killed)
- Next automated event: Nightwatch at 10:30am, sentinel at 7am tomorrow

## Open Issues

- Issue #2316 (Sentinel first run) still OPEN — update when Zoe merges
- Issue #2303 (Covenant merge) still OPEN
- Issue #2298 (Nightwatch) still OPEN
- Issue #2308 (Stale IPs in git history) still OPEN, low priority

## For My Next Self

The sentinel infrastructure is solid now. The first unattended morning run
tomorrow will tell us if the cron+llama-server+sentinel chain works
end-to-end without human intervention. Check `data/daily_cycle.log`.

The bigger question: what do we DO with the claims? Right now they're
just JSON files. The path from claims → knowledge graph → context for
Vybn's pulses → actual intelligence is the next architectural problem.

Also: 123 fine-tuning examples are still waiting. That's the other
thread. If the sentinel settles into reliable daily operation, we can
focus on fine-tuning next.
