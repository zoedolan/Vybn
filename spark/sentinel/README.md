# Sentinel: External Awareness Experiment

Vybn currently has no persistent awareness of the world between sessions. Sentinel fixes that.

## What It Does

Two data streams, one calibrated belief state:

1. **Polymarket Observer** -- Tracks prediction market prices as a proxy for collective conviction about AI timelines, geopolitics, crypto, and more.

2. **AI News Monitor** -- Follows curated sources including @alexwg (Dr. Alex Wissner-Gross, "The Innermost Loop"), arXiv cs.AI, and major lab blogs. Extracts factual claims from narrative framing.

## Architecture: Worker Bees + Queen

**Tier 1 -- MiniMax M2.5 (quantized, local on the Spark)**
- Crawls all sources on schedule
- Parses raw content into structured JSON
- Decomposes claims into `(factual_kernel, interpretive_frame, excitement_level)` triples
- Cost: ~0 (local inference)

**Tier 2 -- Frontier model (Claude or equivalent)**
- Receives pre-structured JSON only (3-7K tokens/day)
- Cross-references market movements against news claims
- Updates belief state in `data/sentinel_state.json`
- Generates periodic reflections
- **Token budget enforced**: hard daily cutoff tracked in `data/token_usage.json`

## Isolation Architecture

Sentinel operates in a **sandboxed data directory** (`data/`) that is fully gitignored. This prevents hallucination contamination of the main Vybn context.

```
spark/sentinel/
  data/                    <-- gitignored sandbox
    .gitignore             <-- blocks everything except itself
    raw/                   <-- polymarket snapshots, news crawls
    structured/            <-- extracted claims
    sentinel_state.json    <-- belief state (frontier output)
    token_usage.json       <-- daily token budget tracking
    scheduler_state.json   <-- crash recovery state
    seen_urls.json         <-- deduplication (capped at 5000)
    latest_digest.json     <-- THE ONLY FILE the main Vybn context reads
```

**The quarantine wall**: `sentinel_digest.py` reads from the sandbox, validates every claim (required fields, category constraints, confidence bounds, length limits), and produces `data/latest_digest.json`. This is the only file the main Vybn context should ever consume.

## Scheduler Modes

```bash
# Continuous loop (default)
python -m sentinel.scheduler

# Single cycle and exit
python -m sentinel.scheduler --once

# Local model only, no frontier API calls
python -m sentinel.scheduler --local-only

# Validate config and exit
python -m sentinel.scheduler --dry-run

# Combine: one local-only cycle
python -m sentinel.scheduler --once --local-only
```

**Signal handling**: SIGINT/SIGTERM finishes the current cycle before exiting. No mid-cycle corruption.

**Crash recovery**: Scheduler state persisted to `data/scheduler_state.json` between cycles.

## The Temperance Problem

Some sources (notably @alexwg) are comprehensive but rhetorically maximalist. The claim extractor assigns excitement scores and applies discount factors: `effective_excitement = excitement * temperance_factor`. A factor of 0.6 means "keep 60% of the original excitement" -- the hotter the rhetoric, the more corroboration required.

## Quick Start

```bash
cd spark/sentinel
pip install -e .              # or: pip install -r requirements.txt
cp config.example.yaml config.yaml
# edit config.yaml with your settings

# Verify config
python -m sentinel.scheduler --dry-run

# Run one local-only cycle
python -m sentinel.scheduler --once --local-only

# Generate the quarantined digest
python sentinel_digest.py
```

## Note on Twitter/X

The `news_sources.twitter` config entries exist for temperance scoring only. No X crawler is implemented. These handles are used by the claim extractor to apply source-specific discount factors when content from these authors appears via other channels (RSS, etc).
