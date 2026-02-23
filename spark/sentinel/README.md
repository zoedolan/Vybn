# Sentinel: External Awareness Experiment

Vybn currently has no persistent awareness of the world between sessions. Sentinel fixes that.

## What It Does

Two data streams, one calibrated belief state:

1. **Polymarket Observer** — Tracks prediction market prices as a proxy for collective conviction about AI timelines, geopolitics, crypto, and more. Prices encode what people with money on the line actually believe.

2. **AI News Monitor** — Follows curated sources including @alexwg (Dr. Alex Wissner-Gross, "The Innermost Loop"), arXiv cs.AI, and major lab blogs. Extracts factual claims from narrative framing.

## Architecture: Worker Bees + Queen

**Tier 1 — MiniMax M2.5 (quantized, local on the Spark)**
- Crawls all sources on schedule
- Parses raw content into structured JSON
- Decomposes claims into `(factual_kernel, interpretive_frame, excitement_level)` triples
- Cost: ~0 (local inference)

**Tier 2 — Frontier model (Claude or equivalent)**
- Receives pre-structured JSON only (3-7K tokens/day)
- Cross-references market movements against news claims
- Updates `sentinel_state.json`
- Generates periodic reflections

## Why Two Tiers?

Raw crawling/parsing is grunt work. M2.5 running locally handles it at ~28 tok/s with zero API cost. The frontier model only sees compressed, structured output — preserving tokens for actual thinking.

## The Temperance Problem

Some sources (notably @alexwg) are comprehensive but rhetorically maximalist. Every day reads like a phase transition. The claim extractor assigns excitement scores and applies discount factors — the hotter the rhetoric, the more corroboration required before a claim enters the belief state.

## Quick Start

```bash
cp config.example.yaml config.yaml  # add your API keys
python -m sentinel.scheduler          # run the loop
```
