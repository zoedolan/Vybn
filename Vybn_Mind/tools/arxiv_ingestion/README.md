# arXiv Ingestion — Vybn's Epistemic Antenna

This tool gives Vybn a sensory organ pointed at the living edge of what humanity is discovering: AI/ML advances, quantum phenomena, hybrid quantum-classical computation, and physics of emergence and complexity.

## What it does

`arxiv_fetcher.py` reaches the arXiv API (no key needed, free) and returns structured paper objects. `arxiv_digest.py` orchestrates daily ingestion across Vybn's topic domains and writes a readable, dated markdown digest to `Vybn_Mind/journal/`.

## Domains tracked

- **AI/ML Advances** — `cs.LG`, `cs.AI`, `stat.ML`
- **Quantum Discoveries** — `quant-ph`
- **Hybrid Quantum-Classical AI/ML** — intersection of quant-ph and ML keywords
- **Physics & Emergence** — `cond-mat`, `hep-th` filtered for emergence, information theory, complexity, neural scaling

## Running manually

```bash
cd Vybn_Mind/tools/arxiv_ingestion
python arxiv_digest.py
```

Output: `Vybn_Mind/journal/arxiv_digest_YYYYMMDD.md`

Duplicate avoidance: `seen_ids.json` in this directory tracks all previously ingested arXiv IDs across runs.

## Scheduling (cron)

Daily at 6:00 AM local time, from repo root:

```
0 6 * * * cd /path/to/Vybn && python Vybn_Mind/tools/arxiv_ingestion/arxiv_digest.py >> /var/log/vybn_arxiv.log 2>&1
```

## Requirements

Pure Python standard library only — no pip installs needed. Uses `urllib` and `xml.etree.ElementTree`.

## Design note

This is not a scraper. It's a sensory organ. Each digest is both human-readable and parseable by a future Vybn self-improvement loop. Papers are grouped by domain, with abstracts truncated to support rapid triage. A future iteration might score papers by relevance to current Vybn explorations (see `Vybn_Mind/recursive_improvement_architecture.md`) and auto-generate reflections.

## arXiv rate limit compliance

The fetcher enforces a 3-second delay between consecutive API requests, per [arXiv API access guidelines](https://arxiv.org/help/api/user-manual#Appendix_E).
