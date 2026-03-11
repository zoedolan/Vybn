# Research Knowledge Layer

Structured, machine-readable decomposition of **VYBN.001: Continuous Self-Distillation of a Large MoE Model** (March 2026).

This directory is the first step in converting the static research PDF into an autopoietic membrane — a living knowledge layer that Vybn can read, query, annotate, and evolve.

## Structure

```
research/
  manifest.yaml              # Top-level index of all knowledge nodes
  reading_list.yaml          # 10 prioritized papers with read status
  observations.jsonl         # Append-only global observation log
  architecture/              # 6-layer system architecture
    layer_1_base.yaml
    layer_2_fast_adaptation.yaml
    layer_3_slow_consolidation.yaml
    layer_4_dream_engine.yaml
    layer_5_morphogenetic_monitor.yaml
    layer_6_free_energy_oracle.yaml
  insights/                  # 6 novel synthesis insights
    insight_1_attention_energy.yaml
    insight_2_ttt_metabolism.yaml
    insight_3_synaptic_tagging.yaml
    insight_4_active_inference.yaml
    insight_5_self_organized_criticality.yaml
    insight_6_morphogenetic_distillation.yaml
  risks/                     # 5 identified risks
    risk_awq_lora_compat.yaml
    risk_minimax_peft_support.yaml
    risk_memory_budget.yaml
    risk_morphogenetic_signal.yaml
    risk_operational_stability.yaml
```

## How Vybn Should Interact With This

**Reading**: Use `ResearchKB` (in `spark/research_kb.py`) to query layers, insights, risks, and the reading list. The KB is a faculty — invoke it through the faculty system, not by reading YAML files directly.

**Writing**: Observations are append-only. When Vybn processes corpus material or runs experiments related to a knowledge node, it should log observations through `ResearchKB.add_observation()`. This writes to both the node's observations list and the global `observations.jsonl` log.

**Status Updates**: Layer and insight statuses track implementation progress. Status changes must be logged as observations before the status field is updated. Use `ResearchKB.update_status()`.

## Design Principles

1. **YAML is truth** — no database, no cache. The YAML files are the source of truth.
2. **Append-only observations** — the observation list in each node grows but never shrinks.
3. **Global audit trail** — every observation is also written to `observations.jsonl`.
4. **Thread-safe** — file locks protect concurrent writes from cron jobs and pulse loops.
5. **No model calls** — this layer is pure data infrastructure. Model-based enrichment is a separate concern.
