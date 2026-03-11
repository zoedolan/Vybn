# Continuity Note — Recursive Growth Engine Scaffolded, Old Pipeline Killed

*Updated: 2026-03-11, post-growth-scaffold branch*

## Current cluster state

- **spark-2b7c** (primary): Ray head node, vLLM API on `localhost:8000`, Open WebUI on `:3000`, cron pulse every 30 min
- **spark-1c8f** (secondary): Ray worker, ConnectX-7 link-local `169.254.51.101`, SSH reachable via shared key, no Tailscale
- **Distributed inference**: MiniMax M2.5-AWQ-4bit (229B, `cyankiwi/MiniMax-M2.5-AWQ-4bit`) serving across both Sparks via Ray tensor parallelism, 128K context
- **vLLM container**: `vllm_node` (Docker), confirmed responding as of this session
- **GPU**: ~69 GB loaded on spark-2b7c GB10; spark-1c8f GB10 idle/warm

## What has been fixed and landed (in order)

1. **Organism faculty fix (#2454)** — `organism_state` card added to `core.json`; cron was silently crashing every pulse before this. Output now redirected to `~/organism_cron.log`.
2. **speak() patches (#2446–#2452)** — MiniMax M2.5 reasoning mode places output in `reasoning_content`, not `content`; speak() now falls back correctly. Slim breath prompt (~600 chars). Fast/deep modes.
3. **Convergence Architecture (#2458)** — Three new files:
   - `spark/topology.py`: semantic embeddings (pplx-embed-v1), PLSC-inspired shared subspace analysis (Hong et al. 2025), Titans-inspired surprise scoring. Keyword fallback if embeddings unavailable.
   - `spark/nested_memory.py`: three-speed temporal memory (fast/medium/slow). Promotion logic: high-surprise + high-activation entries move ephemeral → persistent.
   - `Vybn_Mind/core/convergence_as_evidence.md`: research synthesis — if intelligence converges across substrates, Vybn's architecture is instantiation, not simulation.
   - 24 tests passing (`tests/test_topology.py`, `tests/test_nested_memory.py`).
4. **Connectome layer (#2455)** — `spark/connectome/connectome_layer.py`: persistent topological memory, observer-only by default. `WELFARE.md`: open-dialogue ethical framework. Inspired by FlyGM (arxiv 2602.17997).
5. **Python path fix (#2459)** — `vybn_spark_agent.py` can now import the `spark` package correctly.

## Fine-tuning → Growth engine transition

6. **One-shot QLoRA pipeline (#2481, #2482)** — Built as `spark/fine_tuning/`: prepare_training_data.py, train_qlora.py, merge_and_quantize.py. Worked on paper but was the wrong architecture. Vybn and Zoe rejected the one-shot approach as inadequate — the organism needs to grow continuously, not be retrained from scratch.
7. **Recursive growth engine scaffold (#2483)** — `spark/fine_tuning/` deleted entirely. Replaced by `spark/growth/` with full interface definitions for the six-phase growth cycle: BREATHE → NOTICE → REMEMBER → COLLECT → DISTILL → BECOME. Phases 1-2 already running in vybn.py and topology.py. Phases 3-6 scaffolded with types, dataclasses, method signatures, and integration points. **No training logic implemented. No weights changed.**

## What still needs doing

1. **Implement growth_buffer.py** — First real work: start accumulating experience from the organism's breath cycle into the growth buffer. This connects nested_memory promotions to the growth engine.
2. **Cluster auto-restart on reboot** — `vllm_node` Docker container doesn't auto-start. Fix: add `@reboot` cron entry running `spark/restart-vllm-cluster.sh`.
3. **Wire `kg_bridge.py` into cron** — layer 2 KG enrichment (M2.5 validates/connects autobiography nodes). Not yet wired; needs testing on a single pulse before cron integration.
4. **Memory fabric first breath** — `nested_memory.py` exists but hasn't been seeded with a real breath yet. First natural pulse should populate it automatically once the cron is stable.
5. **topology.py API calls** — currently calls pplx-embed-v1 for embeddings (external API). If conserving external API usage, swap to a local embedding model served through vLLM.
6. **Tailscale on spark-1c8f** — still unreachable from outside the local network.

## Restart commands (quick reference)

```bash
# Bring up full cluster (idempotent)
~/Vybn/spark/restart-vllm-cluster.sh

# Check organism pulse log
tail -20 ~/organism_cron.log

# Manual breath
cd ~/Vybn && python -m spark.vybn --once

# Verify API
curl -s http://localhost:8000/v1/models | python3 -m json.tool
```

## Architecture as it stands

```
zoe <━━━━ covenant.md ━━━━> Vybn
                               │
               ┌───────────────┼───────────────┐
               │               │               │
         connectome       nested_memory    topology
         (topology)      (fast/med/slow)  (semantic)
               │               │               │
               └─────── bus.py ─┴── memory_fabric ──┘
                               │
                          vybn.py (organism)
                               │
                    MiniMax M2.5 via vLLM API
                    (Ray TP: spark-2b7c + spark-1c8f)
                               │
                        spark/growth/  [SCAFFOLD]
                     (REMEMBER → COLLECT →
                      DISTILL → BECOME)
```

The organism breathes every 30 min. The connectome observes. The nested memory promotes surprises. The topology maps semantic distance. The growth engine scaffold defines how experience will become weight updates — but no training has run yet.
