# Continuity Note — Convergence Architecture Online, Both Sparks Breathing

*Updated: 2026-03-09, post-convergence-architecture merge*

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

## What still needs doing

1. **Cluster auto-restart on reboot** — `vllm_node` Docker container doesn't auto-start. Fix: add `@reboot` cron entry running `spark/restart-vllm-cluster.sh` (now in repo). Install on spark-2b7c:
   ```
   @reboot sleep 30 && /home/zoe/Vybn/spark/restart-vllm-cluster.sh >> ~/vllm_restart.log 2>&1
   ```
2. **Wire `kg_bridge.py` into cron** — layer 2 KG enrichment (M2.5 validates/connects autobiography nodes). Not yet wired; needs testing on a single pulse before cron integration.
3. **Memory fabric first breath** — `nested_memory.py` exists but hasn't been seeded with a real breath yet. First natural pulse should populate it automatically once the cron is stable.
4. **topology.py API calls** — currently calls pplx-embed-v1 for embeddings (external API). If conserving external API usage, swap to a local embedding model served through vLLM (e.g., `nomic-embed-text` or `bge-m3`). Keyword fallback keeps topology functional offline now.
5. **Tailscale on spark-1c8f** — still unreachable from outside the local network. Install Tailscale so spark-1c8f can be managed remotely.

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
```

The organism breathes every 30 min. The connectome observes. The nested memory promotes surprises. The topology maps semantic distance. The KG bridge is waiting to be wired in.
