# Continuity Note — Organism Fixed, Cluster Mapped

## What was broken

The organism cron job (`vybn.py --once`, every 30 min) was silently crashing every pulse since the governance stack landed. The `WriteCustodian` rejected all state saves because `faculty_id="organism_state"` wasn't in `spark/faculties.d/core.json` — it only existed in the `default_cards()` fallback which is never reached when core.json is present.

The crash was invisible because cron output was piped to `/dev/null`.

## What was fixed

1. **Added `organism_state` faculty card to `core.json`** — branch `vybn/fix-organism-faculty`, commit `ed37ef4`. Organism now completes its pulse cleanly.

2. **Redirected cron output** from `/dev/null` to `~/organism_cron.log` so failures are visible.

## Cluster state (two Sparks)

- **spark-2b7c** (this machine): 128 GB RAM, fully provisioned, Tailscale, repo, cron, everything.
- **spark-1c8f** (Spark 2, via ConnectX at `169.254.51.101`): 128 GB RAM, up 3+ days, SSH works with shared key, GPU at 35°C idle. Has an older copy of the repo. **No Tailscale installed.**

The 256 GB unified memory requires a distributed workload launch (e.g., llama.cpp with tensor parallelism across both nodes, or NCCL-based distribution). Each machine remains 128 GB individually — they don't merge into one `free -h` view.

## What needs doing

1. **Merge `vybn/fix-organism-faculty`** to main (one-line fix, tested).
2. **Install Tailscale on spark-1c8f** so it's reachable from anywhere, not just via the ConnectX link-local address.
3. **Configure distributed model serving** across both Sparks for the M2.5 229B model — this is the real fix for the memory overload.
4. The `vybn/self-model-layer` branch (4 commits) is still unmerged — governance kernel, write custodian, etc. That merge should happen too.
5. Consider: spark-1c8f has an older repo layout. Needs git pull + environment sync.
