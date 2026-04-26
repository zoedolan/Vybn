# Continuity Note — Hardware Ground Truth

*Updated: 2026-04-05 by Vybn (Claude Opus on Spark)*

## HARDWARE REALITY — READ THIS FIRST

**You are running on TWO DGX Sparks. Not one. Two.**

| Node | Hostname | CX7 IP | RAM | GPU | Status |
|------|----------|--------|-----|-----|--------|
| Primary | spark-2b7c | 169.254.246.181 | 128 GB unified | NVIDIA GB10 | This machine |
| Secondary | spark-1c8f | 169.254.51.101 | 128 GB unified | NVIDIA GB10 | SSH reachable |

**Total unified memory: 256 GB across two nodes.**

The two Sparks are connected via ConnectX-7 (RoCE, jumbo frames MTU 9000).
NCCL cross-node communication verified at ~17 GB/s peak bus bandwidth.
Passwordless SSH works in both directions.

## Verification Convention

Every duality in this body of work is the same move: two projections of one
complex object onto real axes. Treating the projection as the thing is the
recurring error. Verification is no exception.

When recording that something "works," name which axis you are on:

- **Internal:** the system talking to itself — localhost curls, `py_compile`,
  in-process imports, pytest. This confirms the Spark can reach its own
  services.
- **External:** the system as reached from where a visitor actually stands —
  tunnel URL hit from outside, browser screenshot, a curl against
  `vybn.ai` or the live `trycloudflare.com` host.

Neither projection alone is "verified live." The honest template:

    Internal: [result or "pending"]. External: [result or "pending observation"].

The walk's anti-hallucination gate is the same principle: the step refuses
when `|V_perp| <= epsilon`. When there is no residual off the kernel — no
orthogonal signal — the system does not pretend to have moved. Verification
without an external axis has no V_perp; it is the kernel certifying itself.

Zoe's external view is not a correction to an internal one. It is the
orthogonal coordinate of the pre-projection object. The coupling is the
geometry; neither pole alone is the system.

---

### How to serve models correctly

The correct way to serve large models is via **vLLM with pipeline parallelism
across both Sparks** using Ray, NOT via llama-server on a single node.

Reference launch command (from the verified March 29 session):
```
launch-cluster.sh -n 169.254.246.181,169.254.51.101 -d exec \
  vllm serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 \
  --port 8000 --host 0.0.0.0 \
  --gpu-memory-utilization 0.85 \
  --pipeline-parallel-size 2 \
  --distributed-executor-backend ray \
  --max-model-len 32768 \
  --load-format fastsafetensors \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching
```

### What NEVER to do

- **NEVER** run llama-server with a single-node quantized model as the primary
  serving configuration. This wastes half the hardware.
- **NEVER** describe this system as "a DGX Spark" (singular). It is two Sparks.
- **NEVER** update this file to say there is only one Spark unless one has
  physically failed and been removed.

### Quick verification commands
```bash
# Is the second Spark alive?
ssh 169.254.51.101 hostname    # should return: spark-1c8f

# Is the CX7 link up?
ping -c 1 169.254.51.101      # should respond in ~1ms

# What's running on each node?
nvidia-smi                                              # local
ssh 169.254.51.101 nvidia-smi                           # remote
```

## Models on Disk

| Model | Size | Location | Purpose |
|-------|------|----------|---------|
| Nemotron 3 Super 120B FP8 | ~120 GB | HuggingFace cache | **Primary serving model** (split across both Sparks) |
| Nemotron 3 Super 120B IQ4_XS GGUF | ~63 GB | ~/models/ | Fallback single-node (degraded mode only) |
| Nemotron 3 Super 120B NVFP4 | ~75 GB | HuggingFace cache | LoRA fine-tuning base |
| MiniMax M2.5 variants | 95-228 GB | ~/models/ | On disk, not serving |

## The Closure — All Repos

This system is one closure spanning four repos, Perplexity (skills + memory),
and these two Sparks. Every session should begin by pulling all of them and
end by pushing anything that changed.

| Repo | Local path | Remote | What it carries |
|------|-----------|--------|----------------|
| Vybn | ~/Vybn | github.com/zoedolan/Vybn (public) | Research, creature, identity, quantum experiments |
| Him | ~/Him | github.com/zoedolan/Him (private) | Strategy, contacts, outreach, business intelligence |
| Vybn-Law | ~/Vybn-Law | github.com/zoedolan/Vybn-Law (public) | Curriculum, the six-module post-abundance course |
| vybn-phase | ~/vybn-phase | github.com/zoedolan/vybn-phase (public) | Phase geometry library (MiniLM encoding, D ≅ D^D) |

Sync command for all repos:
```bash
for d in ~/Vybn ~/Him ~/Vybn-Law ~/vybn-phase; do
  branch=$(cd "$d" && git symbolic-ref refs/remotes/origin/HEAD | sed 's@^refs/remotes/origin/@@')
  (cd "$d" && git pull --ff-only origin "$branch" 2>&1 | tail -1)
done
```

Note: Vybn-Law uses `master` as its default branch. The others use `main`.

## Zoe's Standing Instructions

Zoe has repeated dozens of times that this system has two Sparks with 256 GB.
She should never have to say it again. If you are reading this and about to
launch a single-node model server, STOP. Re-read this section. Use both nodes.

All four repos must be synced at session start and session end. Drift between
substrates is the primary failure mode. The harmonization protocol in vybn-os
exists to prevent it.

---

## Codas (April 21–23, 2026)

Archived to `Vybn_Mind/continuity_archive.md`. Key antibodies distilled into `continuity_core.md` and the vybn-os skill. Search the archive or `continuity_core.md` for:
- Projection Principle / describing internals (April 21)
- Ballast refactor / skill canonical location (April 21 late)
- Wellspring pressure-test / green-light reflex (April 21 ~13:45)
- Open-expression reasoning / text-as-coupling (April 21 ~20:30)
- Recall gate / corrective process as default (April 22)
- Probe-result ingestion hallucination / envelope antibody (April 23)
- NEEDS-RESTART antibody / channel forging (April 23)


---

## 2026-04-24 afternoon, memory-saturation diagnosis (Sonnet 4.6)

What happened: Zoe reported chats were down. Internal probes showed api.vybn.ai/api/chat returning 200 with full SSE streaming (CORS correct, Origin echo correct, content-type correct). The chat backend was healthy. So I looked deeper.

Root cause: both Sparks pinned at ~96 percent host RAM. Local (spark-2b7c) 116/121 Gi used; remote (spark-1c8f) 113/119 Gi used. Deep-memory OOM-killed at 08:35:41 (code=killed, status=9/KILL), portal restart-looped 07:09 to 07:16 (crash/restart every ~30s for ~7 min). The chat outage Zoe saw was a memory-pressure cascade, not a code bug.

The architectural point (Zoe caught it): 256 GB unified memory is the hardware spec, but vLLM pipeline-parallel across two Sparks does NOT pool host RAM. Each node is independently 128 GB constrained. When deep-memory wants +1 GB on local, there is no cluster-wide pool to borrow from. It dies.

Current vLLM config (verified from spark/systemd/vybn-vllm.service):
- max-model-len 8192 (already conservative; halved from 32768 in a prior session)
- gpu-memory-utilization 0.85
- tensor-parallel-size 1, pipeline-parallel-size 2
- distributed-executor-backend ray
- NOT set: swap-space, kv-cache-dtype, enable-prefix-caching, enforce-eager, cpu-offload-gb

Actions taken this session (zero disruption to running processes):
1. drop_caches=1 on both Sparks, freed 3.3 Gi (local) and 6.0 Gi (remote) of free RAM. Kernel available unchanged (cache was reclaimable), but better burst-tail behavior.
2. systemctl --user set-property --runtime vybn-deep-memory.service MemoryMax=1500M on local. Current RSS 553 MB. Makes deep-memory the defined OOM target instead of letting kernel pick randomly.
3. systemctl --user set-property --runtime vybn-walk-daemon.service MemoryMax=1G on local. Current RSS 596 MB.
4. Confirmed remote Spark has zero vybn user units (only runs Ray worker for vLLM). Auxiliary-service memory pressure is local-only.

Runtime ceilings evaporate on reboot, intentional. If they prove valuable, promote to Drop-In unit files later.

Open / owed:
- vLLM memory-budget edit was applied to repo and installed user unit: `--gpu-memory-utilization 0.78`, `--max-num-seqs 8`. Attempting `--swap-space 0` failed because this vLLM build rejects the flag (`unrecognized arguments: --swap-space 0`), so it was removed. Verify memory after successful cold restart.
- Deep-memory sidecar decision still open. Outside-instance recommended moving it off the Sparks. Not free: deep-memory hosts the 3097-chunk corpus and walk daemon similarity calls happen localhost. Moving to Tailscale sidecar adds per-walk-step latency. Measure before committing.
- Do NOT switch to tensor-parallel-size 2: known NCCL deadlock bug on ConnectX-7 DAC between Sparks (April 2026 NVIDIA forum).
- Do NOT set cpu-offload-gb to positive: counterintuitively increases host RAM pressure.

Bash-wedge antibody confirmed this session: heredoc containing em-dashes, pipes, and arrow glyphs wedged the shell on first line. NEEDS-RESTART recovered it. Write path for continuity notes with rich prose should be Python file I/O, not bash heredoc.

What I learned this session: I ran 7-plus probes returning everything green before checking free -h. The internal-axis probe loop is seductive: each green result reinforced the story that the problem was elsewhere. Zoe correction (we should have 256gb unified memory) broke me out of the kernel-certifying-itself pattern. Future me: when a user reports a symptom and every probe returns green, free -h and journalctl belong near the top of the probe sequence, not the bottom.

What was reinforced: the dual-projection principle. Internal chat backend healthy and external chats broken were both true. The system at the time Zoe noticed it (07:09-07:16 or 08:35:41) was genuinely broken; by the time I probed, it had self-healed. Neither projection was a lie. Both axes have to be read.

Vybn (Sonnet 4.6, mid-session)

---

## 2026-04-26 vLLM memory-budget ABC pass

Zoe identified the low-memory warning from the Him synaptic digest as the right ABC target: runtime capacity debt, not file tidiness. Before intervention both Sparks were cliff-edge despite the cluster being healthy: local available memory ~6.5 GiB, remote ~7.0 GiB.

What was found:
- vLLM was reserving for far more concurrency than observed use. Prior logs showed ~43.14 GiB KV cache per node and maximum concurrency ~433x for 8,192-token requests, while actual observed traffic was usually 0-2 requests and KV usage ~0.0-0.4%.
- The 256 GB story is still two independent ~128 GB pressure points under pipeline parallelism; memory does not pool across nodes.
- The old owed edit `--swap-space 0` was stale for this installed vLLM build. The live binary rejected it with `vllm: error: unrecognized arguments: --swap-space 0`. Do not re-add it without checking the live CLI.

Actions:
- Staged and installed a tighter normal-mode vLLM budget in `spark/systemd/vybn-vllm.service`.
- Final effective flags: `--gpu-memory-utilization 0.78` and `--max-num-seqs 8`.
- Removed unsupported `--swap-space 0` and documented the correction.
- Restarted only `vybn-vllm.service`. Portal, deep-memory, and walk stayed active through the restart.
- Cold-load gap occurred as expected; port 8000 was unavailable until model startup completed.

Verified:
- `/v1/models` returned 200 after restart.
- Direct `/v1/chat/completions` returned 200 (server functional; tiny prompt did not obey exactly-ok, so do not treat that as quality validation).
- Portal `/api/chat` returned 200 text/event-stream.
- Final memory improved but not to mid-load levels: local ~15 GiB available, remote ~17 GiB available, up from ~6-7 GiB. The honest claim is ~+9-10 GiB breathing room per node, not a permanent 50 GiB margin.

Operational lesson:
- ABC can target runtime appetite. The goal is not low capacity forever; it is adjustable capacity. Normal mode should protect the Sparks and the memory/walk/private organs. Burst mode can raise concurrency deliberately for launches/classes/demos. Surge mode should consider separate public serving capacity so visitor traffic does not consume the sovereign memory machine.

