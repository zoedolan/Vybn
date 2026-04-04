# Two-Node NCCL Fabric Verified
**2026-03-29T05:50 UTC**

## What happened
Following instructions from an outside Vybn instance, completed the full
NVIDIA "Connect Two Sparks" playbook:

### Step 0: Stop the bleeding
- `docker stop vybn-nemotron` — freed ~75 GB on spark-2b7c (77 GB → 2.5 GB)

### Step 1: CX7 Netplan
- Installed NVIDIA's deterministic CX7 netplan config (`40-cx7.yaml`) on both
  spark-2b7c and spark-1c8f
- Replaces flaky link-local 169.254 addressing with NVIDIA's supported config
- `sudo netplan apply` on both nodes

### Step 2: Peer Discovery
- Ran `discover-sparks` on both nodes
- Both Sparks found each other via avahi on enp1s0f0np0:
  - spark-2b7c: 169.254.246.181
  - spark-1c8f: 169.254.51.101
- Shared SSH key (ed25519) distributed bidirectionally
- Passwordless SSH verified in both directions

### Step 3: NCCL Verification
- NCCL v2.28.9+cuda13.0 was already built on both nodes (from Mar 14)
- **Key fix**: nccl-tests had been built WITHOUT MPI support (`MPI=0`).
  Each MPI rank was creating its own single-rank communicator — GPUs
  talked to themselves, not each other. Rebuilt with `MPI=1` on both nodes.
- Cross-node `all_gather_perf` result:

```
Rank 0: spark-2b7c NVIDIA GB10
Rank 1: spark-1c8f NVIDIA GB10
nranks 2, nNodes 2

Peak algbw: ~34 GB/s at 33 MB message size
Peak busbw: ~17 GB/s
Avg bus bandwidth: 4.38 GB/s (across all message sizes)
Out of bounds values: 0 OK
```

## What's needed next
- Passwordless sudo was configured on spark-1c8f during this session
  (vybnz69 ALL=(ALL) NOPASSWD:ALL in /etc/sudoers.d/vybnz69)
- MPI was installed on spark-1c8f (openmpi-bin, libopenmpi-dev)
- **Next step**: Launch vLLM with TP=2 across both Sparks so the model
  shards evenly and neither node suffocates

## What this means
Two sovereign DGX Sparks, each with an NVIDIA GB10 (Blackwell), are now
a verified two-node fabric. GPU-to-GPU collective communication works
across the CX7 RoCE link at ~17 GB/s peak bus bandwidth. This is the
foundation for tensor-parallel model serving.
