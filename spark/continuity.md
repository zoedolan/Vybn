# Continuity — 2026-04-30 05:03 PDT

## Where we are

Super is up and healthy (nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8, port 8000).
The fp8-wake-fix patch and updated vllm-exec.sh are committed to origin/main via PR #2942 (da53ff7e).
The repo is current on this Spark.

## What happened this session

The Omni-window experiment ran. Sleep succeeded. The script stalled on a
non-interactive `sudo` cache-flush step, then the wake API crashed with:

    AttributeError: 'list' object has no attribute 'zero_'
    at vllm/v1/worker/gpu_model_runner.py — init_fp8_kv_scales()

Root cause: Nemotron-Super is a hybrid GDN+Attention model. Its kv_caches
contains nested lists for GDN recurrent-state layers. The upstream init_fp8_kv_scales
called .zero_() on every element without an isinstance check. This is now patched.

## The fix (in repo)

spark/systemd/patches/fp8-wake-fix/run.sh
  — injected via launch-cluster.sh --apply-mod whenever --enable-sleep-mode is armed
  — patches the exact function inside the running container before vllm serve starts
  — idempotent; self-skips if upstream has fixed it

spark/systemd/vllm-exec.sh
  — auto-appends --apply-mod fp8-wake-fix when VYBN_VLLM_EXTRA_ARGS contains --enable-sleep-mode

## How to resume the experiment

The path is clean now. Steps:

1. Super now boots with sleep endpoints enabled by default from spark/systemd/vllm-exec.sh:
   VLLM_SERVER_DEV_MODE=1, --enable-sleep-mode, and the fp8 wake patch are single-sourced there.
   Do not add duplicate --enable-sleep-mode in ~/.config/vybn/vllm.env; stale duplicates are ignored.

2. Restart Super only when you actually need a fresh boot:
   systemctl --user restart vybn-vllm.service
   # Cold load can take time. Port 8000 not listening during load is not failure.
   # Wait for logs/readiness instead of clearing caches or restarting again.

3. Test the sleep endpoints:
   curl http://127.0.0.1:8000/is_sleeping
   curl -X POST http://127.0.0.1:8000/sleep?level=2
   curl http://127.0.0.1:8000/is_sleeping  # true

4. Flush unified-memory buffer cache only from an interactive shell with sudo:
   sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'

5. Start Omni on peer (169.254.51.101:8002) while Super sleeps.
   Capture the parallax. Do the thing.

6. Wake Super:
   curl -X POST http://127.0.0.1:8000/wake_up
   # The patch means this should no longer crash on the first clean cycle. Smoke-test with a completion.

## Remaining known risk

vLLM issue #36651: double cuMemRelease and stale error codes on the
second/third sleep-wake cycle. First cycle should be clean. Test two
full cycles before treating sleep mode as durable infrastructure.

## Next step

Resume the experiment. The aperture is worth opening.

