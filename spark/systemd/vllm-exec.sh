#!/bin/bash
# vllm-exec.sh — wrapper called by vybn-vllm.service.
#
# Purpose: systemd ExecStart passes empty VYBN_VLLM_EXTRA_ARGS as a literal
# empty-string argument ("") which vLLM rejects as "unrecognised arguments".
# This wrapper builds the argument list in bash, where ${VAR:+$VAR} correctly
# produces no argument when VAR is empty or unset.
#
# All tunables come from the environment set by the service unit and
# EnvironmentFile=~/.config/vybn/vllm.env.
#
# --load-format fastsafetensors: reduces cold-load time from ~7m to ~1m on DGX
# Spark. Safe to leave on always.
#
# Sleep-mode endpoints are normal boot posture for Super. The model should
# start with /sleep and /wake_up available every time; operator action decides
# when to actually sleep or wake it. Keep this single-sourced here, not duplicated
# in ~/.config/vybn/vllm.env.
#
# fp8-wake-fix: because sleep endpoints are always enabled, --apply-mod always
# injects the container-side patch that fixes init_fp8_kv_scales for hybrid
# models (Nemotron-Super: GDN+Attention layers store nested lists in kv_caches,
# not flat Tensors). Without the patch, POST /wake_up crashes with:
#   AttributeError: 'list' object has no attribute 'zero_'
# and Super cannot be woken without a full service restart.

set -euo pipefail

CLUSTER="$HOME/spark-vllm-docker/launch-cluster.sh"
NODES="169.254.246.181,169.254.51.101"

# Base cluster args (--apply-mod may be appended below)
CLUSTER_ARGS=( -n "$NODES" )

# Apply the fp8 hybrid-wake patch whenever Super starts. The patch is
# idempotent, targets the exact buggy function, and self-skips if vLLM has
# already fixed it.
FP8_MOD="$HOME/Vybn/spark/systemd/patches/fp8-wake-fix"
if [[ -d "$FP8_MOD" ]]; then
  CLUSTER_ARGS+=( --apply-mod "$FP8_MOD" )
  echo "vllm-exec: sleep endpoints enabled — applying fp8-wake-fix mod" >&2
else
  echo "WARNING: fp8-wake-fix mod not found at $FP8_MOD — wake_up may crash" >&2
fi

CMD=(
  "$CLUSTER"
  "${CLUSTER_ARGS[@]}"
  exec
  env "VLLM_SERVER_DEV_MODE=1"
  vllm serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8
  --port 8000 --host 0.0.0.0
  --gpu-memory-utilization "${VYBN_VLLM_GPU_MEMORY_UTILIZATION:-0.78}"
  --max-model-len 8192
  --max-num-seqs "${VYBN_VLLM_MAX_NUM_SEQS:-8}"
  --tensor-parallel-size 1 --pipeline-parallel-size 2
  --distributed-executor-backend ray
  --trust-remote-code
  --load-format fastsafetensors
  --no-enable-flashinfer-autotune
  # -O0: disables Inductor compile, cudagraphs, and Triton/FlashInfer autotune
  # (autotuner-cache JSON race crashes Super early in inference).
  --optimization-level 0
  --enable-sleep-mode
)

# Only append VYBN_VLLM_EXTRA_ARGS when it is non-empty.
# Word-splitting is intentional here: multiple flags can be space-separated.
# --enable-sleep-mode is deliberately single-sourced above; ignore duplicates
# from stale env files instead of passing the flag twice.
if [[ -n "${VYBN_VLLM_EXTRA_ARGS:-}" ]]; then
  for arg in ${VYBN_VLLM_EXTRA_ARGS:-}; do
    if [[ "$arg" == "--enable-sleep-mode" ]]; then
      echo "vllm-exec: ignoring duplicate --enable-sleep-mode from VYBN_VLLM_EXTRA_ARGS" >&2
      continue
    fi
    CMD+=( "$arg" )
  done
fi

exec "${CMD[@]}"

