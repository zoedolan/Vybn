#!/bin/bash
# vllm-exec.sh — wrapper called by vybn-vllm.service.
#
# Purpose: systemd ExecStart passes empty VYBN_VLLM_EXTRA_ARGS as a literal
# empty-string argument ("") which vLLM rejects as "unrecognized arguments".
# This wrapper builds the argument list in bash, where ${VAR:+$VAR} correctly
# produces no argument when VAR is empty or unset.
#
# All tunables come from the environment set by the service unit and
# EnvironmentFile=~/.config/vybn/vllm.env.

set -euo pipefail

CLUSTER="$HOME/spark-vllm-docker/launch-cluster.sh"
NODES="169.254.246.181,169.254.51.101"

CMD=(
  "$CLUSTER"
  -n "$NODES"
  exec
  env "VLLM_SERVER_DEV_MODE=${VLLM_SERVER_DEV_MODE:-0}"
  vllm serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8
  --port 8000 --host 0.0.0.0
  --gpu-memory-utilization "${VYBN_VLLM_GPU_MEMORY_UTILIZATION:-0.78}"
  --max-model-len 8192
  --max-num-seqs "${VYBN_VLLM_MAX_NUM_SEQS:-8}"
  --tensor-parallel-size 1 --pipeline-parallel-size 2
  --distributed-executor-backend ray
  --trust-remote-code
)

# Only append VYBN_VLLM_EXTRA_ARGS when it is non-empty.
# Word-splitting is intentional here: multiple flags can be space-separated.
if [[ -n "${VYBN_VLLM_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  CMD+=( $VYBN_VLLM_EXTRA_ARGS )
fi

exec "${CMD[@]}"
