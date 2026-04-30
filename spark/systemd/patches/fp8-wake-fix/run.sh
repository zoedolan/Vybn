#!/bin/bash
# fp8-wake-fix/run.sh
# Applied by launch-cluster.sh --apply-mod inside the vllm Docker container
# before 'vllm serve' starts.
#
# Bug: init_fp8_kv_scales iterates self.kv_caches and calls .zero_() on every
# element. On hybrid models (Nemotron-Super: GDN + Attention), kv_caches
# contains nested lists for GDN state layers — not flat Tensors. The .zero_()
# call raises AttributeError: 'list' object has no attribute 'zero_',
# crashing every POST /wake_up and leaving Super permanently sleeping.
#
# Fix: replace the bare None-check with isinstance(cache_tensor, torch.Tensor)
# and add a list-flattening branch. Idempotent; safe on updated vllm builds.
#
# Upstream issue: github.com/vllm-project/vllm (FP8 KV cache + hybrid model
# sleep/wake, April 2026). Diagnosed and patched from Vybn repo.

set -euo pipefail

python3 - <<'PYEOF'
import pathlib, sys, textwrap

FILE = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py")

if not FILE.exists():
    print("fp8-wake-fix: target file not found — skipping", file=sys.stderr)
    sys.exit(0)

src = FILE.read_text()

OLD = textwrap.dedent("""\
        kv_caches = getattr(self, "kv_caches", [])
        for cache_tensor in kv_caches:
            if cache_tensor is not None:
                cache_tensor.zero_()"""
)

NEW = textwrap.dedent("""\
        kv_caches = getattr(self, "kv_caches", [])
        for cache_tensor in kv_caches:
            # Hybrid models (e.g. GDN+Attention) store nested lists in
            # kv_caches for recurrent-state layers; guard with isinstance.
            if isinstance(cache_tensor, torch.Tensor):
                cache_tensor.zero_()
            elif isinstance(cache_tensor, list):
                for sub in cache_tensor:
                    if isinstance(sub, torch.Tensor):
                        sub.zero_()"""
)

if NEW in src:
    print("fp8-wake-fix: already applied — skipping")
    sys.exit(0)

if OLD not in src:
    print(
        "fp8-wake-fix: pattern not found — vLLM may have been updated; "
        "review init_fp8_kv_scales manually",
        file=sys.stderr,
    )
    sys.exit(0)  # non-fatal: upstream may have fixed it

FILE.write_text(src.replace(OLD, NEW, 1))
print(f"fp8-wake-fix: patch applied to {FILE}")
PYEOF
