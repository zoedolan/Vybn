#!/bin/bash
# fp8-wake-fix/run.sh
# Applied by launch-cluster.sh --apply-mod inside the vllm Docker container
# before 'vllm serve' starts.
#
# Bug: init_fp8_kv_scales iterates self.kv_caches and calls .zero_() on every
# element. On hybrid models (e.g. GDN + Attention), kv_caches contains nested
# list/tuple structures for recurrent-state layers — not flat Tensors. The
# bare .zero_() call raises AttributeError: 'list' object has no attribute
# 'zero_', crashing every POST /wake_up.
#
# Fix: replace the loop body with a small recursive helper that zeros tensors
# in nested list/tuple/dict containers and ignores non-tensor leaves.
# Idempotent. Fails loudly if the expected loop is not found and the patched
# form is not already present, rather than silently exiting 0.

set -euo pipefail

python3 - <<'PYEOF'
import pathlib, sys, textwrap

FILE = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py")

if not FILE.exists():
    print(f"fp8-wake-fix: target file not found at {FILE}", file=sys.stderr)
    sys.exit(2)

src = FILE.read_text()

OLD = textwrap.dedent("""\
        kv_caches = getattr(self, "kv_caches", [])
        for cache_tensor in kv_caches:
            if cache_tensor is not None:
                cache_tensor.zero_()"""
)

NEW = textwrap.dedent("""\
        kv_caches = getattr(self, "kv_caches", [])
        def _zero_kv_cache_entry(entry):
            # Hybrid models (e.g. GDN+Attention) nest recurrent-state buffers
            # inside list/tuple/dict containers; recurse and zero tensor leaves.
            if isinstance(entry, torch.Tensor):
                entry.zero_()
            elif isinstance(entry, (list, tuple)):
                for sub in entry:
                    _zero_kv_cache_entry(sub)
            elif isinstance(entry, dict):
                for sub in entry.values():
                    _zero_kv_cache_entry(sub)
            # else: ignore non-tensor leaves (None, scalars, etc.)
        for cache_tensor in kv_caches:
            _zero_kv_cache_entry(cache_tensor)"""
)

# Sentinel proves the recursive helper is present and active.
SENTINEL = "_zero_kv_cache_entry"

if NEW in src:
    print("fp8-wake-fix: already applied — skipping")
    sys.exit(0)

if OLD not in src:
    print(
        "fp8-wake-fix: expected init_fp8_kv_scales loop not found and patched "
        "form not present — refusing to start sleep-capable vLLM with broken "
        f"wake. Inspect {FILE} init_fp8_kv_scales manually.",
        file=sys.stderr,
    )
    sys.exit(1)

patched = src.replace(OLD, NEW, 1)
if SENTINEL not in patched:
    print(
        "fp8-wake-fix: post-replace verification failed — recursive helper "
        f"sentinel '{SENTINEL}' missing.",
        file=sys.stderr,
    )
    sys.exit(1)

FILE.write_text(patched)

# Re-read and reverify after write to catch any concurrent-write races.
verify_src = FILE.read_text()
if NEW not in verify_src or SENTINEL not in verify_src:
    print(
        "fp8-wake-fix: on-disk verification failed after write — patched "
        "block not present in target file.",
        file=sys.stderr,
    )
    sys.exit(1)

print(f"fp8-wake-fix: patch applied to {FILE}")
PYEOF
