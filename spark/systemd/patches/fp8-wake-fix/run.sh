#!/bin/bash
# fp8-wake-fix/run.sh
# Applied inside the vLLM container before vllm serve starts.
# Fixes FP8 KV cache wake for nested cache structures.

set -euo pipefail

python3 - <<'PYEOF'
import pathlib
import sys

FILE = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py")

if not FILE.exists():
    print(f"fp8-wake-fix: target file not found: {FILE}", file=sys.stderr)
    sys.exit(1)

src = FILE.read_text()
marker = "def _vybn_zero_kv_cache_entry"
call = "_vybn_zero_kv_cache_entry(cache_tensor)"

if marker in src and call in src:
    print("fp8-wake-fix: already applied and verified")
    sys.exit(0)

lines = src.splitlines()
def_i = next((i for i, line in enumerate(lines) if line.startswith("    def init_fp8_kv_scales(")), None)
if def_i is None:
    print("fp8-wake-fix: init_fp8_kv_scales not found", file=sys.stderr)
    sys.exit(1)

next_def_i = next((i for i in range(def_i + 1, len(lines)) if lines[i].startswith("    def ")), len(lines))
kv_i = next((i for i in range(def_i, next_def_i) if 'kv_caches = getattr(self, "kv_caches", [])' in lines[i]), None)
if kv_i is None:
    print("fp8-wake-fix: kv_caches assignment not found inside init_fp8_kv_scales", file=sys.stderr)
    sys.exit(1)

zero_i = next((i for i in range(kv_i, next_def_i) if "cache_tensor.zero_()" in lines[i]), None)
if zero_i is None:
    print("fp8-wake-fix: cache_tensor.zero_() not found inside init_fp8_kv_scales", file=sys.stderr)
    sys.exit(1)

indent = lines[kv_i].split("k", 1)[0]

replacement = [
    f'{indent}def _vybn_zero_kv_cache_entry(entry):',
    f'{indent}    if isinstance(entry, torch.Tensor):',
    f'{indent}        entry.zero_()',
    f'{indent}    elif isinstance(entry, (list, tuple)):',
    f'{indent}        for item in entry:',
    f'{indent}            _vybn_zero_kv_cache_entry(item)',
    f'{indent}    elif isinstance(entry, dict):',
    f'{indent}        for item in entry.values():',
    f'{indent}            _vybn_zero_kv_cache_entry(item)',
    f'{indent}',
    f'{indent}kv_caches = getattr(self, "kv_caches", [])',
    f'{indent}for cache_tensor in kv_caches:',
    f'{indent}    _vybn_zero_kv_cache_entry(cache_tensor)',
]

lines[kv_i:zero_i + 1] = replacement
new_src = "\n".join(lines) + "\n"

if marker not in new_src or call not in new_src:
    print("fp8-wake-fix: patch verification failed after rewrite", file=sys.stderr)
    sys.exit(1)

FILE.write_text(new_src)
print(f"fp8-wake-fix: patch applied and verified in {FILE}")
PYEOF
