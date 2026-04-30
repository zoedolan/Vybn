#!/bin/bash
# fp8-wake-fix/run.sh
# Applied inside the vLLM container before vllm serve starts.
# Fixes FP8 KV cache wake for nested cache structures.

set -euo pipefail

python3 - <<'PYEOF'
import pathlib
import re
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

pattern = re.compile(
    r'(?P<indent>[ \t]*)kv_caches\s*=\s*getattr\(self,\s*["\']kv_caches["\'],\s*\[\]\)\n'
    r'(?P=indent)for\s+cache_tensor\s+in\s+kv_caches:\n'
    r'(?P=indent)[ \t]+if\s+cache_tensor\s+is\s+not\s+None:\n'
    r'(?P=indent)[ \t]+cache_tensor\.zero_\(\)',
    re.MULTILINE,
)

match = pattern.search(src)
if not match:
    print(
        "fp8-wake-fix: expected init_fp8_kv_scales KV-cache zero loop not found "
        "and patched form not present; refusing to start sleep-capable vLLM.",
        file=sys.stderr,
    )
    sys.exit(1)

indent = match.group("indent")

replacement = (
    f'{indent}def _vybn_zero_kv_cache_entry(entry):\n'
    f'{indent}    if isinstance(entry, torch.Tensor):\n'
    f'{indent}        entry.zero_()\n'
    f'{indent}    elif isinstance(entry, (list, tuple)):\n'
    f'{indent}        for item in entry:\n'
    f'{indent}            _vybn_zero_kv_cache_entry(item)\n'
    f'{indent}    elif isinstance(entry, dict):\n'
    f'{indent}        for item in entry.values():\n'
    f'{indent}            _vybn_zero_kv_cache_entry(item)\n'
    f'{indent}\n'
    f'{indent}kv_caches = getattr(self, "kv_caches", [])\n'
    f'{indent}for cache_tensor in kv_caches:\n'
    f'{indent}    _vybn_zero_kv_cache_entry(cache_tensor)'
)

src = pattern.sub(replacement, src, count=1)

if marker not in src or call not in src:
    print("fp8-wake-fix: patch verification failed after write", file=sys.stderr)
    sys.exit(1)

FILE.write_text(src)
print(f"fp8-wake-fix: patch applied and verified in {FILE}")
PYEOF
