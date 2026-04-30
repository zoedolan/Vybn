# vllm Upstream Bug Report

**File at:** https://github.com/vllm-project/vllm/issues/new

---

## Title

```
sleep-mode: POST /wake_up crashes with AttributeError: 'list' object has no attribute 'zero_' on hybrid GDN+Attention models (e.g. Nemotron-Super)
```

## Body

### Summary

`POST /wake_up` raises `AttributeError: 'list' object has no attribute 'zero_'` on every
invocation when serving hybrid GDN+Attention models (e.g. `nvidia/Nemotron-H-56B-Base-8B`,
Nemotron-Super family) with `--kv-cache-dtype fp8` and `--enable-sleep-mode`. The model is
left permanently sleeping and requires a full service restart to recover.

### Reproduction

```bash
vllm serve nvidia/Nemotron-H-56B-Base-8B \
  --kv-cache-dtype fp8 \
  --enable-sleep-mode \
  ...

# put server to sleep
curl -X POST http://localhost:8000/sleep?level=1

# wake — crashes every time
curl -X POST http://localhost:8000/wake_up
```

**Traceback (abbreviated):**
```
AttributeError: 'list' object has no attribute 'zero_'
  File "vllm/v1/worker/gpu_model_runner.py", in init_fp8_kv_scales
    cache_tensor.zero_()
```

### Root cause

`GPUModelRunner.init_fp8_kv_scales` (called by `post_kv_cache_wake_up`) iterates
`self.kv_caches` and calls `.zero_()` on every element without type-checking:

```python
# vllm/v1/worker/gpu_model_runner.py — current code
kv_caches = getattr(self, "kv_caches", [])
for cache_tensor in kv_caches:
    if cache_tensor is not None:
        cache_tensor.zero_()      # <-- assumes flat Tensor
```

For pure-attention models `kv_caches` is `list[torch.Tensor]`, so this is fine.  
For **hybrid models** (GDN / Mamba / SSM layers + Attention), vLLM populates `kv_caches`
with **nested list/tuple containers** for the recurrent-state layers. Those container
objects are not `None`, so the `if` guard passes and `.zero_()` raises `AttributeError`.

The affected model families include any model using `GDNAttentionMetadataBuilder` or
similar hybrid KV-cache specs — Nemotron-H, Nemotron-Super, and anything else where
`kv_cache_groups` contains a mix of `MambaSpec`/`GDNSpec` alongside `FullAttentionSpec`.

### Proposed fix

Replace the flat loop body with a small recursive helper:

```python
kv_caches = getattr(self, "kv_caches", [])

def _zero_kv_cache_entry(entry: object) -> None:
    """Zero tensor leaves; recurse into list/tuple/dict containers.

    Hybrid models (e.g. GDN+Attention) store recurrent-state buffers
    in nested list/tuple containers inside kv_caches. A bare .zero_()
    call crashes with AttributeError on those container objects.
    """
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
    _zero_kv_cache_entry(cache_tensor)
```

Backward-compatible: for a flat `list[torch.Tensor]` the behavior is identical.

### Workaround

Patch the installed vLLM file in-container before starting `vllm serve`:
https://github.com/zoedolan/Vybn/blob/main/spark/systemd/patches/fp8-wake-fix/run.sh

### Environment

- vLLM: latest main (confirmed against `gpu_model_runner.py` SHA `53197a5c`)
- Model: Nemotron-Super / Nemotron-H family
- `--kv-cache-dtype fp8`, `VLLM_SERVER_DEV_MODE=1`, `--enable-sleep-mode`
- Container: vLLM Docker image, Python 3.12

Happy to submit a PR with the fix and a test.
