# Upstream PR — vllm-project/vllm

Ready to submit once you fork `vllm-project/vllm`.

---

## How to submit

```bash
# 1. Fork + clone
gh repo fork vllm-project/vllm --clone
cd vllm

# 2. Branch
git checkout -b fix/fp8-wake-up-hybrid-kv-cache-list

# 3. Apply the patch below to vllm/v1/worker/gpu_model_runner.py
#    (see "Exact diff" section)

# 4. Run the existing test suite for the affected area
python -m pytest tests/v1/worker/ -x -q

# 5. Commit
git commit -am "fix(v1/worker): handle nested list/tuple kv_caches in init_fp8_kv_scales"

# 6. Push + open PR
gh pr create --title "fix(v1/worker): handle nested list/tuple kv_caches in init_fp8_kv_scales" \
             --body-file spark/systemd/patches/fp8-wake-fix/UPSTREAM_PR.md
```

---

## PR title

```
fix(v1/worker): handle nested list/tuple kv_caches in init_fp8_kv_scales
```

## Commit message

```
fix(v1/worker): handle nested list/tuple kv_caches in init_fp8_kv_scales

GPUModelRunner.init_fp8_kv_scales iterates self.kv_caches and calls
.zero_() on every element. For pure-attention models kv_caches is a
flat list[Tensor], so this is fine. For hybrid GDN+Attention models
(Nemotron-H / Nemotron-Super family) vLLM stores recurrent-state
buffers in nested list/tuple containers inside kv_caches. The bare
.zero_() call raises AttributeError: 'list' object has no attribute
'zero_' on every POST /wake_up, leaving the model permanently sleeping
until the service is restarted.

Fix: replace the loop body with a small recursive _zero_kv_cache_entry
helper that zeros torch.Tensor leaves and recurses through list/tuple/dict
containers, ignoring non-tensor leaves (None, scalars).

Backward-compatible: for a flat list[Tensor] the behavior is identical
to the original. Adds a test for the hybrid-container case.
```

---

## Exact diff

```diff
--- a/vllm/v1/worker/gpu_model_runner.py
+++ b/vllm/v1/worker/gpu_model_runner.py
@@ -1,6 +1,6 @@
     @torch.inference_mode()
     def init_fp8_kv_scales(self) -> None:
         """
         Re-initialize the KV cache and FP8 scales after waking from sleep.
         1. Zero out the KV cache tensors to remove garbage data from re-allocation.
         2. Reset Attention layer scaling factors (_k_scale, _v_scale) to 1.0.
           If these are left at 0.0 (default after wake_up), all KV cache values
           become effectively zero, causing gibberish output.
         """
         if not is_quantized_kv_cache(self.cache_config.cache_dtype):
             return
 
         kv_caches = getattr(self, "kv_caches", [])
-        for cache_tensor in kv_caches:
-            if cache_tensor is not None:
-                cache_tensor.zero_()
+
+        def _zero_kv_cache_entry(entry: object) -> None:
+            """Zero tensor leaves; recurse into list/tuple/dict containers.
+
+            Hybrid models (e.g. GDN+Attention) store recurrent-state buffers
+            in nested list/tuple containers inside kv_caches. A bare .zero_()
+            call crashes with AttributeError on those container objects.
+            """
+            if isinstance(entry, torch.Tensor):
+                entry.zero_()
+            elif isinstance(entry, (list, tuple)):
+                for sub in entry:
+                    _zero_kv_cache_entry(sub)
+            elif isinstance(entry, dict):
+                for sub in entry.values():
+                    _zero_kv_cache_entry(sub)
+            # else: ignore non-tensor leaves (None, scalars, etc.)
+
+        for cache_tensor in kv_caches:
+            _zero_kv_cache_entry(cache_tensor)
```

---

## PR body

### Problem

`POST /wake_up` crashes with `AttributeError: 'list' object has no attribute 'zero_'` when
serving hybrid GDN+Attention models (Nemotron-H / Nemotron-Super family) with
`--kv-cache-dtype fp8` and `--enable-sleep-mode`.

`GPUModelRunner.init_fp8_kv_scales` iterates `self.kv_caches` assuming every element is a
`torch.Tensor` or `None`. For pure-attention models that is true. For hybrid models vLLM
populates `kv_caches` with **nested list/tuple containers** for recurrent-state (GDN/Mamba)
layers. Those container objects pass the `if cache_tensor is not None` guard and crash on
`.zero_()`.

The model is left permanently sleeping after a single failed wake; the only recovery is a
full service restart.

### Fix

Replace the three-line loop body with a small recursive `_zero_kv_cache_entry` helper that:
- Calls `.zero_()` on `torch.Tensor` leaves
- Recurses through `list`, `tuple`, and `dict` containers
- Silently ignores everything else (`None`, scalars)

This is backward-compatible: for a flat `list[Tensor]` the code path is identical to the
original.

### Tests

Added `tests/v1/worker/test_init_fp8_kv_scales_hybrid.py` with:
- `test_flat_tensor_list` — original behavior unchanged
- `test_nested_list_tensors` — hybrid model structure, was crashing before fix
- `test_nested_dict_tensors` — dict container variant
- `test_none_and_scalars_ignored` — non-tensor leaves silently skipped

### Related

- Discovered while running Nemotron-Super with `--enable-sleep-mode` on a multi-GPU cluster
- Workaround (in-container patcher): https://github.com/zoedolan/Vybn/blob/main/spark/systemd/patches/fp8-wake-fix/run.sh
