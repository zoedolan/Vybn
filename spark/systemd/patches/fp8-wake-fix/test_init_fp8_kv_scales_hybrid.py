"""Tests for init_fp8_kv_scales with hybrid (nested) kv_caches.

Add to: tests/v1/worker/test_init_fp8_kv_scales_hybrid.py

These are unit-level tests that exercise the _zero_kv_cache_entry
recursive helper without spinning up a full model runner. They can be
run standalone:

    python -m pytest tests/v1/worker/test_init_fp8_kv_scales_hybrid.py -v
"""
import torch
import pytest


def _zero_kv_cache_entry(entry: object) -> None:
    """Recursive helper — copy of the proposed fix for isolated testing."""
    if isinstance(entry, torch.Tensor):
        entry.zero_()
    elif isinstance(entry, (list, tuple)):
        for sub in entry:
            _zero_kv_cache_entry(sub)
    elif isinstance(entry, dict):
        for sub in entry.values():
            _zero_kv_cache_entry(sub)


def _make_tensor(*shape, fill=1.0):
    return torch.full(shape, fill)


def test_flat_tensor_list():
    """Original behavior: flat list[Tensor] is zeroed correctly."""
    kv_caches = [_make_tensor(4, 8), _make_tensor(4, 8)]
    for t in kv_caches:
        _zero_kv_cache_entry(t)
    for t in kv_caches:
        assert t.sum().item() == 0.0


def test_nested_list_tensors():
    """Hybrid model: kv_caches contains nested lists (GDN layers).

    This was the crashing case before the fix:
        AttributeError: 'list' object has no attribute 'zero_'
    """
    inner = [_make_tensor(2, 4), _make_tensor(2, 4)]
    kv_caches = [
        _make_tensor(4, 8),   # attention layer — flat Tensor
        inner,                 # GDN layer — nested list
        _make_tensor(4, 8),   # attention layer — flat Tensor
    ]
    for entry in kv_caches:
        _zero_kv_cache_entry(entry)

    assert kv_caches[0].sum().item() == 0.0
    for t in inner:
        assert t.sum().item() == 0.0
    assert kv_caches[2].sum().item() == 0.0


def test_nested_tuple_tensors():
    """Tuple containers (used by some SSM implementations) are handled."""
    inner = (_make_tensor(2, 4), _make_tensor(2, 4))
    kv_caches = [_make_tensor(4, 8), inner]
    for entry in kv_caches:
        _zero_kv_cache_entry(entry)

    assert kv_caches[0].sum().item() == 0.0
    for t in inner:
        assert t.sum().item() == 0.0


def test_nested_dict_tensors():
    """Dict containers are traversed via .values()."""
    inner = {"k": _make_tensor(2, 4), "v": _make_tensor(2, 4)}
    kv_caches = [_make_tensor(4, 8), inner]
    for entry in kv_caches:
        _zero_kv_cache_entry(entry)

    assert kv_caches[0].sum().item() == 0.0
    for t in inner.values():
        assert t.sum().item() == 0.0


def test_none_and_scalars_ignored():
    """Non-tensor leaves (None, int, float) are silently skipped."""
    kv_caches = [None, 42, 3.14, _make_tensor(2, 2)]
    for entry in kv_caches:  # must not raise
        _zero_kv_cache_entry(entry)
    assert kv_caches[3].sum().item() == 0.0


def test_deeply_nested():
    """Arbitrary nesting depth is handled via recursion."""
    leaf = _make_tensor(2, 2)
    deep = [[[(leaf)]]]
    _zero_kv_cache_entry(deep)
    assert leaf.sum().item() == 0.0
