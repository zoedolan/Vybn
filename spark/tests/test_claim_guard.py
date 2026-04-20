"""Tests for the claim guard.

Run: python3 spark/tests/test_claim_guard.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.claim_guard import check  # noqa: E402


def test_fabricated_numbers_flagged():
    text = "the mean alpha was 0.302774 in control and 0.297585 in coupled"
    msgs = [{"role": "user", "content": "please run the experiment"}]
    warn = check(text, msgs)
    assert warn is not None, "should flag fabricated numbers"
    assert "0.297585" in warn
    assert "0.302774" in warn


def test_numbers_in_evidence_pass():
    text = "the mean alpha was 0.302774 in control"
    msgs = [{
        "role": "user",
        "content": "[probe result] alpha_mean 0.302774 coupled 0.297585",
    }]
    assert check(text, msgs) is None


def test_partial_support_flags_missing():
    text = "control 0.302774 coupled 0.297585 delta 0.005189"
    msgs = [{
        "role": "user",
        "content": "[probe result] alpha_mean 0.302774 coupled 0.297585",
    }]
    warn = check(text, msgs)
    assert warn is not None
    assert "0.005189" in warn
    assert "0.302774" not in warn
    assert "0.297585" not in warn


def test_no_numbers_returns_none():
    msgs = [{"role": "user", "content": "hi"}]
    assert check("I am uncertain about the path", msgs) is None


def test_empty_text_returns_none():
    assert check("", []) is None
    assert check(None, []) is None


def test_list_content_extracted():
    text = "delta was 0.123456"
    msgs = [{
        "role": "user",
        "content": [{"type": "tool_result", "text": "result 0.123456"}],
    }]
    assert check(text, msgs) is None


def test_nested_content_string_extracted():
    text = "we saw 0.424242"
    msgs = [{
        "role": "user",
        "content": [{"type": "tool_result", "content": "raw 0.424242"}],
    }]
    assert check(text, msgs) is None


def test_window_bounds_discards_old_evidence():
    text = "value 0.999999"
    old = {"role": "user", "content": "[probe result] 0.999999"}
    filler = [{"role": "user", "content": "filler"} for _ in range(10)]
    assert check(text, [old] + filler, window=3) is not None


def test_integer_claims_flagged():
    text = "commit 8234567 landed at turn 142"
    msgs = [{"role": "user", "content": "ok"}]
    warn = check(text, msgs)
    assert warn is not None
    assert "8234567" in warn


def test_short_integers_not_flagged():
    text = "we tried 42 times across 2 shards"
    msgs = [{"role": "user", "content": "ok"}]
    assert check(text, msgs) is None


def test_single_decimal_not_flagged():
    text = "roughly 3.1 seconds"
    msgs = [{"role": "user", "content": "ok"}]
    assert check(text, msgs) is None


if __name__ == "__main__":
    import traceback
    fns = [
        (n, f) for n, f in list(globals().items())
        if n.startswith("test_") and callable(f)
    ]
    passed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"OK  {name}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL {name}: {e}")
            traceback.print_exc()
    print(f"\n{passed}/{len(fns)} passed")
    sys.exit(0 if passed == len(fns) else 1)

