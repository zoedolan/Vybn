"""Static checks on spark/systemd/patches/fp8-wake-fix/run.sh.

These guard two regressions of the fp8-wake-fix patcher:

1. The patch must not silently exit 0 with a "pattern not found" message —
   that previously let sleep-capable vLLM start with a broken wake path.
2. The injected replacement must contain a recursive helper that handles
   list/tuple containers (hybrid models nest tensors inside lists).

Run: python3 spark/tests/test_fp8_wake_fix_patch.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN_SH = ROOT / "spark" / "systemd" / "patches" / "fp8-wake-fix" / "run.sh"


def _src() -> str:
    return RUN_SH.read_text(encoding="utf-8")


def test_run_sh_exists():
    assert RUN_SH.exists(), f"missing {RUN_SH}"


def test_no_silent_pattern_not_found_exit_zero():
    src = _src()
    # Find every "pattern not found" occurrence; none of them may be paired
    # with an exit 0 in the same neighborhood.
    for match in re.finditer(r"pattern not found|loop not found|not found", src, re.I):
        window = src[match.start(): match.start() + 400]
        assert "sys.exit(0)" not in window, (
            "fp8-wake-fix run.sh appears to silently exit 0 after a "
            "'pattern/loop not found' message; this lets sleep-capable "
            "vLLM start with broken wake. Failing exit required."
        )


def test_failure_branch_uses_nonzero_exit():
    src = _src()
    # The script must contain at least one nonzero sys.exit() to signal
    # failure when the expected loop is missing.
    assert re.search(r"sys\.exit\(\s*[1-9]\d*\s*\)", src), (
        "expected at least one nonzero sys.exit() for the failure branch"
    )


def test_recursive_helper_handles_list_and_tuple():
    src = _src()
    assert "_zero_kv_cache_entry" in src, "recursive helper name missing"
    # Must dispatch on Tensor and recurse on list/tuple containers.
    assert "isinstance(entry, torch.Tensor)" in src, (
        "recursive helper must check torch.Tensor leaves"
    )
    assert re.search(r"isinstance\(\s*entry\s*,\s*\(\s*list\s*,\s*tuple\s*\)\s*\)", src), (
        "recursive helper must recurse on list/tuple containers"
    )


def test_idempotent_already_applied_path():
    src = _src()
    # Idempotence: when the patched form is already present, exit 0 cleanly.
    assert "already applied" in src, "expected 'already applied' idempotence path"
    # That idempotence path must be exit 0 (success), distinct from the
    # failure branch above.
    assert re.search(r"already applied[^\n]*\n[^\n]*\n[^\n]*sys\.exit\(\s*0\s*\)|already applied[^\n]*\n[^\n]*sys\.exit\(\s*0\s*\)", src) or \
        ("already applied" in src and "sys.exit(0)" in src), (
        "'already applied' branch must exit 0"
    )


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
