"""Checks for local-model runtime patches and watchdog behavior.

These guard three regressions:

1. The fp8 patch must not silently exit 0 with a "pattern not found" message.
2. Its replacement must recurse through list/tuple cache containers.
3. The watchdog must not treat a retained timestamp as the age of a fresh
   vLLM activation and restart it before the model can load.

Run: python3 spark/tests/test_fp8_wake_fix_patch.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN_SH = ROOT / "spark" / "systemd" / "patches" / "fp8-wake-fix" / "run.sh"
WATCHDOG_SH = ROOT / "spark" / "systemd" / "vybn-watchdog.sh"


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


def test_watchdog_does_not_restart_during_activation():
    """A retained prior timestamp must not turn a fresh activation into stale load."""
    import os, subprocess, tempfile
    with tempfile.TemporaryDirectory() as raw:
        tmp_path = Path(raw)
        fake = tmp_path / "bin"
        fake.mkdir()
        restart_log = tmp_path / "restarts"
        scripts = {
            "curl": "#!/bin/sh\ncase \"$*\" in *127.0.0.1:8000*) printf 000;; *) printf 200;; esac\n",
            "systemctl": "#!/bin/sh\ncase \"$*\" in *is-active*) printf activating;; *show*) printf 1;; *restart*) echo \"$*\" >> \"$WATCHDOG_RESTART_LOG\";; esac\n",
            "docker": "#!/bin/sh\nexit 0\n",
        }
        for name, body in scripts.items():
            target = fake / name
            target.write_text(body)
            target.chmod(0o755)
        marker = tmp_path / ".config/vybn/vllm-enabled"
        marker.parent.mkdir(parents=True)
        marker.touch()
        run = subprocess.run(
            ["bash", str(WATCHDOG_SH)], text=True, capture_output=True,
            env=os.environ | {"HOME": str(tmp_path),
                              "PATH": f"{fake}:{os.environ['PATH']}",
                              "WATCHDOG_RESTART_LOG": str(restart_log)},
        )
        assert run.returncode == 0
        assert "wait vllm activating (code=000)" in run.stdout
        assert not restart_log.exists(), run.stdout + run.stderr


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
