"""Structural regression tests for spark/experiments/omni-window.sh.

These tests don't run the experiment (it requires two live DGX Sparks); they
guard the load-bearing safety latches so that future edits cannot silently
regress them. Keep this file fast and offline.

Run: python3 -m pytest spark/tests/test_omni_window_script.py
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "omni-window.sh"
SLEEP_CYCLE_SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "super-sleep-cycle.sh"


def _src() -> str:
    return SCRIPT.read_text()


def test_script_exists_and_is_executable():
    assert SCRIPT.is_file(), f"missing: {SCRIPT}"
    assert SCRIPT.stat().st_mode & 0o111, "omni-window.sh must be executable"


def test_bash_syntax_clean():
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not on PATH")
    res = subprocess.run([bash, "-n", str(SCRIPT)], capture_output=True, text=True)
    assert res.returncode == 0, f"bash -n failed: {res.stderr}"


def test_strict_mode_enabled():
    assert "set -euo pipefail" in _src(), "must use strict bash mode"


def test_cleanup_trap_present():
    src = _src()
    assert "trap cleanup EXIT" in src, "EXIT trap must be installed"
    assert "SUPER_SLEEPING=true" in src, "sleep latch must be set on confirm"
    assert "SUPER_SLEEPING=false" in src, "sleep latch must be cleared on wake"


def test_wake_fallback_restarts_service():
    src = _src()
    assert "systemctl --user restart vybn-vllm.service" in src, (
        "wake-fail fallback (service restart) must remain in cleanup"
    )


def test_dev_mode_precondition_check():
    """Calling /sleep against a server without dev mode 404s and corrupts state.
    The script must verify /is_sleeping is reachable before sleeping."""
    src = _src()
    assert "/is_sleeping" in src
    assert "dev-mode" in src.lower() or "dev_mode" in src.lower(), (
        "must explicitly check that dev-mode endpoints are live before /sleep"
    )


def test_omni_launch_has_safe_fallback():
    """Aggressive flags (kv-cache fp8, fastsafetensors, moe-backend, parser)
    are the most likely cause of engine-init failure on a fresh build/quant.
    The script must retry once with a stripped baseline so we can distinguish
    'model can't load' from 'specific flag broken'."""
    src = _src()
    assert "build_omni_args" in src
    assert "aggressive" in src and "safe" in src
    assert "launch_omni safe" in src or "launch_omni \"safe\"" in src


def test_omni_log_streamed_to_local_disk():
    """tail -30 over ssh on the peer's container /tmp is too short for a
    Python traceback and disappears when the container reaps tmpfs. The
    full log must be pulled to the primary."""
    src = _src()
    assert "OMNI_LOG_LOCAL" in src
    assert "fetch_omni_log" in src
    assert "tail -n 200" in src, "must print at least 200 lines on failure"


def test_no_sensitive_topology_in_log_lines():
    """Public log/journal text shouldn't hard-code peer IPs in a way that
    survives copy-paste into public channels. PEER must be configurable
    via env so test/staging can override."""
    src = _src()
    assert 'PEER="${OMNI_PEER:-' in src, (
        "PEER must be overridable via OMNI_PEER env var"
    )


def test_ssh_retry_helper_present():
    src = _src()
    assert "ssh_peer()" in src, "ssh_peer one-retry helper must exist"


def test_wake_failure_captures_journal():
    src = _src()
    assert "vybn-vllm-wakefail" in src, (
        "wake-timeout path must dump journalctl tail to disk for diagnosis"
    )



def test_super_semantic_gate_exists_and_uses_deterministic_expected_outputs():
    src = _src()
    assert "super_semantic_gate()" in src
    assert '"temperature": 0' in src
    assert '"max_tokens": 16' in src
    for expected in ["FOUR", "READY", "{\\\"ok\\\":true}"]:
        assert expected in src
    assert 'finish == "length"' in src or 'finish_reason=length' in src


def test_semantic_gate_runs_before_sleep_and_after_wake_before_success():
    src = _src()
    pre = src.index('Running pre-sleep Super semantic gate')
    sleep = src.index('/sleep?level=2')
    wake = src.index('/wake_up')
    gate = src.index('--- semantic wake gate ---')
    clear = src.index('--- clearing sleep mode ---')
    assert pre < sleep
    assert wake < gate < clear
    assert 'failing closed' in src
    assert 'cleanup will force recovery restart' in src


def test_super_sleep_cycle_harness_is_isolated_from_omni_and_defaults_level1():
    assert SLEEP_CYCLE_SCRIPT.is_file(), f"missing: {SLEEP_CYCLE_SCRIPT}"
    src = SLEEP_CYCLE_SCRIPT.read_text()
    assert "OMNI" not in src
    assert 'SLEEP_LEVEL="${SLEEP_LEVEL:-1}"' in src
    assert 'CYCLES="${CYCLES:-5}"' in src
    assert "ALLOW_LEVEL2" in src
    assert "super_semantic_gate()" in src
    assert '"temperature": 0' in src
    assert '"max_tokens": 16' in src


def test_super_sleep_cycle_harness_bash_syntax_clean():
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not on PATH")
    res = subprocess.run([bash, "-n", str(SLEEP_CYCLE_SCRIPT)], capture_output=True, text=True)
    assert res.returncode == 0, f"bash -n failed: {res.stderr}"
