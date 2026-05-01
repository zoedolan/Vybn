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
SEMANTIC_GATE_MODULE = Path(__file__).resolve().parents[1] / "harness" / "semantic_gate.py"


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
    assert "SLEEP_REQUESTED=true" in src, "sleep request latch must be set before /sleep"
    assert "SLEEP_REQUESTED=false" in src, "sleep request latch must be cleared on recovery"


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



def test_super_semantic_gate_exists_and_uses_raw_deterministic_expected_outputs():
    src = _src()
    gate = SEMANTIC_GATE_MODULE.read_text()
    assert "super_semantic_gate()" in src
    assert "python3 -m harness.semantic_gate" in src
    assert 'api_base + "/completions"' in gate
    assert '"temperature": 0' in gate
    assert '"max_tokens": 24' in gate
    assert '"chat_template_kwargs": {"enable_thinking": False}' not in src
    assert "semantic_gate_visible_answer(" in gate
    for expected in ["FOUR", '{"status":"ok"}', "FAIL"]:
        assert expected in gate
    for probe_name in ["known_answer", "structured_shape", "wake_reasoning"]:
        assert probe_name in gate
    assert "corruption_signature=" in gate
    assert 'finish == "length"' in gate or 'finish_reason=length' in gate
    assert "<<'PY'" not in src, "omni-window must not duplicate the canonical semantic gate heredoc"


def test_semantic_gate_runs_before_sleep_and_after_wake_before_success():
    src = _src()
    pre = src.index('Running pre-sleep Super semantic gate')
    sleep = src.index('/sleep?level=${SLEEP_LEVEL}')
    wake_section = src.index('--- waking Super ---')
    wake = src.index('/wake_up', wake_section)
    gate = src.index('--- semantic wake gate ---')
    clear = src.index('--- clearing sleep mode ---')
    assert pre < sleep
    assert wake < gate < clear
    assert 'failing closed' in src
    assert 'recovery restart issued before cleanup' in src
    assert 'restore_non_sleep_super "post-wake semantic failure"' in src[src.index('--- semantic wake gate ---'):]
    assert 'systemctl --user restart vybn-vllm.service' in src
    assert "restored_non_sleep_semantic_passed" in src[src.index('--- clearing sleep mode ---'):]
    assert "final-non-sleep" in src[src.index('--- clearing sleep mode ---'):]


def test_super_sleep_cycle_harness_is_isolated_from_omni_and_defaults_level1():
    assert SLEEP_CYCLE_SCRIPT.is_file(), f"missing: {SLEEP_CYCLE_SCRIPT}"
    src = SLEEP_CYCLE_SCRIPT.read_text()
    gate = SEMANTIC_GATE_MODULE.read_text()
    assert "OMNI" not in src
    assert 'SLEEP_LEVEL="${SLEEP_LEVEL:-1}"' in src
    assert 'CYCLES="${CYCLES:-5}"' in src
    assert "ALLOW_LEVEL2" in src
    assert "VYBN_SLEEP_ACTUATOR_ARM" in src
    assert 'CURL_CONNECT_TIMEOUT="${CURL_CONNECT_TIMEOUT:-5}"' in src
    assert 'CURL_MAX_TIME="${CURL_MAX_TIME:-30}"' in src
    assert "curl_super()" in src
    assert "SLEEP_REQUESTED=true" in src
    assert "sleep request failed/timed out" in src
    assert "wake request failed/timed out" in src
    assert "super_semantic_gate()" in src
    assert "python3 -m harness.semantic_gate" in src
    assert 'api_base + "/completions"' in gate
    assert '"temperature": 0' in gate
    assert '"max_tokens": 24' in gate
    assert '"chat_template_kwargs": {"enable_thinking": False}' not in src
    assert "semantic_gate_visible_answer(" in gate
    for expected in ["FOUR", '{"status":"ok"}', "FAIL"]:
        assert expected in gate
    for probe_name in ["known_answer", "structured_shape", "wake_reasoning"]:
        assert probe_name in gate
    assert "corruption_signature=" in gate
    assert "cold_restart_super_non_sleep" in src
    assert "failing closed after cold restart" in src
    assert "<<'PY'" not in src, "sleep-cycle must not duplicate the canonical semantic gate heredoc"


def test_super_sleep_cycle_harness_bash_syntax_clean():
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not on PATH")
    res = subprocess.run([bash, "-n", str(SLEEP_CYCLE_SCRIPT)], capture_output=True, text=True)
    assert res.returncode == 0, f"bash -n failed: {res.stderr}"


def test_peer_gpu_memory_check_runs_after_confirmed_sleep_before_omni_launch():
    src = _src()
    sleep_confirmed = src.index('Super confirmed sleeping')
    peer_gpu_check = src.index('peer_gpu_memory_check_passed')
    launch = src.index('--- launching Omni on peer')
    assert sleep_confirmed < peer_gpu_check < launch



def test_omni_window_packet_centered_single_controller():
    src = _src()
    assert "PACKET_FILE=" in src
    assert "packet_stub_created" in src
    assert "finalize_packet" in src
    assert "PEER_MAX_GPU_USED_AFTER_SLEEP_MB" in src
    assert "peer_gpu_memory_check_passed" in src
    assert "Peer GPU still has a process using" in src
    assert "super_restart_begin" in src
    assert "post_wake_semantic_passed" in src
    assert "restore_non_sleep_restart" in src
    assert "restored_non_sleep_semantic_passed" in src
    assert "visual/manifold perception organ" in src
    assert "absorption_targets" in src
    assert "semantic_failure_restart_required" in src
    assert "success_condition" in src
    assert "no sensory visual claim" in src


def test_omni_window_accepts_operator_input_packet():
    src = _src()
    assert "OMNI_INPUT_PACKET=\"${VYBN_OMNI_INPUT_PACKET:-}\"" in src
    assert "python3 - \"$OMNI_INPUT_PACKET\"" in src
    assert "input_packet_excerpt" in src
    assert "Artifact packet excerpt" in src
    assert "file-mediated context rather than sensory vision" in src


def test_omni_input_packet_excerpt_is_interpolated():
    src = _src()
    assert "prompt = f\"\"\"You are Nemotron-Nano-Omni" in src
    assert "{input_packet_excerpt}" in src
