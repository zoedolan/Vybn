"""Static guards for spark/experiments/omni-window.sh.

These tests pin invariants we keep getting bitten by:

  1. Omni MUST use `nemotron_v3` (not `nano_v3`, which is the Nano text-only
     parser). The script previously auto-downloaded a `nano_v3_reasoning_parser.py`
     plugin and passed `--reasoning-parser nano_v3`; that was wrong.
  2. The aggressive flags from the model card must all be present:
     --enable-auto-tool-choice, --tool-call-parser qwen3_coder,
     --max-num-seqs 8, --max-num-batched-tokens 32768,
     --allowed-local-media-path=/.
  3. Spark memory overrides must stay in range:
     --max-model-len 32768, --gpu-memory-utilization in [0.70, 0.80].
  4. Sleep mode must be armed with VLLM_SERVER_DEV_MODE=1 (the /sleep,
     /wake_up, /is_sleeping endpoints are dev-mode only).
  5. A safe fallback must exist for missing multimodal audio/video deps:
     the script should *not* unconditionally pass --limit-mm-per-prompt or
     --media-io-kwargs — those flags must be guarded by a probe.
  6. The cleanup latch must wake Super on any exit when SUPER_SLEEPING=true,
     and full Omni server logs must be dumped on launch failure.

Run: python3 spark/tests/test_omni_window.py
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

THIS = Path(__file__).resolve()
SCRIPT = THIS.parent.parent / "experiments" / "omni-window.sh"


def script_text() -> str:
    return SCRIPT.read_text()


class OmniReasoningParserTests(unittest.TestCase):
    """The most important invariants — get the parser right."""

    def test_no_nano_v3_anywhere_in_omni_launch(self):
        """`nano_v3` was the wrong parser (Nano text-only). It must not appear
        in any active Omni-launch code path. The only allowed reference is
        inside an explicit refusal/die guard or a comment explaining why."""
        text = script_text()
        for lineno, line in enumerate(text.splitlines(), 1):
            if "nano_v3" not in line:
                continue
            # Allowed only inside the explicit `die` guard or in comments
            stripped = line.strip()
            is_comment = stripped.startswith("#")
            is_guard = ("die " in line and "Refusing" in line) or '== "nano_v3"' in line
            self.assertTrue(
                is_comment or is_guard,
                f"line {lineno}: stray reference to nano_v3 outside guard/comment: {line!r}",
            )

    def test_default_omni_reasoning_parser_is_nemotron_v3(self):
        text = script_text()
        # The default value of OMNI_REASONING_PARSER must be nemotron_v3.
        m = re.search(
            r'OMNI_REASONING_PARSER="\$\{OMNI_REASONING_PARSER:-([^}]+)\}"', text
        )
        self.assertIsNotNone(m, "OMNI_REASONING_PARSER default not found")
        self.assertEqual(m.group(1), "nemotron_v3")

    def test_no_plugin_download_for_nano_parser(self):
        """The previous version downloaded nano_v3_reasoning_parser.py from HF
        and passed --reasoning-parser-plugin. Both must be gone from active
        code (comments documenting the removal are fine)."""
        text = script_text()
        for lineno, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            self.assertNotIn(
                "nano_v3_reasoning_parser.py", line,
                f"line {lineno}: parser plugin filename in active code: {line!r}",
            )
            self.assertNotIn(
                "--reasoning-parser-plugin", line,
                f"line {lineno}: --reasoning-parser-plugin in active code: {line!r}",
            )

    def test_explicit_refusal_for_nano_v3(self):
        """If someone sets OMNI_REASONING_PARSER=nano_v3, the script must
        refuse rather than silently launch with the wrong parser."""
        text = script_text()
        self.assertRegex(
            text,
            r'OMNI_REASONING_PARSER.*==\s*"nano_v3".*\n\s*die ',
            "expected an explicit `die` guard refusing nano_v3",
        )


class OmniLaunchFlagsTests(unittest.TestCase):
    """Pin the model-card flags in OMNI_ARGS."""

    REQUIRED_FLAGS = [
        "--enable-auto-tool-choice",
        "--tool-call-parser qwen3_coder",
        "--max-num-seqs 8",
        "--max-num-batched-tokens 32768",
        "--max-model-len 32768",
        "--allowed-local-media-path=/",
        "--load-format fastsafetensors",
        "--kv-cache-dtype fp8",
        "--enable-prefix-caching",
        "--moe-backend cutlass",
        "--trust-remote-code",
    ]

    def test_required_flags_present(self):
        text = script_text()
        for flag in self.REQUIRED_FLAGS:
            self.assertIn(
                flag, text, f"missing required Omni launch flag: {flag}"
            )

    def test_gpu_memory_utilization_in_spark_range(self):
        """Spark override must stay between 0.70 and 0.80 inclusive — the model
        card uses 0.85 but Spark's unified memory needs more headroom while
        Super is sleeping but still resident."""
        text = script_text()
        m = re.search(r"--gpu-memory-utilization\s+([0-9.]+)", text)
        self.assertIsNotNone(m, "--gpu-memory-utilization not found")
        gmu = float(m.group(1))
        self.assertGreaterEqual(gmu, 0.70, f"gmu {gmu} below Spark range")
        self.assertLessEqual(gmu, 0.80, f"gmu {gmu} above Spark range")

    def test_max_model_len_is_spark_override(self):
        """Spark override is 32768; model card default is 65536. We must not
        silently revert to 65536 since Spark cannot fit it."""
        text = script_text()
        self.assertRegex(text, r"--max-model-len\s+32768")
        self.assertNotRegex(text, r"--max-model-len\s+65536")


class OmniMultimodalFallbackTests(unittest.TestCase):
    """Multimodal audio/video flags must be opt-in based on a runtime probe,
    because the vllm_node container ships without librosa/soundfile/decord."""

    def test_mm_flags_are_guarded(self):
        text = script_text()
        # The MM-flag block must be inside an `if [[ ... OMNI_ENABLE_MM ... ]]`
        # branch — it must NOT appear at the top level of OMNI_ARGS.
        guarded_block = re.search(
            r'if \[\[ "\$\{OMNI_ENABLE_MM[^}]*\}" == "true" \]\]; then(.*?)fi',
            text,
            re.DOTALL,
        )
        self.assertIsNotNone(
            guarded_block, "expected guarded if-block for OMNI_ENABLE_MM"
        )
        block = guarded_block.group(1)
        self.assertIn("--limit-mm-per-prompt", block)
        self.assertIn("--media-io-kwargs", block)

    def test_mm_probe_present(self):
        """Preflight must probe for librosa+soundfile+decord and set
        OMNI_ENABLE_MM accordingly."""
        text = script_text()
        self.assertIn("import librosa, soundfile, decord", text)
        self.assertRegex(text, r"OMNI_ENABLE_MM=true")
        self.assertRegex(text, r"OMNI_ENABLE_MM=false")


class SuperSleepDevModeTests(unittest.TestCase):
    def test_dev_mode_armed(self):
        """The /sleep, /wake_up, /is_sleeping vLLM endpoints only exist when
        VLLM_SERVER_DEV_MODE=1. The arm-sleep step must set that env var
        AND --enable-sleep-mode."""
        text = script_text()
        self.assertIn("VLLM_SERVER_DEV_MODE=1", text)
        self.assertIn("--enable-sleep-mode", text)


class CleanupLatchTests(unittest.TestCase):
    def test_cleanup_wakes_super_on_exit(self):
        """If SUPER_SLEEPING is true at exit, cleanup must POST /wake_up and
        fall back to systemctl restart of vybn-vllm.service if wake times out."""
        text = script_text()
        self.assertIn("trap cleanup EXIT", text)
        self.assertRegex(text, r'if \$SUPER_SLEEPING; then')
        self.assertIn("/wake_up", text)
        self.assertIn("vybn-vllm.service", text)

    def test_full_logs_on_launch_failure(self):
        """When Omni fails to come up, dump the full server log (not just
        tail -30) so we can diagnose missing-package failures end-to-end."""
        text = script_text()
        # full-cat of the omni server log on failure path
        self.assertRegex(
            text,
            r"docker exec vllm_node cat /tmp/omni-server.log",
            "expected full `cat` of /tmp/omni-server.log on failure path",
        )
        self.assertNotRegex(
            text,
            r"docker exec vllm_node tail -30 /tmp/omni-server.log",
            "old `tail -30` should be replaced with full log dump",
        )


def _assert_script_exists():
    if not SCRIPT.exists():
        raise SystemExit(f"FATAL: script not found at {SCRIPT}")


if __name__ == "__main__":
    _assert_script_exists()
    unittest.main(verbosity=2)

