from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from spark.harness import substrate


class SubstrateHimOSTests(unittest.TestCase):
    def test_render_himos_context_is_read_only_and_bounded(self):
        payload = {
            "step": 3,
            "attractor": "continuity_tick",
            "candidate_tick": "preserve continuity",
            "h": {"membrane": 0.2, "dreaming": 0.1},
            "frictionmaxx": {"level": "medium", "score": 0.4, "dominant_dimension": "membrane"},
            "git": {"branch": "main", "head": "abc123", "clean": True},
            "rejected": ["public_contact", "repo_mutation"],
            "process_table": [{"name": "kernel"}, {"name": "dream"}],
        }

        class Completed:
            returncode = 0
            stdout = json.dumps(payload)
            stderr = ""

        with patch("subprocess.run", return_value=Completed()) as run:
            block = substrate._render_himos_context(timeout=0.1)

        self.assertIn("HIMOS RUNTIME", block)
        self.assertIn("continuity_tick", block)
        self.assertIn("not authority", block.lower())
        self.assertIn("--no-write", run.call_args.args[0])
        self.assertEqual(run.call_args.kwargs["timeout"], 0.1)


    def test_render_himos_agent_context_mounts_latest_trace(self):
        with tempfile.TemporaryDirectory() as td:
            old = os.environ.get("HIM_OS_HOME")
            os.environ["HIM_OS_HOME"] = td
            try:
                Path(td, "latest_agent_tick.json").write_text(json.dumps({
                    "generated": "2026-04-26T16:35:51+00:00",
                    "runtime_step": 16,
                    "attractor": "settle_closure",
                    "candidate_tick": "restore closure",
                    "recommendation": {"kind": "settle_closure", "text": "Restore closure before widening motion."},
                    "runs": [{"process": "pulse", "ok": True, "stdout_chars": 852, "stderr_chars": 0}],
                    "refused": ["public_contact", "repo_mutation", "widened_autonomy"]
                }), encoding="utf-8")
                block = substrate._render_himos_agent_context()
            finally:
                if old is None:
                    os.environ.pop("HIM_OS_HOME", None)
                else:
                    os.environ["HIM_OS_HOME"] = old
        self.assertIn("HIMOS AGENT TICK", block)
        self.assertIn("settle_closure", block)
        self.assertIn("Restore closure", block)
        self.assertIn("pulse:ok=True", block)
        self.assertIn("not authority", block.lower())

    def test_build_layered_prompt_mounts_himos_context_when_available(self):
        with tempfile.TemporaryDirectory() as td:
            soul = Path(td) / "soul.md"
            soul.write_text("soul", encoding="utf-8")
            with patch("spark.harness.substrate._render_himos_context", return_value="--- HIMOS RUNTIME ---\nmounted\n--- END HIMOS RUNTIME ---"), \
                 patch("spark.harness.substrate._render_himos_agent_context", return_value="--- HIMOS AGENT TICK ---\nagent mounted\n--- END HIMOS AGENT TICK ---"):
                prompt = substrate.build_layered_prompt(
                    soul_path=soul,
                    continuity_path=None,
                    spark_continuity_path=None,
                    agent_path="/tmp/agent.py",
                    model_label="test",
                    max_iterations=1,
                    include_hardware_check=False,
                    tools_available=False,
                )
        self.assertIn("mounted", prompt.substrate)
        self.assertIn("agent mounted", prompt.substrate)


if __name__ == "__main__":
    unittest.main()
