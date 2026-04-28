import pathlib
import sys
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "spark"))

from harness.providers import is_parallel_safe, validate_command
import vybn_spark_agent as agent
from spark.harness.subturns import probe_envelope

BAD = "rm" + " -rf" + " /"


class RecursiveUnlockTests(unittest.TestCase):
    def test_destructive_command_still_blocks(self):
        ok, reason = validate_command(BAD)
        self.assertFalse(ok)
        self.assertIn("Blocked", reason or "")

    def test_dangerous_literal_in_readonly_grep_is_data(self):
        cmd = "grep -RIn " + repr(BAD) + " spark/harness"
        self.assertTrue(is_parallel_safe(cmd))
        ok, reason = validate_command(cmd, allow_dangerous_literals_for_readonly=True)
        self.assertTrue(ok, reason)

    def test_cd_readonly_is_parallel_safe(self):
        self.assertTrue(is_parallel_safe("cd ~/Vybn && grep -n foo README.md"))

    def test_cd_git_add_is_not_parallel_safe(self):
        self.assertFalse(is_parallel_safe("cd ~/Vybn && git add README.md"))

    def test_command_substitution_is_blocked_in_probe_channel(self):
        for cmd in [
            "echo `task`",
            "grep -n \"route to `task`\" spark/harness/substrate.py",
            "echo $(whoami)",
        ]:
            self.assertFalse(is_parallel_safe(cmd))
            ok, reason = validate_command(cmd, allow_dangerous_literals_for_readonly=True)
            self.assertFalse(ok)
            self.assertIn("command substitution", reason)

    def test_single_quoted_backticks_remain_literal_readonly_data(self):
        cmd = "grep -n 'route to `task`' spark/harness/substrate.py"
        self.assertTrue(is_parallel_safe(cmd))
        ok, reason = validate_command(cmd, allow_dangerous_literals_for_readonly=True)
        self.assertTrue(ok, reason)

    def test_probe_subturn_uses_fresh_subprocess_for_readonly(self):
        with mock.patch.object(agent, "execute_readonly", return_value="fresh") as er:
            bash = mock.Mock()
            ran, out = agent._run_probe_subturn("echo ok", bash)
        self.assertTrue(ran)
        self.assertEqual(out, "fresh")
        er.assert_called_once()
        bash.execute.assert_not_called()

    def test_timeout_is_not_ordinary_executed_stdout(self):
        with mock.patch.object(agent, "execute_readonly", return_value="[timed out after 1s]"):
            ran, out = agent._run_probe_subturn("echo ok", mock.Mock())
        self.assertFalse(ran)
        self.assertIn("probe timed out", out)

    def test_restart_output_during_probe_is_mismatch(self):
        bash = mock.Mock()
        bash.execute.return_value = "(bash session restarted)"
        with mock.patch.object(agent, "is_parallel_safe", return_value=False):
            ran, out = agent._run_probe_subturn("export X=1", bash)
        self.assertFalse(ran)
        self.assertIn("control-event mismatch", out)

    def test_envelopes_distinguish_restart_and_probe(self):
        probe = probe_envelope(kind="probe", header_fields={"cmd": "echo ok"}, body="ok", ran=True)
        restart = probe_envelope(kind="needs-restart", header_fields={}, body="(bash session restarted)", ran=True)
        self.assertIn("kind: probe", probe)
        self.assertIn("BEGIN_PROBE_STDOUT", probe)
        self.assertIn("kind: needs-restart", restart)
        self.assertIn("BEGIN_NEEDS_RESTART_STDOUT", restart)


if __name__ == "__main__":
    unittest.main()
