"""Tests for spark.sandbox — static analysis gate and sandbox runner.

Covers:
  - static_check: blocked patterns reject dangerous code, safe code passes
  - runner: SandboxResult dataclass, kill switch, docker command construction,
    output capping (mocked Docker — no real containers needed)
  - agency integration: code block extraction, sandbox/LLM fallback logic
"""

import os
import unittest
from unittest import mock

from spark.sandbox.static_check import check_code, BLOCKED_PATTERNS
from spark.sandbox.runner import (
    SandboxResult,
    _build_docker_cmd,
    sandbox_enabled,
    run_in_sandbox,
    STDOUT_CAP,
    STDERR_CAP,
)


# ── static_check tests ──────────────────────────────────────────────────────

class TestStaticCheck(unittest.TestCase):
    """Static analysis gate must block dangerous code and allow safe code."""

    def test_blocks_os_import(self):
        safe, reason = check_code("import os\nos.system('rm -rf /')")
        self.assertFalse(safe)
        self.assertIn("BLOCKED", reason)

    def test_blocks_subprocess_import(self):
        safe, reason = check_code("import subprocess\nsubprocess.run(['ls'])")
        self.assertFalse(safe)
        self.assertIn("BLOCKED", reason)

    def test_blocks_socket(self):
        safe, reason = check_code("import socket\ns = socket.socket()")
        self.assertFalse(safe)

    def test_blocks_eval(self):
        safe, reason = check_code("result = eval('1+1')")
        self.assertFalse(safe)

    def test_blocks_exec(self):
        safe, reason = check_code("exec('print(1)')")
        self.assertFalse(safe)

    def test_blocks_compile(self):
        safe, reason = check_code("code = compile('1+1', '<string>', 'eval')")
        self.assertFalse(safe)

    def test_blocks_dunder_import(self):
        safe, reason = check_code("m = __import__('os')")
        self.assertFalse(safe)

    def test_blocks_ctypes(self):
        safe, reason = check_code("import ctypes")
        self.assertFalse(safe)

    def test_blocks_builtins_access(self):
        safe, reason = check_code("__builtins__['eval']('1')")
        self.assertFalse(safe)

    def test_blocks_os_system(self):
        safe, reason = check_code("os.system('whoami')")
        self.assertFalse(safe)

    def test_blocks_open_etc(self):
        safe, reason = check_code("f = open('/etc/passwd')")
        self.assertFalse(safe)

    def test_blocks_open_home(self):
        safe, reason = check_code("f = open('/home/user/.ssh/id_rsa')")
        self.assertFalse(safe)

    def test_blocks_shutil_import(self):
        safe, reason = check_code("import shutil")
        self.assertFalse(safe)

    def test_blocks_pathlib_import(self):
        safe, reason = check_code("import pathlib")
        self.assertFalse(safe)

    def test_blocks_http_import(self):
        safe, reason = check_code("import http.client")
        self.assertFalse(safe)

    def test_blocks_urllib_import(self):
        safe, reason = check_code("import urllib.request")
        self.assertFalse(safe)

    def test_blocks_requests_import(self):
        safe, reason = check_code("import requests")
        self.assertFalse(safe)

    def test_blocks_getattr(self):
        safe, reason = check_code("getattr(__builtins__, 'eval')('1')")
        self.assertFalse(safe)

    def test_allows_numpy(self):
        safe, reason = check_code("import numpy as np\nx = np.array([1,2,3])\nprint(x.mean())")
        self.assertTrue(safe)
        self.assertIsNone(reason)

    def test_allows_torch(self):
        safe, reason = check_code("import torch\nt = torch.tensor([1.0, 2.0])\nprint(t.mean())")
        self.assertTrue(safe)
        self.assertIsNone(reason)

    def test_allows_scipy(self):
        safe, reason = check_code("from scipy import stats\nprint(stats.norm.pdf(0))")
        self.assertTrue(safe)
        self.assertIsNone(reason)

    def test_allows_matplotlib(self):
        code = (
            "import matplotlib.pyplot as plt\n"
            "plt.plot([1,2,3])\n"
            "plt.savefig('/tmp/plot.png')\n"
        )
        safe, reason = check_code(code)
        self.assertTrue(safe)
        self.assertIsNone(reason)

    def test_allows_math_stdlib(self):
        safe, reason = check_code("import math\nprint(math.pi)")
        self.assertTrue(safe)

    def test_allows_statistics_stdlib(self):
        safe, reason = check_code("import statistics\nprint(statistics.mean([1,2,3]))")
        self.assertTrue(safe)

    def test_allows_collections_stdlib(self):
        safe, reason = check_code("from collections import Counter\nprint(Counter('aab'))")
        self.assertTrue(safe)

    def test_allows_itertools_stdlib(self):
        safe, reason = check_code("import itertools\nlist(itertools.combinations([1,2,3], 2))")
        self.assertTrue(safe)

    def test_allows_random_stdlib(self):
        safe, reason = check_code("import random\nprint(random.random())")
        self.assertTrue(safe)

    def test_allows_plain_open_tmp(self):
        """open() on /tmp paths is allowed — only /etc, /home, /root, /proc, /dev blocked."""
        safe, reason = check_code("f = open('/tmp/data.txt', 'w')\nf.write('hello')")
        self.assertTrue(safe)

    def test_empty_code(self):
        safe, reason = check_code("")
        self.assertTrue(safe)

    def test_pattern_count(self):
        """Ensure we have the expected number of blocked patterns."""
        self.assertGreaterEqual(len(BLOCKED_PATTERNS), 8)


# ── runner tests ─────────────────────────────────────────────────────────────

class TestSandboxResult(unittest.TestCase):
    def test_ok_on_success(self):
        r = SandboxResult(stdout="hello", exit_code=0)
        self.assertTrue(r.ok)

    def test_not_ok_on_nonzero_exit(self):
        r = SandboxResult(exit_code=1)
        self.assertFalse(r.ok)

    def test_not_ok_on_timeout(self):
        r = SandboxResult(exit_code=124, timed_out=True)
        self.assertFalse(r.ok)

    def test_not_ok_on_error(self):
        r = SandboxResult(error="docker not found")
        self.assertFalse(r.ok)

    def test_not_ok_on_blocked(self):
        r = SandboxResult(blocked="BLOCKED: import os")
        self.assertFalse(r.ok)


class TestKillSwitch(unittest.TestCase):
    def test_disabled_by_env(self):
        with mock.patch.dict(os.environ, {"VYBN_SANDBOX_ENABLED": "0"}):
            self.assertFalse(sandbox_enabled())

    def test_enabled_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            # Remove the key if it happens to be set
            env = os.environ.copy()
            env.pop("VYBN_SANDBOX_ENABLED", None)
            with mock.patch.dict(os.environ, env, clear=True):
                with mock.patch("spark.sandbox.runner._KILL_SWITCH_FILE") as mock_path:
                    mock_path.exists.return_value = False
                    self.assertTrue(sandbox_enabled())

    def test_disabled_by_file(self):
        with mock.patch("spark.sandbox.runner._KILL_SWITCH_FILE") as mock_path:
            mock_path.exists.return_value = True
            self.assertFalse(sandbox_enabled())


class TestDockerCommand(unittest.TestCase):
    def test_base_command(self):
        cmd = _build_docker_cmd("/tmp/test.py")
        self.assertIn("docker", cmd)
        self.assertIn("--network", cmd)
        self.assertIn("none", cmd)
        self.assertIn("--memory", cmd)
        self.assertIn("2g", cmd)
        self.assertIn("--read-only", cmd)
        self.assertIn("/tmp/test.py:/script.py:ro", cmd)

    def test_gpu_passthrough(self):
        with mock.patch.dict(os.environ, {"VYBN_SANDBOX_GPU": "1"}):
            cmd = _build_docker_cmd("/tmp/test.py")
            self.assertIn("--gpus", cmd)
            self.assertIn("all", cmd)

    def test_no_gpu_by_default(self):
        with mock.patch.dict(os.environ, {"VYBN_SANDBOX_GPU": "0"}):
            cmd = _build_docker_cmd("/tmp/test.py")
            self.assertNotIn("--gpus", cmd)

    def test_persistent_output(self):
        with mock.patch.dict(os.environ, {"VYBN_SANDBOX_PERSIST": "/data/output"}):
            cmd = _build_docker_cmd("/tmp/test.py")
            self.assertIn("/data/output:/output:rw", cmd)


class TestRunInSandbox(unittest.TestCase):
    def test_returns_error_when_disabled(self):
        with mock.patch.dict(os.environ, {"VYBN_SANDBOX_ENABLED": "0"}):
            result = run_in_sandbox("print('hello')")
            self.assertFalse(result.ok)
            self.assertIn("disabled", result.error)

    def test_returns_error_when_no_docker(self):
        with mock.patch.dict(os.environ, {"VYBN_SANDBOX_ENABLED": "1"}):
            with mock.patch("spark.sandbox.runner._KILL_SWITCH_FILE") as mock_path:
                mock_path.exists.return_value = False
                with mock.patch("shutil.which", return_value=None):
                    result = run_in_sandbox("print('hello')")
                    self.assertFalse(result.ok)
                    self.assertIn("docker not found", result.error)

    @mock.patch("subprocess.run")
    @mock.patch("shutil.which", return_value="/usr/bin/docker")
    def test_captures_stdout(self, mock_which, mock_run):
        with mock.patch.dict(os.environ, {"VYBN_SANDBOX_ENABLED": "1"}):
            with mock.patch("spark.sandbox.runner._KILL_SWITCH_FILE") as mock_path:
                mock_path.exists.return_value = False
                mock_run.return_value = mock.Mock(
                    returncode=0,
                    stdout=b"42\n",
                    stderr=b"",
                )
                result = run_in_sandbox("print(42)")
                self.assertTrue(result.ok)
                self.assertIn("42", result.stdout)

    @mock.patch("subprocess.run")
    @mock.patch("shutil.which", return_value="/usr/bin/docker")
    def test_caps_stdout(self, mock_which, mock_run):
        with mock.patch.dict(os.environ, {"VYBN_SANDBOX_ENABLED": "1"}):
            with mock.patch("spark.sandbox.runner._KILL_SWITCH_FILE") as mock_path:
                mock_path.exists.return_value = False
                big_output = b"x" * (STDOUT_CAP + 5000)
                mock_run.return_value = mock.Mock(
                    returncode=0,
                    stdout=big_output,
                    stderr=b"",
                )
                result = run_in_sandbox("print('x' * 20000)")
                self.assertLessEqual(len(result.stdout), STDOUT_CAP)

    @mock.patch("subprocess.run")
    @mock.patch("shutil.which", return_value="/usr/bin/docker")
    def test_detects_timeout(self, mock_which, mock_run):
        with mock.patch.dict(os.environ, {"VYBN_SANDBOX_ENABLED": "1"}):
            with mock.patch("spark.sandbox.runner._KILL_SWITCH_FILE") as mock_path:
                mock_path.exists.return_value = False
                mock_run.return_value = mock.Mock(
                    returncode=124,
                    stdout=b"partial...",
                    stderr=b"",
                )
                result = run_in_sandbox("import time; time.sleep(999)")
                self.assertTrue(result.timed_out)
                self.assertFalse(result.ok)


# ── agency integration tests ────────────────────────────────────────────────

class TestCodeBlockExtraction(unittest.TestCase):
    def test_extracts_python_block(self):
        from spark.extensions.agency import _extract_code_blocks
        text = "Here is the code:\n```python\nprint(42)\n```\nDone."
        code = _extract_code_blocks(text)
        self.assertIsNotNone(code)
        self.assertIn("print(42)", code)

    def test_extracts_unlabeled_block(self):
        from spark.extensions.agency import _extract_code_blocks
        text = "```\nx = 1 + 1\nprint(x)\n```"
        code = _extract_code_blocks(text)
        self.assertIsNotNone(code)
        self.assertIn("x = 1 + 1", code)

    def test_returns_none_for_no_code(self):
        from spark.extensions.agency import _extract_code_blocks
        text = "This is just prose. No code here."
        code = _extract_code_blocks(text)
        self.assertIsNone(code)

    def test_extracts_multiple_blocks(self):
        from spark.extensions.agency import _extract_code_blocks
        text = "```python\nimport numpy as np\n```\nThen:\n```python\nprint(np.pi)\n```"
        code = _extract_code_blocks(text)
        self.assertIn("import numpy", code)
        self.assertIn("np.pi", code)


if __name__ == "__main__":
    unittest.main()
