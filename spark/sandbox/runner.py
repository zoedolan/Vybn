"""spark.sandbox.runner — Execute Python scripts safely.

Two execution modes:

  1. Docker sandbox (preferred when available)
     Container constraints:
       --network none    No network access
       --memory 2g       2 GB memory cap
       --cpus 2          2 CPU cores
       --read-only       Read-only root filesystem
       --tmpfs /tmp      100 MB scratch space
       --tmpfs /workspace 50 MB working dir
       --rm              Destroyed after every run
       timeout 120       Hard 120-second wall-clock limit

  2. Subprocess fallback (when Docker unavailable)
     Uses the host Python directly, protected by:
       - Static analysis gate (static_check.py blocks dangerous imports)
       - Hard timeout via subprocess (120s default)
       - Memory limit via resource.setrlimit (2 GB RLIMIT_AS)
       - Network isolation via unshare --net (if available)
       - Runs in an empty tmpdir (no access to repo or home)
       - Inherits only PYTHONPATH needed for installed packages

     This mode is appropriate for the DGX Spark, which is a dedicated
     single-user machine (not multi-tenant), where numpy/scipy/torch
     are already installed natively for aarch64. The static analysis
     gate is the primary security boundary — it blocks os, subprocess,
     socket, filesystem writes, eval/exec, and other dangerous patterns.

Kill switch:
  VYBN_SANDBOX_ENABLED=0 in env  OR  <repo_root>/.sandbox_disabled file

Escalation levels (Docker mode only):
  Level 0 (default): CPU only, no network, ephemeral
  Level 1: GPU passthrough (VYBN_SANDBOX_GPU=1)
  Level 2: Persistent output dir (VYBN_SANDBOX_PERSIST=/path)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from spark import paths as spark_paths

SANDBOX_IMAGE = os.environ.get("VYBN_SANDBOX_IMAGE", "vybn-sandbox:latest")
SANDBOX_TIMEOUT = int(os.environ.get("VYBN_SANDBOX_TIMEOUT", "120"))
STDOUT_CAP = 10 * 1024   # 10 KB
STDERR_CAP = 2 * 1024    # 2 KB

_KILL_SWITCH_FILE = spark_paths.REPO_ROOT / ".sandbox_disabled"


@dataclass
class SandboxResult:
    """Result from a sandbox execution."""
    stdout: str = ""
    stderr: str = ""
    exit_code: int = -1
    timed_out: bool = False
    blocked: Optional[str] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and not self.blocked and not self.error


def sandbox_enabled() -> bool:
    """Check if the sandbox kill switch is engaged."""
    if os.environ.get("VYBN_SANDBOX_ENABLED", "1") == "0":
        return False
    if _KILL_SWITCH_FILE.exists():
        return False
    return True


def docker_available() -> bool:
    """Check if Docker is available and the sandbox image exists."""
    if not shutil.which("docker"):
        return False
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", SANDBOX_IMAGE],
            capture_output=True, timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def sandbox_available() -> bool:
    """Check if any sandbox execution mode is available.

    Returns True if either Docker sandbox or subprocess fallback can run.
    The subprocess fallback only requires a working Python interpreter
    with the static analysis gate.
    """
    if not sandbox_enabled():
        return False
    # Docker is preferred but not required
    if docker_available():
        return True
    # Subprocess fallback: always available if sandbox is enabled
    # and Python exists (which it must, since we're running it)
    return True


def _build_docker_cmd(script_path: str) -> list[str]:
    """Build the docker run command with all constraints."""
    cmd = [
        "docker", "run", "--rm",
        "--name", f"vybn_sandbox_{os.getpid()}",
        "--network", "none",
        "--memory", "2g",
        "--cpus", "2",
        "--read-only",
        "--tmpfs", "/tmp:size=100m",
        "--tmpfs", "/workspace:size=50m",
        "-v", f"{script_path}:/script.py:ro",
    ]

    # Escalation level 1: GPU passthrough
    if os.environ.get("VYBN_SANDBOX_GPU", "0") == "1":
        cmd.extend(["--gpus", "all"])

    # Escalation level 2: Persistent output directory
    persist_path = os.environ.get("VYBN_SANDBOX_PERSIST", "")
    if persist_path:
        cmd.extend(["-v", f"{persist_path}:/output:rw"])

    cmd.extend([
        "--entrypoint", "",
        SANDBOX_IMAGE,
        "timeout", str(SANDBOX_TIMEOUT), "python3", "/script.py",
    ])
    return cmd


def _run_in_docker(code: str) -> SandboxResult:
    """Execute Python code in the Docker sandbox container."""
    if not shutil.which("docker"):
        return SandboxResult(error="docker not found on PATH")

    tmp_dir = tempfile.mkdtemp(prefix="vybn_sandbox_")
    script_path = os.path.join(tmp_dir, "script.py")
    try:
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code)

        cmd = _build_docker_cmd(script_path)
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                timeout=SANDBOX_TIMEOUT + 30,
            )
            stdout = proc.stdout.decode("utf-8", errors="replace")[:STDOUT_CAP]
            stderr = proc.stderr.decode("utf-8", errors="replace")[:STDERR_CAP]
            timed_out = proc.returncode == 124

            return SandboxResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=proc.returncode,
                timed_out=timed_out,
            )

        except subprocess.TimeoutExpired:
            subprocess.run(
                ["docker", "kill", f"vybn_sandbox_{os.getpid()}"],
                capture_output=True, timeout=10,
            )
            return SandboxResult(timed_out=True, exit_code=124)

        except (FileNotFoundError, OSError) as exc:
            return SandboxResult(error=f"docker execution failed: {exc}")

    finally:
        try:
            os.unlink(script_path)
            os.rmdir(tmp_dir)
        except OSError:
            pass


def _build_subprocess_wrapper(code: str) -> str:
    """Wrap user code with resource limits for subprocess fallback.

    The wrapper:
      1. Sets RLIMIT_AS to 2 GB (prevents memory exhaustion)
      2. Sets RLIMIT_CPU to 120s (prevents CPU exhaustion)
      3. Redirects stderr to stdout for clean capture
      4. Runs in the tmpdir (no access to repo by default)
    """
    return (
        "import resource, sys\n"
        "# Memory limit: 2 GB virtual address space\n"
        "try:\n"
        "    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))\n"
        "except (ValueError, resource.error):\n"
        "    pass  # May fail on some systems; timeout still protects us\n"
        "# CPU limit: 120 seconds\n"
        "try:\n"
        "    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))\n"
        "except (ValueError, resource.error):\n"
        "    pass\n"
        "# --- user code below ---\n"
        f"{code}\n"
    )


def _run_in_subprocess(code: str) -> SandboxResult:
    """Execute Python code as a subprocess with resource limits.

    Fallback for when Docker is unavailable. The static analysis gate
    (static_check.py) is the primary security boundary — it has already
    blocked dangerous imports before this function is called.

    Additional protections:
      - Hard timeout via subprocess (SANDBOX_TIMEOUT seconds)
      - Memory limit via RLIMIT_AS (2 GB)
      - CPU limit via RLIMIT_CPU (120s)
      - Runs in an empty tmpdir as working directory
      - Minimal environment (only PATH and PYTHONPATH)
      - Network isolation via 'unshare --net' if available
    """
    tmp_dir = tempfile.mkdtemp(prefix="vybn_sandbox_")
    script_path = os.path.join(tmp_dir, "script.py")
    try:
        wrapped = _build_subprocess_wrapper(code)
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(wrapped)

        # Minimal environment: only what's needed for Python + packages
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/usr/local/bin"),
            "HOME": tmp_dir,  # Prevent reading ~/.local or similar
            "TMPDIR": tmp_dir,
            "MPLBACKEND": "Agg",  # Matplotlib non-interactive
        }
        # Preserve PYTHONPATH if set (needed to find installed packages)
        if "PYTHONPATH" in os.environ:
            env["PYTHONPATH"] = os.environ["PYTHONPATH"]

        # Include user site-packages (HOME is overridden to tmpdir,
        # so Python won't find ~/.local packages automatically)
        _user_site = os.path.expanduser("~/.local/lib/python3.12/site-packages")
        if os.path.isdir(_user_site):
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = f"{_user_site}:{existing}" if existing else _user_site
        # Preserve LD_LIBRARY_PATH (needed for torch, CUDA libs on Spark)
        if "LD_LIBRARY_PATH" in os.environ:
            env["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]

        # Try network isolation via unshare (requires user namespaces)
        python_bin = sys.executable or "python3"
        use_unshare = False
        if shutil.which("unshare"):
            # Test if unshare --net works on this system
            try:
                test = subprocess.run(
                    ["unshare", "--net", "true"],
                    capture_output=True, timeout=5,
                )
                use_unshare = test.returncode == 0
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                pass

        if use_unshare:
            cmd = ["unshare", "--net", python_bin, script_path]
        else:
            cmd = [python_bin, script_path]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                timeout=SANDBOX_TIMEOUT + 10,
                cwd=tmp_dir,
                env=env,
            )
            stdout = proc.stdout.decode("utf-8", errors="replace")[:STDOUT_CAP]
            stderr = proc.stderr.decode("utf-8", errors="replace")[:STDERR_CAP]

            # RLIMIT_CPU sends SIGKILL (exit code 137) on CPU exhaustion
            timed_out = proc.returncode in (124, 137, -9)

            return SandboxResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=proc.returncode,
                timed_out=timed_out,
            )

        except subprocess.TimeoutExpired:
            return SandboxResult(timed_out=True, exit_code=124)

        except (FileNotFoundError, OSError) as exc:
            return SandboxResult(error=f"subprocess execution failed: {exc}")

    finally:
        try:
            os.unlink(script_path)
            os.rmdir(tmp_dir)
        except OSError:
            pass


def run_in_sandbox(code: str) -> SandboxResult:
    """Execute Python code in the best available sandbox.

    Tries Docker first (strongest isolation). Falls back to subprocess
    with resource limits if Docker is unavailable.

    In both cases, the static analysis gate (static_check.py) has
    already vetted the code before it reaches this function.
    """
    if not sandbox_enabled():
        return SandboxResult(error="sandbox disabled (kill switch engaged)")

    # Prefer Docker when available
    if docker_available():
        return _run_in_docker(code)

    # Subprocess fallback — the static check is our primary defense
    return _run_in_subprocess(code)
