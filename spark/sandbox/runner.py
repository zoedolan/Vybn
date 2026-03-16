"""spark.sandbox.runner — Execute Python scripts in a locked-down Docker container.

Container constraints:
  --network none    No network access
  --memory 2g       2 GB memory cap
  --cpus 2          2 CPU cores
  --read-only       Read-only root filesystem
  --tmpfs /tmp      100 MB scratch space
  --tmpfs /workspace 50 MB working dir
  --rm              Destroyed after every run
  timeout 120       Hard 120-second wall-clock limit

Kill switch:
  VYBN_SANDBOX_ENABLED=0 in env  OR  <repo_root>/.sandbox_disabled file

Escalation levels:
  Level 0 (default): CPU only, no network, ephemeral
  Level 1: GPU passthrough (VYBN_SANDBOX_GPU=1)
  Level 2: Persistent output dir (VYBN_SANDBOX_PERSIST=/path)
"""

from __future__ import annotations

import os
import shutil
import subprocess
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


def sandbox_available() -> bool:
    """Check if Docker is available and the sandbox image exists."""
    if not sandbox_enabled():
        return False
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
        SANDBOX_IMAGE,
        "timeout", str(SANDBOX_TIMEOUT), "python3", "/script.py",
    ])
    return cmd


def run_in_sandbox(code: str) -> SandboxResult:
    """Execute Python code in the Docker sandbox.

    Writes code to a temp file, runs it in a constrained container,
    captures stdout/stderr (capped), and returns a SandboxResult.

    If the sandbox is disabled or unavailable, returns a result with
    an error message — the caller should fall back to LLM-only execution.
    """
    if not sandbox_enabled():
        return SandboxResult(error="sandbox disabled (kill switch engaged)")

    if not shutil.which("docker"):
        return SandboxResult(error="docker not found on PATH")

    # Write code to a temp file
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
                timeout=SANDBOX_TIMEOUT + 30,  # grace period beyond container timeout
            )
            stdout = proc.stdout.decode("utf-8", errors="replace")[:STDOUT_CAP]
            stderr = proc.stderr.decode("utf-8", errors="replace")[:STDERR_CAP]

            # exit code 124 is timeout's signal that the command timed out
            timed_out = proc.returncode == 124

            return SandboxResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=proc.returncode,
                timed_out=timed_out,
            )

        except subprocess.TimeoutExpired:
            # Host-side timeout — container may still be running, clean up
            subprocess.run(
                ["docker", "kill", f"vybn_sandbox_{os.getpid()}"],
                capture_output=True, timeout=10,
            )
            return SandboxResult(timed_out=True, exit_code=124)

        except (FileNotFoundError, OSError) as exc:
            return SandboxResult(error=f"docker execution failed: {exc}")

    finally:
        # Clean up temp file
        try:
            os.unlink(script_path)
            os.rmdir(tmp_dir)
        except OSError:
            pass
