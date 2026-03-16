"""spark.sandbox — Docker-based code execution for agency experiments.

Lets the agency extension execute small Python scripts from experiment
proposals in a locked-down Docker container.  No network, no GPU (by
default), read-only root filesystem, ephemeral containers destroyed
after every run.

Kill switch:
  VYBN_SANDBOX_ENABLED=0  or  <repo_root>/.sandbox_disabled

Escalation levels (env vars):
  Level 0 (default): CPU only, no network, no GPU, ephemeral
  Level 1: GPU passthrough  (VYBN_SANDBOX_GPU=1)
  Level 2: Persistent output (VYBN_SANDBOX_PERSIST=/path)
  Level 3: Network access    (manual only — never automatic)
"""

from .static_check import check_code, BLOCKED_PATTERNS
from .runner import run_in_sandbox, sandbox_available

__all__ = ["check_code", "BLOCKED_PATTERNS", "run_in_sandbox", "sandbox_available"]
