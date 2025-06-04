"""Sandboxed bash execution tool."""
import subprocess
from typing import Any, Dict

def run(command: str) -> Dict[str, Any]:
    """Run a bash command and capture output."""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }
