"""Python execution skill - run Python code safely.

    SKILL_NAME: python_exec
    TOOL_ALIASES: ["python_exec", "run_python", "py_exec"]
"""

import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

SKILL_NAME = "python_exec"
TOOL_ALIASES = ["python_exec", "run_python", "py_exec"]

# SECURITY: Patterns that indicate dangerous code.
# These are checked before execution to prevent the LLM from being
# tricked into running harmful code via prompt injection.
_BLOCKED_PATTERNS = [
    # Direct shell access
    r"\bos\.system\b",
    r"\bos\.popen\b",
    r"\bos\.exec[lv]p?e?\b",
    r"\bsubprocess\b",
    r"\bcommands\.get",
    # Network access
    r"\bsocket\b",
    r"\burllib\b",
    r"\brequests\b",
    r"\bhttplib\b",
    r"\bhttp\.client\b",
    r"\baiohttp\b",
    # File system escape
    r"\bshutil\.rmtree\b",
    r"\bos\.remove\b",
    r"\bos\.unlink\b",
    r"\bos\.rmdir\b",
    r"\bos\.rename\b",
    # Code loading
    r"\b__import__\b",
    r"\bimportlib\b",
    r"\bexec\s*\(",
    r"\beval\s*\(",
    r"\bcompile\s*\(",
    # Dangerous builtins
    r"\bglobals\s*\(",
    r"\blocals\s*\(",
    r"\bgetattr\s*\(",
    r"\bsetattr\s*\(",
    r"\bdelattr\s*\(",
    r"\b__builtins__\b",
    r"\b__subclasses__\b",
]

_BLOCKED_RE = re.compile("|".join(_BLOCKED_PATTERNS))


def _validate_code(code: str) -> str | None:
    """Check code for dangerous patterns. Returns reason string or None."""
    match = _BLOCKED_RE.search(code)
    if match:
        return f"blocked pattern: '{match.group()}'"
    return None


def execute(action: dict, router) -> str:
    """Execute Python code in a sandboxed subprocess."""
    params = action.get("params", {})
    code = (
        params.get("code", "")
        or params.get("content", "")
        or params.get("script", "")
    )

    if not code:
        # Try to extract from raw response
        raw = action.get("raw", "")
        code = _extract_python_code(raw)

    if not code:
        return "no Python code specified"

    # SECURITY: Pre-execution validation
    violation = _validate_code(code)
    if violation:
        logger.warning("python_exec blocked: %s", violation)
        return f"Code blocked by security policy: {violation}"

    # Create temporary file for code execution
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.py', delete=False
    ) as f:
        f.write(code)
        temp_path = Path(f.name)

    try:
        # SECURITY: Restricted environment for subprocess
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONNOUSERSITE"] = "1"

        result = subprocess.run(
            ["python3", "-I", str(temp_path)],  # -I = isolated mode
            capture_output=True,
            text=True,
            timeout=30,  # Reduced from 60s
            cwd=str(router.repo_root),
            env=env,
        )

        output = result.stdout[:10000]
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr[:2000]}"
        if result.returncode != 0:
            output += f"\n(exit code: {result.returncode})"

        return output or "(no output)"

    except subprocess.TimeoutExpired:
        return "Python execution timed out after 30 seconds"
    except Exception as e:
        return f"Python execution error: {e}"
    finally:
        temp_path.unlink(missing_ok=True)


def _extract_python_code(text: str) -> str:
    """Extract Python code from markdown fence or raw text."""

    # Try markdown fence first
    fence_match = re.search(
        r"```(?:python|py)?\n(.+?)```",
        text,
        re.DOTALL,
    )
    if fence_match:
        return fence_match.group(1).strip()

    # Try indented code block
    lines = text.split("\n")
    code_lines = []
    in_code = False

    for line in lines:
        if line.startswith("    ") or line.startswith("\t"):
            code_lines.append(line.lstrip())
            in_code = True
        elif in_code and not line.strip():
            code_lines.append("")
        elif in_code:
            break

    if code_lines:
        return "\n".join(code_lines).strip()

    return ""
