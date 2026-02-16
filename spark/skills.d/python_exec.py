"""Python execution skill - run Python code safely.

SKILL_NAME: python_exec
TOOL_ALIASES: ["python_exec", "run_python", "py_exec"]
"""

import subprocess
import tempfile
from pathlib import Path

SKILL_NAME = "python_exec"
TOOL_ALIASES = ["python_exec", "run_python", "py_exec"]


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
    
    # Create temporary file for code execution
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_path = Path(f.name)
    
    try:
        result = subprocess.run(
            ["python3", str(temp_path)],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=router.repo_root,
        )
        
        output = result.stdout[:10000]
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr[:2000]}"
        if result.returncode != 0:
            output += f"\n(exit code: {result.returncode})"
        
        return output or "(no output)"
    
    except subprocess.TimeoutExpired:
        return "Python execution timed out after 60 seconds"
    except Exception as e:
        return f"Python execution error: {e}"
    finally:
        temp_path.unlink(missing_ok=True)


def _extract_python_code(text: str) -> str:
    """Extract Python code from markdown fence or raw text."""
    import re
    
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
