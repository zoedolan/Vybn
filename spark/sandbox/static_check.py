"""spark.sandbox.static_check — Pre-execution static analysis gate.

Blocks dangerous imports and patterns before code reaches the container.
This runs on the host, not inside Docker.  If any pattern matches, the
code is rejected and never sent to the sandbox.

The check is deliberately conservative: false positives are acceptable
(the LLM-only fallback still works), false negatives are not.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

BLOCKED_PATTERNS: list[str] = [
    # Dangerous stdlib imports
    r'\bimport\s+(os|sys|subprocess|shutil|pathlib|socket|http|urllib|requests|ftplib)\b',
    # Filesystem probing
    r'\bopen\s*\(.*(/etc|/home|/root|/proc|/dev)',
    # Dynamic execution primitives
    r'\b(exec|eval|compile|__import__|getattr)\s*\(',
    # os module calls (even if aliased)
    r'\bos\.(system|popen|exec|spawn|remove|unlink|rmdir|rename|chmod)',
    # Subprocess variants
    r'\b(subprocess|Popen|call|check_output)\b',
    # Network access
    r'\bsocket\b',
    # C FFI
    r'\bctypes\b',
    # Builtins manipulation
    r'\b__builtins__\b',
]

_COMPILED = [re.compile(p) for p in BLOCKED_PATTERNS]


def check_code(code: str) -> Tuple[bool, Optional[str]]:
    """Check code for blocked patterns.

    Returns:
        (True, None)              — code is safe to execute
        (False, "BLOCKED: ...")   — code contains disallowed pattern
    """
    for pattern, compiled in zip(BLOCKED_PATTERNS, _COMPILED):
        match = compiled.search(code)
        if match:
            return False, f"BLOCKED: code contained disallowed pattern: {match.group()}"
    return True, None
