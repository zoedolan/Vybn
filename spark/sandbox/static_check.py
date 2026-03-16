"""spark.sandbox.static_check — Pre-execution static analysis gate.

Blocks dangerous imports and patterns before code reaches the container.
This runs on the host, not inside Docker.  If any pattern matches, the
code is rejected and never sent to the sandbox.

The check is deliberately conservative: false positives are acceptable
(the LLM-only fallback still works), false negatives are not.

Whitelisted packages (safe for scientific probes):
  numpy, scipy, torch, math, random, collections, itertools,
  functools, json, statistics, cmath, decimal, fractions,
  operator, string, textwrap, re, dataclasses, typing, enum, abc

These may internally 'import os' etc., but the sandbox container is
network-isolated, read-only, resource-bounded, and ephemeral — so
transitive stdlib usage inside trusted packages is not a threat.
The gate only blocks *user-level* use of dangerous modules.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

# Packages the sandbox image ships with that are safe for probes.
# Imports of these (and their submodules) are allowed through the gate.
_WHITELISTED_PACKAGES = frozenset({
    "numpy", "np",
    "scipy",
    "torch",
    "math", "cmath",
    "random",
    "statistics",
    "collections",
    "itertools", "functools", "operator",
    "json",
    "decimal", "fractions",
    "string", "textwrap",
    "re",
    "dataclasses", "typing", "enum", "abc",
    "copy", "pprint",
    "matplotlib",  # useful for numerical probes even without display
    "sklearn",     # scikit-learn, if installed in sandbox image
})


def _is_whitelisted_import(line: str) -> bool:
    """Return True if the line is an import of a whitelisted package.

    Handles:
        import numpy
        import numpy as np
        from numpy import ...
        from numpy.linalg import ...
        import torch, numpy  (multi-import)
    """
    stripped = line.strip()
    # 'from X.sub import ...' or 'from X import ...'
    m = re.match(r'^from\s+([\w.]+)\s+import', stripped)
    if m:
        root = m.group(1).split(".")[0]
        return root in _WHITELISTED_PACKAGES
    # 'import X' or 'import X as Y' or 'import X, Y'
    m = re.match(r'^import\s+(.+)', stripped)
    if m:
        parts = [p.strip().split()[0].split(".")[0] for p in m.group(1).split(",")]
        return all(p in _WHITELISTED_PACKAGES for p in parts)
    return False


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

    Lines that are imports of whitelisted packages are skipped.
    This prevents false positives from e.g. `import numpy` triggering
    the `import os` pattern (numpy internally imports os).
    """
    for line in code.splitlines():
        # Skip blank / comment lines
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # If this line is a whitelisted import, skip all pattern checks
        if _is_whitelisted_import(stripped):
            continue
        for pattern, compiled in zip(BLOCKED_PATTERNS, _COMPILED):
            match = compiled.search(line)
            if match:
                return False, f"BLOCKED: code contained disallowed pattern: {match.group()}"
    return True, None
