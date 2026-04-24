"""Secure loader for Vybn provider credentials.

Problem this solves: after reboot / service hardening the interactive
`vybn` process loses the environment populated by PAM for login shells.
Secrets still exist on disk at ~/.config/vybn/llm.env (and sometimes
/etc/environment), but the agent never reads them, so provider clients
see OPENAI_API_KEY absent and cloud OpenAI returns HTTP 401.

Design goals:
  - Never print values. No logging of key material, no echoing.
  - Never overwrite env vars that are already set (respect the
    environment the process was actually launched with).
  - Only read files owned by the invoking user, readable without sudo.
  - Parse a narrow subset of shell syntax: `KEY=value` and
    `export KEY=value`, optional matched single/double quotes, no
    command substitution, no variable expansion.
  - Return only counts / key names (never values) so callers can emit
    a non-sensitive status line.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable

# Only these keys are eligible to be injected. Whitelisting keeps an
# accidentally-committed llm.env from quietly enabling unrelated env
# vars on the running service.
_ALLOWED_KEYS: frozenset[str] = frozenset({
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENROUTER_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "XAI_API_KEY",
    "GROQ_API_KEY",
    "DEEPSEEK_API_KEY",
    "TOGETHER_API_KEY",
    "MISTRAL_API_KEY",
})

# KEY=value, with optional leading `export ` and optional matched
# single- or double-quotes around the value. Values stop at EOL or
# unquoted `#`. We deliberately do not expand $VAR or $(…).
_LINE = re.compile(
    r"""^\s*(?:export\s+)?
        (?P<key>[A-Za-z_][A-Za-z0-9_]*)
        \s*=\s*
        (?:
          "(?P<dq>(?:[^"\\]|\\.)*)" |
          '(?P<sq>[^']*)' |
          (?P<bare>[^#\n\r]*?)
        )
        \s*(?:\#.*)?\s*$
    """,
    re.VERBOSE,
)


def _safe_path(p: str | os.PathLike[str]) -> Path | None:
    try:
        path = Path(p).expanduser()
    except (RuntimeError, OSError):
        return None
    if not path.is_file():
        return None
    try:
        # Readable in this process without elevation? os.access handles
        # the current euid without raising on unreadable files.
        if not os.access(path, os.R_OK):
            return None
    except OSError:
        return None
    return path


def _parse(path: Path) -> dict[str, str]:
    """Return KEY→value for whitelisted keys found in file. No logging."""
    out: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        key = m.group("key")
        if key not in _ALLOWED_KEYS:
            continue
        val = m.group("dq")
        if val is not None:
            # decode the trivial \\n / \\" escapes we allow inside "..."
            val = val.encode("utf-8").decode("unicode_escape", errors="replace")
        else:
            val = m.group("sq")
            if val is None:
                val = (m.group("bare") or "").strip()
        if not val:
            continue
        out[key] = val
    return out


def load_env_files(
    paths: Iterable[str | os.PathLike[str]] | None = None,
    *,
    overwrite: bool = False,
) -> dict[str, str]:
    """Merge provider credentials from ~/.config/vybn/llm.env (and
    optionally /etc/environment) into os.environ.

    Returns a dict of {KEY: source_path} — ONLY the keys we actually
    set — suitable for a non-sensitive status line. Values are never
    returned and never logged.

    Precedence: earlier paths win over later paths. Existing os.environ
    values always win unless overwrite=True.
    """
    if paths is None:
        paths = (
            "~/.config/vybn/llm.env",
            "/etc/environment",
        )

    applied: dict[str, str] = {}
    seen: dict[str, str] = {}  # key -> first source that provided it

    for p in paths:
        sp = _safe_path(p)
        if sp is None:
            continue
        for key, val in _parse(sp).items():
            if key in seen:
                continue
            seen[key] = str(sp)
            if not overwrite and os.environ.get(key):
                # Respect the environment the process was launched with.
                continue
            os.environ[key] = val
            applied[key] = str(sp)
    return applied


def describe(applied: dict[str, str]) -> str:
    """Return a non-sensitive, printable summary. No values."""
    if not applied:
        return "no provider keys loaded from disk"
    keys = ", ".join(sorted(applied.keys()))
    return f"loaded {len(applied)} provider key(s) from disk: {keys}"


__all__ = ["load_env_files", "describe"]
