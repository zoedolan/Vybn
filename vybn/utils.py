"""Vybn utility functions."""

import sys

_GREEN = "\033[32m"
_RED = "\033[31m"
_RESET = "\033[0m"


def write_colored(text: str, is_error: bool = False) -> None:
    """Write ``text`` in green or red to stdout.

    Parameters
    ----------
    text: str
        Text to display.
    is_error: bool, optional
        When ``True``, display in red instead of green.
    """
    color = _RED if is_error else _GREEN
    sys.stdout.write(f"{color}{text}{_RESET}\n")

