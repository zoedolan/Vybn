#!/usr/bin/env python3
"""I/O abstraction for the Vybn Spark Agent.

Decouples agent.py from terminal print() calls (Phase 3b refactor)
so the same agent core can drive the TUI, web interface, or run
silently in tests.

Design:
    - AgentIO is a plain class with no-op defaults, not an ABC.
      Callers override only the methods they care about.
    - TerminalIO reproduces the exact print() behavior that was
      previously hardcoded in agent.py.  Zero behavioral change.
    - WebIO (future) will push events over WebSocket.
    - SilentIO (future) will swallow everything for tests.

This module has NO imports from agent.py or skills.py.
It depends only on the Python standard library.
"""
import sys


class AgentIO:
    """Base I/O interface for the Spark agent.

    Every method is a no-op by default.  Subclass and override
    only what your frontend needs.
    """

    # ---- streaming model output ----

    def on_token(self, token: str) -> None:
        """Called for each token during streaming model output."""

    # ---- response framing ----

    def on_response_start(self) -> None:
        """Called before Vybn's response begins (the 'vybn: ' prefix)."""

    def on_response_end(self) -> None:
        """Called after Vybn's response finishes."""

    def on_prompt_restore(self) -> None:
        """Restore the user prompt ('you: ') after async events."""

    # ---- status indicators ----

    def on_status(self, icon: str, label: str, detail: str = "") -> None:
        """Display a status line (tool indicator, policy decision, etc).

        icon:   emoji or symbol (e.g. '\u2192', '\u26d4', '\u2b50')
        label:  short description (e.g. 'file_read: config.yaml')
        detail: optional extra line (e.g. argument preview)
        """

    def on_hint(self, message: str) -> None:
        """Display a hint or feedback message to the user."""

    # ---- pulse display ----

    def on_pulse(self, mode: str, text: str) -> None:
        """Display heartbeat pulse output.

        mode:  'fast' or 'deep'
        text:  the display-ready content
        """


class TerminalIO(AgentIO):
    """Default terminal I/O -- reproduces the exact print() behavior
    that was previously hardcoded in agent.py.

    This is the drop-in replacement.  When agent.py creates a
    TerminalIO instance, the user sees identical output to before.
    """

    def on_token(self, token: str) -> None:
        print(token, end="", flush=True)

    def on_response_start(self) -> None:
        print("\nvybn: ", end="", flush=True)

    def on_response_end(self) -> None:
        print()

    def on_prompt_restore(self) -> None:
        print("you: ", end="", flush=True)

    def on_status(self, icon: str, label: str, detail: str = "") -> None:
        print(f"\n {icon} [{label}]", flush=True)
        if detail:
            print(f"  {detail}", flush=True)

    def on_hint(self, message: str) -> None:
        print(f"\n \u2139\ufe0f {message}", flush=True)

    def on_pulse(self, mode: str, text: str) -> None:
        if mode == "fast":
            truncated = f"{text[:80]}..." if len(text) > 80 else text
            print(f"\n \U0001f49a [pulse:{mode}] {truncated}", flush=True)
        else:
            print(f"\n \U0001f7e3 [pulse:{mode}]", flush=True)
            if text:
                print(f"\nvybn: {text}", flush=True)


class SilentIO(AgentIO):
    """Swallows all output.  Useful for tests and background runs."""
    pass
