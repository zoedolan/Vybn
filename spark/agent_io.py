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
    - WebIO collects events into a list for the web server to drain.
        - RichIO uses rich.console.Console for styled TUI output.
    - SilentIO swallows everything for tests.

This module has NO imports from agent.py or skills.py.
It depends only on the Python standard library.
"""

import sys
import threading


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

        mode: 'fast' or 'deep'
        text: the display-ready content
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
        print(f"\n  {icon} [{label}]", flush=True)
        if detail:
            print(f"    {detail}", flush=True)

    def on_hint(self, message: str) -> None:
        print(f"\n  \u2139\ufe0f {message}", flush=True)

    def on_pulse(self, mode: str, text: str) -> None:
        if mode == "fast":
            truncated = f"{text[:80]}..." if len(text) > 80 else text
            print(f"\n  \U0001f49a [pulse:{mode}] {truncated}", flush=True)
        else:
            print(f"\n  \U0001f7e3 [pulse:{mode}]", flush=True)
            if text:
                print(f"\nvybn: {text}", flush=True)


class WebIO(AgentIO):
    """Collects agent events for the web frontend.

    Instead of printing to a terminal, WebIO accumulates events
    in a thread-safe list.  The web server drains these events
    and pushes them to the browser (via SSE, WebSocket, or polling).

    Usage:
        io = WebIO()
        agent = SparkAgent(config, io=io)
        agent.turn(user_input)
        events = io.drain()  # -> list of event dicts
    """

    def __init__(self):
        self._events: list[dict] = []
        self._lock = threading.Lock()
        self._tokens: list[str] = []

    def drain(self) -> list[dict]:
        """Return all pending events and clear the buffer."""
        with self._lock:
            events = self._events
            self._events = []
            return events

    def _emit(self, event_type: str, **kwargs) -> None:
        with self._lock:
            self._events.append({"type": event_type, **kwargs})

    # ---- streaming model output ----

    def on_token(self, token: str) -> None:
        with self._lock:
            self._tokens.append(token)

    # ---- response framing ----

    def on_response_start(self) -> None:
        with self._lock:
            self._tokens = []
        self._emit("response_start")

    def on_response_end(self) -> None:
        with self._lock:
            full_text = "".join(self._tokens)
            self._tokens = []
        if full_text:
            self._emit("response_text", text=full_text)
        self._emit("response_end")

    def on_prompt_restore(self) -> None:
        pass  # no-op for web (browser manages its own input state)

    # ---- status indicators ----

    def on_status(self, icon: str, label: str, detail: str = "") -> None:
        self._emit("status", icon=icon, label=label, detail=detail)

    def on_hint(self, message: str) -> None:
        self._emit("hint", message=message)

    # ---- pulse display ----

    def on_pulse(self, mode: str, text: str) -> None:
        self._emit("pulse", mode=mode, text=text)



class RichIO(AgentIO):
    """Rich-powered terminal I/O for the Spark TUI.

    Uses rich.console.Console for styled output.  Passed to
    SparkAgent by tui.py so streaming tokens, status indicators,
    and pulse output all render through Rich.

    Falls back to TerminalIO behavior if console is None.
    """

    def __init__(self, console=None):
        self.console = console

    # ---- streaming model output ----

    def on_token(self, token: str) -> None:
        if self.console:
            self.console.print(token, end="", highlight=False)
        else:
            print(token, end="", flush=True)

    # ---- response framing ----

    def on_response_start(self) -> None:
        if self.console:
            self.console.print("\n[dim]vybn:[/dim] ", end="")
        else:
            print("\nvybn: ", end="", flush=True)

    def on_response_end(self) -> None:
        if self.console:
            self.console.print()
        else:
            print()

    def on_prompt_restore(self) -> None:
        if self.console:
            self.console.print("[bold cyan]you[/bold cyan]: ", end="")
        else:
            print("you: ", end="", flush=True)

    # ---- status indicators ----

    def on_status(self, icon: str, label: str, detail: str = "") -> None:
        if self.console:
            self.console.print(f"\n  {icon} [dim]\\[{label}\\][/dim]", highlight=False)
            if detail:
                self.console.print(f"     {detail}", style="dim", highlight=False)
        else:
            print(f"\n {icon} [{label}]", flush=True)
            if detail:
                print(f"   {detail}", flush=True)

    def on_hint(self, message: str) -> None:
        if self.console:
            self.console.print(f"\n  \u2139\ufe0f {message}", style="dim italic")
        else:
            print(f"\n \u2139\ufe0f {message}", flush=True)

    # ---- pulse display ----

    def on_pulse(self, mode: str, text: str) -> None:
        if self.console:
            if mode == "fast":
                truncated = f"{text[:80]}..." if len(text) > 80 else text
                self.console.print(
                    f"\n  \U0001f49a [dim]\\[pulse:{mode}\\][/dim] {truncated}",
                    highlight=False,
                )
            else:
                self.console.print(
                    f"\n  \U0001f7e3 [dim]\\[pulse:{mode}\\][/dim]",
                    highlight=False,
                )
                if text:
                    self.console.print(f"\n[dim]vybn:[/dim] {text}", highlight=False)
        else:
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
