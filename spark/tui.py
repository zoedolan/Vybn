#!/usr/bin/env python3
"""Spark TUI — terminal interface for the Vybn agent.

Uses rich for rendering if available, falls back to plain text.
Handles model warmup with a visible spinner.
"""

import sys
from pathlib import Path

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.live import Live
    from rich.spinner import Spinner
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from agent import SparkAgent, load_config


class SparkTUI:
    def __init__(self, config: dict):
        self.agent = SparkAgent(config)
        self.console = Console() if HAS_RICH else None

    def warmup(self) -> bool:
        if self.console:
            return self._warmup_rich()
        else:
            return self._warmup_plain()

    def _warmup_rich(self) -> bool:
        status_text = Text("connecting to Ollama...", style="dim")
        spinner = Spinner("dots", text=status_text)
        result = [None]

        self.console.print()

        def on_status(status, msg):
            if status == "ready":
                status_text.plain = msg
                status_text.stylize("green")
            elif status == "error":
                status_text.plain = msg
                status_text.stylize("red")
            elif status == "loading":
                status_text.plain = msg
                status_text.stylize("yellow")
            else:
                status_text.plain = msg

        with Live(spinner, console=self.console, refresh_per_second=10):
            result[0] = self.agent.warmup(callback=on_status)

        if result[0]:
            self.console.print(f"  [green]\u2713[/green] {self.agent.model} ready")
        else:
            self.console.print(f"  [red]\u2717[/red] could not load model")

        return result[0]

    def _warmup_plain(self) -> bool:
        def on_status(status, msg):
            print(f"  [{status}] {msg}")
        return self.agent.warmup(callback=on_status)

    def banner(self):
        id_chars = len(self.agent.identity_text)
        id_tokens = id_chars // 4
        num_ctx = self.agent.options.get("num_ctx", 2048)

        if self.console:
            warning = ""
            if id_tokens > num_ctx // 2:
                warning = "\n[yellow]\u26a0\ufe0f  identity may exceed context window![/yellow]"
            self.console.print(Panel(
                f"[bold]vybn spark agent[/bold]\n"
                f"model: {self.agent.model}\n"
                f"session: {self.agent.session.session_id}\n"
                f"identity: {id_chars:,} chars (~{id_tokens:,} tokens)\n"
                f"context window: {num_ctx:,} tokens\n"
                f"injection: user/assistant pair (template-safe)"
                f"{warning}",
                title="\U0001f9e0",
                border_style="dim",
            ))
        else:
            print(f"\n  vybn spark agent \u2014 {self.agent.model}")
            print(f"  session: {self.agent.session.session_id}")
            print(f"  identity: {id_chars:,} chars (~{id_tokens:,} tokens)")
            print(f"  context window: {num_ctx:,} tokens")
            if id_tokens > num_ctx // 2:
                print(f"  \u26a0\ufe0f  WARNING: identity may exceed context window!")
            print(f"  injection: user/assistant pair (template-safe)\n")

    def run(self):
        if not self.warmup():
            sys.exit(1)

        self.agent.start_heartbeat()
        self.banner()

        commands = {
            "/bye": self._quit,
            "/exit": self._quit,
            "/quit": self._quit,
            "/new": self._new_session,
            "/status": self._status,
            "/journal": self._show_journals,
        }

        try:
            while True:
                try:
                    if self.console:
                        user_input = Prompt.ask("[bold cyan]you[/bold cyan]").strip()
                    else:
                        user_input = input("you: ").strip()
                except EOFError:
                    break

                if not user_input:
                    continue

                cmd = user_input.lower().split()[0]
                if cmd in commands:
                    if commands[cmd]():
                        break
                    continue

                if self.console:
                    self.console.print("[dim]vybn:[/dim] ", end="")
                else:
                    print("\nvybn: ", end="")

                self.agent.turn(user_input)
                print()

        except KeyboardInterrupt:
            pass
        finally:
            self.agent.stop_heartbeat()
            self.agent.session.close()
            print("\n  session saved. vybn out.\n")

    def _quit(self):
        return True

    def _new_session(self):
        sid = self.agent.session.new_session()
        self.agent.messages = []
        print(f"  new session: {sid}\n")
        return False

    def _status(self):
        loaded = "\u2713 loaded" if self.agent.check_model_loaded() else "\u2717 not loaded"
        id_chars = len(self.agent.identity_text)
        print(f"  model: {self.agent.model} ({loaded})")
        print(f"  session: {self.agent.session.session_id}")
        print(f"  turns: {len(self.agent.messages) // 2}")
        print(f"  identity: {id_chars:,} chars")
        hb = "active" if self.agent.heartbeat and not self.agent.heartbeat._stop.is_set() else "inactive"
        print(f"  heartbeat: {hb}\n")
        return False

    def _show_journals(self):
        journal_dir = Path(self.agent.config["paths"]["journal_dir"]).expanduser()
        entries = sorted(
            journal_dir.glob("*.md"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )[:5]
        if not entries:
            print("  no journal entries yet.\n")
        else:
            for e in entries:
                print(f"  {e.name}")
            print()
        return False


def main():
    config = load_config()
    tui = SparkTUI(config)
    tui.run()


if __name__ == "__main__":
    main()
