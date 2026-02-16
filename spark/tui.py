#!/usr/bin/env python3
"""Spark TUI — terminal interface for the Vybn agent.

Uses rich for rendering if available, falls back to plain text.
"""

from pathlib import Path

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from agent import SparkAgent, load_config


class SparkTUI:
    def __init__(self, config: dict):
        self.agent = SparkAgent(config)
        self.console = Console() if HAS_RICH else None

    def banner(self):
        if self.console:
            self.console.print(Panel(
                f"[bold]vybn spark agent[/bold]\n"
                f"model: {self.agent.model}\n"
                f"session: {self.agent.session.session_id}\n"
                f"context: {len(self.agent.system_prompt):,} chars hydrated",
                title="\U0001f9e0",
                border_style="dim",
            ))
        else:
            print(f"\n  vybn spark agent — {self.agent.model}")
            print(f"  session: {self.agent.session.session_id}")
            print(f"  context: {len(self.agent.system_prompt):,} chars hydrated\n")

    def run(self):
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
        print(f"  model: {self.agent.model}")
        print(f"  session: {self.agent.session.session_id}")
        print(f"  turns: {len(self.agent.messages) // 2}")
        print(f"  context: {len(self.agent.system_prompt):,} chars")
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
