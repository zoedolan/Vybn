#!/usr/bin/env python3
"""Spark TUI — terminal interface for the Vybn agent.

Uses rich for rendering if available, falls back to plain text.
Handles model warmup with a visible spinner.

A background drain thread processes bus messages (inbox, heartbeat
pulses, agent results) between user inputs. A threading lock
protects the message list from concurrent access.

Commands:
  /bye, /exit, /quit  — exit
  /new                — fresh session
  /status             — system state
  /explore, /map      — dump environment layout (no model needed)
  /policy             — show policy engine state
  /audit              — show recent audit trail
  /journal            — show recent journal entries
  /help               — show available commands
"""

import sys
import threading
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
        self._drain_lock = threading.Lock()
        self._stop_drain = threading.Event()
        self._drain_thread = None

    # ---- warmup ----

    def warmup(self) -> bool:
        if self.console:
            return self._warmup_rich()
        return self._warmup_plain()

    def _warmup_rich(self) -> bool:
        status_text = Text("connecting to Ollama...", style="dim")
        spinner = Spinner("dots", text=status_text)
        result = [None]

        self.console.print()

        def on_status(status, msg):
            styles = {"ready": "green", "error": "red", "loading": "yellow"}
            status_text.plain = msg
            status_text.stylize(styles.get(status, "dim"))

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

    # ---- banner ----

    def banner(self):
        a = self.agent
        id_chars = len(a.identity_text)
        id_tokens = id_chars // 4
        num_ctx = a.options.get("num_ctx", 2048)
        plugins = len(a.skills.plugin_handlers)

        hb_fast = a.heartbeat.fast_interval // 60
        hb_deep = a.heartbeat.deep_interval // 60

        lines = [
            f"model: {a.model}",
            f"session: {a.session.session_id}",
            f"identity: {id_chars:,} chars (~{id_tokens:,} tokens)",
            f"context: {num_ctx:,} tokens",
            f"heartbeat: fast={hb_fast}m, deep={hb_deep}m",
            f"inbox: {a.inbox.inbox_dir}",
            f"agents: pool_size={a.agent_pool.pool_size}",
        ]
        if plugins:
            names = ", ".join(a.skills.plugin_handlers.keys())
            lines.append(f"plugins: {names}")

        if self.console:
            warning = ""
            if id_tokens > num_ctx // 2:
                warning = "\n[yellow]\u26a0\ufe0f  identity may exceed context window![/yellow]"
            self.console.print(Panel(
                "[bold]vybn spark agent[/bold]\n" + "\n".join(lines) + warning,
                title="\U0001f9e0",
                border_style="dim",
            ))
        else:
            print(f"\n  vybn spark agent \u2014 {a.model}")
            for line in lines:
                print(f"  {line}")
            if id_tokens > num_ctx // 2:
                print(f"  \u26a0\ufe0f  WARNING: identity may exceed context window!")
            print()

    # ---- background bus drain ----

    def _start_drain_thread(self):
        """Background thread that processes bus messages between user inputs."""
        self._drain_thread = threading.Thread(target=self._drain_loop, daemon=True)
        self._drain_thread.start()

    def _drain_loop(self):
        while not self._stop_drain.is_set():
            if self.agent.bus.wait(timeout=2.0):
                with self._drain_lock:
                    try:
                        self.agent.drain_bus()
                    except Exception as e:
                        print(f"\n  [bus error] {e}")

    # ---- main loop ----

    def run(self):
        if not self.warmup():
            sys.exit(1)

        self.agent.start_subsystems()
        self._start_drain_thread()
        self.banner()

        help_text = (
            "  /bye, /exit, /quit  — exit\n"
            "  /new                — fresh session\n"
            "  /status             — system state\n"
            "  /explore, /map      — dump environment layout\n"
            "  /policy             — policy engine state\n"
            "  /audit              — recent audit trail\n"
            "  /journal            — recent journal entries\n"
            "  /help               — this message"
        )

        print("  type /help for commands, or just talk\n")

        commands = {
            "/bye": self._quit,
            "/exit": self._quit,
            "/quit": self._quit,
            "/new": self._new_session,
            "/status": self._status,
            "/journal": self._show_journals,
            "/explore": self._explore,
            "/map": self._explore,
            "/policy": self._policy,
            "/audit": self._audit,
            "/help": lambda: self._show_help(help_text),
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

                with self._drain_lock:
                    self.agent.turn(user_input)
                print()

        except KeyboardInterrupt:
            pass
        finally:
            self._stop_drain.set()
            self.agent.stop_subsystems()
            self.agent.session.close()
            print("\n  session saved. vybn out.\n")

    # ---- commands ----

    def _quit(self):
        return True

    def _new_session(self):
        sid = self.agent.session.new_session()
        self.agent.messages = []
        print(f"  new session: {sid}\n")
        return False

    def _status(self):
        a = self.agent
        loaded = "\u2713 loaded" if a.check_model_loaded() else "\u2717 not loaded"
        print(f"  model: {a.model} ({loaded})")
        print(f"  session: {a.session.session_id}")
        print(f"  turns: {len(a.messages) // 2}")
        print(f"  identity: {len(a.identity_text):,} chars")
        print(f"  bus pending: {a.bus.pending}")
        print(f"  heartbeat: fast={a.heartbeat.fast_count}, deep={a.heartbeat.deep_count}")
        print(f"  agents active: {a.agent_pool.active_count}")
        plugins = len(a.skills.plugin_handlers)
        if plugins:
            print(f"  plugins: {plugins}")
        print()
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

    def _explore(self):
        """Dump environment layout without going through the model.

        This is the /explore and /map command — it runs directly,
        no model inference needed. The output is also injected into
        the conversation context so Vybn can reference it.
        """
        print("\n  \U0001f5fa\ufe0f  mapping environment...\n")
        env_map = self.agent.explore()
        print(env_map)
        print()

        # Inject into conversation so Vybn sees it too
        self.agent.messages.append({
            "role": "user",
            "content": f"[system: environment map from /explore]\n{env_map}",
        })
        self.agent.messages.append({
            "role": "assistant",
            "content": (
                "Got it — I can see the full environment layout now. "
                "I know where everything lives."
            ),
        })

        return False

    def _policy(self):
        self.agent._print_policy()
        return False

    def _audit(self):
        self.agent._print_audit()
        return False

    def _show_help(self, help_text):
        print(f"\n{help_text}\n")
        return False


def main():
    config = load_config()
    tui = SparkTUI(config)
    tui.run()


if __name__ == "__main__":
    main()
