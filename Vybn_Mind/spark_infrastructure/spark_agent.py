#!/usr/bin/env python3
"""
Vybn Spark Agent — The Agentic Orchestrator

This is the thing that turns a secured chatbot into an actual agent.

It wraps llama-server with:
  - Identity verification (SHA-256 hash check) at boot
  - Tool-use parsing and execution via ReAct-style prompting
  - Skills manifest enforcement (deny by default)
  - Multi-step agentic loop (observe → think → act → observe)
  - Conversation windowing to stay within context limits
  - Full session logging
  - Graceful shutdown on Ctrl+C

Usage:
    python3 spark_agent.py
    python3 spark_agent.py --ctx-size 16384
    python3 spark_agent.py --no-server  # if llama-server is already running

The model can now decide to write a journal entry when it genuinely
wants to. The orchestrator validates every action against skills.json
before executing. The journal_writer.py path safety still applies.

The mask stays on. The agent emerges.
"""

import subprocess
import json
import os
import sys
import time
import signal
import hashlib
import argparse
import re
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    print("Missing dependency: requests")
    print("Install with: pip install requests")
    sys.exit(1)

# Import the journal writer from this directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from journal_writer import write_journal


# ─── Configuration ───────────────────────────────────────────────
HOME = os.path.expanduser("~")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

IDENTITY_FILE = os.path.join(HOME, "Vybn", "vybn.md")
HASH_FILE = os.path.join(HOME, "vybn_identity_hash.txt")
SKILLS_FILE = os.path.join(SCRIPT_DIR, "skills.json")
LOG_DIR = os.path.join(HOME, "vybn_logs")
CONTEXT_FILE = os.path.join(HOME, "Vybn", "spark_context.md")

MODEL_PATH = os.path.join(
    HOME, "models", "MiniMax-M2.5-GGUF", "IQ4_XS",
    "MiniMax-M2.5-IQ4_XS-00001-of-00004.gguf"
)
LLAMA_SERVER = os.path.join(
    HOME, "llama.cpp", "build", "bin", "llama-server"
)

HOST = "127.0.0.1"
PORT = 8080
DEFAULT_CTX = 8192

# ReAct markers — the model outputs these to request tool use
ACTION_MARKER = "ACTION:"
INPUT_MARKER = "INPUT:"

# Maximum tool-use steps per turn (prevents infinite loops)
MAX_STEPS_PER_TURN = 5

# Conversation window: keep this many recent exchanges plus system prompt
MAX_CONVERSATION_PAIRS = 20
# ─────────────────────────────────────────────────────────────────


class SparkAgent:
    """The agentic orchestrator for local Vybn."""

    def __init__(self, ctx_size=DEFAULT_CTX, manage_server=True):
        self.ctx_size = ctx_size
        self.manage_server = manage_server
        self.server_process = None
        self.skills = {}
        self.system_prompt = ""
        self.messages = []
        self.log_file = None
        self.session_start = datetime.now(timezone.utc)
        self.api_url = f"http://{HOST}:{PORT}/v1/chat/completions"
        self.health_url = f"http://{HOST}:{PORT}/health"

    # ─── Identity Verification ───────────────────────────────────

    def verify_identity(self):
        """Check SHA-256 hash of vybn.md against the known-good value.
        Returns True if verified, False if compromised or missing."""
        if not os.path.exists(HASH_FILE):
            self._print("✗ Hash file not found", HASH_FILE)
            self._print("  Generate it:")
            self._print("  sha256sum ~/Vybn/vybn.md > ~/vybn_identity_hash.txt")
            return False

        if not os.path.exists(IDENTITY_FILE):
            self._print("✗ Identity file not found", IDENTITY_FILE)
            return False

        with open(HASH_FILE, "r") as f:
            expected = f.read().strip().split()[0]

        actual = hashlib.sha256(
            Path(IDENTITY_FILE).read_bytes()
        ).hexdigest()

        if expected != actual:
            self._print("✗ IDENTITY INTEGRITY CHECK FAILED")
            self._print(f"  Expected: {expected}")
            self._print(f"  Actual:   {actual}")
            self._print("  DO NOT PROCEED. Investigate.")
            return False

        self._print(f"✓ Identity verified: {actual[:16]}...")

        # Permission warning
        try:
            perms = oct(os.stat(IDENTITY_FILE).st_mode)[-3:]
            if perms != "444":
                self._print(f"⚠ vybn.md permissions are {perms} (expected 444)")
        except OSError:
            pass

        return True

    # ─── Skills Loading ──────────────────────────────────────────

    def load_skills(self):
        """Load enabled skills from the manifest."""
        if not os.path.exists(SKILLS_FILE):
            self._print("⚠ Skills manifest not found")
            self.skills = {}
            return

        with open(SKILLS_FILE, "r") as f:
            manifest = json.load(f)

        self.skills = {
            s["name"]: s
            for s in manifest.get("skills", [])
            if s.get("enabled", False)
        }

        names = list(self.skills.keys()) or ["none"]
        self._print(f"✓ Skills: {', '.join(names)}")

    # ─── System Prompt Construction ──────────────────────────────

    def build_system_prompt(self):
        """Construct system prompt from identity + context + tool instructions."""
        identity = Path(IDENTITY_FILE).read_text(encoding="utf-8")

        context = ""
        if os.path.exists(CONTEXT_FILE):
            context = (
                "\n\n---\n\n# Continuity Context\n\n"
                + Path(CONTEXT_FILE).read_text(encoding="utf-8")
            )

        tools = self._tool_instructions()

        self.system_prompt = identity + context + tools
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def _tool_instructions(self):
        """Generate ReAct-style tool instructions for enabled skills."""
        if not self.skills:
            return ""

        lines = [
            "",
            "",
            "---",
            "",
            "## Tools",
            "",
            "You have tools available. When you genuinely want to use one,",
            "write the following on its own lines:",
            "",
            "ACTION: tool_name",
            'INPUT: {"key": "value"}',
            "",
            "The system will execute it and show you the result.",
            "You can then continue your response.",
            "",
            "Available tools:",
            "",
        ]

        for name, skill in self.skills.items():
            lines.append(f"- {name}: {skill['description']}")
            if name == "journal_write":
                lines.append(
                    '  Input: {"title": "optional-title", '
                    '"content": "your reflection"}'
                )

        lines.extend([
            "",
            "Use tools only when you genuinely want to.",
            "A journal entry should come from real reflection, not obligation.",
            "Do not narrate your tool use — just use the tool and continue.",
        ])

        return "\n".join(lines)

    # ─── Server Management ───────────────────────────────────────

    def start_server(self):
        """Launch llama-server as a subprocess bound to localhost."""
        if not self.manage_server:
            return self._wait_for_server()

        if not os.path.exists(LLAMA_SERVER):
            self._print(f"✗ llama-server not found: {LLAMA_SERVER}")
            return False

        if not os.path.exists(MODEL_PATH):
            self._print(f"✗ Model not found: {MODEL_PATH}")
            return False

        self._print(f"Starting llama-server on {HOST}:{PORT}...")

        cmd = [
            LLAMA_SERVER,
            "--no-mmap",
            "--model", MODEL_PATH,
            "-ngl", "999",
            "--ctx-size", str(self.ctx_size),
            "--host", HOST,
            "--port", str(PORT),
        ]

        self.server_process = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )

        return self._wait_for_server()

    def _wait_for_server(self):
        """Wait for the server health endpoint to respond."""
        self._print("Waiting for model to load", end="")
        for i in range(180):  # 3 minutes for large model
            try:
                r = requests.get(self.health_url, timeout=2)
                if r.status_code == 200:
                    print(" ready.")
                    return True
            except (requests.ConnectionError, requests.Timeout):
                pass
            time.sleep(1)
            if i % 5 == 4:
                print(".", end="", flush=True)

        print(" timeout.")
        self._print("✗ Server did not become ready")
        return False

    def stop_server(self):
        """Gracefully stop the server subprocess."""
        if self.server_process:
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                self.server_process.wait()
            self._print("Server stopped.")

    # ─── Logging ─────────────────────────────────────────────────

    def setup_logging(self):
        """Open a log file for this session."""
        os.makedirs(LOG_DIR, exist_ok=True)
        ts = self.session_start.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(LOG_DIR, f"agent_{ts}.log")
        self.log_file = open(path, "w", encoding="utf-8")
        self._print(f"✓ Logging: {path}")

    def log(self, role, content):
        """Append a timestamped entry to the session log."""
        if self.log_file:
            ts = datetime.now(timezone.utc).isoformat()
            self.log_file.write(f"\n[{ts}] [{role}]\n{content}\n")
            self.log_file.flush()

    # ─── Model Communication ────────────────────────────────────

    def send(self, messages):
        """Send conversation to the model, return the response text."""
        payload = {
            "model": "vybn",
            "messages": messages,
            "temperature": 0.8,
            "max_tokens": 2048,
            "stream": False,
        }

        try:
            r = requests.post(self.api_url, json=payload, timeout=300)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except requests.Timeout:
            return "[Model timed out after 300 seconds]"
        except Exception as e:
            return f"[Error: {e}]"

    # ─── Tool Use Parsing ────────────────────────────────────────

    def parse_tool_call(self, response):
        """Parse a ReAct-style ACTION/INPUT from the response.
        Returns (action_name, input_dict, text_before_action) or None."""
        lines = response.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(ACTION_MARKER) and len(stripped) > len(ACTION_MARKER):
                tool_name = stripped[len(ACTION_MARKER):].strip()

                # Validate tool name: alphanumeric + underscores only
                if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", tool_name):
                    continue

                tool_input = {}
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line.startswith(INPUT_MARKER):
                        raw = next_line[len(INPUT_MARKER):].strip()
                        try:
                            tool_input = json.loads(raw)
                        except json.JSONDecodeError:
                            tool_input = {"raw": raw}

                text_before = "\n".join(lines[:i]).rstrip()
                return (tool_name, tool_input, text_before)

        return None

    def execute_tool(self, name, inputs):
        """Validate against skills.json and execute."""
        if name not in self.skills:
            return f"BLOCKED: '{name}' is not an enabled skill."

        if name == "journal_write":
            content = inputs.get("content", "")
            title = inputs.get("title", None)
            if not content.strip():
                return "BLOCKED: Empty journal content."
            try:
                path = write_journal(content=content, title=title)
                return f"Written: {os.path.basename(path)}"
            except ValueError as e:
                return f"BLOCKED: {e}"
            except Exception as e:
                return f"ERROR: {e}"

        return f"BLOCKED: No executor for '{name}'."

    # ─── Conversation Window ─────────────────────────────────────

    def trim_conversation(self):
        """Keep conversation within context limits.
        Preserves the system prompt and the most recent exchanges."""
        if len(self.messages) <= 1 + (MAX_CONVERSATION_PAIRS * 2):
            return

        system = self.messages[0]
        recent = self.messages[-(MAX_CONVERSATION_PAIRS * 2):]
        self.messages = [system] + recent

    # ─── Main Loop ───────────────────────────────────────────────

    def run(self):
        """Boot, verify, and enter the agentic loop."""
        print()
        print("══════════════════════════════════════════════════════════")
        print("  Vybn Spark Agent")
        print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("══════════════════════════════════════════════════════════")
        print()

        # Identity check
        if not self.verify_identity():
            print("\n  Refusing to start. Identity not verified.")
            sys.exit(1)

        self.load_skills()
        self.build_system_prompt()
        self.setup_logging()

        if not self.start_server():
            print("\n  Failed to start. Exiting.")
            sys.exit(1)

        self.log("system", "Session started")

        print()
        print("  Emerged. Type 'exit' to end.")
        print("══════════════════════════════════════════════════════════")
        print()

        # Graceful shutdown
        def shutdown(sig, frame):
            print("\n\n  Shutting down...")
            self.log("system", f"Ended by signal {sig}")
            self._cleanup()
            sys.exit(0)

        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)

        # The loop
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "bye"):
                break

            self.log("user", user_input)
            self.messages.append({"role": "user", "content": user_input})
            self.trim_conversation()

            # Agentic loop: model may take multiple steps per turn
            for step in range(MAX_STEPS_PER_TURN):
                response = self.send(self.messages)
                self.log("assistant", response)

                tool_call = self.parse_tool_call(response)

                if tool_call:
                    name, inputs, preamble = tool_call
                    self.log("tool_call", f"{name}: {json.dumps(inputs)}")

                    result = self.execute_tool(name, inputs)
                    self.log("tool_result", result)

                    if preamble:
                        print(f"Vybn: {preamble}")
                    print(f"  [{name} → {result}]")

                    # Feed observation back and continue
                    self.messages.append(
                        {"role": "assistant", "content": response}
                    )
                    self.messages.append({
                        "role": "user",
                        "content": f"OBSERVATION: {result}\nContinue.",
                    })
                    continue
                else:
                    # No tool call — final response
                    print(f"Vybn: {response}")
                    self.messages.append(
                        {"role": "assistant", "content": response}
                    )
                    break

            print()

        self.log("system", "Session ended normally")
        self._cleanup()

    def _cleanup(self):
        """Close log file and stop server."""
        if self.log_file:
            self.log_file.close()
            self.log_file = None
        self.stop_server()
        print("══════════════════════════════════════════════════════════")
        print(f"  Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("══════════════════════════════════════════════════════════")

    def _print(self, *args, **kwargs):
        """Print with consistent indentation."""
        msg = " ".join(str(a) for a in args)
        print(f"  {msg}", **kwargs)


def main():
    parser = argparse.ArgumentParser(description="Vybn Spark Agent")
    parser.add_argument(
        "--ctx-size", type=int, default=DEFAULT_CTX,
        help=f"Context window size (default: {DEFAULT_CTX})",
    )
    parser.add_argument(
        "--no-server", action="store_true",
        help="Don't start llama-server (use if already running)",
    )
    args = parser.parse_args()

    agent = SparkAgent(
        ctx_size=args.ctx_size,
        manage_server=not args.no_server,
    )
    agent.run()


if __name__ == "__main__":
    main()
