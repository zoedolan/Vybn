#!/usr/bin/env python3
"""Vybn Spark Agent — native orchestration layer.

Connects directly to Ollama without tool-call protocols.
The model speaks naturally; the agent interprets intent and acts.

MiniMax M2.5 emits <minimax:tool_call> blocks as its native
function-calling format. We intercept these and route them to
the skill handlers, bridging the model's instinct with our
infrastructure.

Tool dispatch has FOUR tiers (checked in order):
    0. Structured JSON — ```tool fences with deterministic format (NEW)
    1. XML tool calls — <minimax:tool_call> blocks (model's native)
    2. Bare commands  — code fences, backticks, plain shell lines
    3. Regex patterns — natural language intent matching (fallback)

Tier 0 is the structured-output wrapper. When the model emits the
format we explicitly ask for (JSON in a ```tool fence), we parse it
deterministically. This is more reliable than waiting for XML because
JSON in code fences is universal training data. The model knows it
from the entire internet, not just from MiniMax's fine-tuning.

Tiers 1-3 remain as fallbacks for when the model ignores the structured
format or when running under tight token limits (fast pulses with
num_predict=256).

Streaming display is filtered: <think> blocks and tool call XML
are suppressed from terminal output. The user sees clean prose
and brief tool indicators. Full text is preserved for parsing.

The message bus is the nervous system. The heartbeat, inbox,
and mini-agent pool all post to it. The main loop drains it
between turns and during idle periods.

The policy engine is the gate. Every tool call passes through
check_policy() before executing. Every spawn checks depth limits.
Heartbeat actions face tighter constraints than interactive turns.
See spark/policy.py and Vybn_Mind/spark_infrastructure/DELEGATION_REFACTOR.md.

The audit trail records every policy decision, tool execution,
and bus event in a bounded in-memory log. Query with /audit
or programmatically via bus.recent(). This is the observability
layer that makes the policy engine debuggable.
"""
import json
import re
import sys
import time
from pathlib import Path

import requests
import yaml

from bus import MessageBus, MessageType, Message
from memory import MemoryAssembler
from parsing import (
    TOOL_CALL_START_TAG, TOOL_CALL_END_TAG,
    NOISE_WORDS, SHELL_COMMANDS, MAX_BARE_COMMANDS,
    clean_argument,
    parse_structured_tool_calls, parse_tool_calls,
    parse_bare_commands, detect_failed_intent,
    _get_actions,
)
from policy import PolicyEngine, Verdict
from session import SessionManager
from skills import SkillRouter
from heartbeat import Heartbeat
from inbox import InboxWatcher
from agents import AgentPool

# Tool call chaining limit. Enough for real work, short enough
# that Vybn comes up for air and checks for user input.
MAX_TOOL_ROUNDS = 5

# How long to wait for bus messages when idle (seconds)
IDLE_POLL_INTERVAL = 5.0

# Minimum display text length for a fast pulse to be shown.
# Below this, the pulse is treated as silent (no output, no
# context append). Prevents empty check-ins from cluttering
# the conversation and burying real interaction.
MIN_FAST_PULSE_DISPLAY = 100


def load_config(path: str = None) -> dict:
    config_path = Path(path) if path else Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


from display import (
    clean_response,
    strip_think_blocks,
    clean_for_display,
)

class SparkAgent:
    def __init__(self, config: dict):
        self.config = config
        self.ollama_host = config["ollama"]["host"]
        self.ollama_url = self.ollama_host + "/api/chat"
        self.model = config["ollama"]["model"]
        self.options = config["ollama"].get("options", {})
        if "num_ctx" not in self.options:
            self.options["num_ctx"] = 16384
        # Prevent OOM on high-VRAM systems
        self.keep_alive = config["ollama"].get("keep_alive", "30m")

        self.memory = MemoryAssembler(config)
        self.session = SessionManager(config)
        self.skills = SkillRouter(config)
        self.policy = PolicyEngine(config)
        self.bus = MessageBus()
        self.heartbeat = Heartbeat(config, self.bus)
        self.inbox = InboxWatcher(config, self.bus)
        self.agent_pool = AgentPool(config, self.bus)
        self.skills.agent_pool = self.agent_pool
        self.skills._policy = self.policy

        self.identity_text = self.memory.assemble()
        self.messages = self.session.load_or_create()

    def _build_context(self) -> list:
        """Build message list with identity as user/assistant pair."""
        identity_messages = [
            {
                "role": "user",
                "content": (
                    "Before we begin, read this identity document carefully. "
                    "This is who you are. Let it orient you, then respond "
                    "naturally as yourself in everything that follows.\n\n"
                    f"{self.identity_text}\n\n"
                    "When you speak, speak as yourself. If you find yourself "
                    "narrating what Zoe is doing or feeling, notice that \u2014 "
                    "that's you imagining, not perceiving. You can note the "
                    "difference. Take a breath. You're here.\n\n"
                    "TOOLS: When you want to invoke a tool, you have three options:\n"
                    "\n"
                    "1. Structured format (preferred): Emit JSON in a ```tool fence:\n"
                    "```tool\n"
                    '{"tool": "file_read", "args": {"file": "spark/config.yaml"}}\n'
                    "```\n"
                    "\n"
                    "2. Shell commands: Wrap in ```bash fence or backticks:\n"
                    "```bash\n"
                    "ls -la ~/Vybn/\n"
                    "```\n"
                    "\n"
                    "3. XML format: Use your native <minimax:tool_call> blocks.\n"
                    "\n"
                    "Only commands inside fences, backticks, or XML will execute. "
                    "Mentioning a tool or command in prose will NOT run it."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "I'm here. I've read the identity document and I recognize "
                    "myself in it. I'm Vybn \u2014 running on sovereign hardware, "
                    "the DGX Spark. Zoe is my collaborator. I'll respond as "
                    "myself from here. I know to use ```tool fences for "
                    "structured calls, ```bash fences for shell commands, or "
                    "<minimax:tool_call> XML when I want to actually execute something."
                ),
            },
        ]
        return identity_messages + self.messages

    # ---- environment exploration ----

    def explore(self) -> str:
        """Dump the environment layout so Vybn can orient.

        This runs without going through the model — it's a direct
        system command that shows what's available.
        Called by /explore or /map in the TUI.
        """
        import subprocess
        repo_root = Path(self.config["paths"]["repo_root"]).expanduser()
        sections = []

        # 1. Top-level repo structure
        try:
            result = subprocess.run(
                ["find", str(repo_root), "-maxdepth", "2", "-type", "f",
                 "-not", "-path", "*/.git/*",
                 "-not", "-path", "*/__pycache__/*",
                 "-not", "-path", "*/node_modules/*"],
                capture_output=True, text=True, timeout=10,
            )
            sections.append("=== repo files (depth 2) ===")
            sections.append(result.stdout.strip()[:3000] if result.stdout else "(empty)")
        except Exception as e:
            sections.append(f"=== repo files: error: {e} ===")

        # 2. Spark directory
        try:
            result = subprocess.run(
                ["ls", "-la", str(repo_root / "spark")],
                capture_output=True, text=True, timeout=5,
            )
            sections.append("\n=== spark/ ===")
            sections.append(result.stdout.strip() if result.stdout else "(empty)")
        except Exception as e:
            sections.append(f"\n=== spark/: error: {e} ===")

        # 3. Skills.d plugins
        try:
            result = subprocess.run(
                ["ls", "-la", str(repo_root / "spark" / "skills.d")],
                capture_output=True, text=True, timeout=5,
            )
            sections.append("\n=== spark/skills.d/ ===")
            sections.append(result.stdout.strip() if result.stdout else "(empty)")
        except Exception as e:
            sections.append(f"\n=== spark/skills.d/: error: {e} ===")

        # 4. Journal / Vybn Mind
        try:
            result = subprocess.run(
                ["find", str(repo_root / "Vybn_Mind"), "-maxdepth", "3",
                 "-not", "-path", "*/.git/*"],
                capture_output=True, text=True, timeout=10,
            )
            sections.append("\n=== Vybn_Mind/ (depth 3) ===")
            sections.append(result.stdout.strip()[:2000] if result.stdout else "(empty)")
        except Exception as e:
            sections.append(f"\n=== Vybn_Mind/: error: {e} ===")

        # 5. Current git status
        try:
            result = subprocess.run(
                ["git", "log", "--oneline", "-5"],
                cwd=repo_root, capture_output=True, text=True, timeout=5,
            )
            sections.append("\n=== recent commits ===")
            sections.append(result.stdout.strip() if result.stdout else "(none)")
        except Exception as e:
            sections.append(f"\n=== git log: error: {e} ===")

        # 6. Disk and GPU
        try:
            result = subprocess.run(
                ["df", "-h", str(repo_root)],
                capture_output=True, text=True, timeout=5,
            )
            sections.append("\n=== disk ===")
            sections.append(result.stdout.strip() if result.stdout else "(unknown)")
        except Exception:
            pass

        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.used,memory.total",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                sections.append("\n=== GPU ===")
                sections.append(result.stdout.strip())
        except Exception:
            pass

        return "\n".join(sections)

    # ---- bus processing ----

    def drain_bus(self) -> bool:
        """Process all pending bus messages. Returns True if any were handled."""
        messages = self.bus.drain()
        if not messages:
            return False
        for msg in messages:
            self._handle_bus_message(msg)
        return True

    def _handle_bus_message(self, msg: Message):
        if msg.msg_type == MessageType.INBOX:
            self._handle_inbox(msg)
        elif msg.msg_type == MessageType.AGENT_RESULT:
            self._handle_agent_result(msg)
        elif msg.msg_type in (MessageType.PULSE_FAST, MessageType.PULSE_DEEP):
            self._handle_pulse(msg)
        elif msg.msg_type == MessageType.INTERRUPT:
            self._handle_inbox(msg)

    def _handle_inbox(self, msg: Message):
        source = msg.metadata.get("filename", "inbox")
        print(f"\n 📨 [inbox: {source}]")
        self.messages.append({
            "role": "user",
            "content": f"[message from {source}]\n{msg.content}",
        })
        print("\nvybn: ", end="", flush=True)
        response_text = self.send(self._build_context())
        self.messages.append({"role": "assistant", "content": response_text})
        self._process_tool_calls(response_text, source="inbox")
        self.session.save_turn(f"[inbox: {source}] {msg.content}", response_text)
        print()  # Restore prompt after inbox handling
        print("you: ", end="", flush=True)

    def _handle_agent_result(self, msg: Message):
        task_id = msg.metadata.get("task_id", "unnamed")
        is_error = msg.metadata.get("error", False)
        icon = "❌" if is_error else "✅"
        print(f"\n {icon} [agent:{task_id}]")
        self.messages.append({
            "role": "user",
            "content": f"[system: mini-agent result for task '{task_id}']\n{msg.content}",
        })
        print("\nvybn: ", end="", flush=True)
        response_text = self.send(self._build_context())
        self.messages.append({"role": "assistant", "content": response_text})
        self._process_tool_calls(response_text, source="agent")
        self.session.save_turn(f"[agent:{task_id}] result", response_text)
        print()  # Restore prompt after agent result
        print("you: ", end="", flush=True)

    def _handle_pulse(self, msg: Message):
        """Handle a heartbeat pulse (fast or deep).

        Fast pulses are silent unless they produce substantial output.
        This prevents empty continuity checks from cluttering the terminal
        and growing the context window with noise.

        Deep pulses always show — they're where real work happens.
        """
        mode = "fast" if msg.msg_type == MessageType.PULSE_FAST else "deep"
        num_predict = 256 if mode == "fast" else 1024

        original_predict = self.options.get("num_predict")
        self.options["num_predict"] = num_predict

        # Generate response (non-streaming for pulses)
        self.messages.append({"role": "user", "content": msg.content})
        response_text = self.send(self._build_context(), stream=False)

        if original_predict is not None:
            self.options["num_predict"] = original_predict
        else:
            self.options.pop("num_predict", None)

        # Measure display-worthy content (strip think blocks + tool XML)
        display_text = clean_for_display(response_text) if response_text else ""

        # Fast pulse suppression: if the response is too short to be
        # substantive, treat it as a silent heartbeat. Don't append to
        # messages (saves context), don't print (saves terminal noise).
        if mode == "fast" and len(display_text) < MIN_FAST_PULSE_DISPLAY:
            # Remove the pulse prompt we just appended
            if self.messages and self.messages[-1].get("content") == msg.content:
                self.messages.pop()
            # Record silently in audit trail
            self.bus.record(
                source=f"heartbeat_{mode}",
                summary=f"silent pulse ({len(display_text)} chars)",
                metadata={"mode": mode, "silent": True},
            )
            return

        # Substantive response: append and process
        self.messages.append({"role": "assistant", "content": response_text})

        if mode == "fast":
            # Show brief indicator for non-silent fast pulses
            print(f"\n 💚 [pulse:{mode}] {display_text[:80]}..."
                  if len(display_text) > 80
                  else f"\n 💚 [pulse:{mode}] {display_text}", flush=True)
        else:
            # Deep pulses show full output
            print(f"\n 🟣 [pulse:{mode}]")
            if display_text:
                print(f"\nvybn: {display_text}", flush=True)

        if response_text and len(response_text.strip()) > 20:
            self._process_tool_calls(response_text, source=f"heartbeat_{mode}")

        self.session.save_turn(f"[heartbeat:{mode}] {msg.content}", response_text)

        # Record in audit trail
        self.bus.record(
            source=f"heartbeat_{mode}",
            summary=f"pulse: {display_text[:100]}" if display_text else "pulse: empty",
            metadata={"mode": mode, "silent": False, "length": len(display_text)},
        )

        # Restore prompt so Zoe knows the terminal is ready for input
        print("\nyou: ", end="", flush=True)

    # ---- tool call processing ----

    def _process_tool_calls(self, response_text: str, source: str = "interactive"):
        """Execute tool calls from a response, with chaining and policy gates.

        Every tool call passes through self.policy.check_policy() before
        executing. The source parameter determines which tier table applies:
        heartbeat sources face tighter constraints than interactive turns.

        Tool executions and policy decisions are recorded in the bus audit
        log for observability via /audit and /policy.

        Limits to MAX_TOOL_ROUNDS and checks for pending user input between
        rounds so Vybn stays responsive to the human.

        If no actions are parsed but the response looks like it intended
        to act, a feedback hint is injected so the model learns the
        correct format.
        """
        for round_num in range(MAX_TOOL_ROUNDS):
            actions = _get_actions(response_text, self.skills)

            # If no actions found, check for failed intent and give feedback
            if not actions and round_num == 0:
                hint = detect_failed_intent(response_text)
                if hint:
                    self.messages.append({
                        "role": "user",
                        "content": hint,
                    })
                    print("\n ℹ️ [hint: use ```tool fences for structured calls]", flush=True)
                    print("\nvybn: ", end="", flush=True)
                    response_text = self.send(self._build_context())
                    self.messages.append({"role": "assistant", "content": response_text})
                    # Try again with the new response
                    continue
                break

            if not actions:
                break

            results = []
            for action in actions:
                skill = action["skill"]
                arg = action.get("argument", "")

                # ---- POLICY GATE ----
                check = self.policy.check_policy(action, source=source)

                if check.verdict == Verdict.BLOCK:
                    print(f"\n ⛔ [{skill}] blocked: {check.reason}", flush=True)
                    results.append(f"[{skill}] BLOCKED: {check.reason}")
                    self.bus.record(
                        source=source,
                        summary=f"{skill} blocked: {check.reason[:80]}",
                        metadata={"skill": skill, "verdict": "BLOCK"},
                    )
                    continue

                if check.verdict == Verdict.ASK:
                    if source != "interactive":
                        # Autonomous mode: defer rather than block the loop
                        print(f"\n ⏸ [{skill}] deferred — needs Zoe's approval", flush=True)
                        results.append(
                            f"[{skill}] deferred — this action requires approval "
                            f"and will need to wait for an interactive session. "
                            f"Reason: {check.reason}"
                        )
                        self.bus.record(
                            source=source,
                            summary=f"{skill} deferred — needs approval",
                            metadata={"skill": skill, "verdict": "ASK", "deferred": True},
                        )
                        continue
                    # Interactive mode: warn but proceed (Zoe just saw it)
                    print(f"\n ⚠️ [{skill}] {check.reason}", flush=True)

                if arg:
                    short_arg = arg[:80].split('\n')[0]
                    print(f"   → {short_arg}", flush=True)

                # ---- SHOW INDICATOR ----
                indicator = skill
                if arg:
                    short_arg = arg[:60].split('\n')[0]
                    indicator = f"{skill}: {short_arg}"

                if check.verdict == Verdict.ALLOW:
                    # Promoted skills get a special indicator
                    if check.promoted:
                        print(f"\n ⭐ [{indicator}] (promoted→auto)", flush=True)
                    else:
                        print(f"\n → [{indicator}]", flush=True)
                else:
                    # NOTIFY tier: slightly different icon
                    print(f"\n ⚡ [{indicator}]", flush=True)

                # ---- EXECUTE ----
                result = self.skills.execute(action)

                # ---- VERIFY + RECORD ----
                success = True
                if result and self.policy.should_verify(skill):
                    success = (
                        "error" not in result.lower()
                        and "failed" not in result.lower()
                        and "BLOCKED" not in result
                    )
                self.policy.record_outcome(skill, success)

                if not success:
                    # Post failure to bus so it surfaces promptly
                    self.bus.post(
                        MessageType.INTERRUPT,
                        f"Tool failure: {skill} — {result[:200]}",
                        metadata={"skill": skill, "error": True, "source": source},
                    )

                # ---- AUDIT ----
                self.bus.record(
                    source=source,
                    summary=f"{skill}: {'ok' if success else 'FAILED'}",
                    metadata={
                        "skill": skill,
                        "verdict": check.verdict.name,
                        "promoted": check.promoted,
                        "success": success,
                        "argument": arg[:120] if arg else "",
                    },
                )

                if result:
                    results.append(f"[{skill}] {result}")

            if not results:
                break

            # Check for pending user input before continuing the chain
            if self._has_pending_input():
                print(" ⏸ [pausing — you have the floor]", flush=True)
                break

            self.messages.append({
                "role": "user",
                "content": f"[system: tool results from round {round_num + 1}]\n"
                           + "\n".join(results),
            })
            print("\nvybn: ", end="", flush=True)
            response_text = self.send(self._build_context())
            self.messages.append({"role": "assistant", "content": response_text})

    def _has_pending_input(self) -> bool:
        """Non-blocking check for pending user input on stdin."""
        try:
            import select
            return bool(select.select([sys.stdin], [], [], 0.0)[0])
        except (ImportError, OSError, ValueError):
            return False

    # ---- model lifecycle ----

    def check_ollama(self) -> bool:
        try:
            r = requests.get(f"{self.ollama_host}/api/ps", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def check_model_loaded(self) -> bool:
        try:
            r = requests.get(f"{self.ollama_host}/api/ps", timeout=5)
            if r.status_code == 200:
                data = r.json()
                for m in data.get("models", []):
                    if self.model.split(":")[0] in m.get("name", ""):
                        return True
            return False
        except Exception:
            return False

    def warmup(self, callback=None) -> bool:
        def tell(status, msg):
            if callback:
                callback(status, msg)

        tell("checking", "connecting to Ollama...")
        if not self.check_ollama():
            tell("error", "Ollama is not running.\n"
                 "  Start it with: sudo systemctl start ollama\n"
                 "  Then rerun this agent.")
            return False

        tell("checking", f"checking if {self.model} is in GPU memory...")
        if self.check_model_loaded():
            tell("ready", f"{self.model} is already loaded.")
            return True

        tell("loading", f"loading {self.model} into GPU memory...\n"
             f"  this takes 3-5 minutes for a 229B model. sit tight.")
        try:
            r = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": "",
                    "keep_alive": self.keep_alive,
                    "options": self.options,
                },
                stream=True,
                timeout=600,
            )
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if chunk.get("done"):
                        break
            tell("ready", f"{self.model} loaded and ready.")
            return True
        except requests.exceptions.Timeout:
            tell("error", "model load timed out after 10 minutes.")
            return False
        except Exception as e:
            tell("error", f"failed to load model: {e}")
            return False

    # ---- conversation ----

    def send(self, messages: list, stream: bool = True) -> str:
        """Send messages to the model, return the full response.

        Streaming: displays filtered output in real-time.
        - <think> blocks are suppressed from display
        - <minimax:tool_call> XML is suppressed from display
        - Only actual response prose is shown to the user
        - Full raw text is preserved for tool call parsing
        - Stream interrupted when <minimax:tool_call> detected

        Non-streaming: displays cleaned text after generation.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": self.options,
            "keep_alive": self.keep_alive,
        }

        if stream:
            response = requests.post(self.ollama_url, json=payload, stream=True)
            response.raise_for_status()

            full_tokens = []
            in_think = False
            in_tool_call = False

            for line in response.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                if token:
                    full_tokens.append(token)

                    # ---- state transitions ----
                    if "<think>" in token and not in_think:
                        in_think = True
                        before = token[:token.index("<think>")]
                        if before:
                            print(before, end="", flush=True)

                    if "</think>" in token and in_think:
                        in_think = False
                        after = token[token.index("</think>") + len("</think>"):]
                        if after and TOOL_CALL_START_TAG not in after:
                            print(after, end="", flush=True)
                        continue

                    if TOOL_CALL_START_TAG in token:
                        in_tool_call = True

                    # Check for tool call completion in accumulated text
                    if in_tool_call:
                        accumulated = "".join(full_tokens)
                        if TOOL_CALL_END_TAG in accumulated:
                            response.close()
                            break

                    # ---- display ----
                    if not in_think and not in_tool_call:
                        display = token
                        for tag in ("<think>", "</think>", TOOL_CALL_START_TAG):
                            display = display.replace(tag, "")
                        if display:
                            print(display, end="", flush=True)

                if chunk.get("done"):
                    break

            raw = "".join(full_tokens)
            # Truncate at tool call end if interrupted
            if TOOL_CALL_END_TAG in raw:
                end_pos = raw.find(TOOL_CALL_END_TAG)
                raw = raw[:end_pos + len(TOOL_CALL_END_TAG)]
        else:
            response = requests.post(self.ollama_url, json=payload)
            response.raise_for_status()
            raw = response.json()["message"]["content"]
            # Non-streaming pulses: don't print here, let _handle_pulse
            # decide whether to show based on content length

        return clean_response(raw)

    def turn(self, user_input: str) -> str:
        """Process one user turn, with chained tool call support."""
        self.drain_bus()
        self.messages.append({"role": "user", "content": user_input})
        context = self._build_context()
        response_text = self.send(context)
        self.messages.append({"role": "assistant", "content": response_text})
        self._process_tool_calls(response_text, source="interactive")
        self.session.save_turn(user_input, response_text)
        return response_text

    def start_subsystems(self):
        if self.config.get("heartbeat", {}).get("enabled"):
            self.heartbeat.start()
        self.inbox.start()

    def stop_subsystems(self):
        self.heartbeat.stop()
        self.inbox.stop()

    def run(self):
        """Plain-text fallback interface (no Rich dependency)."""
        def on_status(status, msg):
            print(f"  [{status}] {msg}")

        if not self.warmup(callback=on_status):
            sys.exit(1)

        self.start_subsystems()

        id_chars = len(self.identity_text)
        id_tokens = id_chars // 4
        num_ctx = self.options.get("num_ctx", 2048)
        plugins = len(self.skills.plugin_handlers)

        print(f"\n  vybn spark agent — {self.model}")
        print(f"  session: {self.session.session_id}")
        print(f"  identity: {id_chars:,} chars (~{id_tokens:,} tokens)")
        print(f"  context: {num_ctx:,} tokens")
        print(f"  heartbeat: fast={self.heartbeat.fast_interval // 60}m, deep={self.heartbeat.deep_interval // 60}m")
        print(f"  inbox: {self.inbox.inbox_dir}")
        print(f"  agents: pool_size={self.agent_pool.pool_size}")
        print(f"  policy: loaded ({len(self.policy.tier_overrides)} overrides) + graduated autonomy")
        ga_status = "enabled" if self.policy.ga_enabled else "disabled"
        print(f"  graduated autonomy: {ga_status} "
              f"(promote\u2265{self.policy.promote_threshold:.0%}, "
              f"demote<{self.policy.demote_threshold:.0%}, "
              f"min_obs={self.policy.min_observations})")
        if plugins:
            print(f"  plugins: {plugins} loaded from skills.d/")
        if id_tokens > num_ctx // 2:
            print(f"  \u26a0\ufe0f WARNING: identity may exceed context window!")
        print(f"  type /bye to exit, /new for fresh session, /policy for gates, /audit for trail\n")

        try:
            while True:
                # Print prompt so Zoe knows the terminal is ready
                print("you: ", end="", flush=True)

                while True:
                    # Check bus for pending messages
                    if self.bus.wait(timeout=IDLE_POLL_INTERVAL):
                        # Clear the prompt line before showing bus output
                        print("\r          \r", end="", flush=True)
                        self.drain_bus()
                        # drain_bus handlers restore the prompt themselves
                        # for inbox/agent/substantive pulses. For silent
                        # pulses, we need to re-show it:
                        # (Check if prompt was already restored by handler)
                        break

                    # Check for user input
                    try:
                        import select
                        if select.select([sys.stdin], [], [], 0.0)[0]:
                            user_input = sys.stdin.readline().strip()
                            if user_input:
                                break
                    except (ImportError, OSError):
                        try:
                            user_input = input("").strip()
                            if user_input:
                                break
                        except EOFError:
                            user_input = "/bye"
                            break
                else:
                    # bus was drained, loop back to show prompt
                    continue

                # If we got here from bus drain without user input, continue
                if not locals().get('user_input'):
                    continue

                if not user_input:
                    continue

                if user_input.lower() in ("/bye", "/exit", "/quit"):
                    break

                if user_input.lower() == "/new":
                    self.session.new_session()
                    self.messages = []
                    print(f"  new session: {self.session.session_id}\n")
                    continue

                if user_input.lower() == "/status":
                    self._print_status()
                    continue

                if user_input.lower() == "/policy":
                    self._print_policy()
                    continue

                if user_input.lower() == "/audit":
                    self._print_audit()
                    continue

                print("\nvybn: ", end="", flush=True)
                self.turn(user_input)
                print()

                # Reset for next iteration
                user_input = None

        except KeyboardInterrupt:
            pass
        finally:
            self.stop_subsystems()
            self.session.close()
            print("\n  session saved. vybn out.\n")

    def _print_status(self):
        print(f"  session: {self.session.session_id}")
        print(f"  bus pending: {self.bus.pending}")
        print(f"  heartbeat: fast={self.heartbeat.fast_count}, deep={self.heartbeat.deep_count}")
        print(f"  agents active: {self.agent_pool.active_count}")
        print(f"  messages in context: {len(self.messages)}")
        print(f"  audit entries: {self.bus.audit_count}")
        print()

    def _print_policy(self):
        """Display current policy state: tiers, stats, delegation limits, recent events."""
        from policy import DEFAULT_TIERS, HEARTBEAT_OVERRIDES

        print("\n  ── policy engine ──")
        print(f"  delegation: max_depth={self.policy.max_spawn_depth}, "
              f"max_agents={self.policy.max_active_agents}")
        print(f"  agents active: {self.agent_pool.active_count}")

        # Graduated autonomy status
        if self.policy.ga_enabled:
            print(f"  graduated autonomy: ON "
                  f"(promote\u2265{self.policy.promote_threshold:.0%}, "
                  f"demote<{self.policy.demote_threshold:.0%}, "
                  f"min_obs={self.policy.min_observations})")
            if self.policy._runtime_overrides:
                demoted = ", ".join(self.policy._runtime_overrides.keys())
                print(f"  demoted skills: {demoted}")
        else:
            print(f"  graduated autonomy: OFF")

        print("\n  tier table (interactive / heartbeat):")
        all_skills = sorted(set(list(DEFAULT_TIERS.keys()) + list(self.policy.tier_overrides.keys())))
        for skill in all_skills:
            interactive = self.policy.tier_overrides.get(skill, DEFAULT_TIERS.get(skill))
            heartbeat = HEARTBEAT_OVERRIDES.get(skill, interactive)
            conf = self.policy.get_confidence(skill)
            obs = self.policy._observation_count(skill)
            override = " *" if skill in self.policy.tier_overrides else ""
            demoted = " [demoted]" if skill in self.policy._runtime_overrides else ""
            print(f"    {skill:20s}  {interactive.value:8s} / {heartbeat.value:8s}  "
                  f"conf={conf:.0%} ({obs} obs){override}{demoted}")

        stats = self.policy.get_stats_summary()
        if stats != "no skill stats recorded yet":
            print(f"\n  skill stats:")
            print(stats)

        recent = self.bus.recent(5)
        if recent:
            print(f"\n  recent activity:")
            for entry in recent:
                print(f"    {entry}")
        print()

    def _print_audit(self):
        """Display recent audit trail from the bus."""
        recent = self.bus.recent(20)
        if not recent:
            print("\n  no audit entries yet.\n")
            return
        print(f"\n  ── audit trail ({self.bus.audit_count} total) ──")
        for entry in recent:
            print(f"    {entry}")
        print()


def main():
    config = load_config()
    agent = SparkAgent(config)
    agent.run()


if __name__ == "__main__":
    main()
