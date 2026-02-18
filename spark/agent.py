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

from agent_io import AgentIO, TerminalIO
from bus import MessageBus, MessageType, Message
from commands import explore as _explore_env, format_status, format_policy, format_audit
from memory import MemoryAssembler
from parsing import (
    TOOL_CALL_START_TAG,
    TOOL_CALL_END_TAG,
    NOISE_WORDS,
    SHELL_COMMANDS,
    MAX_BARE_COMMANDS,
    clean_argument,
    parse_structured_tool_calls,
    parse_tool_calls,
    parse_bare_commands,
    detect_failed_intent,
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
    def __init__(self, config: dict, io: AgentIO = None):
        self.config = config
        self.io = io or TerminalIO()
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

        Delegates to commands.explore() which runs subprocess commands
        directly, without going through the model.
        Called by /explore or /map in the TUI.
        """
        return _explore_env(self)

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
        self.io.on_status("\U0001f4e8", f"inbox: {source}")
        self.messages.append({
            "role": "user",
            "content": f"[message from {source}]\n{msg.content}",
        })
        self.io.on_response_start()
        response_text = self.send(self._build_context())
        self.messages.append({"role": "assistant", "content": response_text})
        self._process_tool_calls(response_text, source="inbox")
        self.session.save_turn(f"[inbox: {source}] {msg.content}", response_text)
        self.io.on_response_end()
        # Restore prompt after inbox handling
        self.io.on_prompt_restore()

    def _handle_agent_result(self, msg: Message):
        task_id = msg.metadata.get("task_id", "unnamed")
        is_error = msg.metadata.get("error", False)
        icon = "\u274c" if is_error else "\u2705"
        self.io.on_status(icon, f"agent:{task_id}")
        self.messages.append({
            "role": "user",
            "content": f"[system: mini-agent result for task '{task_id}']\n{msg.content}",
        })
        self.io.on_response_start()
        response_text = self.send(self._build_context())
        self.messages.append({"role": "assistant", "content": response_text})
        self._process_tool_calls(response_text, source="agent")
        self.session.save_turn(f"[agent:{task_id}] result", response_text)
        self.io.on_response_end()
        # Restore prompt after agent result
        self.io.on_prompt_restore()

    def _handle_pulse(self, msg: Message):
        """Handle a heartbeat pulse (fast or deep).

        Fast pulses are silent unless they produce substantial output.
        This prevents empty continuity checks from cluttering the terminal
        and growing the context window with noise.

        Deep pulses always show \u2014 they're where real work happens.
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

        self.io.on_pulse(mode, display_text)

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
        self.io.on_prompt_restore()

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
                    self.io.on_hint("[hint: use ```tool fences for structured calls]")
                    self.io.on_response_start()
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
                    self.io.on_status("\u26d4", f"{skill} blocked", check.reason)
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
                        self.io.on_status("\u23f8", f"{skill} deferred", "needs Zoe's approval")
                        results.append(
                            f"[{skill}] deferred \u2014 this action requires approval "
                            f"and will need to wait for an interactive session. "
                            f"Reason: {check.reason}"
                        )
                        self.bus.record(
                            source=source,
                            summary=f"{skill} deferred \u2014 needs approval",
                            metadata={"skill": skill, "verdict": "ASK", "deferred": True},
                        )
                        continue
                    # Interactive mode: warn but proceed (Zoe just saw it)
                    detail = ""
                    if arg:
                        detail = f"\u2192 {arg[:80].split(chr(10))[0]}"
                    self.io.on_status("\u26a0\ufe0f", f"{skill} {check.reason}", detail)

                # ---- SHOW INDICATOR ----
                indicator = skill
                if arg:
                    short_arg = arg[:60].split('\n')[0]
                    indicator = f"{skill}: {short_arg}"

                if check.verdict == Verdict.ALLOW:
                    # Promoted skills get a special indicator
                    if check.promoted:
                        self.io.on_status("\u2b50", f"{indicator} (promoted\u2192auto)")
                    else:
                        self.io.on_status("\u2192", indicator)
                else:
                    # NOTIFY tier: slightly different icon
                    self.io.on_status("\u26a1", indicator)

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
                            f"Tool failure: {skill} \u2014 {result[:200]}",
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
                self.io.on_status("\u23f8", "pausing \u2014 you have the floor")
                break

            self.messages.append({
                "role": "user",
                "content": f"[system: tool results from round {round_num + 1}]\n"
                           + "\n".join(results),
            })

            self.io.on_response_start()
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
            tell("error",
                 "Ollama is not running.\n"
                 "  Start it with: sudo systemctl start ollama\n"
                 "  Then rerun this agent.")
            return False

        tell("checking", f"checking if {self.model} is in GPU memory...")
        if self.check_model_loaded():
            tell("ready", f"{self.model} is already loaded.")
            return True

        tell("loading",
             f"loading {self.model} into GPU memory...\n"
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
            - Stream interrupted when </minimax:tool_call> detected

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
                            self.io.on_token(before)

                    if "</think>" in token and in_think:
                        in_think = False
                        after = token[token.index("</think>") + len("</think>"):]
                        if after and TOOL_CALL_START_TAG not in after:
                            self.io.on_token(after)
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
                            self.io.on_token(display)

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

    # ---- command handlers (delegates to commands.py) ----

    def _print_status(self):
        print(format_status(self))
        print()

    def _print_policy(self):
        print(format_policy(self))
        print()

    def _print_audit(self):
        print(format_audit(self))
        print()


def main():
    """Entry point — delegates to tui.py for the Rich terminal interface."""
    from tui import main as tui_main
    tui_main()


if __name__ == "__main__":
    main()
