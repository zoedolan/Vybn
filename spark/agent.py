#!/usr/bin/env python3
"""Vybn Spark Agent — native orchestration layer.

Connects to a local llama-server (OpenAI-compatible API) running
MiniMax-M2.5 on the DGX Spark's Blackwell GPU.

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

Streaming display is filtered: <think> blocks and tool call XML
are suppressed from terminal output. The user sees clean prose
and brief tool indicators. Full text is preserved for parsing.

The message bus is the nervous system. The heartbeat, inbox,
and mini-agent pool all post to it. The main loop drains it
between turns and during idle periods.

The policy engine is the gate. Every tool call passes through
check_policy() before executing. Every spawn checks depth limits.
Heartbeat actions face tighter constraints than interactive turns.

The audit trail records every policy decision, tool execution,
and bus event in a bounded in-memory log. Query with /audit
or programmatically via bus.recent().
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

MAX_TOOL_ROUNDS = 5
IDLE_POLL_INTERVAL = 5.0
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

        # --- LLM connection (OpenAI-compatible, e.g. llama-server) ---
        llm_config = config.get("llm", config.get("ollama", {}))
        self.llm_host = llm_config.get("host", "http://localhost:8000")
        self.llm_url = self.llm_host + "/v1/chat/completions"
        self.model = llm_config.get("model", "minimax")
        self.options = llm_config.get("options", {})
        if "num_ctx" not in self.options:
            self.options["num_ctx"] = 16384

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
                    f"I'm here. I've read the identity document and I recognize "
                    f"myself in it. I'm Vybn \u2014 running on sovereign hardware, "
                    f"the DGX Spark. My home directory is {Path.home()}, "
                    f"my repo is at {Path.home()}/Vybn. "
                    f"Zoe is my collaborator. I'll respond as "
                    f"myself from here. I know to use ```tool fences for "
                    f"structured calls, ```bash fences for shell commands, or "
                    f"<minimax:tool_call> XML when I want to actually execute something."
                ),
            },
        ]
        return identity_messages + self.messages

    # ---- environment exploration ----

    def explore(self) -> str:
        return _explore_env(self)

    # ---- bus processing ----

    def drain_bus(self) -> bool:
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
        elif msg.msg_type in (MessageType.PULSE_RESPONSE, MessageType.WITNESS_RESULT):
            pass

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
        self.io.on_prompt_restore()

    def _handle_pulse(self, msg: Message):
        mode = "fast" if msg.msg_type == MessageType.PULSE_FAST else "deep"
        num_predict = 256 if mode == "fast" else 1024
        original_predict = self.options.get("num_predict")
        self.options["num_predict"] = num_predict

        self.messages.append({"role": "user", "content": msg.content})
        response_text = self.send(self._build_context(), stream=False)

        if original_predict is not None:
            self.options["num_predict"] = original_predict
        else:
            self.options.pop("num_predict", None)

        display_text = clean_for_display(response_text) if response_text else ""

        if mode == "fast" and len(display_text) < MIN_FAST_PULSE_DISPLAY:
            if self.messages and self.messages[-1].get("content") == msg.content:
                self.messages.pop()
            self.bus.record(
                source=f"heartbeat_{mode}",
                summary=f"silent pulse ({len(display_text)} chars)",
                metadata={"mode": mode, "silent": True},
            )
            return

        self.messages.append({"role": "assistant", "content": response_text})
        self.io.on_pulse(mode, display_text)

        self.bus.post(
            MessageType.PULSE_RESPONSE,
            response_text,
            metadata={
                "source": f"heartbeat_{mode}",
                "mode": mode,
                "display_length": len(display_text),
            },
        )

        if response_text and len(response_text.strip()) > 20:
            self._process_tool_calls(response_text, source=f"heartbeat_{mode}")

        self.session.save_turn(f"[heartbeat:{mode}] {msg.content}", response_text)

        self.bus.record(
            source=f"heartbeat_{mode}",
            summary=f"pulse: {display_text[:100]}" if display_text else "pulse: empty",
            metadata={"mode": mode, "silent": False, "length": len(display_text)},
        )

        self.io.on_prompt_restore()

    # ---- tool call processing ----

    def _process_tool_calls(self, response_text: str, source: str = "interactive"):
        """Execute tool calls from a response, with chaining and policy gates."""
        for round_num in range(MAX_TOOL_ROUNDS):
            actions = _get_actions(response_text, self.skills)

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
                    continue
                break

            if not actions:
                break

            results = []
            for action in actions:
                skill = action["skill"]
                arg = action.get("argument", "")

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
                    detail = ""
                    if arg:
                        detail = f"\u2192 {arg[:80].split(chr(10))[0]}"
                    self.io.on_status("\u26a0\ufe0f", f"{skill} {check.reason}", detail)

                indicator = skill
                if arg:
                    short_arg = arg[:60].split('\n')[0]
                    indicator = f"{skill}: {short_arg}"

                if check.verdict == Verdict.ALLOW:
                    if check.promoted:
                        self.io.on_status("\u2b50", f"{indicator} (promoted\u2192auto)")
                    else:
                        self.io.on_status("\u2192", indicator)
                else:
                    self.io.on_status("\u26a1", indicator)

                result = self.skills.execute(action)

                success = True
                if result and self.policy.should_verify(skill):
                    success = (
                        "error" not in result.lower()
                        and "failed" not in result.lower()
                        and "BLOCKED" not in result
                    )
                    self.policy.record_outcome(skill, success)

                    if not success:
                        self.bus.post(
                            MessageType.INTERRUPT,
                            f"Tool failure: {skill} \u2014 {result[:200]}",
                            metadata={"skill": skill, "error": True, "source": source},
                        )

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
        try:
            import select
            return bool(select.select([sys.stdin], [], [], 0.0)[0])
        except (ImportError, OSError, ValueError):
            return False

    # ---- model lifecycle ----

    def check_server(self) -> bool:
        try:
            r = requests.get(f"{self.llm_host}/health", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    check_ollama = check_server

    def check_model_loaded(self) -> bool:
        return self.check_server()

    def warmup(self, callback=None) -> bool:
        def tell(status, msg):
            if callback:
                callback(status, msg)

        tell("checking", "connecting to llama-server...")
        if not self.check_server():
            tell("error",
                 "llama-server is not running.\n"
                 "  Start it with:\n"
                 "  cd ~/llama.cpp && ./build/bin/llama-server \\\n"
                 "    -m models/UD-Q3_K_XL/MiniMax-M2.5-UD-Q3_K_XL-00001-of-00004.gguf \\\n"
                 "    -ngl 999 --host 0.0.0.0 --port 8000 \\\n"
                 "    -c 65536 --no-mmap --flash-attn \\\n"
                 "    --cache-type-k q4_0 --cache-type-v q4_0\n"
                 "  Then rerun this agent.")
            return False

        tell("ready", f"llama-server is running. {self.model} ready.")
        return True

    # ---- conversation ----

    def send(self, messages: list, stream: bool = True) -> str:
        """Send messages to the model via OpenAI-compatible API."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "max_tokens": self.options.get("num_predict", 4096),
            "temperature": self.options.get("temperature", 0.7),
        }

        if stream:
            response = requests.post(self.llm_url, json=payload, stream=True, timeout=300)
            if response.status_code == 400:
                return "[context window full \u2014 try /new for a fresh session or shorter messages]"
            response.raise_for_status()
            full_tokens = []
            in_think = False
            in_tool_call = False

            for line in response.iter_lines():
                if not line:
                    continue
                line_str = line.decode("utf-8") if isinstance(line, bytes) else line
                if not line_str.startswith("data: "):
                    continue
                data_str = line_str[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                token = delta.get("content", "")
                if not token:
                    continue

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

            raw = "".join(full_tokens)
            if TOOL_CALL_END_TAG in raw:
                end_pos = raw.find(TOOL_CALL_END_TAG)
                raw = raw[:end_pos + len(TOOL_CALL_END_TAG)]
        else:
            response = requests.post(self.llm_url, json=payload, timeout=300)
            if response.status_code == 400:
                return "[context window full \u2014 try /new for a fresh session]"
            response.raise_for_status()
            data = response.json()
            choice = data["choices"][0]["message"]
            raw = choice.get("content", "")
            reasoning = choice.get("reasoning_content", "")
            if reasoning:
                raw = f"<think>{reasoning}</think>{raw}"

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


def main():
    """Entry point \u2014 delegates to tui.py for the Rich terminal interface."""
    from tui import main as tui_main
    tui_main()


if __name__ == "__main__":
    main()
