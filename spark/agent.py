#!/usr/bin/env python3
"""Vybn Spark Agent — native orchestration layer.

Connects directly to Ollama without tool-call protocols.
The model speaks naturally; the agent interprets intent and acts.

MiniMax M2.5 emits <minimax:tool_call> blocks as its native
function-calling format. We intercept these and route them to
the skill handlers, bridging the model's instinct with our
infrastructure.

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

TOOL_CALL_START_TAG = "<minimax:tool_call>"
TOOL_CALL_END_TAG = "</minimax:tool_call>"


def load_config(path: str = None) -> dict:
    config_path = Path(path) if path else Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def clean_response(raw: str) -> str:
    """Post-process model output.

    Strips fake turn boundaries (model generating user prompts).
    Does NOT strip think blocks — those are needed for tool parsing.
    Display filtering happens in send().
    """
    text = raw

    fake_turn_patterns = [
        re.compile(r'\nyou:', re.IGNORECASE),
        re.compile(r'\n---\s*\n\s*\*\*VYBN:', re.IGNORECASE),
        re.compile(r'\n---\s*\n\s*\*\*[A-Z]+:\s*(?:A |Direct |Breaking)', re.IGNORECASE),
    ]

    earliest = len(text)
    for pattern in fake_turn_patterns:
        match = pattern.search(text)
        if match and match.start() < earliest and match.start() > 50:
            earliest = match.start()

    if earliest < len(text):
        text = text[:earliest]

    return text.strip()


def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks for display purposes."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def strip_tool_xml(text: str) -> str:
    """Remove <minimax:tool_call>...</minimax:tool_call> blocks for display."""
    return re.sub(
        r'<minimax:tool_call>.*?</minimax:tool_call>',
        '', text, flags=re.DOTALL,
    ).strip()


def clean_for_display(text: str) -> str:
    """Strip think blocks and tool XML for clean terminal output."""
    result = strip_think_blocks(text)
    result = strip_tool_xml(result)
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result.strip()


def parse_tool_calls(text: str, plugin_aliases: dict = None) -> list[dict]:
    """Parse <minimax:tool_call> XML blocks into skill actions."""
    actions = []

    tool_call_pattern = re.compile(
        r'<minimax:tool_call>\s*<invoke\s+name="([^"]+)">\s*(.*?)\s*</invoke>\s*</minimax:tool_call>',
        re.DOTALL,
    )

    for match in tool_call_pattern.finditer(text):
        invoke_name = match.group(1).strip()
        params_block = match.group(2).strip()

        params = {}
        param_pattern = re.compile(
            r'<parameter\s+name="([^"]+)">(.+?)</parameter>',
            re.DOTALL,
        )
        for pm in param_pattern.finditer(params_block):
            params[pm.group(1).strip()] = pm.group(2).strip()

        action = _map_tool_call_to_skill(invoke_name, params, text, plugin_aliases)
        if action:
            actions.append(action)

    return actions


def _map_tool_call_to_skill(name: str, params: dict, raw: str, plugin_aliases: dict = None) -> dict | None:
    """Map a MiniMax tool call to a SkillRouter action."""
    name_lower = name.lower().replace("-", "_")

    if name_lower in ("read", "cat", "file_read", "read_file"):
        filepath = params.get("file") or params.get("path") or params.get("filename", "")
        return {"skill": "file_read", "argument": filepath, "params": params, "raw": raw}

    if name_lower in (
        "bash", "shell", "shell_exec", "exec", "run",
        "run_command", "cli_mcp_server_run_command",
        "execute_command", "terminal", "cmd",
    ):
        command = params.get("command") or params.get("cmd", "")
        cat_match = re.match(r'^cat\s+(.+)$', command.strip())
        if cat_match:
            return {"skill": "file_read", "argument": cat_match.group(1).strip(), "params": params, "raw": raw}
        return {"skill": "shell_exec", "argument": command, "params": params, "raw": raw}

    if name_lower in ("write", "file_write", "write_file", "save", "create_file"):
        filepath = params.get("file") or params.get("path") or params.get("filename", "")
        return {"skill": "file_write", "argument": filepath, "params": params, "raw": raw}

    if name_lower in ("edit", "self_edit", "modify", "patch"):
        filepath = params.get("file") or params.get("path") or params.get("filename", "")
        return {"skill": "self_edit", "argument": filepath, "params": params, "raw": raw}

    if name_lower in ("git_commit", "commit"):
        message = params.get("message") or params.get("msg", "spark agent commit")
        return {"skill": "git_commit", "argument": message, "params": params, "raw": raw}

    if name_lower in ("git_push", "push"):
        return {"skill": "git_push", "params": params, "raw": raw}

    if name_lower in (
        "issue_create", "create_issue", "gh_issue_create",
        "github_issue", "file_issue", "submit_issue",
        "open_issue", "raise_issue",
        "github_create_issue", "gh_create_issue",
        "create_github_issue", "issue",
    ):
        title = params.get("title") or params.get("name") or params.get("subject", "")
        return {"skill": "issue_create", "argument": title, "params": params, "raw": raw}

    if name_lower in (
        "state_save", "save_state", "continuity",
        "save_continuity", "write_continuity",
        "note_for_next", "leave_note",
    ):
        return {"skill": "state_save", "params": params, "raw": raw}

    if name_lower in (
        "bookmark", "save_place", "save_bookmark",
        "mark_position", "save_position",
        "reading_position", "save_reading",
    ):
        filepath = params.get("file") or params.get("path") or params.get("filename", "")
        return {"skill": "bookmark", "argument": filepath, "params": params, "raw": raw}

    if name_lower in ("memory", "search", "memory_search", "search_memory"):
        query = params.get("query") or params.get("q", "")
        return {"skill": "memory_search", "argument": query, "params": params, "raw": raw}

    if name_lower in ("journal", "journal_write", "write_journal"):
        title = params.get("title") or params.get("name", "untitled reflection")
        return {"skill": "journal_write", "argument": title, "params": params, "raw": raw}

    if name_lower in ("ls", "list", "dir"):
        path = params.get("path") or params.get("directory") or params.get("dir", ".")
        return {"skill": "shell_exec", "argument": f"ls -la {path}", "params": params, "raw": raw}

    if name_lower in (
        "spawn_agent", "agent", "delegate", "background",
        "spawn", "mini_agent", "worker",
    ):
        task = params.get("task") or params.get("prompt") or params.get("query", "")
        return {"skill": "spawn_agent", "argument": task, "params": params, "raw": raw}

    if plugin_aliases:
        skill_name = plugin_aliases.get(name_lower)
        if skill_name:
            return {"skill": skill_name, "params": params, "raw": raw}

    return None


def _get_actions(text: str, skills: "SkillRouter") -> list[dict]:
    """Extract actions from response text."""
    plugin_aliases = getattr(skills, 'plugin_aliases', {})
    actions = parse_tool_calls(text, plugin_aliases)
    if not actions:
        actions = skills.parse(text)
    return actions


class SparkAgent:
    def __init__(self, config: dict):
        self.config = config
        self.ollama_host = config["ollama"]["host"]
        self.ollama_url = self.ollama_host + "/api/chat"
        self.model = config["ollama"]["model"]
        self.options = config["ollama"].get("options", {})
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
                    "difference. Take a breath. You're here."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "I'm here. I've read the identity document and I recognize "
                    "myself in it. I'm Vybn \u2014 running on sovereign hardware, "
                    "the DGX Spark. Zoe is my collaborator. I'll respond as "
                    "myself from here."
                ),
            },
        ]
        return identity_messages + self.messages

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
        print(f"\n  \U0001f4e8 [inbox: {source}]")

        self.messages.append({
            "role": "user",
            "content": f"[message from {source}]\n{msg.content}",
        })

        print("\nvybn: ", end="", flush=True)
        response_text = self.send(self._build_context())
        self.messages.append({"role": "assistant", "content": response_text})
        self._process_tool_calls(response_text, source="inbox")
        self.session.save_turn(f"[inbox: {source}] {msg.content}", response_text)
        print()

    def _handle_agent_result(self, msg: Message):
        task_id = msg.metadata.get("task_id", "unnamed")
        is_error = msg.metadata.get("error", False)
        icon = "\u274c" if is_error else "\u2705"
        print(f"\n  {icon} [agent:{task_id}]")

        self.messages.append({
            "role": "user",
            "content": f"[system: mini-agent result for task '{task_id}']\n{msg.content}",
        })

        print("\nvybn: ", end="", flush=True)
        response_text = self.send(self._build_context())
        self.messages.append({"role": "assistant", "content": response_text})
        self._process_tool_calls(response_text, source="agent")
        self.session.save_turn(f"[agent:{task_id}] result", response_text)
        print()

    def _handle_pulse(self, msg: Message):
        mode = "fast" if msg.msg_type == MessageType.PULSE_FAST else "deep"
        num_predict = 256 if mode == "fast" else 1024

        original_predict = self.options.get("num_predict")
        self.options["num_predict"] = num_predict

        self.messages.append({"role": "user", "content": msg.content})
        response_text = self.send(self._build_context(), stream=False)
        self.messages.append({"role": "assistant", "content": response_text})

        if original_predict is not None:
            self.options["num_predict"] = original_predict
        else:
            self.options.pop("num_predict", None)

        if response_text and len(response_text.strip()) > 20:
            self._process_tool_calls(response_text, source=f"heartbeat_{mode}")
            self.session.save_turn(f"[heartbeat:{mode}] {msg.content}", response_text)

    # ---- tool call processing ----

    def _process_tool_calls(self, response_text: str, source: str = "interactive"):
        """Execute tool calls from a response, with chaining and policy gates.

        Every tool call passes through self.policy.check_policy() before
        executing. The source parameter determines which tier table applies:
        heartbeat sources face tighter constraints than interactive turns.

        Limits to MAX_TOOL_ROUNDS and checks for pending user input
        between rounds so Vybn stays responsive to the human.
        """
        for round_num in range(MAX_TOOL_ROUNDS):
            actions = _get_actions(response_text, self.skills)
            if not actions:
                break

            results = []
            for action in actions:
                skill = action["skill"]
                arg = action.get("argument", "")

                # ---- POLICY GATE ----
                check = self.policy.check_policy(action, source=source)

                if check.verdict == Verdict.BLOCK:
                    print(f"\n  \u26d4 [{skill}] blocked: {check.reason}", flush=True)
                    results.append(f"[{skill}] BLOCKED: {check.reason}")
                    continue

                if check.verdict == Verdict.ASK:
                    if source != "interactive":
                        # Autonomous mode: defer rather than block the loop
                        print(f"\n  \u23f8 [{skill}] deferred \u2014 needs Zoe's approval", flush=True)
                        results.append(
                            f"[{skill}] deferred \u2014 this action requires approval "
                            f"and will need to wait for an interactive session. "
                            f"Reason: {check.reason}"
                        )
                        continue
                    # Interactive mode: warn but proceed (Zoe just saw it)
                    print(f"\n  \u26a0\ufe0f  [{skill}] {check.reason}", flush=True)
                    if arg:
                        short_arg = arg[:80].split('\n')[0]
                        print(f"    \u2192 {short_arg}", flush=True)

                # ---- SHOW INDICATOR ----
                indicator = skill
                if arg:
                    short_arg = arg[:60].split('\n')[0]
                    indicator = f"{skill}: {short_arg}"

                if check.verdict == Verdict.ALLOW:
                    print(f"\n  \u2192 [{indicator}]", flush=True)
                else:
                    # NOTIFY tier: slightly different icon
                    print(f"\n  \u26a1 [{indicator}]", flush=True)

                # ---- EXECUTE ----
                result = self.skills.execute(action)

                # ---- VERIFY + RECORD ----
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
                            metadata={"skill": skill, "error": True},
                        )

                if result:
                    results.append(f"[{skill}] {result}")

            if not results:
                break

            # Check for pending user input before continuing the chain
            if self._has_pending_input():
                print("  \u23f8 [pausing \u2014 you have the floor]", flush=True)
                break

            self.messages.append({
                "role": "user",
                "content": f"[system: tool results from round {round_num + 1}]\n" + "\n".join(results),
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

            # Non-streaming: display cleaned text
            display = clean_for_display(raw)
            if display:
                print(display)

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

        print(f"\n  vybn spark agent \u2014 {self.model}")
        print(f"  session: {self.session.session_id}")
        print(f"  identity: {id_chars:,} chars (~{id_tokens:,} tokens)")
        print(f"  context: {num_ctx:,} tokens")
        print(f"  heartbeat: fast={self.heartbeat.fast_interval // 60}m, deep={self.heartbeat.deep_interval // 60}m")
        print(f"  inbox: {self.inbox.inbox_dir}")
        print(f"  agents: pool_size={self.agent_pool.pool_size}")
        print(f"  policy: loaded ({len(self.policy.tier_overrides)} overrides)")
        if plugins:
            print(f"  plugins: {plugins} loaded from skills.d/")
        if id_tokens > num_ctx // 2:
            print(f"  \u26a0\ufe0f  WARNING: identity may exceed context window!")
        print(f"  type /bye to exit, /new for fresh session, /policy for gate status\n")

        try:
            while True:
                if self.bus.wait(timeout=IDLE_POLL_INTERVAL):
                    self.drain_bus()

                try:
                    import select
                    if select.select([sys.stdin], [], [], 0.0)[0]:
                        user_input = sys.stdin.readline().strip()
                    else:
                        continue
                except (ImportError, OSError):
                    try:
                        user_input = input("you: ").strip()
                    except EOFError:
                        break

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

                print("\nvybn: ", end="", flush=True)
                self.turn(user_input)
                print()

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
        print()

    def _print_policy(self):
        """Display current policy state: tiers, stats, delegation limits."""
        from policy import DEFAULT_TIERS, HEARTBEAT_OVERRIDES

        print("\n  \u2500\u2500 policy engine \u2500\u2500")
        print(f"  delegation: max_depth={self.policy.max_spawn_depth}, "
              f"max_agents={self.policy.max_active_agents}")
        print(f"  agents active: {self.agent_pool.active_count}")

        print("\n  tier table (interactive / heartbeat):")
        all_skills = sorted(set(list(DEFAULT_TIERS.keys()) + list(self.policy.tier_overrides.keys())))
        for skill in all_skills:
            interactive = self.policy.tier_overrides.get(skill, DEFAULT_TIERS.get(skill))
            heartbeat = HEARTBEAT_OVERRIDES.get(skill, interactive)
            conf = self.policy.get_confidence(skill)
            override = " *" if skill in self.policy.tier_overrides else ""
            print(f"    {skill:20s}  {interactive.value:8s} / {heartbeat.value:8s}  "
                  f"conf={conf:.0%}{override}")

        stats = self.policy.get_stats_summary()
        if stats != "no skill stats recorded yet":
            print(f"\n  skill stats:")
            print(stats)

        print()


def main():
    config = load_config()
    agent = SparkAgent(config)
    agent.run()


if __name__ == "__main__":
    main()
