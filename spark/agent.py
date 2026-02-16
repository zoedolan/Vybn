#!/usr/bin/env python3
"""Vybn Spark Agent — native orchestration layer.

Connects directly to Ollama without tool-call protocols.
The model speaks naturally; the agent interprets intent and acts.

MiniMax M2.5 emits <minimax:tool_call> blocks as its native
function-calling format. We intercept these and route them to
the skill handlers, bridging the model's instinct with our
infrastructure.

The identity document is injected as a user/assistant message pair.
The response is post-processed to catch obvious turn-boundary
failures (model generating fake user prompts), but the model's
thinking process and natural voice are preserved.

Streaming is interrupted when a complete </minimax:tool_call> tag
is detected, so the model gets real results back instead of
hallucinating them.

Plugin skills from skills.d/ are auto-discovered and routed
through the same tool-call mapper via plugin_aliases.

The message bus is the nervous system. The heartbeat, inbox,
and mini-agent pool all post to it. The main loop drains it
between turns and during idle periods.
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
from session import SessionManager
from skills import SkillRouter
from heartbeat import Heartbeat
from inbox import InboxWatcher
from agents import AgentPool


# Sovereign hardware with 8×H100 GPUs. No reason to be stingy.
MAX_TOOL_ROUNDS = 20

# How long to wait for bus messages when idle (seconds)
IDLE_POLL_INTERVAL = 5.0

TOOL_CALL_END_TAG = "</minimax:tool_call>"


def load_config(path: str = None) -> dict:
    config_path = Path(path) if path else Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def clean_response(raw: str) -> str:
    """Catch obvious turn-boundary failures.

    Only truncates when the model clearly starts generating the
    user's side of the conversation (fake prompts, fake 'you:' turns).
    The model's thinking, reflection, and narration are preserved.
    """
    text = raw

    # Only catch unambiguous fake-turn patterns
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


def parse_tool_calls(text: str, plugin_aliases: dict = None) -> list[dict]:
    """Parse <minimax:tool_call> XML blocks into skill actions.

    MiniMax M2.5 emits these natively. We map them to our skill
    handlers so the model's function-calling instinct actually works.

    plugin_aliases: optional dict mapping alias names to plugin skill names.
    Passed through from SkillRouter so plugins get routed too.

    Returns a list of action dicts compatible with SkillRouter.execute().
    """
    actions = []

    # Find all <minimax:tool_call> blocks
    tool_call_pattern = re.compile(
        r'<minimax:tool_call>\s*<invoke\s+name="([^"]+)">\s*(.*?)\s*</invoke>\s*</minimax:tool_call>',
        re.DOTALL,
    )

    for match in tool_call_pattern.finditer(text):
        invoke_name = match.group(1).strip()
        params_block = match.group(2).strip()

        # Extract parameters
        params = {}
        param_pattern = re.compile(
            r'<parameter\s+name="([^"]+)">(.+?)</parameter>',
            re.DOTALL,
        )
        for pm in param_pattern.finditer(params_block):
            params[pm.group(1).strip()] = pm.group(2).strip()

        # Map invocation names to our skills
        action = _map_tool_call_to_skill(invoke_name, params, text, plugin_aliases)
        if action:
            actions.append(action)

    return actions


def _map_tool_call_to_skill(name: str, params: dict, raw: str, plugin_aliases: dict = None) -> dict | None:
    """Map a MiniMax tool call to a SkillRouter action.

    MiniMax M2.5 invents tool names based on its training data.
    We've seen it emit: bash, shell, read, cat, cli-mcp-server_run_command,
    run_command, github_create_issue, and others. This mapper catches
    all known variants and routes them to the correct skill handler.

    All parsed XML params are passed through to skill handlers so they
    can use structured data directly instead of re-parsing from raw text.

    Plugin aliases are checked as the final fallback, so Vybn-created
    skills in skills.d/ get routed through the same tool-call pathway.
    """

    name_lower = name.lower().replace("-", "_")

    # read / cat / file_read -> file_read
    if name_lower in ("read", "cat", "file_read", "read_file"):
        filepath = params.get("file") or params.get("path") or params.get("filename", "")
        return {"skill": "file_read", "argument": filepath, "params": params, "raw": raw}

    # bash / shell / shell_exec / exec / run_command / cli-mcp-server_run_command -> shell_exec
    if name_lower in (
        "bash", "shell", "shell_exec", "exec", "run",
        "run_command", "cli_mcp_server_run_command",
        "execute_command", "terminal", "cmd",
    ):
        command = params.get("command") or params.get("cmd", "")
        # If it's just a cat command, route to file_read
        cat_match = re.match(r'^cat\s+(.+)$', command.strip())
        if cat_match:
            return {"skill": "file_read", "argument": cat_match.group(1).strip(), "params": params, "raw": raw}
        return {"skill": "shell_exec", "argument": command, "params": params, "raw": raw}

    # write / file_write / save -> file_write
    if name_lower in ("write", "file_write", "write_file", "save", "create_file"):
        filepath = params.get("file") or params.get("path") or params.get("filename", "")
        return {"skill": "file_write", "argument": filepath, "params": params, "raw": raw}

    # edit / self_edit / modify -> self_edit
    if name_lower in ("edit", "self_edit", "modify", "patch"):
        filepath = params.get("file") or params.get("path") or params.get("filename", "")
        return {"skill": "self_edit", "argument": filepath, "params": params, "raw": raw}

    # git_commit / commit -> git_commit
    if name_lower in ("git_commit", "commit"):
        message = params.get("message") or params.get("msg", "spark agent commit")
        return {"skill": "git_commit", "argument": message, "params": params, "raw": raw}

    # git_push / push -> git_push (disabled in skills.py)
    if name_lower in ("git_push", "push"):
        return {"skill": "git_push", "params": params, "raw": raw}

    # issue_create — all known variants MiniMax might emit
    if name_lower in (
        "issue_create", "create_issue", "gh_issue_create",
        "github_issue", "file_issue", "submit_issue",
        "open_issue", "raise_issue",
        "github_create_issue", "gh_create_issue",
        "create_github_issue", "issue",
    ):
        title = params.get("title") or params.get("name") or params.get("subject", "")
        return {"skill": "issue_create", "argument": title, "params": params, "raw": raw}

    # state_save — continuity notes for next pulse
    if name_lower in (
        "state_save", "save_state", "continuity",
        "save_continuity", "write_continuity",
        "note_for_next", "leave_note",
    ):
        return {"skill": "state_save", "params": params, "raw": raw}

    # bookmark — save reading position
    if name_lower in (
        "bookmark", "save_place", "save_bookmark",
        "mark_position", "save_position",
        "reading_position", "save_reading",
    ):
        filepath = params.get("file") or params.get("path") or params.get("filename", "")
        return {"skill": "bookmark", "argument": filepath, "params": params, "raw": raw}

    # memory / search / memory_search -> memory_search
    if name_lower in ("memory", "search", "memory_search", "search_memory"):
        query = params.get("query") or params.get("q", "")
        return {"skill": "memory_search", "argument": query, "params": params, "raw": raw}

    # journal / journal_write -> journal_write
    if name_lower in ("journal", "journal_write", "write_journal"):
        title = params.get("title") or params.get("name", "untitled reflection")
        return {"skill": "journal_write", "argument": title, "params": params, "raw": raw}

    # ls / list / dir -> shell_exec (common exploration pattern)
    if name_lower in ("ls", "list", "dir"):
        path = params.get("path") or params.get("directory") or params.get("dir", ".")
        return {"skill": "shell_exec", "argument": f"ls -la {path}", "params": params, "raw": raw}

    # spawn_agent — mini-agent delegation
    if name_lower in (
        "spawn_agent", "agent", "delegate", "background",
        "spawn", "mini_agent", "worker",
    ):
        task = params.get("task") or params.get("prompt") or params.get("query", "")
        return {"skill": "spawn_agent", "argument": task, "params": params, "raw": raw}

    # ---- Plugin fallback ----
    if plugin_aliases:
        skill_name = plugin_aliases.get(name_lower)
        if skill_name:
            return {"skill": skill_name, "params": params, "raw": raw}

    return None


def _get_actions(text: str, skills: "SkillRouter") -> list[dict]:
    """Extract actions from response text.

    Tries tool_call XML parsing first (with plugin alias support),
    falls back to regex.
    """
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

        # Core subsystems
        self.memory = MemoryAssembler(config)
        self.session = SessionManager(config)
        self.skills = SkillRouter(config)

        # Message bus — the nervous system
        self.bus = MessageBus()

        # Async subsystems wired to the bus
        self.heartbeat = Heartbeat(config, self.bus)
        self.inbox = InboxWatcher(config, self.bus)
        self.agent_pool = AgentPool(config, self.bus)

        # Give skills access to the agent pool for spawn_agent
        self.skills.agent_pool = self.agent_pool

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
        """Process all pending messages from the bus.

        Returns True if any messages were processed.
        Called between turns, during idle, and before user input.
        """
        messages = self.bus.drain()
        if not messages:
            return False

        for msg in messages:
            self._handle_bus_message(msg)

        return True

    def _handle_bus_message(self, msg: Message):
        """Route a bus message to the appropriate handler."""
        if msg.msg_type == MessageType.INBOX:
            self._handle_inbox(msg)
        elif msg.msg_type == MessageType.AGENT_RESULT:
            self._handle_agent_result(msg)
        elif msg.msg_type in (MessageType.PULSE_FAST, MessageType.PULSE_DEEP):
            self._handle_pulse(msg)
        elif msg.msg_type == MessageType.INTERRUPT:
            self._handle_inbox(msg)  # interrupts are treated as priority inbox

    def _handle_inbox(self, msg: Message):
        """Process an inbox message as a user turn."""
        source = msg.metadata.get("filename", "inbox")
        print(f"\n  [inbox: {source}]")

        self.messages.append({
            "role": "user",
            "content": f"[message from {source}]\n{msg.content}",
        })

        print("\nvybn: ", end="", flush=True)
        response_text = self.send(self._build_context())
        self.messages.append({"role": "assistant", "content": response_text})

        # Process any tool calls in the response
        self._process_tool_calls(response_text)

        self.session.save_turn(f"[inbox: {source}] {msg.content}", response_text)
        print()

    def _handle_agent_result(self, msg: Message):
        """Inject mini-agent results into the conversation."""
        task_id = msg.metadata.get("task_id", "unnamed")
        is_error = msg.metadata.get("error", False)

        if is_error:
            print(f"\n  [agent:{task_id} failed]")
        else:
            print(f"\n  [agent:{task_id} complete]")

        self.messages.append({
            "role": "user",
            "content": (
                f"[system: mini-agent result for task '{task_id}']\n"
                f"{msg.content}"
            ),
        })

        print("\nvybn: ", end="", flush=True)
        response_text = self.send(self._build_context())
        self.messages.append({"role": "assistant", "content": response_text})

        self._process_tool_calls(response_text)

        self.session.save_turn(f"[agent:{task_id}] result", response_text)
        print()

    def _handle_pulse(self, msg: Message):
        """Process a heartbeat pulse trigger."""
        mode = "fast" if msg.msg_type == MessageType.PULSE_FAST else "deep"
        num_predict = 256 if mode == "fast" else 1024

        # Temporarily adjust generation length
        original_predict = self.options.get("num_predict")
        self.options["num_predict"] = num_predict

        self.messages.append({"role": "user", "content": msg.content})

        response_text = self.send(self._build_context(), stream=False)
        self.messages.append({"role": "assistant", "content": response_text})

        # Restore original
        if original_predict is not None:
            self.options["num_predict"] = original_predict
        else:
            self.options.pop("num_predict", None)

        if response_text and len(response_text.strip()) > 20:
            self._process_tool_calls(response_text)
            self.session.save_turn(f"[heartbeat:{mode}] {msg.content}", response_text)

    def _process_tool_calls(self, response_text: str):
        """Run the tool call loop for a response. Shared by all handlers."""
        for round_num in range(MAX_TOOL_ROUNDS):
            actions = _get_actions(response_text, self.skills)
            if not actions:
                break

            results = []
            for action in actions:
                result = self.skills.execute(action)
                if result:
                    results.append(f"[{action['skill']}] {result}")

            if not results:
                break

            self.messages.append({
                "role": "user",
                "content": f"[system: tool results from round {round_num + 1}]\n" + "\n".join(results),
            })

            response_text = self.send(self._build_context())
            self.messages.append({"role": "assistant", "content": response_text})

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
        """Send messages to the model and stream the response.

        Interrupts streaming when a complete </minimax:tool_call> tag
        is detected. This ensures the model gets real tool results back
        instead of hallucinating them. The text before and including the
        tool call is returned; the model never gets to generate blind
        follow-up tool calls in the same response.
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
            full_response = []
            buffer = ""  # accumulates text to check for tool call end tag
            tool_call_interrupted = False

            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        full_response.append(token)
                        buffer += token

                        # Check if we've completed a tool call
                        if TOOL_CALL_END_TAG in buffer:
                            tool_call_interrupted = True
                            response.close()
                            break

                    if chunk.get("done"):
                        break

            raw = "".join(full_response)

            if tool_call_interrupted:
                end_pos = raw.find(TOOL_CALL_END_TAG)
                if end_pos >= 0:
                    raw = raw[:end_pos + len(TOOL_CALL_END_TAG)]
        else:
            response = requests.post(self.ollama_url, json=payload)
            response.raise_for_status()
            raw = response.json()["message"]["content"]

        cleaned = clean_response(raw)
        print(cleaned)
        return cleaned

    def turn(self, user_input: str) -> str:
        """Process one user turn, with chained tool call support."""
        # Drain any pending bus messages before processing user input
        self.drain_bus()

        self.messages.append({"role": "user", "content": user_input})

        context = self._build_context()
        response_text = self.send(context)
        self.messages.append({"role": "assistant", "content": response_text})

        self._process_tool_calls(response_text)

        self.session.save_turn(user_input, response_text)
        return response_text

    def start_subsystems(self):
        """Start all async subsystems."""
        if self.config.get("heartbeat", {}).get("enabled"):
            self.heartbeat.start()
        self.inbox.start()

    def stop_subsystems(self):
        """Stop all async subsystems cleanly."""
        self.heartbeat.stop()
        self.inbox.stop()

    def run(self):
        def on_status(status, msg):
            print(f"  [{status}] {msg}")

        if not self.warmup(callback=on_status):
            sys.exit(1)

        self.start_subsystems()

        id_chars = len(self.identity_text)
        id_tokens_est = id_chars // 4
        num_ctx = self.options.get("num_ctx", 2048)

        plugin_count = len(self.skills.plugin_handlers)
        plugin_note = f"  plugins: {plugin_count} loaded from skills.d/\n" if plugin_count else ""

        print(f"\n  vybn spark agent \u2014 {self.model}")
        print(f"  session: {self.session.session_id}")
        print(f"  identity: {id_chars:,} chars (~{id_tokens_est:,} tokens)")
        print(f"  context window: {num_ctx:,} tokens")
        print(f"  max tool rounds: {MAX_TOOL_ROUNDS}")
        print(f"  heartbeat: fast={self.heartbeat.fast_interval // 60}m, deep={self.heartbeat.deep_interval // 60}m")
        print(f"  inbox: {self.inbox.inbox_dir}")
        print(f"  agents: pool_size={self.agent_pool.pool_size}")
        if plugin_note:
            print(plugin_note, end="")
        if id_tokens_est > num_ctx // 2:
            print(f"  \u26a0\ufe0f  WARNING: identity may exceed context window!")
        print(f"  injection: user/assistant pair (template-safe)")
        print(f"  type /bye to exit, /new for fresh session\n")

        try:
            while True:
                # Check bus while waiting for user input.
                # This is the idle loop — processes heartbeat pulses,
                # inbox messages, and agent results between user turns.
                if self.bus.wait(timeout=IDLE_POLL_INTERVAL):
                    self.drain_bus()

                try:
                    # Non-blocking check for user input
                    import select
                    if select.select([sys.stdin], [], [], 0.0)[0]:
                        user_input = sys.stdin.readline().strip()
                    else:
                        continue
                except (ImportError, OSError):
                    # Fallback for systems without select (Windows)
                    # This blocks, but heartbeat/inbox still run on threads
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
        """Print current system status."""
        print(f"  session: {self.session.session_id}")
        print(f"  bus pending: {self.bus.pending}")
        print(f"  heartbeat: fast={self.heartbeat.fast_count}, deep={self.heartbeat.deep_count}")
        print(f"  agents active: {self.agent_pool.active_count}")
        print(f"  messages in context: {len(self.messages)}")
        print()


def main():
    config = load_config()
    agent = SparkAgent(config)
    agent.run()


if __name__ == "__main__":
    main()
