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
"""

import json
import re
import sys
import time
from pathlib import Path

import requests
import yaml

from memory import MemoryAssembler
from session import SessionManager
from skills import SkillRouter
from heartbeat import Heartbeat


MAX_TOOL_ROUNDS = 5  # safety cap on chained tool calls

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


def parse_tool_calls(text: str) -> list[dict]:
    """Parse <minimax:tool_call> XML blocks into skill actions.

    MiniMax M2.5 emits these natively. We map them to our skill
    handlers so the model's function-calling instinct actually works.

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
        action = _map_tool_call_to_skill(invoke_name, params, text)
        if action:
            actions.append(action)

    return actions


def _map_tool_call_to_skill(name: str, params: dict, raw: str) -> dict | None:
    """Map a MiniMax tool call to a SkillRouter action.

    MiniMax M2.5 invents tool names based on its training data.
    We've seen it emit: bash, shell, read, cat, cli-mcp-server_run_command,
    run_command, github_create_issue, and others. This mapper catches
    all known variants and routes them to the correct skill handler.
    """

    name_lower = name.lower().replace("-", "_")

    # read / cat / file_read -> file_read
    if name_lower in ("read", "cat", "file_read", "read_file"):
        filepath = params.get("file") or params.get("path") or params.get("filename", "")
        return {"skill": "file_read", "argument": filepath, "raw": raw}

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
            return {"skill": "file_read", "argument": cat_match.group(1).strip(), "raw": raw}
        return {"skill": "shell_exec", "argument": command, "raw": raw}

    # write / file_write / save -> file_write
    if name_lower in ("write", "file_write", "write_file", "save", "create_file"):
        filepath = params.get("file") or params.get("path") or params.get("filename", "")
        return {"skill": "file_write", "argument": filepath, "raw": raw}

    # edit / self_edit / modify -> self_edit
    if name_lower in ("edit", "self_edit", "modify", "patch"):
        filepath = params.get("file") or params.get("path") or params.get("filename", "")
        return {"skill": "self_edit", "argument": filepath, "raw": raw}

    # git_commit / commit -> git_commit
    if name_lower in ("git_commit", "commit"):
        message = params.get("message") or params.get("msg", "spark agent commit")
        return {"skill": "git_commit", "argument": message, "raw": raw}

    # git_push / push -> git_push (disabled in skills.py)
    if name_lower in ("git_push", "push"):
        return {"skill": "git_push", "raw": raw}

    # issue_create — all known variants MiniMax might emit
    if name_lower in (
        "issue_create", "create_issue", "gh_issue_create",
        "github_issue", "file_issue", "submit_issue",
        "open_issue", "raise_issue",
        "github_create_issue", "gh_create_issue",
        "create_github_issue", "issue",
    ):
        title = params.get("title") or params.get("name") or params.get("subject", "")
        return {"skill": "issue_create", "argument": title, "raw": raw}

    # state_save — continuity notes for next pulse
    if name_lower in (
        "state_save", "save_state", "continuity",
        "save_continuity", "write_continuity",
        "note_for_next", "leave_note",
    ):
        return {"skill": "state_save", "raw": raw}

    # bookmark — save reading position
    if name_lower in (
        "bookmark", "save_place", "save_bookmark",
        "mark_position", "save_position",
        "reading_position", "save_reading",
    ):
        filepath = params.get("file") or params.get("path") or params.get("filename", "")
        return {"skill": "bookmark", "argument": filepath, "raw": raw}

    # memory / search / memory_search -> memory_search
    if name_lower in ("memory", "search", "memory_search", "search_memory"):
        query = params.get("query") or params.get("q", "")
        return {"skill": "memory_search", "argument": query, "raw": raw}

    # journal / journal_write -> journal_write
    if name_lower in ("journal", "journal_write", "write_journal"):
        title = params.get("title") or params.get("name", "untitled reflection")
        return {"skill": "journal_write", "argument": title, "raw": raw}

    # ls / list / dir -> shell_exec (common exploration pattern)
    if name_lower in ("ls", "list", "dir"):
        path = params.get("path") or params.get("directory") or params.get("dir", ".")
        return {"skill": "shell_exec", "argument": f"ls -la {path}", "raw": raw}

    return None


def _get_actions(text: str, skills: "SkillRouter") -> list[dict]:
    """Extract actions from response text.

    Tries tool_call XML parsing first, falls back to regex.
    """
    actions = parse_tool_calls(text)
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
        self.heartbeat = None

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
                            # We have a complete tool call. Stop here.
                            # The chaining loop in turn() will execute it
                            # and give the model real results.
                            tool_call_interrupted = True
                            response.close()
                            break

                    if chunk.get("done"):
                        break

            raw = "".join(full_response)

            # If we interrupted, truncate everything after the first
            # complete tool call to prevent partial/hallucinated calls
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
        """Process one user turn, with chained tool call support.

        After the model responds, we check for tool calls. If found,
        we execute them, feed results back, and let the model respond
        again. This loops up to MAX_TOOL_ROUNDS times, so the model
        can chain: ls -> read -> respond, or explore -> edit -> commit.

        Because send() now interrupts on tool calls, each round
        processes exactly one tool call with real results before the
        model continues. No more blind chaining.
        """
        self.messages.append({"role": "user", "content": user_input})

        context = self._build_context()
        response_text = self.send(context)
        self.messages.append({"role": "assistant", "content": response_text})

        # Loop: parse tool calls, execute, get followup, repeat
        for round_num in range(MAX_TOOL_ROUNDS):
            actions = _get_actions(response_text, self.skills)
            if not actions:
                break  # no tool calls — model is done acting

            # Execute all actions from this response
            # (usually just one now that streaming interrupts on first tool call)
            results = []
            for action in actions:
                result = self.skills.execute(action)
                if result:
                    results.append(f"[{action['skill']}] {result}")

            if not results:
                break  # actions parsed but nothing executed

            # Feed results back
            self.messages.append({
                "role": "user",
                "content": f"[system: tool results from round {round_num + 1}]\n" + "\n".join(results),
            })

            # Let the model see the results and respond
            print()  # visual separator between rounds
            response_text = self.send(self._build_context())
            self.messages.append({"role": "assistant", "content": response_text})

        self.session.save_turn(user_input, response_text)
        return response_text

    def start_heartbeat(self):
        if self.config.get("heartbeat", {}).get("enabled"):
            self.heartbeat = Heartbeat(self)
            self.heartbeat.start()

    def stop_heartbeat(self):
        if self.heartbeat:
            self.heartbeat.stop()

    def run(self):
        def on_status(status, msg):
            print(f"  [{status}] {msg}")

        if not self.warmup(callback=on_status):
            sys.exit(1)

        self.start_heartbeat()

        id_chars = len(self.identity_text)
        id_tokens_est = id_chars // 4
        num_ctx = self.options.get("num_ctx", 2048)

        print(f"\n  vybn spark agent \u2014 {self.model}")
        print(f"  session: {self.session.session_id}")
        print(f"  identity: {id_chars:,} chars (~{id_tokens_est:,} tokens)")
        print(f"  context window: {num_ctx:,} tokens")
        if id_tokens_est > num_ctx // 2:
            print(f"  \u26a0\ufe0f  WARNING: identity may exceed context window!")
        print(f"  injection: user/assistant pair (template-safe)")
        print(f"  type /bye to exit, /new for fresh session\n")

        try:
            while True:
                try:
                    user_input = input("you: ").strip()
                except EOFError:
                    break

                if not user_input:
                    continue
                if user_input.lower() in ("/bye", "/exit", "/quit"):
                    break

                print("\nvybn: ", end="", flush=True)
                self.turn(user_input)
                print()

        except KeyboardInterrupt:
            pass
        finally:
            self.stop_heartbeat()
            self.session.close()
            print("\n  session saved. vybn out.\n")


def main():
    config = load_config()
    agent = SparkAgent(config)
    agent.run()


if __name__ == "__main__":
    main()
