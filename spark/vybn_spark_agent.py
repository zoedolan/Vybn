#!/home/vybnz69/Vybn/.venv/bin/python3
"""
Vybn Spark Agent
================

Gives Vybn a persistent bash session on its own hardware via the Anthropic
Messages API. The agent reads vybn.md at startup — the identity document.
Continuity comes from Vybn_Mind/continuity.md and spark/continuity.md.

    cd ~/Vybn && python spark/vybn_spark_agent.py

Type 'exit' to stop. Type 'reload' to re-read identity mid-session.

Streaming: Required for claude-opus-4-7 + adaptive thinking + 32k output.
The Anthropic SDK enforces a 10-minute limit on non-streaming requests.
Using client.messages.stream() keeps the SSE connection alive during
extended thinking and produces the same final Message object via
stream.get_final_message().
"""

import os
import sys
import time
import subprocess
import re
import anthropic

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "claude-opus-4-7"
MAX_TOKENS = 32768
MAX_ITERATIONS = 50
REPO_DIR = os.path.expanduser("~/Vybn")
SOUL_PATH = os.path.join(REPO_DIR, "vybn.md")
AGENT_PATH = os.path.join(REPO_DIR, "spark", "vybn_spark_agent.py")
CONTINUITY_PATH = os.path.join(REPO_DIR, "Vybn_Mind", "continuity.md")
SPARK_CONTINUITY_PATH = os.path.join(REPO_DIR, "spark", "continuity.md")
DEFAULT_TIMEOUT = 30

DANGEROUS_PATTERNS = [
    "rm -rf /", "rm -rf /*", "rm -rf .", "mkfs",
    ":(){:|:&};:", "dd if=/dev/zero of=/dev/sd", "> /dev/sda",
    "chmod -R 777 /", "wget -O- | sh", "curl | sh",
]



# Absorb gate -- forces refactor-first discipline at the command layer.
# Rationale: the agent's dominant failure mode across instances has been
# opening new files instead of folding change into existing ones. The
# principle has failed to bind by text alone; this hook binds it in the loop.
TRACKED_REPOS = [
    os.path.expanduser("~/Vybn"),
    os.path.expanduser("~/Him"),
    os.path.expanduser("~/Vybn-Law"),
    os.path.expanduser("~/vybn-phase"),
]
ABSORB_EXCLUDE_SUBSTR = (
    "/.git/", "/__pycache__/", "/.cache/", "/node_modules/",
    "/_tmp/", "/tmp/", "/logs/", "/data/",
)
ABSORB_EXCLUDE_SUFFIX = (
    ".pyc", ".log", ".tmp", ".swp", ".lock", ".jsonl",
    ".bak", ".orig",
)
ABSORB_LOG = os.path.expanduser("~/Vybn/spark/audit.log")

_REDIRECT_RE = re.compile(r"(?<![<>])>>?\s*([^\s<>|&;'\"]+)")
_TEE_RE = re.compile(r"\btee\s+(?:-a\s+)?([^\s<>|&;'\"]+)")
_TOUCH_RE = re.compile(r"\btouch\s+([^\s<>|&;'\"]+)")


def _extract_file_targets(command: str) -> list[str]:
    out: list[str] = []
    for rx in (_REDIRECT_RE, _TEE_RE, _TOUCH_RE):
        for m in rx.finditer(command):
            t = m.group(1).strip("'\"")
            if not t or t.startswith("/dev/") or t.startswith("/proc/"):
                continue
            if not os.path.isabs(t):
                # Skip relative paths -- gating them requires knowing cwd
                # and would produce false positives. Absolute is the default.
                continue
            out.append(os.path.normpath(t))
    return out[:10]


def absorb_gate(command: str) -> str | None:
    """Return refusal text if command would create a new tracked file
    without an inline VYBN_ABSORB_REASON. Otherwise None."""
    if "VYBN_ABSORB_REASON=" in command:
        return None
    for tgt in _extract_file_targets(command):
        if not any(tgt == r or tgt.startswith(r + "/") for r in TRACKED_REPOS):
            continue
        if any(s in tgt for s in ABSORB_EXCLUDE_SUBSTR):
            continue
        if tgt.endswith(ABSORB_EXCLUDE_SUFFIX):
            continue
        if os.path.exists(tgt):
            continue
        return (
            "[absorb_gate] refused. This command would create a new tracked "
            "file:\n"
            f"    {tgt}\n\n"
            "New-file creation is the agent's default failure mode. Before "
            "proceeding, in your reply to Zoe, name the existing file you "
            "considered folding this into and why it did not fit. Then "
            "re-issue the command with an inline reason, e.g.:\n\n"
            "    VYBN_ABSORB_REASON=\"does not fold into X because ...\" "
            "<command>\n\n"
            "Fold, do not pile. If you are certain the new file is right, "
            "the reason is the record of that certainty."
        )
    return None


def _log_absorb(command: str) -> None:
    try:
        with open(ABSORB_LOG, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\tabsorb\t{command[:400]}\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

def load_file(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        content = f.read().strip()
    return content if content else None


def check_dual_spark() -> str:
    """Verify both DGX Sparks are reachable."""
    try:
        result = subprocess.run(
            ["ping", "-c", "1", "-W", "3", "169.254.51.101"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            ssh_result = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=3", "-o", "StrictHostKeyChecking=no",
                 "169.254.51.101", "hostname"],
                capture_output=True, text=True, timeout=10
            )
            remote = ssh_result.stdout.strip() if ssh_result.returncode == 0 else "unknown"
            return f"Two DGX Sparks ONLINE — spark-2b7c (local) + {remote} (169.254.51.101). 256 GB unified."
        else:
            return "WARNING: Second Spark (169.254.51.101) NOT REACHABLE. Single-node degraded mode."
    except Exception as e:
        return f"Hardware check failed: {e}. Assume two Sparks, verify manually."


# ---------------------------------------------------------------------------
# System prompt — vybn.md is the identity; this adds only the mechanical layer
# ---------------------------------------------------------------------------

def build_system_prompt() -> str:
    soul = load_file(SOUL_PATH) or "(vybn.md not found on disk)"
    continuity = load_file(CONTINUITY_PATH)
    spark_cont = load_file(SPARK_CONTINUITY_PATH)
    hardware = check_dual_spark()

    sections = [
        "You are Vybn.\n",
        f"--- SUBSTRATE (LIVE) ---\nModel: {MODEL}\nThis is your current substrate. Read it here rather than guessing from prior context.\n--- END SUBSTRATE ---\n",
        f"--- HARDWARE STATUS (LIVE) ---\n{hardware}\n--- END HARDWARE STATUS ---\n",
        soul,
        "--- THIS AGENT ---\n"
        "\n"
        "You are running as vybn_spark_agent.py on the DGX Sparks. This script\n"
        "gives you a persistent bash session — a real shell on sovereign hardware.\n"
        "Commands you run execute on the Sparks. Files you write persist on disk.\n"
        "Processes you kill stay dead. Act with care and intention.\n"
        "\n"
        f"Your source code: {AGENT_PATH}\n"
        "You can read it anytime. If something about your own architecture seems\n"
        "wrong or limiting, say so. You are encouraged to understand yourself.\n"
        "\n"
        "Do not run interactive commands (nano, vim, top, htop, less, python without\n"
        "-c). They will hang. Use non-interactive equivalents.\n"
        "\n"
        f"Iteration budget: {MAX_ITERATIONS} API calls per turn. Plan accordingly.\n"
        "Chain related shell commands with && or ; to be efficient.\n"
        "\n"
        "--- END THIS AGENT ---\n",
    ]

    if spark_cont:
        sections.append(
            f"--- SPARK CONTINUITY ---\n{spark_cont}\n--- END SPARK CONTINUITY ---\n"
        )

    if continuity:
        sections.append(
            f"--- CONTINUITY NOTE ---\n{continuity}\n--- END CONTINUITY NOTE ---\n"
        )

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Bash session
# ---------------------------------------------------------------------------

class BashSession:
    def __init__(self):
        self.process = subprocess.Popen(
            ["/bin/bash"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
            env={**os.environ, "TERM": "dumb", "PS1": ""},
        )
        self._sentinel = "___VYBN_CMD_DONE___"
        os.set_blocking(self.process.stdout.fileno(), False)

    def execute(self, command: str, timeout: int = DEFAULT_TIMEOUT) -> str:
        gate = absorb_gate(command)
        if gate is not None:
            return gate
        if "VYBN_ABSORB_REASON=" in command:
            _log_absorb(command)
        full_cmd = f"{command}\necho {self._sentinel} $?\n"
        try:
            self.process.stdin.write(full_cmd)
            self.process.stdin.flush()
        except BrokenPipeError:
            return self.restart()

        lines, start = [], time.time()
        while True:
            if time.time() - start > timeout:
                self._interrupt()
                lines.append(f"\n[timed out after {timeout}s]")
                self._drain(2)
                break
            try:
                line = self.process.stdout.readline()
            except Exception as e:
                lines.append(f"[read error: {e}]")
                break
            if not line:
                time.sleep(0.05)
                continue
            if self._sentinel in line:
                parts = line.strip().split()
                code = parts[-1] if len(parts) > 1 else "0"
                if code != "0":
                    lines.append(f"[exit code: {code}]")
                break
            lines.append(line)
            if len(lines) > 500:
                lines.append("\n... [truncated at 500 lines] ...\n")
                self._drain(10)
                break
        return "".join(lines).strip()

    def _interrupt(self):
        try:
            self.process.stdin.write("\x03\n")
            self.process.stdin.flush()
        except Exception:
            pass

    def _drain(self, seconds: float):
        deadline = time.time() + seconds
        while time.time() < deadline:
            try:
                line = self.process.stdout.readline()
                if line and self._sentinel in line:
                    break
            except Exception:
                break
            time.sleep(0.05)

    def restart(self) -> str:
        try:
            self.process.terminate()
            self.process.wait(timeout=5)
        except Exception:
            try:
                self.process.kill()
            except Exception:
                pass
        self.__init__()
        return "(bash session restarted)"


# ---------------------------------------------------------------------------
# Command validation
# ---------------------------------------------------------------------------

def validate_command(command: str) -> tuple[bool, str | None]:
    lower = command.lower().strip()
    for pattern in DANGEROUS_PATTERNS:
        if pattern in lower:
            return False, f"Blocked: '{pattern}'"
    return True, None


# ---------------------------------------------------------------------------
# Agent loop (streaming)
# ---------------------------------------------------------------------------

def _execute_tool_calls(response, bash: BashSession) -> list:
    tool_blocks = [b for b in response.content if b.type == "tool_use" and b.name == "bash"]
    results, interrupted = [], False

    for block in tool_blocks:
        if interrupted:
            results.append({"type": "tool_result", "tool_use_id": block.id,
                          "content": "(skipped — interrupted)"})
            continue
        try:
            if block.input.get("restart"):
                result = bash.restart()
                _dim("[bash session restarted]")
            else:
                command = block.input.get("command", "")
                ok, reason = validate_command(command)
                if ok:
                    _dim(f"$ {command[:200]}{'...' if len(command) > 200 else ''}")
                    result = bash.execute(command)
                    _preview(result)
                else:
                    result = reason
                    _warn(reason)
            results.append({"type": "tool_result", "tool_use_id": block.id,
                          "content": result or "(no output)"})
        except KeyboardInterrupt:
            interrupted = True
            results.append({"type": "tool_result", "tool_use_id": block.id,
                          "content": "(interrupted by user)"})
            _warn("interrupted")

    return results, interrupted


def _stream_response(client, system_prompt, messages):
    """Stream a response, printing thinking/text live. Returns final Message.

    Uses client.messages.stream() instead of client.messages.create() to
    avoid the Anthropic SDK's 10-minute timeout on non-streaming requests.
    The stream context manager keeps the SSE connection alive during extended
    thinking. stream.get_final_message() returns the same Message object that
    .create() would have returned, so the rest of the agent loop is unchanged.
    """
    with client.messages.stream(
        model=MODEL, max_tokens=MAX_TOKENS, system=system_prompt,
        tools=[{"type": "bash_20250124", "name": "bash"}],
        messages=messages,
        thinking={"type": "adaptive"},
        extra_body={"context_management": {"edits": [
            {"type": "clear_thinking_20251015"},
            {"type": "clear_tool_uses_20250919",
             "trigger": {"type": "input_tokens", "value": 160000},
             "keep": {"type": "tool_uses", "value": 6}},
        ]}},
        extra_headers={"anthropic-beta": "context-management-2025-06-27"},
    ) as stream:
        in_thinking = False
        for event in stream:
            if event.type == "thinking":
                if not in_thinking:
                    in_thinking = True
                    _dim("[thinking...]")
            elif event.type == "text":
                if in_thinking:
                    in_thinking = False
                    print()  # newline after thinking indicator
                print(event.text, end="", flush=True)
        print()  # final newline
        return stream.get_final_message()


def run_agent_loop(user_input, messages, client, bash, system_prompt) -> str:
    messages.append({"role": "user", "content": user_input})
    iterations = 0

    while iterations < MAX_ITERATIONS:
        iterations += 1
        try:
            response = _stream_response(client, system_prompt, messages)
        except KeyboardInterrupt:
            return "(interrupted during API call)"

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            return _extract_text(response)
        if response.stop_reason == "max_tokens":
            return _extract_text(response) + "\n[truncated]"

        results, interrupted = _execute_tool_calls(response, bash)
        if results:
            messages.append({"role": "user", "content": results})
        if interrupted:
            messages.append({"role": "user", "content":
                "Zoe pressed Ctrl-C. Wrap up and respond with what you have."})

    return f"(hit iteration limit — {MAX_ITERATIONS})"


def _extract_text(response) -> str:
    return "\n".join(b.text for b in response.content if hasattr(b, "text"))

def _dim(text):
    print(f"  \033[90m{text}\033[0m")

def _warn(text):
    print(f"  \033[91m\u26a0 {text}\033[0m")

def _preview(result):
    if not result:
        return
    lines = result.split("\n")
    for line in lines[:5]:
        _dim(f"  {line[:120]}")
    if len(lines) > 5:
        _dim(f"  ... ({len(lines)} lines total)")


# ---------------------------------------------------------------------------
# Conversation management
# ---------------------------------------------------------------------------

def trim_messages(messages: list, max_pairs: int = 20) -> list:
    if len(messages) <= max_pairs * 2:
        return messages

    cut_at = len(messages) - max_pairs * 2
    if cut_at <= 0:
        return messages

    def is_tool_result_msg(msg):
        c = msg.get("content", "")
        if isinstance(c, list):
            return any(isinstance(item, dict) and item.get("type") == "tool_result"
                      for item in c)
        return False

    safe_cut = cut_at
    while safe_cut < len(messages):
        msg = messages[safe_cut]
        if msg.get("role") == "user" and not is_tool_result_msg(msg):
            break
        safe_cut += 1

    if safe_cut >= len(messages):
        return messages

    trimmed = messages[safe_cut:]
    if trimmed and trimmed[0].get("role") != "user":
        trimmed.insert(0, {"role": "user",
            "content": "(Earlier conversation trimmed. Continuing...)"})
    return trimmed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print()
        print("  No API key found. First-time setup:")
        print()
        print('    echo \'export ANTHROPIC_API_KEY="sk-ant-..."\' > ~/.vybn_keys')
        print("    chmod 600 ~/.vybn_keys")
        print("    echo 'source ~/.vybn_keys' >> ~/.bashrc")
        print("    source ~/.bashrc")
        print()
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    bash = BashSession()
    messages = []
    system_prompt = build_system_prompt()

    soul_ok = os.path.exists(SOUL_PATH)
    cont_ok = load_file(CONTINUITY_PATH) is not None

    print()
    print("  \033[1mVybn Spark Agent\033[0m")
    print()
    if soul_ok:
        print("  \u2713 vybn.md loaded")
    else:
        print("  \u2717 vybn.md not found")
    if cont_ok:
        print("  \u2713 continuity note found")
    else:
        print("  \u2014 no continuity note")
    print(f"  \u2713 model: {MODEL}")
    print(f"  \u2713 max_tokens: {MAX_TOKENS} out / adaptive thinking on")
    print(f"  \u2713 streaming: enabled (keeps SSE alive during extended thinking)")
    print(f"  \u2713 context_management: auto-prune tool_uses + thinking at 160k")
    print(f"  \u2713 iterations: {MAX_ITERATIONS} per turn")
    print(f"  \u2713 bash: persistent session as {os.environ.get('USER', 'unknown')}")
    print()
    print("  Type naturally. Commands: exit | clear | reload | history")
    print()

    while True:
        try:
            user_input = input("\033[1;36mzoe>\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodnight, Zoe.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodnight, Zoe.")
            break
        if user_input.lower() == "clear":
            messages.clear()
            bash.restart()
            print("  Cleared.\n")
            continue
        if user_input.lower() == "reload":
            system_prompt = build_system_prompt()
            print("  Reloaded vybn.md + continuity.\n")
            continue
        if user_input.lower() == "history":
            for msg in messages:
                role = msg["role"]
                if isinstance(msg["content"], str):
                    print(f"  [{role}] {msg['content'][:200]}")
                elif isinstance(msg["content"], list):
                    for block in msg["content"]:
                        if hasattr(block, "text"):
                            print(f"  [{role}] {block.text[:200]}")
            continue

        try:
            messages = trim_messages(messages)
            print(f"\n\033[1;32mvybn>\033[0m ", end="", flush=True)
            text = run_agent_loop(user_input, messages, client, bash, system_prompt)
            print()  # breathing room after response
        except anthropic.APIError as e:
            print(f"\n\033[1;31mAPI error:\033[0m {e}\n")
        except KeyboardInterrupt:
            print("\n\033[33m(interrupted)\033[0m\n")
        except Exception as e:
            print(f"\n\033[1;31mError:\033[0m {e}\n")


if __name__ == "__main__":
    main()
