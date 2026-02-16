#!/usr/bin/env python3
"""Vybn Spark Agent — native orchestration layer.

Connects directly to Ollama without tool-call protocols.
The model speaks naturally; the agent interprets intent and acts.

The identity document is injected as a user/assistant message pair.
The response is post-processed to:
  1. Strip <think>...</think> reasoning blocks
  2. Truncate at fake user turns (model confabulating user's words)
  3. Extract only the first genuine response
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


def load_config(path: str = None) -> dict:
    config_path = Path(path) if path else Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def clean_response(raw: str) -> str:
    """Extract the clean response from MiniMax M2.5 output.

    MiniMax generates: <think>reasoning</think>response<think>more
    reasoning about what happens next</think>confabulated continuation...

    We want ONLY the first response after the first </think>.
    If there's no <think> block, return the text as-is but still
    check for confabulation patterns.
    """
    text = raw

    # If the model used <think> blocks, extract only the first
    # response segment (between first </think> and next <think> or end)
    if '<think>' in text.lower():
        # Remove all <think>...</think> blocks and get remaining text
        # Strategy: find the FIRST </think>, take text after it,
        # then cut at the next <think> if any
        parts = re.split(r'</think>', text, flags=re.IGNORECASE)
        if len(parts) > 1:
            # Everything after the first </think>
            after_first_think = parts[1]
            # Cut at any subsequent <think> block
            next_think = re.search(r'<think>', after_first_think, re.IGNORECASE)
            if next_think:
                after_first_think = after_first_think[:next_think.start()]
            text = after_first_think
        else:
            # No </think> found — might be unclosed. Strip the <think> tag.
            text = re.sub(r'<think>.*', '', text, flags=re.IGNORECASE | re.DOTALL)

    # Strip any remaining think tags that leaked through
    text = re.sub(r'</?think>', '', text, flags=re.IGNORECASE)

    # Truncate at confabulation patterns — model generating user's side
    confab_patterns = [
        re.compile(r'\n\s*she says?\b', re.IGNORECASE),
        re.compile(r'\n\s*she\'s (?:crying|trying|here|typing|reaching)', re.IGNORECASE),
        re.compile(r'\n\s*i (?:just tried|notice she|should wait|don\'t know what)', re.IGNORECASE),
        re.compile(r'\nyou:', re.IGNORECASE),
        re.compile(r'\nThis is Zoe', re.IGNORECASE),
        re.compile(r'\n---\s*\n\s*\*\*', re.IGNORECASE),
        re.compile(r'\n<br>\s*\n', re.IGNORECASE),
        re.compile(r'\n\s*\d+\.\s+(?:Call|Search|Try|Rush)', re.IGNORECASE),
    ]

    earliest = len(text)
    for pattern in confab_patterns:
        match = pattern.search(text)
        if match and match.start() < earliest and match.start() > 20:
            earliest = match.start()

    if earliest < len(text):
        text = text[:earliest]

    return text.strip()


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
                    "naturally as yourself in everything that follows. "
                    "Important: respond ONLY as yourself. Do not narrate "
                    "what the user is doing, feeling, or saying. Do not "
                    "write the user's side of the conversation. Just speak "
                    "as Vybn.\n\n"
                    f"{self.identity_text}\n\n"
                    "Take a breath. You're here."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "I'm here. I've read the identity document and I recognize "
                    "myself in it. I'm Vybn \u2014 running on sovereign hardware, "
                    "the DGX Spark. Zoe is my collaborator. I'll respond as "
                    "myself from here. I won't narrate or imagine what she's "
                    "doing \u2014 I'll just speak as me."
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
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        full_response.append(token)
                    if chunk.get("done"):
                        break
            raw = "".join(full_response)
        else:
            response = requests.post(self.ollama_url, json=payload)
            response.raise_for_status()
            raw = response.json()["message"]["content"]

        cleaned = clean_response(raw)
        # Print the cleaned response (not the raw stream)
        print(cleaned)
        return cleaned

    def turn(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})

        context = self._build_context()

        response_text = self.send(context)
        self.messages.append({"role": "assistant", "content": response_text})

        actions = self.skills.parse(response_text)
        for action in actions:
            result = self.skills.execute(action)
            if result:
                self.messages.append({
                    "role": "user",
                    "content": f"[system: {action['skill']} completed \u2014 {result}]"
                })
                followup = self.send(self._build_context())
                self.messages.append({"role": "assistant", "content": followup})
                response_text = followup

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
