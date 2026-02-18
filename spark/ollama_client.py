#!/usr/bin/env python3
"""Ollama client — model communication layer for the Vybn Spark Agent.

Handles all direct interaction with the Ollama API:
  - Connection health checks (check_ollama, check_model_loaded)
  - Model loading / warmup
  - Streaming and non-streaming chat completions

Extracted from agent.py in Phase 3 of the refactoring.

No agent state, no conversation history, no tool dispatch.
Pure model I/O — takes messages in, returns text out.
"""

import json

import requests

from parsing import TOOL_CALL_START_TAG, TOOL_CALL_END_TAG
from display import clean_response


class OllamaClient:
    """Thin wrapper around the Ollama /api/chat endpoint."""

    def __init__(self, config: dict):
        self.ollama_host = config["ollama"]["host"]
        self.ollama_url = self.ollama_host + "/api/chat"
        self.model = config["ollama"]["model"]
        self.options = config["ollama"].get("options", {})
        if "num_ctx" not in self.options:
            self.options["num_ctx"] = 16384
        self.keep_alive = config["ollama"].get("keep_alive", "30m")

    # ---- health checks ----

    def check_ollama(self) -> bool:
        """Return True if the Ollama server is reachable."""
        try:
            r = requests.get(f"{self.ollama_host}/api/ps", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def check_model_loaded(self) -> bool:
        """Return True if the configured model is already in GPU memory."""
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

    # ---- warmup ----

    def warmup(self, callback=None) -> bool:
        """Load the model into GPU memory if not already loaded.

        *callback(status, msg)* is called with progress updates:
          status: "checking" | "loading" | "ready" | "error"
        """
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

    # ---- chat completions ----

    def send(self, messages: list, stream: bool = True) -> str:
        """Send messages to the model, return the full response.

        Streaming: displays filtered output in real-time.
          - <think> blocks are suppressed from display
          - <minimax:tool_call> XML is suppressed from display
          - Only actual response prose is shown to the user
          - Full raw text is preserved for tool call parsing
          - Stream interrupted when </minimax:tool_call> detected

        Non-streaming: returns cleaned text silently.
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
