#!/usr/bin/env python3
"""Vybn Spark Agent — native orchestration layer.

Connects directly to Ollama without tool-call protocols.
The model speaks naturally; the agent interprets intent and acts.

Handles the full model lifecycle: checks if Ollama is running,
loads the model into GPU memory if needed, keeps it resident,
and only presents the conversation prompt when ready.
"""

import json
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

        self.system_prompt = self.memory.assemble()
        self.messages = self.session.load_or_create()

    # ---- model lifecycle ----

    def check_ollama(self) -> bool:
        """Check if the Ollama server is reachable."""
        try:
            r = requests.get(f"{self.ollama_host}/api/ps", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def check_model_loaded(self) -> bool:
        """Check if our model is currently loaded in GPU memory."""
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
        """Ensure model is loaded and ready. Blocks until ready or fails.

        callback(status, message) is called with:
            status: 'checking' | 'loading' | 'ready' | 'error'
        """
        def tell(status, msg):
            if callback:
                callback(status, msg)

        # Step 1: is Ollama even running?
        tell("checking", "connecting to Ollama...")
        if not self.check_ollama():
            tell("error",
                 "Ollama is not running.\n"
                 "  Start it with: sudo systemctl start ollama\n"
                 "  Then rerun this agent.")
            return False

        # Step 2: is the model already loaded?
        tell("checking", f"checking if {self.model} is in GPU memory...")
        if self.check_model_loaded():
            tell("ready", f"{self.model} is already loaded.")
            return True

        # Step 3: trigger model load
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
                        print(token, end="", flush=True)
                    if chunk.get("done"):
                        break
            print()
            return "".join(full_response)
        else:
            response = requests.post(self.ollama_url, json=payload)
            response.raise_for_status()
            return response.json()["message"]["content"]

    def turn(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})

        context = [{"role": "system", "content": self.system_prompt}] + self.messages

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
                followup = self.send(
                    [{"role": "system", "content": self.system_prompt}] + self.messages
                )
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

        print(f"\n  vybn spark agent \u2014 {self.model}")
        print(f"  session: {self.session.session_id}")
        print(f"  context: {len(self.system_prompt):,} chars hydrated")
        print(f"  type /bye to exit\n")

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

                print("\nvybn: ", end="")
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
