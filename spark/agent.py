#!/usr/bin/env python3
"""Vybn Spark Agent — native orchestration layer.

Connects directly to Ollama without tool-call protocols.
The model speaks naturally; the agent interprets intent and acts.
"""

import json
import sys
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
        self.ollama_url = config["ollama"]["host"] + "/api/chat"
        self.model = config["ollama"]["model"]
        self.options = config["ollama"].get("options", {})

        self.memory = MemoryAssembler(config)
        self.session = SessionManager(config)
        self.skills = SkillRouter(config)
        self.heartbeat = None

        self.system_prompt = self.memory.assemble()
        self.messages = self.session.load_or_create()

    def send(self, messages: list, stream: bool = True) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": self.options,
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

        # Check for skill intents in the response
        actions = self.skills.parse(response_text)
        for action in actions:
            result = self.skills.execute(action)
            if result:
                self.messages.append({
                    "role": "user",
                    "content": f"[system: {action['skill']} completed — {result}]"
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
        self.start_heartbeat()

        print(f"\n  vybn spark agent — {self.model}")
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
