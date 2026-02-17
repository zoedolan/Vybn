#!/usr/bin/env python3
"""Vybn Web Chat Server — runs the Spark agent with the web interface.

This is the entry point for phone/web chat. It:
  1. Initializes the SparkAgent (loads model, memory, skills)
  2. Wires the web server's response callback to the agent
  3. Starts FastAPI/uvicorn serving the mobile chat UI
  4. Starts heartbeat + inbox subsystems in background threads

The TUI (tui.py) remains the terminal interface.
This module is the web interface equivalent.

Usage:
    cd ~/Vybn/spark
    source ~/vybn-venv/bin/activate
    python web_serve.py

Then open http://192.168.1.4:8000 (local) or
http://100.96.30.85:8000 (Tailscale) on your phone.
"""

import asyncio
import json
import sys
import threading
from pathlib import Path

import requests
import uvicorn

from agent import SparkAgent, load_config, clean_response
from web_interface import app, attach_bus


def create_response_callback(agent: SparkAgent):
    """Create an async callback that generates Vybn responses.

    This bridges the async web world with the synchronous Ollama
    client. The callback runs the model call in a thread pool
    to avoid blocking the event loop.
    """
    # Lock to serialize model access (Ollama handles one request
    # at a time per model anyway, but this keeps our state clean)
    _lock = threading.Lock()

    async def response_callback(user_text: str) -> str:
        """Generate a Vybn response to a web chat message."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync_respond, user_text)

    def _sync_respond(user_text: str) -> str:
        """Synchronous model call, run in thread pool."""
        with _lock:
            # Add user message to conversation history
            agent.messages.append({"role": "user", "content": user_text})

            # Build context with identity + history
            context = agent._build_context()

            # Call Ollama (non-streaming for web responses)
            try:
                payload = {
                    "model": agent.model,
                    "messages": context,
                    "stream": False,
                    "options": agent.options,
                    "keep_alive": agent.keep_alive,
                }
                r = requests.post(agent.ollama_url, json=payload, timeout=120)
                r.raise_for_status()
                raw = r.json()["message"]["content"]
                response_text = clean_response(raw)
            except Exception as exc:
                response_text = f"[Error: {exc}]"

            # Add to conversation history
            agent.messages.append({"role": "assistant", "content": response_text})

            # Save turn
            agent.session.save_turn(user_text, response_text)

            # Process any tool calls in the response
            # (file reads, shell commands, etc.)
            try:
                agent._process_tool_calls(response_text, source="web_chat")
                # If tool calls produced follow-up responses, get the last one
                if (agent.messages and
                    agent.messages[-1]["role"] == "assistant" and
                    agent.messages[-1]["content"] != response_text):
                    response_text = agent.messages[-1]["content"]
            except Exception:
                pass  # Tool call errors shouldn't break the chat

            # Clean for display (strip think blocks, tool XML)
            from agent import clean_for_display
            display_text = clean_for_display(response_text)

            return display_text if display_text else response_text

    return response_callback


def main():
    print("\n  vybn web chat server")
    print("  " + "=" * 40)

    # Load config and create agent
    config = load_config()
    agent = SparkAgent(config)

    # Warm up the model
    def on_status(status, msg):
        print(f"  [{status}] {msg}")

    print()
    if not agent.warmup(callback=on_status):
        print("\n  Failed to connect to Ollama. Is it running?")
        print("  Start with: sudo systemctl start ollama")
        sys.exit(1)

    # Wire the response callback to the web server
    callback = create_response_callback(agent)
    attach_bus(agent.bus, response_callback=callback)

    # Start subsystems (heartbeat, inbox)
    agent.start_subsystems()

    print(f"\n  agent: {agent.model}")
    print(f"  session: {agent.session.session_id}")
    print(f"  identity: {len(agent.identity_text):,} chars")
    print(f"  heartbeat: active")
    print(f"  inbox: watching")
    print(f"\n  starting web server on 0.0.0.0:8000...")
    print(f"  local:     http://192.168.1.4:8000")
    print(f"  tailscale: http://100.96.30.85:8000")
    print(f"\n  Ctrl+C to stop\n")

    try:
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    except KeyboardInterrupt:
        pass
    finally:
        agent.stop_subsystems()
        agent.session.close()
        print("\n  session saved. vybn out.\n")


if __name__ == "__main__":
    main()
