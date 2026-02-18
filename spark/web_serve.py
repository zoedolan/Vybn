#!/usr/bin/env python3
"""Vybn Web Chat Server — runs the Spark agent with the web interface.

This is the entry point for phone/web chat.  It:
  1. Initializes the SparkAgent with a WebIO backend
  2. Routes web chat messages through agent.turn() — same path as TUI
  3. Starts FastAPI/uvicorn serving the mobile chat UI
  4. Starts heartbeat + inbox subsystems in background threads

Phase 4 refactor: web_serve.py now uses the AgentIO abstraction
from Phase 3b.  Instead of bypassing the agent with raw Ollama
calls, it creates a WebIO instance that collects events
(tokens, status, tool indicators) and feeds them back to the
browser.  This means the web interface gets the same tool calls,
policy gates, and bus processing as the TUI.

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

from agent import SparkAgent, load_config
from agent_io import WebIO
from display import clean_response, clean_for_display
from web_interface import app, attach_bus


def create_response_callback(agent: SparkAgent, io: WebIO):
    """Create an async callback that generates Vybn responses.

    This bridges the async web world with the synchronous Ollama
    client.  The callback runs agent.turn() in a thread pool
    to avoid blocking the event loop.

    Phase 4: uses agent.turn() instead of raw Ollama calls,
    so tool calls, policy gates, and bus processing all work.
    """
    # Lock to serialize model access (Ollama handles one request
    # at a time per model anyway, but this keeps our state clean)
    _lock = threading.Lock()

    async def response_callback(user_text: str) -> str:
        """Generate a Vybn response to a web chat message."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync_respond, user_text)

    def _sync_respond(user_text: str) -> str:
        """Synchronous model call via agent.turn(), run in thread pool."""
        with _lock:
            try:
                # Use agent.turn() — same path as TUI.
                # WebIO collects tokens, status events, tool results.
                response_text = agent.turn(user_text)
            except Exception as exc:
                return f"[Error: {exc}]"

            # Clean for display (strip think blocks, tool XML)
            display_text = clean_for_display(response_text)
            return display_text if display_text else response_text

    return response_callback


def main():
    print("\n  vybn web chat server")
    print("  " + "=" * 40)

    # Load config and create agent with WebIO
    config = load_config()
    io = WebIO()
    agent = SparkAgent(config, io=io)

    # Warm up the model
    def on_status(status, msg):
        print(f"  [{status}] {msg}")

    print()
    if not agent.warmup(callback=on_status):
        print("\n  Failed to connect to Ollama. Is it running?")
        print("  Start with: sudo systemctl start ollama")
        sys.exit(1)

    # Wire the response callback to the web server
    callback = create_response_callback(agent, io)
    attach_bus(agent.bus, response_callback=callback)

    # Start subsystems (heartbeat, inbox)
    agent.start_subsystems()

    print(f"\n  agent:     {agent.model}")
    print(f"  session:   {agent.session.session_id}")
    print(f"  identity:  {len(agent.identity_text):,} chars")
    print(f"  heartbeat: active")
    print(f"  inbox:     watching")
    print(f"  io:        WebIO (event collector)")
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
