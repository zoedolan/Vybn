#!/usr/bin/env python3
"""Mini-agent pool — Vybn spawns lightweight workers.

When Vybn wants something done in the background (search journals,
analyze a file, draft something), it spawns a mini-agent. The worker
runs on a parallel Ollama context slot and posts its result to the
message bus when done.

The main loop never blocks waiting for agents. Results arrive
asynchronously on the bus and get integrated on the next drain.

Ollama supports OLLAMA_NUM_PARALLEL for concurrent context slots
sharing the same model weights. No duplicate VRAM usage.
"""

import json
import threading
from datetime import datetime, timezone

import requests

from bus import MessageBus, MessageType


class AgentPool:
    def __init__(self, config: dict, bus: MessageBus):
        self.bus = bus
        self.ollama_host = config["ollama"]["host"]
        self.main_model = config["ollama"]["model"]
        self.keep_alive = config["ollama"].get("keep_alive", "30m")

        agent_config = config.get("agents", {})
        self.pool_size = agent_config.get("pool_size", 2)
        self.model = agent_config.get("model") or self.main_model
        self.default_num_predict = agent_config.get("num_predict", 512)

        self._semaphore = threading.Semaphore(self.pool_size)
        self._active = 0
        self._lock = threading.Lock()

    @property
    def active_count(self) -> int:
        with self._lock:
            return self._active

    def spawn(self, task: str, context: str = "", task_id: str = "") -> bool:
        """Spawn a mini-agent to handle a task.

        Returns True if the agent was spawned, False if the pool is full.
        The result lands on the bus as AGENT_RESULT when complete.
        """
        if not self._semaphore.acquire(blocking=False):
            return False

        with self._lock:
            self._active += 1

        thread = threading.Thread(
            target=self._run_agent,
            args=(task, context, task_id),
            daemon=True,
        )
        thread.start()
        return True

    def _run_agent(self, task: str, context: str, task_id: str):
        """Execute a task on a parallel Ollama context slot."""
        try:
            messages = []

            if context:
                messages.append({
                    "role": "system",
                    "content": context,
                })

            messages.append({
                "role": "user",
                "content": task,
            })

            response = requests.post(
                f"{self.ollama_host}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "num_predict": self.default_num_predict,
                        "temperature": 0.5,  # agents should be focused
                    },
                    "keep_alive": self.keep_alive,
                },
                timeout=120,
            )
            response.raise_for_status()
            result = response.json()["message"]["content"]

            self.bus.post(
                MessageType.AGENT_RESULT,
                result,
                metadata={
                    "task_id": task_id or "unnamed",
                    "task": task[:200],
                    "model": self.model,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                },
            )

        except Exception as e:
            self.bus.post(
                MessageType.AGENT_RESULT,
                f"Agent failed: {e}",
                metadata={
                    "task_id": task_id or "unnamed",
                    "task": task[:200],
                    "error": True,
                },
            )

        finally:
            with self._lock:
                self._active -= 1
            self._semaphore.release()
