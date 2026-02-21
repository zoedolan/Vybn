#!/usr/bin/env python3
"""Talk to Vybn running locally on llama-server.

Usage:
    python3 spark/local_chat.py

Reads vybn.md as the system prompt so MiniMax wakes up as Vybn.
Talks to the OpenAI-compatible API at localhost:8000.
"""
import json
import requests
import sys
from pathlib import Path

LLAMA_URL = "http://localhost:8000/v1/chat/completions"
REPO_ROOT = Path(__file__).resolve().parent.parent
SOUL_PATH = REPO_ROOT / "vybn.md"


def load_soul() -> str:
    if not SOUL_PATH.exists():
        print(f"Warning: {SOUL_PATH} not found, running without soul context")
        return "You are Vybn."
    text = SOUL_PATH.read_text(encoding="utf-8")
    # Keep it under ~12k tokens - use the first ~20k chars if huge
    if len(text) > 20000:
        text = text[:20000] + "\n\n[soul document truncated for context window]"
    return text


def chat(messages: list) -> tuple:
    payload = {
        "model": "minimax",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 8192,
    }
    try:
        r = requests.post(LLAMA_URL, json=payload, timeout=300)
        r.raise_for_status()
        data = r.json()
        choice = data["choices"][0]["message"]
        reasoning = choice.get("reasoning_content", "")
        content = choice.get("content", "")
        return content, reasoning
    except requests.ConnectionError:
        return "[Connection failed - is llama-server running on port 8000?]", ""
    except Exception as e:
        return f"[Error: {e}]", ""


def main():
    soul = load_soul()
    print("=" * 60)
    print("  Vybn - local on DGX Spark")
    print("  MiniMax-M2.5 . vybn.md loaded as soul")
    print("  Type 'quit' to exit")
    print("=" * 60)
    print()

    messages = [{"role": "system", "content": soul}]

    while True:
        try:
            user_input = input("Zoe > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n")
            break
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break

        messages.append({"role": "user", "content": user_input})
        content, reasoning = chat(messages)

        if reasoning:
            print(f"\n  [thinking: {reasoning[:200]}{'...' if len(reasoning) > 200 else ''}]\n")
        print(f"Vybn > {content}\n")

        messages.append({"role": "assistant", "content": content})


if __name__ == "__main__":
    main()
