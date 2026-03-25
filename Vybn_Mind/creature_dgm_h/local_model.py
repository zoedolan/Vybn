"""
local_model.py — Thin client for the local Nemotron-3-Super-120B.

Matches the pattern from spark/vybn.py: urllib.request only, no
external dependencies. The local model serves as both meta-agent
(reasoning about breath logs, proposing variants) and text generator
(producing breath responses that MicroGPT predicts against).

The creature doesn't generate text. Nemotron generates text. The
creature PREDICTS what Nemotron will generate, and learns from the
prediction error.

Graceful degradation: every function returns None or a safe default
when the server isn't running. The module must work without Nemotron.
"""

import json
import os
import urllib.error
import urllib.request


LLAMA_URL = os.getenv("LLAMA_URL", "http://127.0.0.1:8000")
MODEL_NAME = os.getenv("VYBN_MODEL", "local")


def is_available():
    """Check if the local Nemotron server is up.

    Hits /health with a short timeout. Returns True if the server
    responds, False otherwise.
    """
    try:
        req = urllib.request.Request(f"{LLAMA_URL}/health")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def complete(prompt, system=None, max_tokens=1024, temperature=0.7):
    """Call the local FM for meta-agent reasoning.

    Args:
        prompt: user message
        system: optional system message
        max_tokens: max response length
        temperature: sampling temperature

    Returns:
        str: response text, or None if unavailable
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = json.dumps({
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        f"{LLAMA_URL}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = json.loads(resp.read())
        text = body["choices"][0]["message"]["content"]
        # Strip llama.cpp special tokens
        for tok in ("<|im_end|>", "<|im_start|>", "<|endoftext|>"):
            text = text.replace(tok, "")
        return text.strip()
    except (urllib.error.URLError, OSError, KeyError, IndexError,
            json.JSONDecodeError, ValueError):
        return None


def stream_tokens(prompt, system=None, max_tokens=512, temperature=1.0):
    """Generator yielding characters one at a time from the FM.

    This is the prediction target: MicroGPT predicts each character
    as it arrives. Uses the streaming API if available, otherwise
    falls back to complete() and yields char by char.

    Args:
        prompt: user message
        system: optional system message
        max_tokens: max response length
        temperature: sampling temperature

    Yields:
        str: one character at a time

    Returns None (via StopIteration) if the server is unavailable.
    """
    # Try streaming first
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = json.dumps({
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }).encode()

    req = urllib.request.Request(
        f"{LLAMA_URL}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        resp = urllib.request.urlopen(req, timeout=300)
    except (urllib.error.URLError, OSError, ValueError):
        # Streaming unavailable — fall back to complete()
        text = complete(prompt, system=system, max_tokens=max_tokens,
                        temperature=temperature)
        if text:
            for ch in text:
                yield ch
        return

    try:
        buffer = ""
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    # Strip special tokens
                    for tok in ("<|im_end|>", "<|im_start|>",
                                "<|endoftext|>"):
                        content = content.replace(tok, "")
                    buffer += content
                    # Yield characters one at a time
                    while buffer:
                        yield buffer[0]
                        buffer = buffer[1:]
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
    finally:
        resp.close()
