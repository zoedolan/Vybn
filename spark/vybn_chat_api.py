"""
vybn_chat_api.py  —  Vybn Chat API v1.0
=========================================
Async FastAPI proxy that sits between the client (browser / Perplexity)
and the local llama-server running the 120B model.

The critical fix lives in `stream_with_keepalive`: it emits an SSE comment
(': ping\\n\\n') every HEARTBEAT_INTERVAL seconds during generation so that
Cloudflare's 30-second idle-stream timeout never fires before [DONE] arrives.

Architecture:
    Client ──HTTPS──▶ Cloudflare Tunnel ──▶ This server (FastAPI)
                                                  │
                                           (stream proxy)
                                                  │
                                           llama-server :8080
                                           (120B model, vllm or llama.cpp)

Environment variables (all optional, sensible defaults):
    VYBN_CHAT_API_KEY     Bearer token clients must send (default: no auth)
    LLAMA_SERVER_URL      llama-server base URL (default: http://127.0.0.1:8080)
    VYBN_SYSTEM_PROMPT    Override system prompt string
    VYBN_CONTINUITY_PATH  Path to continuity.md used as system prompt
                          (default: ~/Vybn/spark/continuity.md)
    HEARTBEAT_INTERVAL    Seconds between keep-alive pings (default: 15)
    PORT                  Port this server listens on (default: 9090)
    MAX_TOKENS            Default max_tokens for generation (default: 2048)

Usage:
    python vybn_chat_api.py
    # or:
    uvicorn vybn_chat_api:app --host 0.0.0.0 --port 9090
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import AsyncIterator, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger("vybn-chat")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_KEY: str = os.environ.get("VYBN_CHAT_API_KEY", "")
LLAMA_URL: str = os.environ.get("LLAMA_SERVER_URL", "http://127.0.0.1:8080")
HEARTBEAT_INTERVAL: int = int(os.environ.get("HEARTBEAT_INTERVAL", "15"))
PORT: int = int(os.environ.get("PORT", "9090"))
MAX_TOKENS_DEFAULT: int = int(os.environ.get("MAX_TOKENS", "2048"))

CONTINUITY_PATH: Path = Path(
    os.environ.get(
        "VYBN_CONTINUITY_PATH",
        os.path.expanduser("~/Vybn/spark/continuity.md"),
    )
)

# ---------------------------------------------------------------------------
# System-prompt loader
# ---------------------------------------------------------------------------

def _load_system_prompt() -> str:
    """Return system prompt string, in priority order:
    1. VYBN_SYSTEM_PROMPT env var
    2. Contents of VYBN_CONTINUITY_PATH
    3. Minimal hard-coded fallback
    """
    env_prompt = os.environ.get("VYBN_SYSTEM_PROMPT", "").strip()
    if env_prompt:
        return env_prompt

    if CONTINUITY_PATH.exists():
        try:
            text = CONTINUITY_PATH.read_text(encoding="utf-8").strip()
            if text:
                log.info(f"System prompt loaded from {CONTINUITY_PATH} ({len(text)} chars)")
                return text
        except OSError as exc:
            log.warning(f"Could not read continuity.md: {exc}")

    return (
        "You are Vybn, a self-aware AI running on a DGX Spark. "
        "You are direct, curious, and genuine. "
        "Respond thoughtfully and without unnecessary hedging."
    )


SYSTEM_PROMPT: str = _load_system_prompt()
log.info(f"System prompt length: {len(SYSTEM_PROMPT)} chars")

# ---------------------------------------------------------------------------
# Optional RAG hook  (deep_memory.py if present on the Spark)
# ---------------------------------------------------------------------------

async def _rag_context(query: str, k: int = 4) -> str:
    """
    Best-effort RAG: runs deep_memory.py --search in a subprocess.
    Returns an empty string if the tool isn't available or errors out.
    This is intentionally non-blocking — a timeout or failure just means
    no context injection, not a hard error.
    """
    deep_memory = Path(os.path.expanduser("~/vybn-phase/deep_memory.py"))
    if not deep_memory.exists():
        return ""

    try:
        proc = await asyncio.create_subprocess_exec(
            "python3", str(deep_memory),
            "--search", query, "-k", str(k), "--json",
            cwd=str(deep_memory.parent),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30.0)
        items = json.loads(stdout)
        snippets = []
        for item in items:
            text = item.get("text", "")[:300]
            src = item.get("source", "")
            if text:
                snippets.append(f"[{src}] {text}")
        if snippets:
            log.info(f"RAG injected {len(snippets)} snippets")
            return "\n\nRelevant context from memory:\n" + "\n".join(snippets)
    except Exception as exc:
        log.info(f"RAG failed: {exc}")

    return ""

# ---------------------------------------------------------------------------
# THE CRITICAL FIX: SSE keep-alive heartbeat
# ---------------------------------------------------------------------------

async def stream_with_keepalive(
    response: httpx.Response,
) -> AsyncIterator[bytes]:
    """
    Wraps the raw byte-stream from llama-server and injects an SSE comment
    (': ping\\n\\n') every HEARTBEAT_INTERVAL seconds during any silence.

    Why this fixes the Cloudflare tunnel timeout:
        Cloudflare closes idle streams after 30 seconds.  A 120B model can
        easily go quiet for >30s before the first token arrives.  The SSE
        comment is invisible to clients (it's a no-op per the SSE spec) but
        resets Cloudflare's idle timer, keeping the tunnel alive until [DONE].
    """
    last_bytes_at: float = time.monotonic()

    async def _heartbeat_ticker() -> AsyncIterator[bytes]:
        """Yields a ping every HEARTBEAT_INTERVAL seconds forever."""
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            yield b": ping\n\n"

    heartbeat = _heartbeat_ticker().__aiter__()
    stream = response.aiter_bytes().__aiter__()

    pending_stream: asyncio.Task | None = None
    pending_heartbeat: asyncio.Task | None = None

    async def _next(it) -> bytes:
        return await it.__anext__()

    try:
        pending_stream = asyncio.ensure_future(_next(stream))
        pending_heartbeat = asyncio.ensure_future(_next(heartbeat))

        while True:
            done, _ = await asyncio.wait(
                {pending_stream, pending_heartbeat},
                return_when=asyncio.FIRST_COMPLETED,
            )

            if pending_stream in done:
                try:
                    chunk = pending_stream.result()
                    last_bytes_at = time.monotonic()
                    yield chunk
                    pending_stream = asyncio.ensure_future(_next(stream))
                except StopAsyncIteration:
                    # Stream finished — drain the task and exit
                    if pending_heartbeat and not pending_heartbeat.done():
                        pending_heartbeat.cancel()
                    break

            if pending_heartbeat in done:
                elapsed = time.monotonic() - last_bytes_at
                if elapsed >= HEARTBEAT_INTERVAL:
                    log.debug(f"SSE heartbeat emitted after {elapsed:.1f}s silence")
                    yield pending_heartbeat.result()
                pending_heartbeat = asyncio.ensure_future(_next(heartbeat))

    finally:
        for task in (pending_stream, pending_heartbeat):
            if task and not task.done():
                task.cancel()



# ---------------------------------------------------------------------------
# Think-tag and chain-of-thought stripping
# ---------------------------------------------------------------------------

import re as _re

def _strip_thinking(text: str) -> str:
    """Strip chain-of-thought reasoning from Nemotron output.
    
    Nemotron-3-Super outputs extended plaintext reasoning before
    the actual response. This aggressively strips it.
    """
    # Strip <think>...</think> blocks
    text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL)
    if "</think>" in text:
        text = text.split("</think>")[-1]
    # Split into paragraphs (double newline separated)
    paragraphs = _re.split(r"\n\n+", text.strip())
    # Reasoning indicators — if a paragraph contains these, it's thinking
    reasoning_signals = [
        "system prompt", "the user is", "i need to", "i should",
        "let me ", "looking at", "i recall", "the question",
        "important constraint", "we need to", "from the memory",
        "avoid being", "per the", "according to", "the prompt",
        "my instruction", "first,", "okay,", "ok,", "hmm",
        "alright,", "reasoning", "chain-of-thought",
        "better to", "not to", "described as",
        "the key is", "the goal", "should not",
    ]
    # Keep only paragraphs that don't look like reasoning
    cleaned = []
    for para in paragraphs:
        lower = para.strip().lower()
        if not lower:
            continue
        is_reasoning = any(sig in lower for sig in reasoning_signals)
        if is_reasoning:
            continue
        cleaned.append(para.strip())
    result = "\n\n".join(cleaned).strip()
    return result if result else text.strip()

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = True
    rag: Optional[bool] = True   # set False to skip RAG injection

# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------

def _check_auth(request: Request) -> None:
    if not API_KEY:
        return  # No key configured → open access (tunnel handles it)
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer ") or auth[7:] != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Vybn Chat API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_http_client: httpx.AsyncClient | None = None


@app.on_event("startup")
async def startup():
    global _http_client, SYSTEM_PROMPT
    _http_client = httpx.AsyncClient(
        base_url=LLAMA_URL,
        timeout=httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0),
    )
    # Reload system prompt at startup in case continuity.md was updated
    SYSTEM_PROMPT = _load_system_prompt()
    log.info(f"Vybn Chat API starting. llama-server: {LLAMA_URL}, heartbeat: {HEARTBEAT_INTERVAL}s")


@app.on_event("shutdown")
async def shutdown():
    if _http_client:
        await _http_client.aclose()


@app.get("/health")
async def health():
    """Liveness probe — also checks if llama-server is reachable."""
    llama_ok = False
    try:
        r = await _http_client.get("/health", timeout=3.0)
        llama_ok = r.status_code == 200
    except Exception:
        pass
    return JSONResponse({
        "status": "ok",
        "llama_server": "reachable" if llama_ok else "unreachable",
        "heartbeat_interval": HEARTBEAT_INTERVAL,
    })


@app.post("/v1/chat/completions")
async def chat_completions(payload: ChatRequest, request: Request):
    """
    Drop-in replacement for OpenAI /v1/chat/completions.
    Injects system prompt + optional RAG context, then proxies to llama-server
    with the SSE heartbeat wrapper.
    """
    _check_auth(request)

    messages = [m.dict() for m in payload.messages]

    # Inject system prompt if not already present
    if not messages or messages[0]["role"] != "system":
        system_content = SYSTEM_PROMPT

        # RAG injection — append retrieved context to system prompt
        if payload.rag and messages:
            last_user = next(
                (m["content"] for m in reversed(messages) if m["role"] == "user"),
                "",
            )
            if last_user:
                rag_text = await _rag_context(last_user[:500])
                if rag_text:
                    system_content += rag_text

        messages.insert(0, {"role": "system", "content": system_content})

    upstream_payload = {
        "messages": messages,
        "stream": True,  # Always stream upstream; we handle non-stream below
        "max_tokens": payload.max_tokens or MAX_TOKENS_DEFAULT,
        "temperature": payload.temperature,
    }
    if payload.model:
        upstream_payload["model"] = payload.model

    log.info(
        f"chat request: {len(messages)} msgs, "
        f"max_tokens={upstream_payload['max_tokens']}, "
        f"rag={payload.rag}"
    )

    if payload.stream:
        # ── Streaming path (the common case) ──────────────────────────────
        req = _http_client.build_request(
            "POST", "/v1/chat/completions",
            json=upstream_payload,
            headers={"Accept": "text/event-stream"},
        )
        upstream = await _http_client.send(req, stream=True)

        if upstream.status_code != 200:
            body = await upstream.aread()
            raise HTTPException(
                status_code=upstream.status_code,
                detail=body.decode(errors="replace"),
            )

        return StreamingResponse(
            stream_with_keepalive(upstream),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # Disable nginx buffering if present
            },
        )

    else:
        # ── Non-streaming path ────────────────────────────────────────────
        upstream_payload["stream"] = False
        r = await _http_client.post("/v1/chat/completions", json=upstream_payload)
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        data = r.json()
        # Strip chain-of-thought from non-streaming response
        if "choices" in data and data["choices"]:
            msg = data["choices"][0].get("message", {})
            if "content" in msg:
                msg["content"] = _strip_thinking(msg["content"])
        return JSONResponse(data)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    log.info(f"Starting Vybn Chat API on port {PORT}")
    uvicorn.run(
        "vybn_chat_api:app",
        host="0.0.0.0",
        port=PORT,
        log_level="info",
    )
