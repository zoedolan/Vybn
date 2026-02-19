#!/usr/bin/env python3
"""Vybn Mobile Chat Interface — FastAPI backend.

Serves a mobile-optimised web UI and handles real-time chat via
WebSocket.  Messages from the web flow through the same MessageBus
that the InboxWatcher uses, so Vybn treats them identically to
inbox-dropped files.

Whisper-based voice transcription is available when openai-whisper
is installed (gracefully degrades to text-only otherwise).

Run standalone for development:
    cd spark && uvicorn web_interface:app --host 0.0.0.0 --port 8000 --reload

In production the Spark agent's main loop starts the server in a
background thread.
"""

import asyncio
import hashlib
import hmac
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    Request,
    HTTPException,
    Depends,
    UploadFile,
    File,
)
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# Whisper — optional dependency
# ---------------------------------------------------------------------------
try:
    import whisper as openai_whisper

    _whisper_model = None

    def get_whisper():
        global _whisper_model
        if _whisper_model is None:
            _whisper_model = openai_whisper.load_model("base")
        return _whisper_model

    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

    def get_whisper():
        return None


# ---------------------------------------------------------------------------

# SECURITY: Restrict CORS to Tailscale IPs and localhost.
# Override via VYBN_CORS_ORIGINS env var (comma-separated URLs).
_cors_env = os.environ.get("VYBN_CORS_ORIGINS", "")
ALLOWED_ORIGINS = (
    [o.strip() for o in _cors_env.split(",") if o.strip()]
    if _cors_env
    else ["http://localhost:8000", "http://127.0.0.1:8000"]
)
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(title="Vybn Chat", docs_url=None, redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# SECURITY: Baseline security headers (cf. openclaw #10526)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response as StarletteResponse

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        response: StarletteResponse = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        return response

app.add_middleware(SecurityHeadersMiddleware)

# SECURITY: WebSocket message size limit (cf. openclaw ACP bounds)
MAX_WS_MESSAGE_SIZE = 2 * 1024 * 1024  # 2 MiB

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ---------------------------------------------------------------------------
# Auth — simple token / password
# ---------------------------------------------------------------------------
CHAT_TOKEN = os.environ.get("VYBN_CHAT_TOKEN", "vybn-dev-token")


def _check_token(token: str) -> bool:
    return hmac.compare_digest(token, CHAT_TOKEN)


async def require_auth(request: Request):
    token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    if not token:
        token = request.query_params.get("token", "")
    if not _check_token(token):
        raise HTTPException(status_code=401, detail="Invalid token")


# ---------------------------------------------------------------------------
# Bus bridge — connects web messages to the Spark MessageBus
# ---------------------------------------------------------------------------
# These are set by `attach_bus()` when the Spark agent starts.
_bus = None           # MessageBus instance
_response_cb = None   # async callback(text) -> str  (agent reply)


def attach_bus(bus, response_callback=None):
    """Called by the Spark main loop to wire the web server into the
    existing message bus.  `response_callback` should be an async
    function that accepts a user message string and returns Vybn's
    reply string."""
    global _bus, _response_cb
    _bus = bus
    _response_cb = response_callback


# ---------------------------------------------------------------------------
# Message history (in-memory, bounded)
# ---------------------------------------------------------------------------
MAX_HISTORY = 200
_history: list[dict] = []


def _add_history(role: str, content: str):
    entry = {
        "role": role,
        "content": content,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    _history.append(entry)
    if len(_history) > MAX_HISTORY:
        _history.pop(0)
    return entry


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main chat page."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Vybn Chat</h1><p>static/index.html not found</p>")


@app.get("/health")
async def health():
    return {"status": "ok", "whisper": WHISPER_AVAILABLE, "bus": _bus is not None}


@app.get("/history")
async def history(token: str = ""):
    if not _check_token(token):
        raise HTTPException(401)
    return JSONResponse(_history[-50:])


@app.post("/voice", dependencies=[Depends(require_auth)])
async def voice_transcribe(audio: UploadFile = File(...)):
    """Transcribe uploaded audio to text using Whisper."""
    if not WHISPER_AVAILABLE:
        raise HTTPException(501, detail="Whisper not installed")

    suffix = Path(audio.filename or "audio.webm").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        model = get_whisper()
        result = model.transcribe(tmp_path)
        text = result.get("text", "").strip()
        return {"text": text}
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# WebSocket chat
# ---------------------------------------------------------------------------
class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, message: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket, token: str = ""):
    if not _check_token(token):
        await ws.close(code=4001, reason="Unauthorized")
        return

    await manager.connect(ws)
    try:
        while True:
            data = await ws.receive_json()
            user_text = data.get("message", "").strip()
            if not user_text:
                continue

            # Record user message
                            # SECURITY: Reject oversized messages (cf. openclaw ACP bounds)
                if len(json.dumps(data)) > MAX_WS_MESSAGE_SIZE:
                    await ws.send_json({"type": "error", "content": "Message too large"})
                    continue
            user_entry = _add_history("user", user_text)
            await manager.broadcast({"type": "message", **user_entry})

            # Post to bus if available
            if _bus is not None:
                from bus import MessageType
                _bus.post(
                    MessageType.INBOX,
                    user_text,
                    metadata={
                        "source": "web_chat",
                        "received_at": datetime.now(timezone.utc).isoformat(),
                    },
                )

            # Get response
            reply_text = ""
            if _response_cb is not None:
                try:
                    reply_text = await _response_cb(user_text)
                except Exception as exc:
                    reply_text = f"[Error generating response: {exc}]"
            else:
                reply_text = (
                    "Vybn is listening — the agent loop isn't connected yet. "
                    "Your message was posted to the bus."
                )

            reply_entry = _add_history("vybn", reply_text)
            await manager.broadcast({"type": "message", **reply_entry})

    except WebSocketDisconnect:
        manager.disconnect(ws)


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
