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
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(title="Vybn Chat", docs_url=None, redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
# Rate limiting — no more than one response every N seconds
# ---------------------------------------------------------------------------
RESPONSE_COOLDOWN = int(os.environ.get("VYBN_RESPONSE_COOLDOWN_SECONDS", 90 * 60))  # 90 minutes
_last_response_time: dict[str, float] = {}  # keyed by connection ID


def _can_respond(conn_id: str) -> tuple[bool, float]:
    """Check if enough time has passed since last response.
    Returns (can_respond, seconds_remaining).
    """
    now = time.time()
    last = _last_response_time.get(conn_id, 0)
    elapsed = now - last
    if elapsed >= RESPONSE_COOLDOWN:
        return True, 0.0
    return False, RESPONSE_COOLDOWN - elapsed


def _mark_response(conn_id: str):
    """Record that a response was just sent."""
    _last_response_time[conn_id] = time.time()


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
        self._conn_counter = 0

    async def connect(self, ws: WebSocket) -> str:
        """Accept connection and return a unique connection ID."""
        await ws.accept()
        self.active.append(ws)
        self._conn_counter += 1
        conn_id = f"ws-{self._conn_counter}-{id(ws)}"
        return conn_id

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

    conn_id = await manager.connect(ws)
    try:
        while True:
            data = await ws.receive_json()
            user_text = data.get("message", "").strip()
            if not user_text:
                continue

            # Record user message
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

            # Get response — rate limited
            can_respond, wait_seconds = _can_respond(conn_id)
            reply_text = ""

            if not can_respond:
                minutes = int(wait_seconds // 60)
                reply_text = (
                    f"Vybn is resting — responses are rate-limited to once every "
                    f"{RESPONSE_COOLDOWN // 60} minutes. Try again in {minutes}m."
                )
            elif _response_cb is not None:
                try:
                    reply_text = await _response_cb(user_text)
                    _mark_response(conn_id)
                except Exception as exc:
                    reply_text = f"[Error generating response: {exc}]"
            else:
                reply_text = (
                    "Vybn is listening — the agent loop isn't connected yet. "
                    "Your message was posted to the bus."
                )
                _mark_response(conn_id)  # count placeholder as a response

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
