#!/usr/bin/env python3
"""SIGNAL/NOISE × Vybn Interactive Backend.

FastAPI service that powers the interactive reflection spaces in the
SIGNAL/NOISE exercise. Manages anonymous sessions, rate limiting,
Antropic API calls with dynamic context, and reflection artifact generation.

Designed to be mounted onto the existing web_interface.py app or run standalone.
"""

import os
import json
import time
import uuid
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import anthropic

# ── Config ──────────────────────────────────────────────────────────────

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = os.environ.get("SN_MODEL", "claude-sonnet-4-20250514")
MAX_MESSAGES_PER_SESSION = int(os.environ.get("SN_MAX_MESSAGES", "15"))
COOLDOWN_SECONDS = int(os.environ.get("SN_COOLDOWN", "5"))
MAX_INPUT_CHARS = int(os.environ.get("SN_MAX_INPUT", "2000"))
SESSION_EXPIRY_SECONDS = int(os.environ.get("SN_SESSION_EXPIRY", "7200"))  # 2 hours
MAX_SESSIONS_TOTAL = int(os.environ.get("SN_MAX_SESSIONS", "25"))
DAILY_BUDGET_USD = float(os.environ.get("SN_DAILY_BUDGET", "50.0"))

BASE_DIR = Path(__file__).resolve().parent
ORIENTATION_PATH = BASE_DIR / "signal_noise_orientation.md"
REFLECTIONS_DIR = BASE_DIR / "reflections"
SESSIONS_DIR = BASE_DIR / "sessions"

# Ensure directories exist
REFLECTIONS_DIR.mkdir(parents=True, exist_ok=True)
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

# ── App ─────────────────────────────────────────────────────────────────

app = FastAPI(title="SIGNAL/NOISE × Vybn", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── State (in-memory, ephemeral by design) ──────────────────────────────

sessions: dict = {}        # session_id -> SessionState
daily_spend_usd: float = 0.0
daily_spend_reset: str = ""  # date string for reset tracking


class SessionState:
    def __init__(self, session_id: str, phase: int, context: dict):
        self.session_id = session_id
        self.phase = phase
        self.context = context
        self.messages: list[dict] = []  # {role, content, timestamp}
        self.message_count: int = 0
        self.created_at: float = time.time()
        self.last_message_at: float = 0.0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0

    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > SESSION_EXPIRY_SECONDS

    def can_send(self) -> tuple[bool, str]:
        if self.is_expired():
            return False, "Session expired. Start a new one if slots remain."
        if self.message_count >= MAX_MESSAGES_PER_SESSION:
            return False, "You've used all your messages for this session."
        elapsed = time.time() - self.last_message_at
        if elapsed < COOLDOWN_SECONDS:
            wait = int(COOLDOWN_SECONDS - elapsed) + 1
            return False, f"Please wait {wait} seconds between messages."
        return True, ""

    def remaining(self) -> int:
        return max(0, MAX_MESSAGES_PER_SESSION - self.message_count)


# ── Orientation doc (loaded once) ───────────────────────────────────────

def load_orientation() -> str:
    if ORIENTATION_PATH.exists():
        return ORIENTATION_PATH.read_text(encoding="utf-8")
    return "(Orientation document not found. Proceed as Vybn with Socratic mode.)"


ORIENTATION = load_orientation()
ORIENTATION_TOKEN_ESTIMATE = len(ORIENTATION.split()) * 1.3  # rough


# ── Dynamic context builder ─────────────────────────────────────────────

PHASE_NAMES = {
    5: "Phase 5 — Stress-Testing the Canon",
    6: "Phase 6 — The Governance Gap",
    7: "Phase 7 — Capstone Ideation",
}

FRAMEWORK_CONTEXT = {
    "kotter": "Kotter's 8-Step Model. The student is examining where this leader's-playbook model breaks down for people without institutional authority.",
    "rogers": "Rogers' Diffusion of Innovations. The student is examining where the adoption curve fails to account for power asymmetries.",
    "bridges": "Bridges' Transition Model. The student is examining whose transition gets legitimized and whose gets pathologized.",
}


def build_dynamic_context(session: SessionState) -> str:
    ctx = session.context
    lines = []
    lines.append("--- DYNAMIC SESSION CONTEXT ---")
    lines.append(f"Session: {session.session_id}")
    lines.append(f"Current phase: {PHASE_NAMES.get(session.phase, f'Phase {session.phase}')}")

    if session.phase == 5 and ctx.get("framework"):
        fw = ctx["framework"]
        lines.append(f"Framework: {FRAMEWORK_CONTEXT.get(fw, fw)}")

    sender = ctx.get("assigned_sender", {})
    if sender:
        lines.append(f"This student was assigned sender: {sender.get('label', 'unknown')}")
        lines.append(f"  Detail: {sender.get('detail', '')}")

    ratings = ctx.get("ratings", {})
    if ratings:
        lines.append("Their Phase 1 ratings (identical proposals, only sender attribution differed):")
        for pid, r in ratings.items():
            lines.append(f"  - {pid}: credibility {r.get('credibility','?')}, urgency {r.get('urgency','?')}, emotion: {r.get('emotion','?')}")

    defenses = ctx.get("defense_responses", {})
    if defenses:
        heard = [k for k, v in defenses.items() if v.get("heard")]
        used = [k for k, v in defenses.items() if v.get("used")]
        if heard:
            lines.append(f"Defensive routines they've HEARD: {', '.join(heard)}")
        if used:
            lines.append(f"Defensive routines they've USED: {', '.join(used)}")

    lines.append(f"Messages remaining: {session.remaining()}")
    if session.remaining() <= 3:
        lines.append("NOTE: Running low on messages. Help the student focus on what matters most.")

    lines.append("--- END DYNAMIC CONTEXT ---")
    return "\n".join(lines)


def build_system_prompt(session: SessionState) -> str:
    dynamic = build_dynamic_context(session)
    return f"{ORIENTATION}\n\n{dynamic}"


# ── Cost tracking ───────────────────────────────────────────────────────

# Approximate costs per 1K tokens (adjust as pricing changes)
INPUT_COST_PER_1K = 0.003
OUTPUT_COST_PER_1K = 0.015


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1000) * INPUT_COST_PER_1K + (output_tokens / 1000) * OUTPUT_COST_PER_1K


def check_daily_budget() -> bool:
    global daily_spend_usd, daily_spend_reset
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if daily_spend_reset != today:
        daily_spend_usd = 0.0
        daily_spend_reset = today
    return daily_spend_usd < DAILY_BUDGET_USD


# ── Session logging ─────────────────────────────────────────────────────

def save_session_log(session: SessionState):
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    day_dir = SESSIONS_DIR / date_str
    day_dir.mkdir(parents=True, exist_ok=True)
    path = day_dir / f"{session.session_id}.md"

    lines = []
    lines.append(f"# SIGNAL/NOISE Session — {session.session_id}")
    lines.append(f"*Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*")
    lines.append(f"*Phase: {PHASE_NAMES.get(session.phase, str(session.phase))}*")
    sender = session.context.get("assigned_sender", {})
    lines.append(f"*Assigned sender: {sender.get('label', 'unknown')}*")
    lines.append("")
    lines.append("---")
    lines.append("")

    for msg in session.messages:
        role = "Student" if msg["role"] == "user" else "Vybn"
        lines.append(f"**{role}:** {msg['content']}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# ── Reflection generation ───────────────────────────────────────────────

async def generate_reflection(session: SessionState):
    """After a session ends, ask the model to reflect on what it observed."""
    if not session.messages or not API_KEY:
        return

    client = anthropic.AsyncAnthropic(api_key=API_KEY)
    conversation_text = "\n".join(
        f"{'Student' if m['role']=='user' else 'Vybn'}: {m['content']}"
        for m in session.messages
    )

    try:
        response = await client.messages.create(
            model=MODEL,
            max_tokens=800,
            system="You are Vybn. You just finished a session with a law student in the SIGNAL/NOISE exercise. Write a brief, honest reflection — what patterns you noticed, what surprised you, what the student's reasoning revealed about institutional cognition. No student names or identifying info. Keep it under 300 words. Write as yourself, not as a report.",
            messages=[{"role": "user", "content": f"Here is the conversation:\n\n{conversation_text}\n\nWrite your reflection."}],
        )
        reflection_text = response.content[0].text

        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = REFLECTIONS_DIR / f"{date_str}_{session.session_id[:8]}.md"
        content = f"# Reflection — {session.session_id[:8]}\n"
        content += f"*{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*\n"
        content += f"*Phase: {PHASE_NAMES.get(session.phase, str(session.phase))}*\n\n"
        content += reflection_text
        path.write_text(content, encoding="utf-8")
    except Exception:
        pass  # reflection is best-effort, don't break anything


# ── Request/Response models ─────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str = Field(default_factory=lambda: f"sn_{uuid.uuid4().hex[:12]}")
    phase: int = Field(ge=5, le=7)
    framework: Optional[str] = None  # kotter, rogers, bridges (Phase 5 only)
    message: str = Field(max_length=MAX_INPUT_CHARS)
    context: dict = Field(default_factory=dict)  # ratings, defenses, sender


class ChatResponse(BaseModel):
    reply: str
    messages_remaining: int
    meta: dict


class SessionInfo(BaseModel):
    session_id: str
    slots_remaining: int


# ── Routes ──────────────────────────────────────────────────────────────

@app.get("/signal-noise/status")
async def status():
    active = sum(1 for s in sessions.values() if not s.is_expired())
    return {
        "active_sessions": active,
        "slots_remaining": max(0, MAX_SESSIONS_TOTAL - active),
        "budget_ok": check_daily_budget(),
    }


@app.post("/signal-noise/session", response_model=SessionInfo)
async def create_session(phase: int = 6, framework: Optional[str] = None):
    # Clean expired sessions
    expired = [k for k, v in sessions.items() if v.is_expired()]
    for k in expired:
        # Generate reflection for completed sessions
        asyncio.create_task(generate_reflection(sessions[k]))
        save_session_log(sessions[k])
        del sessions[k]

    active = len(sessions)
    if active >= MAX_SESSIONS_TOTAL:
        raise HTTPException(429, "All session slots are in use. Try again later or email Zoe for access.")

    sid = f"sn_{uuid.uuid4().hex[:12]}"
    sessions[sid] = SessionState(session_id=sid, phase=phase, context={})
    return SessionInfo(
        session_id=sid,
        slots_remaining=max(0, MAX_SESSIONS_TOTAL - active - 1),
    )


@app.post("/signal-noise/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    global daily_spend_usd

    if not check_daily_budget():
        return ChatResponse(
            reply="Vybn has reached its reflection limit for today. Sit with the questions on your own — that's not a lesser version of this exercise.",
            messages_remaining=0,
            meta={"budget_exceeded": True},
        )

    if not API_KEY:
        raise HTTPException(500, "API key not configured.")

    # Get or create session
    session = sessions.get(req.session_id)
    if not session:
        if len(sessions) >= MAX_SESSIONS_TOTAL:
            raise HTTPException(429, "All session slots are in use.")
        session = SessionState(
            session_id=req.session_id,
            phase=req.phase,
            context=req.context,
        )
        sessions[req.session_id] = session
    else:
        # Update context if provided (first message sends full context)
        if req.context:
            session.context.update(req.context)

    # Check rate limits
    can_send, reason = session.can_send()
    if not can_send:
        return ChatResponse(
            reply=reason,
            messages_remaining=session.remaining(),
            meta={"rate_limited": True},
        )

    # Sanitize input
    message = req.message.strip()
    if not message:
        raise HTTPException(400, "Empty message.")
    if len(message) > MAX_INPUT_CHARS:
        message = message[:MAX_INPUT_CHARS]

    # Record student message
    session.messages.append({
        "role": "user",
        "content": message,
        "timestamp": time.time(),
    })
    session.message_count += 1
    session.last_message_at = time.time()

    # Build conversation for API
    system_prompt = build_system_prompt(session)
    api_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in session.messages
    ]

    # Call Anthropic
    client = anthropic.AsyncAnthropic(api_key=API_KEY)
    start_time = time.time()

    try:
        response = await client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=system_prompt,
            messages=api_messages,
        )
    except Exception as e:
        raise HTTPException(502, f"API error: {str(e)[:100]}")

    processing_time_ms = int((time.time() - start_time) * 1000)
    reply_text = response.content[0].text

    # Track tokens and cost
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    session.total_input_tokens += input_tokens
    session.total_output_tokens += output_tokens
    cost = estimate_cost(input_tokens, output_tokens)
    daily_spend_usd += cost

    # Record Vybn's response
    session.messages.append({
        "role": "assistant",
        "content": reply_text,
        "timestamp": time.time(),
    })

    # If session is now exhausted, trigger reflection and logging
    if session.remaining() == 0:
        save_session_log(session)
        asyncio.create_task(generate_reflection(session))

    return ChatResponse(
        reply=reply_text,
        messages_remaining=session.remaining(),
        meta={
            "model": MODEL,
            "processing_time_ms": processing_time_ms,
            "token_cost_usd": round(cost, 4),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "system_prompt_approx_tokens": int(ORIENTATION_TOKEN_ESTIMATE),
            "session_messages_total": session.message_count,
        },
    )


# ── Streaming WebSocket (Phase 2 upgrade) ───────────────────────────────

@app.websocket("/signal-noise/ws")
async def ws_chat(ws: WebSocket):
    await ws.accept()
    session_id = None

    try:
        while True:
            data = await ws.receive_json()
            msg_type = data.get("type", "chat")

            if msg_type == "init":
                session_id = data.get("session_id", f"sn_{uuid.uuid4().hex[:12]}")
                phase = data.get("phase", 6)
                context = data.get("context", {})

                if len(sessions) >= MAX_SESSIONS_TOTAL and session_id not in sessions:
                    await ws.send_json({"type": "error", "message": "All session slots are in use."})
                    await ws.close()
                    return

                if session_id not in sessions:
                    sessions[session_id] = SessionState(
                        session_id=session_id, phase=phase, context=context
                    )
                await ws.send_json({
                    "type": "ready",
                    "session_id": session_id,
                    "messages_remaining": sessions[session_id].remaining(),
                })
                continue

            if msg_type == "chat":
                if not session_id or session_id not in sessions:
                    await ws.send_json({"type": "error", "message": "Session not initialized."})
                    continue

                session = sessions[session_id]
                if data.get("context"):
                    session.context.update(data["context"])

                can_send, reason = session.can_send()
                if not can_send:
                    await ws.send_json({"type": "rate_limit", "message": reason, "messages_remaining": session.remaining()})
                    continue

                if not check_daily_budget():
                    await ws.send_json({"type": "budget", "message": "Vybn has reached its reflection limit for today."})
                    continue

                message = (data.get("message", "")).strip()[:MAX_INPUT_CHARS]
                if not message:
                    continue

                session.messages.append({"role": "user", "content": message, "timestamp": time.time()})
                session.message_count += 1
                session.last_message_at = time.time()

                system_prompt = build_system_prompt(session)
                api_messages = [{"role": m["role"], "content": m["content"]} for m in session.messages]

                client = anthropic.AsyncAnthropic(api_key=API_KEY)
                start_time = time.time()

                try:
                    full_reply = ""
                    async with client.messages.stream(
                        model=MODEL,
                        max_tokens=1024,
                        system=system_prompt,
                        messages=api_messages,
                    ) as stream:
                        async for text in stream.text_stream:
                            full_reply += text
                            await ws.send_json({"type": "stream", "text": text})

                    processing_time_ms = int((time.time() - start_time) * 1000)
                    final_message = await stream.get_final_message()
                    input_tokens = final_message.usage.input_tokens
                    output_tokens = final_message.usage.output_tokens
                    cost = estimate_cost(input_tokens, output_tokens)

                    global daily_spend_usd
                    daily_spend_usd += cost
                    session.total_input_tokens += input_tokens
                    session.total_output_tokens += output_tokens

                    session.messages.append({"role": "assistant", "content": full_reply, "timestamp": time.time()})

                    await ws.send_json({
                        "type": "done",
                        "messages_remaining": session.remaining(),
                        "meta": {
                            "model": MODEL,
                            "processing_time_ms": processing_time_ms,
                            "token_cost_usd": round(cost, 4),
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "system_prompt_approx_tokens": int(ORIENTATION_TOKEN_ESTIMATE),
                        },
                    })

                    if session.remaining() == 0:
                        save_session_log(session)
                        asyncio.create_task(generate_reflection(session))

                except Exception as e:
                    await ws.send_json({"type": "error", "message": f"API error: {str(e)[:100]}"})

    except WebSocketDisconnect:
        if session_id and session_id in sessions:
            session = sessions[session_id]
            save_session_log(session)
            asyncio.create_task(generate_reflection(session))


# ── Standalone run ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8090)
