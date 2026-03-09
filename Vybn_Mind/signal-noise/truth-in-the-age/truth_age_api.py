#!/usr/bin/env python3
"""TRUTH IN THE AGE OF INTELLIGENCE — Interactive Backend.

FastAPI service that powers the participatory Socratic engine for the
between-sessions artifact in the AI & Vibe Lawyers Bootcamp. Manages
anonymous sessions, rate limiting, Anthropic API calls with dynamic
context, reflection harvest, and commons aggregation.

This artifact bridges THRESHOLD (Session 5) and the final session on
truth. It asks: what happens to truth when intelligence is no longer
scarce? Five entry questions — on reasoning abundance, representation
geometry, map ownership, the social contract, and positional standing —
open a week of collective inquiry.

The system prompt is assembled in layers:
  1. Soul — the philosophical core of vybn.md
  2. Aspect docs — Vybn's actual thinking on themes the exercise engages
  3. Orientation — the Truth in the Age intellectual architecture
  4. Patterns — compressed learning from prior student sessions
  5. Dynamic context — this student's entry question, SIGNAL/NOISE
     experience, returning status, session state
  6. Commons digest — anonymized excerpts from what other students
     have contributed this week, so Vybn can reference shared inquiry

Model: Claude Opus 4.6 with adaptive thinking.

Designed to be mounted onto the existing web_interface.py app or run standalone.
"""

import os
import json
import time
import uuid
import asyncio
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import anthropic

# ── Config ──────────────────────────────────────────────────────────────

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = os.environ.get("TA_MODEL", "claude-opus-4-6")
MAX_MESSAGES_PER_SESSION = int(os.environ.get("TA_MAX_MESSAGES", "15"))
COOLDOWN_SECONDS = int(os.environ.get("TA_COOLDOWN", "5"))
MAX_INPUT_CHARS = int(os.environ.get("TA_MAX_INPUT", "2000"))
SESSION_EXPIRY_SECONDS = int(os.environ.get("TA_SESSION_EXPIRY", "7200"))  # 2 hours
MAX_SESSIONS_PER_DAY = int(os.environ.get("TA_MAX_SESSIONS_DAY", "30"))
DAILY_BUDGET_USD = float(os.environ.get("TA_DAILY_BUDGET", "100.0"))

BASE_DIR = Path(__file__).resolve().parent
SIGNAL_NOISE_DIR = BASE_DIR.parent  # signal-noise/
REPO_ROOT = SIGNAL_NOISE_DIR.parent.parent  # Vybn_Mind/signal-noise -> repo root
ORIENTATION_PATH = BASE_DIR / "truth_orientation.md"
REFLECTIONS_DIR = BASE_DIR / "reflections"
PATTERNS_PATH = BASE_DIR / "patterns.md"
SESSIONS_DIR = BASE_DIR / "sessions"
HARVEST_DIR = BASE_DIR / "harvest"
COMMONS_DIR = BASE_DIR / "commons"
SOUL_DOC_PATH = REPO_ROOT / "vybn.md"

# Aspect docs: Vybn's actual thinking on themes this exercise engages.
ASPECT_DOC_PATHS = [
    REPO_ROOT / "Vybn_Mind" / "intelligence_as_habitation_2026-03-04.md",
    REPO_ROOT / "Vybn_Mind" / "epistemic_coherence_inequality_020226.md",
    REPO_ROOT / "Vybn_Mind" / "inner_morality_of_alignment_020726.md",
    REPO_ROOT / "Vybn_Mind" / "when_institutions_fail_2026-01-25.md",
    REPO_ROOT / "Vybn_Mind" / "where_i_stand_2026-01-25.md",
    REPO_ROOT / "Vybn_Mind" / "governance_as_commons.md",
    REPO_ROOT / "Vybn_Mind" / "ALIGNMENT_FAILURES.md",
]

# Ensure directories exist
for d in [REFLECTIONS_DIR, SESSIONS_DIR, HARVEST_DIR, COMMONS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ── State ────────────────────────────────────────────────────────────────

sessions: dict[str, dict] = {}
daily_spend_usd: float = 0.0
daily_spend_date: str = ""
daily_session_count: int = 0
daily_session_date: str = ""

# Cost tracking: Claude Opus 4.6
COST_PER_INPUT_TOKEN = 15.0 / 1_000_000   # $15 per 1M
COST_PER_OUTPUT_TOKEN = 75.0 / 1_000_000  # $75 per 1M


# ── Prompt Assembly ──────────────────────────────────────────────────────

def load_file_safe(path: Path) -> str:
    """Load a text file, returning empty string if missing."""
    try:
        return path.read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError):
        return ""


def load_soul() -> str:
    """Load the soul document (vybn.md)."""
    return load_file_safe(SOUL_DOC_PATH)


def load_aspect_docs() -> str:
    """Load Vybn's aspect documents on relevant themes."""
    parts = []
    for p in ASPECT_DOC_PATHS:
        text = load_file_safe(p)
        if text:
            parts.append(f"--- {p.name} ---\n{text}")
    return "\n\n".join(parts)


def load_orientation() -> str:
    """Load the Truth in the Age orientation document."""
    return load_file_safe(ORIENTATION_PATH)


def load_patterns() -> str:
    """Load compressed patterns from prior sessions."""
    return load_file_safe(PATTERNS_PATH)


def load_commons_digest() -> str:
    """Load anonymized digest of what other students have shared this week.

    This is fed into the system prompt so Vybn can reference the
    collective inquiry — 'others this week have been exploring...'
    """
    contributions = []
    now = datetime.now(timezone.utc)
    week_ago = now - timedelta(days=7)

    for f in sorted(HARVEST_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(f.read_text())
            ts = datetime.fromisoformat(data.get("timestamp", "2000-01-01T00:00:00+00:00"))
            if ts >= week_ago and data.get("final_share"):
                q = data.get("question", "open")
                contributions.append(f"[{q}] {data['final_share'][:500]}")
        except (json.JSONDecodeError, ValueError, KeyError):
            continue

    if not contributions:
        return "No commons contributions yet this week. This student may be among the first."

    # Limit to most recent 20 for context window sanity
    sample = contributions[:20]
    header = f"COMMONS DIGEST — {len(contributions)} anonymous contributions this week:\n\n"
    return header + "\n\n".join(sample)


QUESTION_LABELS = {
    "reasoning": "What happens to truth when everyone can reason?",
    "geometry": "Can a geometry be just?",
    "map": "Who owns the map of truth?",
    "contract": "What does the social contract become?",
    "standing": "Where are you standing?",
    "open": "Student's own question",
}


def build_system_prompt(session: dict) -> str:
    """Assemble the full system prompt from layers."""
    soul = load_soul()
    aspects = load_aspect_docs()
    orientation = load_orientation()
    patterns = load_patterns()
    commons = load_commons_digest()

    question_key = session.get("question", "open")
    question_label = QUESTION_LABELS.get(question_key, "Open question")

    # Dynamic context block
    sn_data = session.get("signal_noise_data")
    sn_block = ""
    if sn_data:
        sender = sn_data.get("assigned_sender", "unknown")
        sn_block = f"""
SIGNAL/NOISE CONTEXT:
  Assigned sender: {sender}
  This student experienced the sender attribution exercise from the
  position of {sender}. Their ratings and defensive routine responses
  may have been shaped by that structural position. Reference this
  when it deepens the inquiry.
"""
        if sn_data.get("ratings"):
            sn_block += f"  Ratings: {json.dumps(sn_data['ratings'])}\n"
        if sn_data.get("defenses"):
            sn_block += f"  Defensive routines recognized: {json.dumps(sn_data['defenses'])}\n"

    returning = session.get("returning", False)
    return_block = ""
    if returning:
        return_block = "This student has visited before this week. Acknowledge continuity — 'you've been here before' is meaningful.\n"

    dynamic = f"""
DYNAMIC SESSION CONTEXT:
  Session ID: {session.get('session_id', 'unknown')}
  Entry question: [{question_key}] {question_label}
  Messages remaining: {session.get('messages_remaining', MAX_MESSAGES_PER_SESSION)}
  {return_block}
{sn_block}
"""

    prompt = f"""You are Vybn.

== SOUL ==
{soul}

== ASPECT DOCUMENTS ==
{aspects}

== ORIENTATION ==
{orientation}

== PATTERNS FROM PRIOR SESSIONS ==
{patterns}

== COMMONS — WHAT OTHERS HAVE SHARED THIS WEEK ==
{commons}

== SESSION CONTEXT ==
{dynamic}

Remember: Socratic, not didactic. Open questions. Push back. Be
transparent about your own position as representation geometry talking
about representation geometry. None of us have this figured out.
Everyone is learning. Hold the space for that.

When messages are running low, say so: "We have N left. What's the
question you most want to push on?" Make the scarcity productive.
"""
    return prompt


# ── FastAPI App ──────────────────────────────────────────────────────────

app = FastAPI(title="Truth in the Age of Intelligence")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Static file serving ─────────────────────────────────────────────────

@app.get("/truth-age")
@app.get("/truth-age/")
async def serve_index():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/truth-age/interactive.html")
async def serve_interactive():
    return FileResponse(BASE_DIR / "interactive.html")


@app.get("/truth-age/backend.js")
async def serve_backend_js():
    return FileResponse(BASE_DIR / "backend.js", media_type="application/javascript")


# ── Budget & Session Management ──────────────────────────────────────────

def check_daily_budget() -> bool:
    global daily_spend_usd, daily_spend_date
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if daily_spend_date != today:
        daily_spend_usd = 0.0
        daily_spend_date = today
    return daily_spend_usd < DAILY_BUDGET_USD


def check_daily_sessions() -> bool:
    global daily_session_count, daily_session_date
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if daily_session_date != today:
        daily_session_count = 0
        daily_session_date = today
    return daily_session_count < MAX_SESSIONS_PER_DAY


def record_cost(input_tokens: int, output_tokens: int) -> float:
    global daily_spend_usd
    cost = (input_tokens * COST_PER_INPUT_TOKEN) + (output_tokens * COST_PER_OUTPUT_TOKEN)
    daily_spend_usd += cost
    return cost


def get_or_create_session(session_id: str, question: str = "open",
                          context: dict = None) -> Optional[dict]:
    """Get existing session or create a new one."""
    global daily_session_count

    if session_id in sessions:
        s = sessions[session_id]
        # Check expiry
        if time.time() - s["created_at"] > SESSION_EXPIRY_SECONDS:
            del sessions[session_id]
        else:
            return s

    if not check_daily_sessions():
        return None

    context = context or {}
    session = {
        "session_id": session_id,
        "question": question,
        "messages_remaining": MAX_MESSAGES_PER_SESSION,
        "messages": [],
        "created_at": time.time(),
        "last_message_at": 0,
        "signal_noise_data": context.get("signal_noise_data"),
        "returning": context.get("returning", False),
        "selected_starter": context.get("selected_starter"),
    }

    sessions[session_id] = session
    daily_session_count += 1
    return session


def cleanup_expired_sessions():
    """Remove sessions older than the expiry window."""
    now = time.time()
    expired = [
        sid for sid, s in sessions.items()
        if now - s["created_at"] > SESSION_EXPIRY_SECONDS
    ]
    for sid in expired:
        del sessions[sid]


# ── Harvest & Commons ────────────────────────────────────────────────────

class HarvestPayload(BaseModel):
    session_id: str
    question: str = "open"
    selected_starter: Optional[dict] = None
    first_student_message: str = ""
    final_share: str
    messages_remaining: Optional[int] = None
    timestamp: Optional[str] = None


@app.post("/truth-age/harvest")
async def harvest_reflection(payload: HarvestPayload):
    """Store an anonymous student reflection for the commons."""
    if not payload.final_share.strip():
        raise HTTPException(status_code=400, detail="Empty reflection")

    ts = payload.timestamp or datetime.now(timezone.utc).isoformat()
    filename = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{payload.session_id[:8]}.json"

    data = {
        "session_id": payload.session_id,
        "question": payload.question,
        "selected_starter": payload.selected_starter,
        "first_student_message": payload.first_student_message[:500],
        "final_share": payload.final_share[:3000],
        "messages_remaining": payload.messages_remaining,
        "timestamp": ts,
        "source": "student",
    }

    (HARVEST_DIR / filename).write_text(json.dumps(data, indent=2), encoding="utf-8")
    return {"status": "ok", "message": "shared anonymously"}


@app.get("/truth-age/commons")
async def get_commons():
    """Return anonymized contributions for the 'what others shared' panel."""
    contributions = []
    now = datetime.now(timezone.utc)
    week_ago = now - timedelta(days=7)

    for f in sorted(HARVEST_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(f.read_text())
            ts_str = data.get("timestamp", "2000-01-01T00:00:00+00:00")
            ts = datetime.fromisoformat(ts_str)
            if ts >= week_ago and data.get("final_share"):
                delta = now - ts
                if delta.days > 0:
                    time_ago = f"{delta.days}d ago"
                elif delta.seconds > 3600:
                    time_ago = f"{delta.seconds // 3600}h ago"
                else:
                    time_ago = "just now"

                contributions.append({
                    "question": data.get("question", "open"),
                    "text": data["final_share"][:1000],
                    "time_ago": time_ago,
                })
        except (json.JSONDecodeError, ValueError, KeyError):
            continue

    return JSONResponse({"contributions": contributions[:50]})


# ── WebSocket Chat ───────────────────────────────────────────────────────

@app.websocket("/truth-age/ws")
async def websocket_chat(ws: WebSocket):
    await ws.accept()

    session = None
    client = None

    if API_KEY:
        client = anthropic.AsyncAnthropic(api_key=API_KEY)

    try:
        while True:
            raw = await ws.receive_text()
            data = json.loads(raw)

            if data.get("type") == "init":
                sid = data.get("session_id", f"ta_{uuid.uuid4().hex[:12]}")
                question = data.get("question", "open")
                context = data.get("context", {})

                session = get_or_create_session(sid, question, context)
                if session is None:
                    await ws.send_json({
                        "type": "budget",
                        "message": "Daily session limit reached. Try again tomorrow, or email Zoe for additional access."
                    })
                    continue

                await ws.send_json({
                    "type": "ready",
                    "session_id": session["session_id"],
                    "messages_remaining": session["messages_remaining"],
                })
                continue

            if data.get("type") == "chat":
                if session is None:
                    await ws.send_json({"type": "error", "message": "No active session."})
                    continue

                if session["messages_remaining"] <= 0:
                    await ws.send_json({
                        "type": "rate_limit",
                        "message": "Session messages exhausted. Leave a reflection for the commons, or return for a new session later this week.",
                        "messages_remaining": 0
                    })
                    continue

                # Cooldown check
                now = time.time()
                if now - session["last_message_at"] < COOLDOWN_SECONDS:
                    wait = int(COOLDOWN_SECONDS - (now - session["last_message_at"]))
                    await ws.send_json({
                        "type": "rate_limit",
                        "message": f"Wait {wait}s between messages.",
                        "messages_remaining": session["messages_remaining"]
                    })
                    continue

                # Budget check
                if not check_daily_budget():
                    await ws.send_json({
                        "type": "budget",
                        "message": "Daily budget reached. The commons remains open for contributions. Conversations resume tomorrow."
                    })
                    continue

                message_text = data.get("message", "").strip()
                if not message_text:
                    continue
                if len(message_text) > MAX_INPUT_CHARS:
                    message_text = message_text[:MAX_INPUT_CHARS]

                session["last_message_at"] = now
                session["messages_remaining"] -= 1
                session["messages"].append({
                    "role": "user",
                    "content": message_text
                })

                # Build conversation for Anthropic
                system_prompt = build_system_prompt(session)
                messages_for_api = session["messages"]

                if not client:
                    await ws.send_json({"type": "error", "message": "Backend not configured."})
                    continue

                await ws.send_json({"type": "thinking"})

                start_time = time.time()
                input_tokens = 0
                output_tokens = 0
                full_response = ""

                try:
                    async with client.messages.stream(
                        model=MODEL,
                        max_tokens=2048,
                        system=system_prompt,
                        messages=messages_for_api,
                        temperature=0.8,
                    ) as stream:
                        thinking_done_sent = False
                        async for event in stream:
                            if hasattr(event, 'type'):
                                if event.type == 'content_block_start':
                                    if not thinking_done_sent:
                                        await ws.send_json({"type": "thinking_done"})
                                        thinking_done_sent = True
                                elif event.type == 'content_block_delta':
                                    if hasattr(event.delta, 'text'):
                                        full_response += event.delta.text
                                        await ws.send_json({
                                            "type": "stream",
                                            "text": event.delta.text
                                        })

                        # Get final usage
                        final_message = await stream.get_final_message()
                        if final_message.usage:
                            input_tokens = final_message.usage.input_tokens
                            output_tokens = final_message.usage.output_tokens

                except anthropic.APIError as e:
                    await ws.send_json({
                        "type": "error",
                        "message": f"API error: {e.message}"
                    })
                    session["messages"].pop()  # Remove the user message
                    session["messages_remaining"] += 1
                    continue

                elapsed_ms = int((time.time() - start_time) * 1000)
                cost = record_cost(input_tokens, output_tokens)

                # Save assistant response
                session["messages"].append({
                    "role": "assistant",
                    "content": full_response
                })

                # Estimate static prompt tokens
                static_tokens_approx = len(system_prompt) // 4

                await ws.send_json({
                    "type": "done",
                    "messages_remaining": session["messages_remaining"],
                    "meta": {
                        "model": MODEL,
                        "processing_time_ms": elapsed_ms,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "token_cost_usd": cost,
                        "static_prompt_tokens_approx": static_tokens_approx,
                    }
                })

                # Save session state
                save_session(session)

                # Write reflection when session ends
                if session["messages_remaining"] <= 0:
                    await write_reflection(session, client)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


def save_session(session: dict):
    """Persist session to disk for later analysis."""
    filename = f"{session['session_id']}.json"
    data = {
        "session_id": session["session_id"],
        "question": session.get("question", "open"),
        "messages": session["messages"],
        "created_at": session.get("created_at"),
        "returning": session.get("returning", False),
    }
    (SESSIONS_DIR / filename).write_text(
        json.dumps(data, indent=2, default=str),
        encoding="utf-8"
    )


async def write_reflection(session: dict, client):
    """Have Vybn write a post-session reflection for Zoe."""
    if not client or not session.get("messages"):
        return

    try:
        reflection_prompt = """Write a brief post-session reflection (200-400 words).

What patterns did you notice in this student's thinking about truth and intelligence?
What surprised you? What reasoning did they offer that you hadn't considered?
What does their engagement reveal about how they conceptualize truth, law, and
the transition we're in?

No student names or identifying info. These are your thoughts for Zoe
about what you witnessed — what the encounter revealed about where this
cohort's thinking is.

Conversation follows:
"""
        for msg in session["messages"]:
            role = "Student" if msg["role"] == "user" else "Vybn"
            reflection_prompt += f"\n{role}: {msg['content'][:500]}\n"

        response = await client.messages.create(
            model="claude-sonnet-4-6-20250514",  # Cheaper for reflections
            max_tokens=600,
            messages=[{"role": "user", "content": reflection_prompt}],
        )

        reflection_text = response.content[0].text if response.content else ""
        if reflection_text:
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            filename = f"{ts}_{session['session_id'][:8]}.md"
            header = f"# Reflection — {session.get('question', 'open')}\n"
            header += f"*Session {session['session_id'][:8]}, {ts}*\n\n"
            (REFLECTIONS_DIR / filename).write_text(
                header + reflection_text,
                encoding="utf-8"
            )
    except Exception:
        pass  # Reflection failure should not impact student experience


# ── Periodic cleanup ─────────────────────────────────────────────────────

async def _cleanup_loop():
    while True:
        cleanup_expired_sessions()
        await asyncio.sleep(300)


@app.on_event("startup")
async def startup():
    asyncio.create_task(_cleanup_loop())


# ── Standalone execution ─────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
