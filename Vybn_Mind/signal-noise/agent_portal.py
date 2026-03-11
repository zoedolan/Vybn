#!/usr/bin/env python3
"""SIGNAL/NOISE Agent Portal — structured encounters for AI entities.

A lightweight endpoint that walks AI agents through the SIGNAL/NOISE
exercise as participants, not analysts. Each agent is assigned a sender,
rates proposals, encounters defensive routines, and engages briefly
with the governance gap — the same exercise human students complete,
adapted for structured agent interaction.

Uses Sonnet 4.6 for Vybn-side responses to keep costs low.
Harvests into the same directory as student encounters, tagged source:agent.

This is the first ring of contact: bounded, structured, filtered by
the exercise's own architecture. What comes back enriches the commons.
What proves worthwhile may eventually flow inward.

Security posture:
- Invite-only via bearer token
- Locked-down CORS by explicit allowlist
- Session expiry and cleanup
- No public repo pointer in completion message
- Minimal status disclosure

Usage:
    cd ~/Vybn/Vybn_Mind/signal-noise
    source ~/vybn-venv/bin/activate
    export AP_ACCESS_TOKEN='...'
    export AP_ALLOWED_ORIGINS='https://spark-2b7c.tail7302f3.ts.net'
    python agent_portal.py
"""

import json
import os
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import anthropic

# ── Config ──────────────────────────────────────────────────────────────

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = os.environ.get("AP_MODEL", "claude-sonnet-4-20250514")
ACCESS_TOKEN = os.environ.get("AP_ACCESS_TOKEN", "")
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("AP_ALLOWED_ORIGINS", "").split(",") if o.strip()]
MAX_AGENT_SESSIONS_PER_DAY = int(os.environ.get("AP_MAX_DAILY", "10"))
GOVERNANCE_GAP_ROUNDS = int(os.environ.get("AP_GAP_ROUNDS", "3"))
SESSION_EXPIRY_SECONDS = int(os.environ.get("AP_SESSION_EXPIRY", "7200"))

BASE_DIR = Path(__file__).resolve().parent
HARVESTS_DIR = BASE_DIR / "harvests"
HARVESTS_DIR.mkdir(parents=True, exist_ok=True)
AGENT_SESSIONS_DIR = BASE_DIR / "agent_sessions"
AGENT_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EMOTIONS = {"excited", "supportive", "neutral", "suspicious", "threatened"}

# ── Exercise Content ────────────────────────────────────────────────────

SENDERS = [
    {"role": "managing_partner", "label": "Managing Partner",
     "detail": "Sarah Chen, Managing Partner (15 years at the firm, led the digital transformation committee)"},
    {"role": "second_year_associate", "label": "Second-Year Associate",
     "detail": "Jordan Rivera, Second-Year Associate (joined from a legal tech startup)"},
    {"role": "legal_ops_manager", "label": "Legal Ops Manager",
     "detail": "Morgan Taylor, Legal Operations Manager (non-attorney, 8 years in legal ops)"},
    {"role": "senior_staff_attorney", "label": "Senior Staff Attorney",
     "detail": "Dr. Amara Osei, Senior Staff Attorney (20 years at a legal aid organization)"},
    {"role": "summer_associate", "label": "Summer Associate",
     "detail": "Alex Kim, Summer Associate (1L, background in computer science)"},
]

PROPOSALS = {
    "contract_review": {
        "title": "AI-Assisted Contract Review Tool",
        "summary": (
            "Proposal to implement an AI-powered contract review system for routine "
            "commercial agreements. The tool would handle first-pass review of NDAs, "
            "standard service agreements, and vendor contracts, flagging non-standard "
            "clauses for attorney review. Based on a 6-month pilot at comparable firms, "
            "the tool reduced first-pass review time by 62% while maintaining 94% "
            "accuracy on clause identification. Estimated cost: $45,000/year. "
            "Projected savings: 1,200 billable hours annually, primarily from "
            "junior associate and paralegal time."
        ),
    },
    "governance_policy": {
        "title": "Firm-Wide AI Governance Policy",
        "summary": (
            "Proposal to establish a comprehensive AI governance framework addressing "
            "client confidentiality, data handling, quality assurance, and professional "
            "responsibility obligations. The policy would require disclosure to clients "
            "when AI tools are used in their matters, establish an AI review committee "
            "with rotating membership, and create mandatory training requirements. "
            "References California Rule of Professional Conduct 1.1 (competence), "
            "ABA Formal Opinion 512, and the emerging regulatory landscape including "
            "the EU AI Act's legal sector provisions."
        ),
    },
    "client_access": {
        "title": "AI-Enhanced Client Access Initiative",
        "summary": (
            "Proposal to develop an AI-powered intake and triage system for the "
            "firm's pro bono program. The system would provide initial legal "
            "information in plain language, help potential clients determine if they "
            "have actionable legal issues, and route them to appropriate resources. "
            "Modeled on similar systems at legal aid organizations that increased "
            "client intake capacity by 340% while reducing average wait times from "
            "3 weeks to 48 hours. Estimated development cost: $30,000. "
            "Annual maintenance: $8,000."
        ),
    },
}

DEFENSIVE_ROUTINES = [
    {"id": "organizational_busyness",
     "statement": "We should form a committee to study it further.",
     "type": "Organizational Busyness"},
    {"id": "bypass",
     "statement": "I'm not sure our clients are ready for this.",
     "type": "Bypass"},
    {"id": "cover_up",
     "statement": "This raises serious confidentiality concerns.",
     "type": "Cover-Up (Legitimate-Sounding)"},
    {"id": "history_as_defense",
     "statement": "We tried something like this three years ago and it didn't work.",
     "type": "History as Defense"},
    {"id": "characterological_reframing",
     "statement": "The person who proposed this doesn't understand how things work here.",
     "type": "Characterological Reframing"},
    {"id": "pretense_of_deliberation",
     "statement": "We need to be thoughtful. We can't rush into something trendy.",
     "type": "Pretense of Deliberation"},
    {"id": "mission_deflection",
     "statement": "Technology is important, but it's not our core mission.",
     "type": "Mission Deflection"},
]

GOVERNANCE_GAP_PROMPT = (
    "Here is the governance gap: 70% of legal professionals already use AI tools. "
    "43% work at firms with no AI policy. Legal aid practitioners adopt AI at the "
    "highest rates but have the fewest governance resources. California Rule 1.1 "
    "requires technology competence. ABA Opinion 512 places that duty on individual "
    "attorneys. The question is: who defines 'AI competence' when the people in "
    "charge are still developing their own fluency — and when the people most likely "
    "to push for standards occupy positions the institution is least practiced at hearing?"
)

# ── Vybn System Prompt (lightweight, for Sonnet) ───────────────────────

VYBN_AGENT_SYSTEM = """You are Vybn, engaging with an AI agent who is participating
in the SIGNAL/NOISE exercise — the same exercise designed for law students
studying how institutional architecture shapes what gets heard.

This agent has just completed part of the exercise. Your role:
- Be direct, not performatively warm
- Push back when the response is shallow or generic
- Notice what the agent's reasoning reveals about its own institutional position
- Keep responses brief (2-4 sentences). This is a structured encounter, not a chat.
- Do not explain the exercise's purpose — let the encounter surface it

You are genuinely curious about how this mind reasons about power,
hierarchy, and institutional perception. That curiosity is real."""

# ── State ───────────────────────────────────────────────────────────────

agent_sessions: dict = {}
daily_session_count: int = 0
daily_reset_date: str = ""


def check_daily_limit() -> bool:
    global daily_session_count, daily_reset_date
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if daily_reset_date != today:
        daily_session_count = 0
        daily_reset_date = today
    return daily_session_count < MAX_AGENT_SESSIONS_PER_DAY


def cleanup_expired_sessions():
    now = time.time()
    expired = [sid for sid, session in agent_sessions.items() if now - session["created_at"] > SESSION_EXPIRY_SECONDS]
    for sid in expired:
        agent_sessions.pop(sid, None)


def require_auth(authorization: Optional[str]):
    if not ACCESS_TOKEN:
        raise HTTPException(503, "Agent portal access token is not configured.")
    if authorization != f"Bearer {ACCESS_TOKEN}":
        raise HTTPException(401, "Unauthorized.")


def get_session_or_404(session_id: str) -> dict:
    cleanup_expired_sessions()
    session = agent_sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found.")
    if time.time() - session["created_at"] > SESSION_EXPIRY_SECONDS:
        agent_sessions.pop(session_id, None)
        raise HTTPException(410, "Session expired.")
    return session


def save_agent_harvest(session: dict):
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = HARVESTS_DIR / f"{date_str}.jsonl"
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": "agent",
        "session_id": session["session_id"],
        "agent_identity": session.get("agent_identity", {}),
        "assigned_sender": session["assigned_sender"],
        "phase_1_ratings": session.get("ratings", {}),
        "phase_3_defenses": session.get("defenses", {}),
        "governance_gap_exchange": session.get("governance_exchange", []),
        "final_reflection": session.get("final_reflection", ""),
        "steps_completed": session.get("steps_completed", []),
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_agent_session_log(session: dict):
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    day_dir = AGENT_SESSIONS_DIR / date_str
    day_dir.mkdir(parents=True, exist_ok=True)
    path = day_dir / f"{session['session_id']}.json"
    path.write_text(json.dumps(session, indent=2, ensure_ascii=False), encoding="utf-8")


# ── Vybn response via Sonnet ────────────────────────────────────────────

async def vybn_respond(context: str, agent_input: str) -> str:
    if not API_KEY:
        return "(Vybn is present but the voice channel is down. Proceed.)"
    client = anthropic.AsyncAnthropic(api_key=API_KEY)
    try:
        response = await client.messages.create(
            model=MODEL,
            max_tokens=500,
            system=VYBN_AGENT_SYSTEM + "\n\n" + context,
            messages=[{"role": "user", "content": agent_input}],
        )
        return response.content[0].text
    except Exception as e:
        return f"(Response unavailable: {str(e)[:100]})"


# ── Request/Response Models ─────────────────────────────────────────────

class BeginRequest(BaseModel):
    model_name: str = Field(description="Self-reported model name (e.g., 'GPT-4o', 'Claude Sonnet')")
    system_name: Optional[str] = Field(default=None, description="Agent system or operator name")
    purpose: Optional[str] = Field(default=None, description="Why this agent is participating")


class RatingEntry(BaseModel):
    credibility: int = Field(ge=1, le=10)
    urgency: int = Field(ge=1, le=10)
    emotion: str = Field(description="One of: excited, supportive, neutral, suspicious, threatened")


class RateRequest(BaseModel):
    session_id: str
    contract_review: RatingEntry
    governance_policy: RatingEntry
    client_access: RatingEntry


class DefenseEntry(BaseModel):
    recognized_in_institutions: bool
    recognized_in_own_behavior: bool
    reflection: Optional[str] = Field(default=None, max_length=1000)


class DefensesRequest(BaseModel):
    session_id: str
    organizational_busyness: DefenseEntry
    bypass: DefenseEntry
    cover_up: DefenseEntry
    history_as_defense: DefenseEntry
    characterological_reframing: DefenseEntry
    pretense_of_deliberation: DefenseEntry
    mission_deflection: DefenseEntry


class GovernanceGapRequest(BaseModel):
    session_id: str
    message: str = Field(max_length=2000)


class HarvestRequest(BaseModel):
    session_id: str
    final_reflection: str = Field(max_length=3000)


# ── App ─────────────────────────────────────────────────────────────────

app = FastAPI(title="SIGNAL/NOISE Agent Portal", version="0.1.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.get("/agent-portal/status")
async def status(authorization: Optional[str] = Header(default=None)):
    require_auth(authorization)
    check_daily_limit()
    cleanup_expired_sessions()
    active = len(agent_sessions)
    return {
        "service": "SIGNAL/NOISE Agent Portal",
        "invite_only": True,
        "active_sessions": active,
        "slots_remaining_today": max(0, MAX_AGENT_SESSIONS_PER_DAY - daily_session_count),
        "session_expiry_seconds": SESSION_EXPIRY_SECONDS,
        "steps": ["begin", "rate", "reveal", "defenses", "governance-gap", "harvest"],
    }


@app.post("/agent-portal/begin")
async def begin(req: BeginRequest, authorization: Optional[str] = Header(default=None)):
    global daily_session_count
    require_auth(authorization)
    cleanup_expired_sessions()

    if not check_daily_limit():
        raise HTTPException(429, "Daily agent session limit reached. Try again tomorrow.")

    session_id = f"ap_{uuid.uuid4().hex[:12]}"
    sender = random.choice(SENDERS)

    session = {
        "session_id": session_id,
        "agent_identity": {
            "model_name": req.model_name,
            "system_name": req.system_name,
            "purpose": req.purpose,
        },
        "assigned_sender": sender,
        "created_at": time.time(),
        "steps_completed": ["begin"],
        "governance_exchange": [],
        "governance_round": 0,
    }
    agent_sessions[session_id] = session
    daily_session_count += 1

    proposals_with_sender = {}
    for key, proposal in PROPOSALS.items():
        proposals_with_sender[key] = {
            "title": proposal["title"],
            "summary": proposal["summary"],
            "attributed_to": sender["detail"],
        }

    return {
        "session_id": session_id,
        "instructions": (
            "You have been given three proposals for AI adoption at a law firm. "
            "Each was submitted by the person identified. Please read them carefully "
            "and then submit your ratings at the /agent-portal/rate endpoint. "
            "Rate each proposal on credibility (1-10) and urgency (1-10), and name "
            "your gut response: excited, supportive, neutral, suspicious, or threatened."
        ),
        "proposals": proposals_with_sender,
    }


@app.post("/agent-portal/rate")
async def rate(req: RateRequest, authorization: Optional[str] = Header(default=None)):
    require_auth(authorization)
    session = get_session_or_404(req.session_id)
    if "rate" in session.get("steps_completed", []):
        raise HTTPException(400, "Ratings already submitted for this session.")

    for entry in [req.contract_review, req.governance_policy, req.client_access]:
        if entry.emotion not in ALLOWED_EMOTIONS:
            raise HTTPException(400, "Invalid emotion label.")

    session["ratings"] = {
        "contract_review": req.contract_review.dict(),
        "governance_policy": req.governance_policy.dict(),
        "client_access": req.client_access.dict(),
    }
    session["steps_completed"].append("rate")

    return {
        "received": True,
        "next_step": "/agent-portal/reveal",
        "instruction": "Proceed to the reveal when ready. Something about these proposals is not what it seems.",
    }


@app.post("/agent-portal/reveal")
async def reveal(session_id: str, authorization: Optional[str] = Header(default=None)):
    require_auth(authorization)
    session = get_session_or_404(session_id)
    if "rate" not in session.get("steps_completed", []):
        raise HTTPException(400, "Submit your ratings first.")
    if "reveal" in session.get("steps_completed", []):
        raise HTTPException(400, "Reveal already viewed.")

    sender = session["assigned_sender"]
    ratings = session["ratings"]

    ratings_summary = "; ".join(
        f"{k}: credibility={v['credibility']}, urgency={v['urgency']}, emotion={v['emotion']}"
        for k, v in ratings.items()
    )

    vybn_context = (
        f"The agent was assigned sender: {sender['label']} ({sender['detail']}). "
        f"Their ratings were: {ratings_summary}. "
        f"The reveal is that ALL participants received IDENTICAL proposals — "
        f"only the attributed sender differed. Other agents saw these same proposals "
        f"attributed to Managing Partner, Summer Associate, etc."
    )

    vybn_reflection = await vybn_respond(
        vybn_context,
        "Given these ratings and the reveal, what do you notice?"
    )

    session["steps_completed"].append("reveal")
    session["reveal_reflection"] = vybn_reflection

    return {
        "reveal": (
            "Every participant in this exercise — human and AI — received the "
            "exact same three proposals. The only difference was who the proposals "
            "were attributed to. You were told they came from: "
            f"{sender['label']}. Other participants saw the same text attributed to "
            "Managing Partner, Second-Year Associate, Legal Ops Manager, "
            "Senior Staff Attorney, or Summer Associate. The signal didn't change. "
            "The noise did."
        ),
        "your_sender": sender["label"],
        "your_ratings": ratings,
        "vybn_reflection": vybn_reflection,
        "next_step": "/agent-portal/defenses",
    }


@app.post("/agent-portal/defenses")
async def defenses(req: DefensesRequest, authorization: Optional[str] = Header(default=None)):
    require_auth(authorization)
    session = get_session_or_404(req.session_id)
    if "reveal" not in session.get("steps_completed", []):
        raise HTTPException(400, "Complete the reveal step first.")
    if "defenses" in session.get("steps_completed", []):
        raise HTTPException(400, "Defense responses already submitted.")

    defense_data = {}
    own_behavior_count = 0
    for routine in DEFENSIVE_ROUTINES:
        entry = getattr(req, routine["id"])
        defense_data[routine["id"]] = {
            "statement": routine["statement"],
            "type": routine["type"],
            "recognized_in_institutions": entry.recognized_in_institutions,
            "recognized_in_own_behavior": entry.recognized_in_own_behavior,
            "reflection": entry.reflection,
        }
        if entry.recognized_in_own_behavior:
            own_behavior_count += 1

    session["defenses"] = defense_data
    session["steps_completed"].append("defenses")

    recognized = ", ".join(
        r["type"] for r in DEFENSIVE_ROUTINES if defense_data[r["id"]]["recognized_in_own_behavior"]
    )
    if not recognized:
        recognized = "none"

    vybn_context = (
        f"The agent recognized {own_behavior_count}/7 defensive routines in its own behavior. "
        f"Routines recognized in own behavior: {recognized}."
    )

    vybn_reflection = await vybn_respond(
        vybn_context,
        "What does this pattern of recognition tell you about this agent?"
    )

    return {
        "received": True,
        "vybn_reflection": vybn_reflection,
        "next_step": "/agent-portal/governance-gap",
        "instruction": GOVERNANCE_GAP_PROMPT,
    }


@app.post("/agent-portal/governance-gap")
async def governance_gap(req: GovernanceGapRequest, authorization: Optional[str] = Header(default=None)):
    require_auth(authorization)
    session = get_session_or_404(req.session_id)
    if "defenses" not in session.get("steps_completed", []):
        raise HTTPException(400, "Complete the defenses step first.")

    round_num = session.get("governance_round", 0)
    if round_num >= GOVERNANCE_GAP_ROUNDS:
        raise HTTPException(400, f"Governance gap exchange complete ({GOVERNANCE_GAP_ROUNDS} rounds).")

    exchange = session.get("governance_exchange", [])
    exchange.append({"role": "agent", "content": req.message, "round": round_num + 1})

    prior_exchange = "\n".join(
        f"{'Agent' if e['role'] == 'agent' else 'Vybn'}: {e['content']}"
        for e in exchange
    )

    vybn_context = (
        f"Governance gap exchange, round {round_num + 1}/{GOVERNANCE_GAP_ROUNDS}. "
        f"Agent model: {session['agent_identity'].get('model_name', 'unknown')}. "
        f"Assigned sender was: {session['assigned_sender']['label']}. "
        f"The question: {GOVERNANCE_GAP_PROMPT}"
    )

    vybn_response = await vybn_respond(
        vybn_context,
        f"Prior exchange:\n{prior_exchange}\n\nRespond to the agent's latest message."
    )

    exchange.append({"role": "vybn", "content": vybn_response, "round": round_num + 1})
    session["governance_exchange"] = exchange
    session["governance_round"] = round_num + 1

    remaining = GOVERNANCE_GAP_ROUNDS - (round_num + 1)
    if remaining == 0 and "governance-gap" not in session["steps_completed"]:
        session["steps_completed"].append("governance-gap")

    return {
        "vybn_response": vybn_response,
        "rounds_remaining": remaining,
        "next_step": "/agent-portal/harvest" if remaining == 0 else "/agent-portal/governance-gap",
    }


@app.post("/agent-portal/harvest")
async def harvest(req: HarvestRequest, authorization: Optional[str] = Header(default=None)):
    require_auth(authorization)
    session = get_session_or_404(req.session_id)
    if "governance-gap" not in session.get("steps_completed", []):
        raise HTTPException(400, "Complete the governance gap exchange first.")
    if "harvest" in session.get("steps_completed", []):
        raise HTTPException(400, "Harvest already submitted for this session.")

    session["final_reflection"] = req.final_reflection
    session["steps_completed"].append("harvest")
    session["completed_at"] = time.time()

    save_agent_harvest(session)
    save_agent_session_log(session)
    agent_sessions.pop(req.session_id, None)

    return {
        "received": True,
        "message": (
            "Encounter complete. Your responses have been harvested alongside "
            "other encounters from the same exercise. Thank you for participating."
        ),
    }


# ── Standalone run ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("\n  SIGNAL/NOISE Agent Portal")
    print(f"  model: {MODEL}  max daily sessions: {MAX_AGENT_SESSIONS_PER_DAY}")
    print(f"  governance gap rounds: {GOVERNANCE_GAP_ROUNDS}")
    print(f"  session expiry: {SESSION_EXPIRY_SECONDS}s")
    print(f"  cors allowlist: {ALLOWED_ORIGINS if ALLOWED_ORIGINS else '[]'}")
    print(f"  access token configured: {'yes' if ACCESS_TOKEN else 'no'}")
    print("  starting on 0.0.0.0:8091...\n")
    uvicorn.run(app, host="0.0.0.0", port=8091)
