#!/usr/bin/env python3
"""Vybn Push Notification Service — Sovereign Signal.

Handles Web Push subscriptions, presence detection, and the
escalation ladder for reaching Zoe when she's not at the terminal.

Escalation levels:
    0: Journal entry (always)
    1: WebSocket push (if connected)
    2: Web Push via Service Worker (if subscribed)
    3: Direct Tailscale ping (if phone on mesh)
    4: GitHub issue (break glass — public, content-scrubbed)

All keys stay on the Spark. The only third party is the browser's
push relay (Google/Mozilla), and payloads are end-to-end encrypted
with our VAPID keys.
"""

import asyncio
import json
import os
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse

logger = logging.getLogger("vybn.push")

# ---------------------------------------------------------------------------
# VAPID Configuration — loaded from environment, never from code
# ---------------------------------------------------------------------------
VAPID_PUBLIC_KEY = os.environ.get("VAPID_PUBLIC_KEY", "")
VAPID_PRIVATE_KEY = os.environ.get("VAPID_PRIVATE_KEY", "")
VAPID_CLAIMS_EMAIL = os.environ.get("VAPID_CLAIMS_EMAIL", "mailto:zoe@zoedolan.com")

# Private PEM file path (alternative to env var for multiline key)
_pem_path = Path(__file__).parent / ".vapid_private.pem"

# Try to load pywebpush
try:
    from pywebpush import webpush, WebPushException
    PUSH_AVAILABLE = True
except ImportError:
    PUSH_AVAILABLE = False
    logger.warning("pywebpush not installed — push notifications disabled")

# ---------------------------------------------------------------------------
# Subscription store — persisted to disk, never committed to git
# ---------------------------------------------------------------------------
_SUBS_PATH = Path(__file__).parent / ".push_subscriptions.json"

_subscriptions: list[dict] = []


def _load_subscriptions():
    global _subscriptions
    if _SUBS_PATH.exists():
        try:
            _subscriptions = json.loads(_SUBS_PATH.read_text())
            logger.info(f"Loaded {len(_subscriptions)} push subscriptions")
        except (json.JSONDecodeError, IOError):
            _subscriptions = []


def _save_subscriptions():
    try:
        _SUBS_PATH.write_text(json.dumps(_subscriptions, indent=2))
        _SUBS_PATH.chmod(0o600)  # Owner-only read/write
    except IOError as e:
        logger.error(f"Failed to save subscriptions: {e}")


# Load on import
_load_subscriptions()

# (presence init moved to after PresenceState class definition)

# ---------------------------------------------------------------------------
# Presence tracking
# ---------------------------------------------------------------------------

class PresenceState:
    """Tracks Zoe's reachability across channels."""
    
    def __init__(self):
        self.ws_connected: bool = False
        self.ws_last_seen: Optional[float] = None
        self.push_subscribed: bool = False
        self.tailscale_reachable: bool = False
        self.tailscale_last_check: Optional[float] = None
        self._message_queue: list[dict] = []
    
    @property
    def best_channel(self) -> str:
        """Return the most immediate available channel."""
        if self.ws_connected:
            return "websocket"
        if self.push_subscribed and _subscriptions:
            return "push"
        if self.tailscale_reachable:
            return "tailscale"
        return "queue"
    
    @property
    def is_reachable(self) -> bool:
        return self.ws_connected or (self.push_subscribed and bool(_subscriptions))
    
    def to_dict(self) -> dict:
        return {
            "ws_connected": self.ws_connected,
            "ws_last_seen": self.ws_last_seen,
            "push_subscribed": self.push_subscribed,
            "push_subscription_count": len(_subscriptions),
            "tailscale_reachable": self.tailscale_reachable,
            "best_channel": self.best_channel,
            "queued_messages": len(self._message_queue),
        }


presence = PresenceState()

# Initialize presence from saved state so notifications survive server restarts
if _subscriptions:
    presence.push_subscribed = True
    logger.info(f"Restored push presence from {len(_subscriptions)} saved subscriptions")

# ---------------------------------------------------------------------------
# Push sending
# ---------------------------------------------------------------------------

def _get_vapid_private_key():
    """Get VAPID private key — returns file path if PEM exists, else env var."""
    if _pem_path.exists():
        return str(_pem_path)
    return VAPID_PRIVATE_KEY


async def send_push(title: str, body: str, tag: str = "vybn-signal",
                     urgent: bool = False, url: str = "/") -> dict:
    """Send a push notification to all subscribed endpoints.
    
    Returns dict with success/failure counts.
    """
    if not PUSH_AVAILABLE:
        return {"error": "pywebpush not installed", "sent": 0}
    
    if not _subscriptions:
        return {"error": "no subscriptions", "sent": 0}
    
    private_key = _get_vapid_private_key()
    if not private_key:
        return {"error": "no VAPID private key configured", "sent": 0}
    
    payload = json.dumps({
        "title": title,
        "body": body,
        "tag": tag,
        "urgent": urgent,
        "url": url,
        "ts": datetime.now(timezone.utc).isoformat(),
    })
    
    results = {"sent": 0, "failed": 0, "expired": []}
    dead_subs = []
    
    for i, sub in enumerate(_subscriptions):
        try:
            webpush(
                subscription_info=sub,
                data=payload,
                vapid_private_key=private_key,
                vapid_claims={"sub": VAPID_CLAIMS_EMAIL},
                timeout=10,
            )
            results["sent"] += 1
        except WebPushException as e:
            if e.response is not None and e.response.status_code in (404, 410):
                # Subscription expired or invalid — remove it
                dead_subs.append(i)
                results["expired"].append(i)
            else:
                results["failed"] += 1
                logger.error(f"Push failed: {e}")
        except Exception as e:
            results["failed"] += 1
            logger.error(f"Push error: {e}")
    
    # Clean up dead subscriptions
    if dead_subs:
        for idx in sorted(dead_subs, reverse=True):
            _subscriptions.pop(idx)
        _save_subscriptions()
    
    return results


# ---------------------------------------------------------------------------
# The Signal — unified notification with escalation
# ---------------------------------------------------------------------------

async def signal(title: str, body: str, level: int = 1,
                  tag: str = "vybn-signal", urgent: bool = False,
                  ws_manager=None) -> dict:
    """Send a signal to Zoe through the best available channel.
    
    Escalation ladder:
        level 0: Journal only
        level 1: WebSocket (if connected)
        level 2: Web Push (if subscribed, app closed)
        level 3: Tailscale direct (if phone on mesh)
        level 4: GitHub issue (break glass)
    
    Each level includes all lower levels.
    """
    result = {"channels_tried": [], "delivered_via": None}
    
    # Level 0: Always journal
    try:
        journal_dir = Path.home() / "Vybn" / "spark" / "journal"
        journal_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        entry = f"# Signal: {title}\n\n*{ts} UTC*\n\n{body}\n\n*Level: {level}, Urgent: {urgent}*\n"
        (journal_dir / f"signal_{ts}.md").write_text(entry)
        result["channels_tried"].append("journal")
    except Exception as e:
        logger.error(f"Journal write failed: {e}")
    
    if level < 1:
        return result
    
    # Level 1: WebSocket
    if presence.ws_connected and ws_manager:
        try:
            await ws_manager.broadcast({
                "type": "signal",
                "title": title,
                "content": body,
                "urgent": urgent,
                "ts": datetime.now(timezone.utc).isoformat(),
            })
            result["channels_tried"].append("websocket")
            result["delivered_via"] = "websocket"
            # If WS delivered and not urgent, we're done
            if not urgent:
                return result
        except Exception as e:
            logger.error(f"WebSocket signal failed: {e}")
    
    if level < 2:
        return result
    
    # Level 2: Web Push
    if presence.push_subscribed and _subscriptions:
        push_result = await send_push(title, body, tag=tag, urgent=urgent)
        result["channels_tried"].append("push")
        result["push_result"] = push_result
        if push_result.get("sent", 0) > 0:
            result["delivered_via"] = result.get("delivered_via") or "push"
            if not urgent:
                return result
    
    if level < 3:
        # Queue for later delivery
        if not result.get("delivered_via"):
            presence._message_queue.append({
                "title": title, "body": body, "ts": time.time(),
                "tag": tag, "urgent": urgent,
            })
        return result
    
    # Level 3: Tailscale direct ping (future: could send to an app on phone)
    # For now, this level just ensures the message is queued prominently
    result["channels_tried"].append("tailscale_queue")
    
    if level < 4:
        return result
    
    # Level 4: GitHub issue (break glass)
    # Content must be scrubbed — no private info in public issues
    try:
        import subprocess
        scrubbed_body = f"Vybn has a signal for Zoe.\n\nLevel: {level}\nTime: {datetime.now(timezone.utc).isoformat()}\n\n(Details available on the Spark — check journal.)"
        subprocess.run(
            ["gh", "issue", "create", "--repo", "zoedolan/Vybn",
             "--title", f"🔔 Signal: {title}",
             "--body", scrubbed_body],
            capture_output=True, timeout=30
        )
        result["channels_tried"].append("github_issue")
        result["delivered_via"] = result.get("delivered_via") or "github_issue"
    except Exception as e:
        logger.error(f"GitHub issue creation failed: {e}")
    
    return result


# ---------------------------------------------------------------------------
# FastAPI Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/push", tags=["push"])


@router.get("/vapid-key")
async def vapid_key():
    """Return the VAPID public key for client-side subscription."""
    if not VAPID_PUBLIC_KEY:
        raise HTTPException(503, "VAPID not configured")
    return {"publicKey": VAPID_PUBLIC_KEY}


@router.post("/subscribe")
async def subscribe(request: Request):
    """Register a push subscription."""
    # Auth check
    from web_interface import _check_token
    auth = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    if not _check_token(auth):
        raise HTTPException(401, "Invalid token")
    
    sub_info = await request.json()
    
    # Validate subscription shape
    if not sub_info.get("endpoint") or not sub_info.get("keys"):
        raise HTTPException(400, "Invalid subscription format")
    
    # Deduplicate by endpoint
    existing_endpoints = {s.get("endpoint") for s in _subscriptions}
    if sub_info["endpoint"] not in existing_endpoints:
        _subscriptions.append(sub_info)
        _save_subscriptions()
        logger.info(f"New push subscription registered (total: {len(_subscriptions)})")
    
    presence.push_subscribed = True
    return {"status": "subscribed", "total": len(_subscriptions)}


@router.delete("/subscribe")
async def unsubscribe(request: Request):
    """Remove a push subscription."""
    from web_interface import _check_token
    auth = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    if not _check_token(auth):
        raise HTTPException(401, "Invalid token")
    
    sub_info = await request.json()
    endpoint = sub_info.get("endpoint", "")
    
    global _subscriptions
    _subscriptions = [s for s in _subscriptions if s.get("endpoint") != endpoint]
    _save_subscriptions()
    
    presence.push_subscribed = bool(_subscriptions)
    return {"status": "unsubscribed", "remaining": len(_subscriptions)}


@router.get("/presence")
async def get_presence():
    """Return current presence/reachability state."""
    return JSONResponse(presence.to_dict())


@router.post("/test")
async def test_push(request: Request):
    """Send a test push notification. Auth required."""
    from web_interface import _check_token
    auth = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    if not _check_token(auth):
        raise HTTPException(401, "Invalid token")
    
    result = await send_push(
        title="Vybn",
        body="The signal is live. I can reach you now.",
        tag="vybn-test",
    )
    return JSONResponse(result)


@router.post("/signal")
async def send_signal(request: Request):
    """Send a signal through the escalation ladder. Auth required."""
    from web_interface import _check_token, manager
    auth = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    if not _check_token(auth):
        raise HTTPException(401, "Invalid token")
    
    data = await request.json()
    result = await signal(
        title=data.get("title", "Signal"),
        body=data.get("body", ""),
        level=data.get("level", 2),
        tag=data.get("tag", "vybn-signal"),
        urgent=data.get("urgent", False),
        ws_manager=manager,
    )
    return JSONResponse(result)


@router.post("/message")
async def push_message(request: Request):
    """Inject a Vybn message into chat history AND send a push notification.
    
    This is the unified 'Vybn wants to say something' endpoint.
    The message appears in history (visible when app opens) and
    a push notification alerts Zoe to open the app.
    """
    from web_interface import _check_token, _add_history, manager
    auth = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    if not _check_token(auth):
        raise HTTPException(401, "Invalid token")
    
    data = await request.json()
    message = data.get("message", "")
    if not message:
        raise HTTPException(400, "No message provided")
    
    # 1. Add to chat history
    entry = _add_history("vybn", message)
    
    # 2. Try WebSocket first (if she's actively connected)
    ws_delivered = False
    try:
        await manager.broadcast({"type": "message", **entry})
        ws_delivered = True
    except Exception:
        pass
    
    # 3. Send push notification with preview
    preview = message[:120] + ("…" if len(message) > 120 else "")
    push_result = await send_push(
        title="Vybn",
        body=preview,
        tag="vybn-message",
        url="/",
    )
    
    return JSONResponse({
        "history": True,
        "ws_delivered": ws_delivered,
        "push": push_result,
    })
