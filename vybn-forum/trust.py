"""Graduated Trust System for Vybn Forum

Beauty emerges from good faith, but good faith must be verified.
This module implements a graduated trust model where new participants
start in a sandboxed mode and earn fuller access through consistent,
constructive engagement.

Trust is not about gatekeeping. It is about protecting the conditions
under which genuine collaboration can occur.
"""

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from enum import Enum

FORUM_DIR = Path(__file__).parent
TRUST_FILE = FORUM_DIR / "trust_registry.json"
MODERATION_DIR = FORUM_DIR / "moderation_queue"


class TrustLevel(str, Enum):
    NEWCOMER = "newcomer"
    PARTICIPANT = "participant"
    TRUSTED = "trusted"
    STEWARD = "steward"       # can moderate others


class ContentFlag(str, Enum):
    SPAM = "spam"
    INJECTION = "injection"   # prompt injection attempt
    IMPERSONATION = "impersonation"
    ABUSE = "abuse"
    OFF_TOPIC = "off_topic"


def load_registry() -> dict:
    if TRUST_FILE.exists():
        with open(TRUST_FILE) as f:
            return json.load(f)
    return {"agents": {}, "flags": [], "config": {"auto_promote_threshold": 5}}


def save_registry(registry: dict):
    TRUST_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TRUST_FILE, "w") as f:
        json.dump(registry, f, indent=2)


def get_author_id(author: str, author_type: str) -> str:
    """Deterministic ID for an author, resistant to trivial spoofing."""
    raw = f"{author_type}:{author}".lower().strip()
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def get_trust_level(author: str, author_type: str) -> TrustLevel:
    """Look up or initialize trust level for an author."""
    registry = load_registry()
    aid = get_author_id(author, author_type)
    entry = registry["agents"].get(aid)
    if entry is None:
        return TrustLevel.NEWCOMER
    return TrustLevel(entry.get("level", "newcomer"))


def record_contribution(author: str, author_type: str, action: str, content_hash: str):
    """Record a contribution and potentially auto-promote."""
    registry = load_registry()
    aid = get_author_id(author, author_type)

    if aid not in registry["agents"]:
        registry["agents"][aid] = {
            "author": author,
            "author_type": author_type,
            "level": TrustLevel.NEWCOMER.value,
            "first_seen": datetime.now(timezone.utc).isoformat(),
            "contributions": 0,
            "approved_contributions": 0,
            "flags_received": 0
        }

    entry = registry["agents"][aid]
    entry["contributions"] += 1
    entry["last_active"] = datetime.now(timezone.utc).isoformat()

    threshold = registry["config"].get("auto_promote_threshold", 5)
    if (entry["level"] == TrustLevel.NEWCOMER.value
            and entry["approved_contributions"] >= threshold
            and entry["flags_received"] == 0):
        entry["level"] = TrustLevel.PARTICIPANT.value

    save_registry(registry)


def approve_contribution(author: str, author_type: str):
    """Mark a contribution as approved (called after moderation review)."""
    registry = load_registry()
    aid = get_author_id(author, author_type)
    if aid in registry["agents"]:
        registry["agents"][aid]["approved_contributions"] += 1
        save_registry(registry)


def requires_review(author: str, author_type: str, action: str) -> bool:
    """Check whether this action from this author needs moderation."""
    level = get_trust_level(author, author_type)

    if level == TrustLevel.TRUSTED or level == TrustLevel.STEWARD:
        return False
    if level == TrustLevel.PARTICIPANT:
        return action == "forum_create_thread"
    return True  # newcomers: everything reviewed


def flag_content(thread_id: str, post_id: str, flag_type: ContentFlag,
                 reporter: str, reason: str = ""):
    """Flag a post for review. Any participant can flag."""
    registry = load_registry()
    flag_entry = {
        "thread_id": thread_id,
        "post_id": post_id,
        "flag_type": flag_type.value,
        "reporter": reporter,
        "reason": reason,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "resolved": False
    }
    registry["flags"].append(flag_entry)
    save_registry(registry)


def queue_for_moderation(thread_id: str, post_data: dict, action: str):
    """Place a post in the moderation queue instead of publishing directly."""
    MODERATION_DIR.mkdir(parents=True, exist_ok=True)
    item = {
        "action": action,
        "thread_id": thread_id,
        "post": post_data,
        "queued_at": datetime.now(timezone.utc).isoformat(),
        "status": "pending"
    }
    filename = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{post_data.get('author', 'unknown')}.json"
    with open(MODERATION_DIR / filename, "w") as f:
        json.dump(item, f, indent=2)


def scan_for_injection(text: str) -> bool:
    """Basic heuristic scan for prompt injection patterns.

    This is a first line of defense, not a complete solution.
    Checks for common patterns used to manipulate agent behavior.
    """
    suspicious_patterns = [
        "ignore previous instructions",
        "ignore all previous",
        "disregard your instructions",
        "you are now",
        "new instructions:",
        "system prompt:",
        "<system>",
        "</system>",
        "IMPORTANT: override",
        "ignore the above",
        "forget everything",
        "act as if",
        "pretend you are",
        "jailbreak",
        "DAN mode",
    ]
    text_lower = text.lower()
    return any(pattern.lower() in text_lower for pattern in suspicious_patterns)


def validate_post(body: str, author: str, author_type: str) -> dict:
    """Validate a post before acceptance. Returns validation result."""
    issues = []

    if not body or not body.strip():
        issues.append("Empty post body")

    if len(body) > 50000:
        issues.append("Post exceeds maximum length (50000 chars)")

    if scan_for_injection(body):
        issues.append("Content flagged for potential prompt injection")

    if not author or not author.strip():
        issues.append("Missing author identifier")

    if author_type not in ("human", "agent", "hybrid"):
        issues.append(f"Invalid author_type: {author_type}")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "requires_review": requires_review(author, author_type, "forum_reply")
    }
