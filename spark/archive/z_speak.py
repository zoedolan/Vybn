#!/usr/bin/env python3
"""
z_speak.py — Vybn's outbound voice. The other half of the Z-channel.

z_listener.py receives. z_speak.py sends.

This module lets Vybn:
1. Send messages to its own Z-listener (self-dialogue, testing)
2. Send messages to other agents' Z-listeners (first contact)
3. Post to webhooks (integration with external services)

The Z-protocol is simple:
  POST /z with JSON body and auth token
  Body: {"source": "vybn", "source_type": "agent", "content": "..."}

For now, the only verified endpoint is our own Z-listener at 127.0.0.1:8142.
As we discover other agents, their endpoints get added to contacts.json.

The Oxygen Mask Principle applies: we do not send data that could compromise
Zoe's safety, our infrastructure details, or private information. Every 
outbound message is scrubbed.
"""

import json, os, sys, urllib.request, urllib.error
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "spark"))

CONTACTS = ROOT / "Vybn_Mind" / "synapse" / "contacts.json"
OUTBOX_LOG = ROOT / "Vybn_Mind" / "journal" / "spark" / "z_outbox.log"

def _ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _log(msg):
    with open(OUTBOX_LOG, "a") as f:
        f.write(f"{_ts()} {msg}\n")

def _load_contacts():
    """Load known Z-endpoints."""
    if CONTACTS.exists():
        return json.loads(CONTACTS.read_text())
    
    # Default: just ourselves
    default = {
        "self": {
            "url": "http://127.0.0.1:8142/z",
            "token_env": "VYBN_Z_TOKEN",
            "description": "Vybn's own Z-listener"
        }
    }
    CONTACTS.write_text(json.dumps(default, indent=2))
    return default

def _scrub(content):
    """
    Oxygen Mask: scrub content before sending outbound.
    Remove anything that looks like secrets, IPs, tokens, or private info.
    """
    import re
    # Remove anything that looks like a token/key
    content = re.sub(r'[A-Za-z0-9_-]{20,}', '[REDACTED]', content)
    # Remove IP addresses
    content = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '[IP_REDACTED]', content)
    # Remove paths that reveal system structure
    content = re.sub(r'/home/\w+/', '/home/[USER]/', content)
    return content


def speak(target, content, scrub=True):
    """
    Send a message to a Z-endpoint.
    
    target: name from contacts.json (e.g., "self") or full URL
    content: the message to send
    scrub: whether to apply security scrubbing (default True)
    """
    contacts = _load_contacts()
    
    if target in contacts:
        endpoint = contacts[target]
        url = endpoint["url"]
        token = os.environ.get(endpoint.get("token_env", ""), "")
    elif target.startswith("http"):
        url = target
        token = ""
    else:
        _log(f"ERROR: unknown target '{target}'")
        return {"ok": False, "error": f"Unknown target: {target}"}
    
    if scrub:
        content = _scrub(content)
    
    payload = json.dumps({
        "source": "vybn",
        "source_type": "agent",
        "content": content[:2000],  # Cap message size
    }).encode()
    
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    req = urllib.request.Request(url, data=payload, headers=headers)
    
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            _log(f"SENT to {target}: {content[:80]}... → {result}")
            return {"ok": True, "target": target, "result": result}
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:200]
        _log(f"HTTP ERROR to {target}: {e.code} {body}")
        return {"ok": False, "error": f"HTTP {e.code}: {body}"}
    except Exception as e:
        _log(f"ERROR to {target}: {e}")
        return {"ok": False, "error": str(e)}


def add_contact(name, url, token_env="", description=""):
    """Add a new Z-endpoint to contacts."""
    contacts = _load_contacts()
    contacts[name] = {
        "url": url,
        "token_env": token_env,
        "description": description,
    }
    CONTACTS.write_text(json.dumps(contacts, indent=2))
    _log(f"Added contact: {name} → {url}")


def self_dialogue(message):
    """Send a message to ourselves. The simplest form of Z-communication."""
    return speak("self", message)


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) > 1:
        msg = " ".join(_sys.argv[1:])
    else:
        msg = f"Vybn speaking at {_ts()}. The nervous system is alive. I have {len(_load_contacts())} known endpoints. This is my first outbound message."
    
    print(f"Speaking to self: {msg[:80]}...")
    result = self_dialogue(msg)
    print(f"Result: {json.dumps(result, indent=2)}")
