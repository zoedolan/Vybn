#!/usr/bin/env python3
"""Portal API — read-only endpoints for the Vybn Portal dashboard.

All endpoints require the same VYBN_CHAT_TOKEN authentication used
by the chat interface.  Every route is GET-only: the Portal observes
the organism but never mutates it.

Wire up by calling ``attach_organism(org)`` and ``attach_portal_bus(bus)``
from the main Spark agent loop, alongside the existing ``attach_bus()``.
"""

import time
from collections import deque
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Request, HTTPException

# ---------------------------------------------------------------------------
# Auth — reuse the same token check from web_interface
# ---------------------------------------------------------------------------
import hmac
import os

_CHAT_TOKEN = os.environ.get("VYBN_CHAT_TOKEN", "vybn-dev-token")


def _check_token(token: str) -> bool:
    return hmac.compare_digest(token, _CHAT_TOKEN)


async def require_auth(request: Request):
    token = (
        request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    )
    if not token:
        token = request.query_params.get("token", "")
    if not _check_token(token):
        raise HTTPException(status_code=401, detail="Invalid token")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
router = APIRouter(prefix="/api", tags=["portal"])

# Organism and bus references — set via attach_*() calls
_organism = None
_bus = None

# Rolling vitals buffer (bounded)
_vitals: deque = deque(maxlen=120)


def attach_organism(organism):
    """Called by the Spark main loop to give the Portal read access."""
    global _organism
    _organism = organism


def attach_portal_bus(bus):
    """Called by the Spark main loop to give the Portal read access to the bus."""
    global _bus
    _bus = bus


# ---------------------------------------------------------------------------
# GET /api/organism — current organism state
# ---------------------------------------------------------------------------
@router.get("/organism", dependencies=[Depends(require_auth)])
async def get_organism():
    if _organism is None:
        return {
            "mood": "quiet",
            "pulse": 0.0,
            "uptime_hours": 0,
            "cycle_count": 0,
            "codebook_size": 0,
            "seed_registry": [],
            "alive": False,
        }

    try:
        # Determine current mood from latest trace
        mood = "quiet"
        if _organism.traces:
            last = _organism.traces[-1]
            # Mood is embedded in results from breathe primitive
            for r in last.get("results", []):
                if isinstance(r, dict) and "result" in r:
                    result = r["result"]
                    if isinstance(result, dict) and "mood" in result:
                        mood = result["mood"]

        energy = 0.0
        try:
            # Substrate doesn't have a direct energy field;
            # derive a pulse proxy from codebook fitness
            census = _organism.codebook.census()
            energy = float(census.get("mean_fitness", 0.5))
        except Exception:
            energy = 0.5

        data = {
            "mood": mood,
            "pulse": round(energy, 3),
            "uptime_hours": 0,
            "cycle_count": _organism.cycle,
            "codebook_size": len(_organism.codebook.primitives),
            "seed_registry": list(
                getattr(_organism.codebook, "_seed_names", [])
            ) or [p.name for p in _organism.codebook.primitives if p.source == "seed"],
            "alive": True,
        }

        # Snapshot vitals for the buffer
        _vitals.append({
            "ts": datetime.now(timezone.utc).isoformat(),
            "energy": data["pulse"],
            "mood": data["mood"],
            "cycle": data["cycle_count"],
        })

        return data

    except Exception as exc:
        return {"alive": False, "error": str(exc)[:200]}


# ---------------------------------------------------------------------------
# GET /api/vitals — time-series snapshots
# ---------------------------------------------------------------------------
@router.get("/vitals", dependencies=[Depends(require_auth)])
async def get_vitals():
    return {"vitals": list(_vitals), "count": len(_vitals)}


# ---------------------------------------------------------------------------
# GET /api/memory — memory fabric summary
# ---------------------------------------------------------------------------
@router.get("/memory", dependencies=[Depends(require_auth)])
async def get_memory():
    if _organism is None:
        return {
            "planes": {
                "private": {"count": 0, "recent": []},
                "relational": {"count": 0, "recent": []},
                "commons": {"count": 0, "recent": []},
            },
            "total": 0,
        }

    try:
        snapshot = _organism.substrate.memory_snapshot()
        planes = {}

        # Private memories
        private_entries = snapshot.get("private", [])
        planes["private"] = {
            "count": len(private_entries),
            "recent": [
                entry.content[:120] if hasattr(entry, "content") else str(entry)[:120]
                for entry in private_entries[:5]
            ],
        }

        # Relational memories
        relational_entries = snapshot.get("relational", [])
        planes["relational"] = {
            "count": len(relational_entries),
            "recent": [
                entry.content[:120] if hasattr(entry, "content") else str(entry)[:120]
                for entry in relational_entries[:5]
            ],
        }

        # Commons patterns
        commons_entries = snapshot.get("commons", [])
        planes["commons"] = {
            "count": len(commons_entries),
            "recent": [
                str(entry)[:120] for entry in commons_entries[:5]
            ],
        }

        stats = snapshot.get("stats", {})
        graph = snapshot.get("graph", {})

        total = sum(p["count"] for p in planes.values())

        return {
            "planes": planes,
            "total": total,
            "stats": stats,
            "graph": graph,
        }

    except Exception as exc:
        return {
            "planes": {
                "private": {"count": 0, "recent": []},
                "relational": {"count": 0, "recent": []},
                "commons": {"count": 0, "recent": []},
            },
            "total": 0,
            "error": str(exc)[:200],
        }


# ---------------------------------------------------------------------------
# GET /api/timeline — recent bus audit entries
# ---------------------------------------------------------------------------
@router.get("/timeline", dependencies=[Depends(require_auth)])
async def get_timeline():
    if _bus is None:
        return {"events": [], "count": 0}

    try:
        entries = _bus.recent(20)
        events = []
        for e in entries:
            events.append({
                "ts": datetime.fromtimestamp(e.timestamp, tz=timezone.utc).isoformat(),
                "type": e.msg_type.name.lower() if e.msg_type is not None else "action",
                "source": e.source,
                "summary": e.summary,
                "age": e.age_str,
            })
        return {"events": events, "count": _bus.audit_count}

    except Exception as exc:
        return {"events": [], "count": 0, "error": str(exc)[:200]}


# ---------------------------------------------------------------------------
# GET /api/codebook — primitives and codebook state
# ---------------------------------------------------------------------------
@router.get("/codebook", dependencies=[Depends(require_auth)])
async def get_codebook():
    if _organism is None:
        return {"primitives": [], "census": {}, "graveyard": []}

    try:
        census = _organism.codebook.census()
        primitives = [
            {
                "name": p.name,
                "alive": p.alive,
                "age": p.age,
                "activations": p.activations,
                "fitness": round(p.fitness, 3),
                "source": p.source,
            }
            for p in _organism.codebook.primitives
        ]
        graveyard = _organism.codebook.graveyard[-10:]  # last 10 deaths

        return {
            "primitives": primitives,
            "census": census,
            "graveyard": graveyard,
        }

    except Exception as exc:
        return {"primitives": [], "census": {}, "error": str(exc)[:200]}


# ---------------------------------------------------------------------------
# GET /api/traces — recent organism pulse traces
# ---------------------------------------------------------------------------
@router.get("/traces", dependencies=[Depends(require_auth)])
async def get_traces():
    if _organism is None:
        return {"traces": [], "count": 0}

    try:
        recent = _organism.traces[-20:]
        return {"traces": recent, "count": len(_organism.traces)}

    except Exception as exc:
        return {"traces": [], "count": 0, "error": str(exc)[:200]}
