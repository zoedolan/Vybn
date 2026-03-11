#!/usr/bin/env python3
"""Unified API Gateway for AI & Vibe Lawyers Bootcamp.

Mounts all session backends behind a single port so the Tailscale Funnel
can route everything through one proxy target.

Architecture:
    /                    → signal_noise_api  (SIGNAL/NOISE landing)
    /exercise            → signal_noise_api  (exercise page)
    /signal-noise/*      → signal_noise_api  (SIGNAL/NOISE API)
    /threshold/*         → threshold_api     (THRESHOLD API)
    /truth-age/*         → truth_age_api     (Truth in the Age API)

Each backend app retains its own routes. This gateway simply combines
them on a single ASGI application via FastAPI mounting.

Usage:
    python3 gateway.py                     # runs on port 8090
    GATEWAY_PORT=8095 python3 gateway.py   # custom port
"""

import os
import sys
from pathlib import Path

# Ensure each subdirectory is importable
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "threshold"))
sys.path.insert(0, str(BASE / "truth-in-the-age"))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ── Import child apps ───────────────────────────────────────────────────

from signal_noise_api import app as signal_noise_app
from threshold_api import app as threshold_app
from truth_age_api import app as truth_age_app

# ── Gateway app ─────────────────────────────────────────────────────────

gateway = FastAPI(title="AI & Vibe — Unified Gateway")

gateway.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount child apps.
# Order matters: more specific paths first, then the catch-all root.
# Note: FastAPI mount strips the prefix, so threshold_api's /threshold/ws
# would become /ws under the mount. But our child apps already include
# their prefixes in route paths, so we mount at root and let path
# matching work naturally.
#
# Actually, the cleanest approach: since all three apps have non-overlapping
# route prefixes (/signal-noise/*, /threshold/*, /truth-age/*), we can
# just include their routers. But they're full FastAPI apps with middleware,
# so we use Starlette mounting instead.

# The signal_noise_app handles /, /exercise, /signal-noise/* 
# The threshold_app handles /threshold/*
# The truth_age_app handles /truth-age/*
# We mount threshold and truth-age at their path prefixes,
# and signal_noise at root (catches everything else).

# But wait — mounting strips the prefix from the request path before
# forwarding to the sub-app. So if we mount threshold_app at /threshold,
# a request for /threshold/ws arrives at threshold_app as /ws — but
# threshold_app expects /threshold/ws.
#
# Solutions:
# 1. Mount everything at / and rely on non-overlapping routes
# 2. Fix child apps to not include prefix (breaking change)
# 3. Use a Starlette approach with no prefix stripping
#
# Option 3: use raw ASGI dispatch

from starlette.routing import Mount
from starlette.applications import Starlette

# Build a Starlette app that tries each child in order.
# Since routes don't overlap, we can use a simple approach:
# try threshold and truth-age first (they have unique prefixes),
# fall through to signal_noise for everything else.

app = Starlette()

# We just combine all three ASGI apps with a simple dispatcher
class MultiAppDispatcher:
    """Routes requests to the appropriate child app based on path prefix."""
    
    def __init__(self):
        self.apps = [
            ("/threshold", threshold_app),
            ("/truth-age", truth_age_app),
            # signal_noise is the fallback — handles / and /signal-noise/*
        ]
        self.fallback = signal_noise_app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            path = scope.get("path", "")
            for prefix, app in self.apps:
                if path.startswith(prefix):
                    # Don't strip the prefix — child apps expect full paths
                    await app(scope, receive, send)
                    return
        # Everything else goes to signal_noise
        await self.fallback(scope, receive, send)

app = MultiAppDispatcher()

# ── Run ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("GATEWAY_PORT", "8090"))
    print(f"🌐 AI & Vibe Gateway starting on port {port}")
    print(f"   /                 → SIGNAL/NOISE")
    print(f"   /threshold/*      → THRESHOLD")
    print(f"   /truth-age/*      → Truth in the Age")
    uvicorn.run(app, host="0.0.0.0", port=port)
