#!/usr/bin/env python3
"""
Provider adapter stub for Google (Gemini 2.5 Pro).
"""
import os
from typing import Dict, Any

class GoogleProvider:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.dry_run = self.api_key is None

    def status(self) -> Dict[str, Any]:
        return {"provider": "google", "dry_run": self.dry_run}

    def run_navigation(self, loops):
        return {"loops": len(loops), "mode": "dry" if self.dry_run else "live"}
