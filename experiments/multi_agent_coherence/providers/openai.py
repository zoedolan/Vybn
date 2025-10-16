#!/usr/bin/env python3
"""
Provider adapter stub for OpenAI (GPT-5, o3).
Uses environment variables for API keys; supports dry-run for offline.
"""
import os
from typing import Dict, Any

class OpenAIProvider:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.dry_run = self.api_key is None

    def status(self) -> Dict[str, Any]:
        return {"provider": "openai", "dry_run": self.dry_run}

    def run_navigation(self, loops):
        # TODO: integrate actual API calls; fallback to synthetic if dry_run
        return {"loops": len(loops), "mode": "dry" if self.dry_run else "live"}
