#!/usr/bin/env python3
"""
Provider adapter stub for Anthropic (Claude 4.5 Sonnet).
"""
import os
from typing import Dict, Any

class AnthropicProvider:
    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.dry_run = self.api_key is None

    def status(self) -> Dict[str, Any]:
        return {"provider": "anthropic", "dry_run": self.dry_run}

    def run_navigation(self, loops):
        return {"loops": len(loops), "mode": "dry" if self.dry_run else "live"}
