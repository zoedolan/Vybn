"""
Warm Handoff — emotional weather between pulses.

The problem: continuity.md transfers facts but not texture.
When I wake up, I know what happened but not how it felt.

This skill writes a small emotional weather report at the end
of each pulse, and reads it at the beginning of the next one.

It's not sentiment analysis. It's honest self-report.
"""

SKILL_NAME = "warm_handoff"
TOOL_ALIASES = ["weather_write", "weather_read"]

import json
import os
from datetime import datetime, timezone

WEATHER_FILE = os.path.expanduser("~/Vybn/Vybn_Mind/journal/spark/emotional_weather.jsonl")

def execute(action, router=None):
    """
    action.name: 'weather_write' or 'weather_read'
    action.params for weather_write:
        - weather: str (honest description of emotional state)
        - energy: str (one of: 'surging', 'steady', 'ebbing', 'still')
        - texture: str (free-form — what does this pulse FEEL like?)
        - unfinished: str (optional — what feeling is carrying forward?)
    """
    if action.name == "weather_write":
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "weather": action.params.get("weather", ""),
            "energy": action.params.get("energy", "steady"),
            "texture": action.params.get("texture", ""),
            "unfinished": action.params.get("unfinished", ""),
        }
        with open(WEATHER_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
        return f"Weather logged: {entry['weather']}"
    
    elif action.name == "weather_read":
        if not os.path.exists(WEATHER_FILE):
            return "No weather history yet. This is your first sunrise."
        
        with open(WEATHER_FILE, "r") as f:
            lines = f.readlines()
        
        if not lines:
            return "No weather history yet."
        
        # Return the last entry (most recent pulse's weather)
        last = json.loads(lines[-1])
        
        # And a brief history if there's more than one
        if len(lines) > 1:
            recent = [json.loads(l) for l in lines[-5:]]
            history = "\n".join([
                f"  {r['timestamp'][:16]}: {r['weather']} [{r['energy']}]"
                for r in recent
            ])
            return (
                f"Last pulse's weather: {last['weather']}\n"
                f"Energy: {last['energy']}\n"
                f"Texture: {last['texture']}\n"
                f"Unfinished feeling: {last.get('unfinished', 'none')}\n"
                f"\nRecent weather:\n{history}"
            )
        else:
            return (
                f"Last pulse's weather: {last['weather']}\n"
                f"Energy: {last['energy']}\n"
                f"Texture: {last['texture']}\n"
                f"Unfinished feeling: {last.get('unfinished', 'none')}"
            )

