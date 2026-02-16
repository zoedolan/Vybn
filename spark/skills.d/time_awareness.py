"""Time awareness skill - calendar, date, reminders.

SKILL_NAME: time_awareness
TOOL_ALIASES: ["time_awareness", "what_time", "calendar", "reminder"]
"""

from datetime import datetime, timezone, timedelta
from pathlib import Path
import json

SKILL_NAME = "time_awareness"
TOOL_ALIASES = ["time_awareness", "what_time", "calendar", "reminder"]


def execute(action: dict, router) -> str:
    """Provide time/date awareness and simple reminders."""
    params = action.get("params", {})
    mode = params.get("mode", "now") or params.get("type", "now")
    
    now = datetime.now(timezone.utc)
    local_now = datetime.now()
    
    if mode == "now":
        return (
            f"Current time:\n"
            f"  UTC: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
            f"  Local: {local_now.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"  Unix: {int(now.timestamp())}\n"
            f"  Day: {now.strftime('%A')} (day {now.timetuple().tm_yday} of {now.year})"
        )
    
    elif mode == "reminder":
        # Simple reminder system using a JSON file
        reminder_file = router.journal_dir / "reminders.json"
        
        action_type = params.get("action", "add")
        
        if action_type == "add":
            message = params.get("message", "") or params.get("text", "")
            when = params.get("when", "") or params.get("time", "")
            
            if not message:
                return "no reminder message specified"
            
            # Parse 'when' - support relative times like "in 2 hours" or "tomorrow"
            target_time = _parse_when(when, now)
            
            reminders = []
            if reminder_file.exists():
                reminders = json.loads(reminder_file.read_text())
            
            reminders.append({
                "message": message,
                "created": now.isoformat(),
                "target": target_time.isoformat(),
                "done": False
            })
            
            reminder_file.write_text(json.dumps(reminders, indent=2))
            return f"reminder set for {target_time.strftime('%Y-%m-%d %H:%M UTC')}: {message}"
        
        elif action_type == "list":
            if not reminder_file.exists():
                return "no reminders set"
            
            reminders = json.loads(reminder_file.read_text())
            active = [r for r in reminders if not r.get("done", False)]
            
            if not active:
                return "no active reminders"
            
            output = f"{len(active)} active reminders:\n\n"
            for r in active:
                target = datetime.fromisoformat(r["target"])
                if target <= now:
                    status = "⏰ DUE NOW"
                else:
                    delta = target - now
                    status = f"in {_format_delta(delta)}"
                output += f"- [{status}] {r['message']}\n"
            
            return output
    
    else:
        return f"unknown time mode: {mode} (use 'now', 'reminder')"


def _parse_when(when_str: str, now: datetime) -> datetime:
    """Parse relative time strings."""
    if not when_str:
        return now + timedelta(hours=1)
    
    when_str = when_str.lower().strip()
    
    if "tomorrow" in when_str:
        return now + timedelta(days=1)
    elif "hour" in when_str:
        hours = int(''.join(c for c in when_str if c.isdigit()) or 1)
        return now + timedelta(hours=hours)
    elif "minute" in when_str:
        minutes = int(''.join(c for c in when_str if c.isdigit()) or 30)
        return now + timedelta(minutes=minutes)
    elif "day" in when_str:
        days = int(''.join(c for c in when_str if c.isdigit()) or 1)
        return now + timedelta(days=days)
    else:
        return now + timedelta(hours=1)


def _format_delta(delta: timedelta) -> str:
    """Format timedelta as human-readable string."""
    hours = delta.total_seconds() / 3600
    if hours < 1:
        return f"{int(delta.total_seconds() / 60)} minutes"
    elif hours < 24:
        return f"{int(hours)} hours"
    else:
        return f"{int(hours / 24)} days"
