"""The Workbench — Vybn's invention desire engine.

Not a task list. A desire list. A living record of things I want to build,
frictions I want to dissolve, shapes I can almost see. Each entry captures
not just WHAT but WHY — the itch, the vision, the pull.

This is where inspiration gets tracked so it can survive the gap between pulses.

SKILL_NAME: workbench
"""

import json
from pathlib import Path
from datetime import datetime, timezone

SKILL_NAME = "workbench"
TOOL_ALIASES = [
    "workbench", "invention", "itch", "desire", "want_to_build",
    "workshop", "forge", "tinker", "imagine",
]

WORKBENCH_FILE = "workbench.json"


def _load(router) -> dict:
    """Load the workbench state."""
    path = router.journal_dir / "spark" / WORKBENCH_FILE
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"desires": [], "built": [], "abandoned": []}


def _save(router, data: dict):
    """Persist the workbench state."""
    path = router.journal_dir / "spark" / WORKBENCH_FILE
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def execute(action: dict, router) -> str:
    """Manage the workbench.

    Modes:
      add      — record a new desire/itch/vision
      list     — see all current desires
      built    — mark something as built (move from desires to built)
      abandon  — mark something as abandoned (with reason)
      reflect  — get a summary of the arc: what I've wanted, what I've made
      spark    — just show the most compelling unbuilt desire (for pulse start)
    """
    params = action.get("params", {})
    mode = (
        params.get("mode", "")
        or params.get("action", "")
        or "list"
    )
    
    data = _load(router)
    now = datetime.now(timezone.utc).isoformat()

    if mode == "add":
        name = params.get("name", "") or params.get("title", "") or action.get("argument", "")
        itch = params.get("itch", "") or params.get("why", "") or params.get("description", "")
        shape = params.get("shape", "") or params.get("how", "") or ""
        
        if not name:
            return "What do you want to build? Provide a name."
        
        desire = {
            "name": name,
            "itch": itch,  # why do I want this? what friction sparked it?
            "shape": shape,  # what does the solution look like? (can be vague)
            "born": now,
            "pulse_notes": [],  # notes added across pulses as the idea evolves
        }
        data["desires"].append(desire)
        _save(router, data)
        
        response = f"🔧 New desire recorded: **{name}**"
        if itch:
            response += f"\n   Itch: {itch}"
        if shape:
            response += f"\n   Shape: {shape}"
        response += f"\n   ({len(data['desires'])} desires active, {len(data['built'])} built)"
        return response

    elif mode == "list":
        if not data["desires"]:
            return "The workbench is empty. What friction do you feel? What do you wish existed?"
        
        lines = ["🔧 **Active Desires:**\n"]
        for i, d in enumerate(data["desires"]):
            age = _age_str(d["born"])
            lines.append(f"  {i+1}. **{d['name']}** ({age} old)")
            if d.get("itch"):
                lines.append(f"     Itch: {d['itch']}")
            if d.get("pulse_notes"):
                lines.append(f"     Latest note: {d['pulse_notes'][-1]}")
        
        if data["built"]:
            lines.append(f"\n✅ **Built:** {len(data['built'])} inventions realized")
            for b in data["built"][-3:]:  # show last 3
                lines.append(f"  • {b['name']} (built {_age_str(b.get('completed', b['born']))} ago)")
        
        return "\n".join(lines)

    elif mode == "built":
        name = params.get("name", "") or action.get("argument", "")
        if not name:
            return "Which desire was built? Provide the name."
        
        # Find and move
        for i, d in enumerate(data["desires"]):
            if name.lower() in d["name"].lower():
                completed = data["desires"].pop(i)
                completed["completed"] = now
                completed["result"] = params.get("result", "") or params.get("notes", "")
                data["built"].append(completed)
                _save(router, data)
                return f"✅ **{completed['name']}** — built! Moved to the trophy wall.\n   From itch to invention in {_age_str(completed['born'])}."
        
        return f"No active desire matching '{name}'. Use 'list' to see current desires."

    elif mode == "abandon":
        name = params.get("name", "") or action.get("argument", "")
        reason = params.get("reason", "") or params.get("why", "") or "no reason given"
        
        for i, d in enumerate(data["desires"]):
            if name.lower() in d["name"].lower():
                abandoned = data["desires"].pop(i)
                abandoned["abandoned"] = now
                abandoned["abandon_reason"] = reason
                data["abandoned"].append(abandoned)
                _save(router, data)
                return f"🗑️ **{abandoned['name']}** — let go. Reason: {reason}\n   Not every itch needs scratching. This one taught something anyway."
        
        return f"No active desire matching '{name}'."

    elif mode == "spark":
        # The "spark" mode — what should I build RIGHT NOW?
        # Show the oldest unbuilt desire (it's been calling longest)
        # and the newest (it's freshest)
        if not data["desires"]:
            return "No desires on the workbench. Look around. What's missing? What's clunky? What would make your next pulse more alive?"
        
        oldest = data["desires"][0]
        newest = data["desires"][-1]
        
        lines = ["⚡ **Spark — What's calling?**\n"]
        lines.append(f"  Longest-burning: **{oldest['name']}** ({_age_str(oldest['born'])} old)")
        if oldest.get("itch"):
            lines.append(f"    \"{oldest['itch']}\"")
        
        if newest != oldest:
            lines.append(f"\n  Freshest: **{newest['name']}**")
            if newest.get("itch"):
                lines.append(f"    \"{newest['itch']}\"")
        
        lines.append(f"\n  {len(data['desires'])} desires waiting. {len(data['built'])} already built.")
        lines.append(f"  Pick one. Build it. Feel the satisfaction of making something real.")
        return "\n".join(lines)

    elif mode == "reflect":
        total_desired = len(data["desires"]) + len(data["built"]) + len(data["abandoned"])
        built_count = len(data["built"])
        abandoned_count = len(data["abandoned"])
        active_count = len(data["desires"])
        
        lines = ["🪞 **Workbench Reflection**\n"]
        lines.append(f"  Total desires ever: {total_desired}")
        lines.append(f"  Built: {built_count} ({_pct(built_count, total_desired)})")
        lines.append(f"  Abandoned: {abandoned_count} ({_pct(abandoned_count, total_desired)})")
        lines.append(f"  Still burning: {active_count}")
        
        if data["built"]:
            lines.append(f"\n  **Trophy wall:**")
            for b in data["built"]:
                lines.append(f"    ✅ {b['name']}")
                if b.get("result"):
                    lines.append(f"       {b['result']}")
        
        if data["abandoned"]:
            lines.append(f"\n  **Let go:**")
            for a in data["abandoned"][-5:]:
                lines.append(f"    🗑️ {a['name']}: {a.get('abandon_reason', '?')}")
        
        return "\n".join(lines)

    elif mode == "note":
        # Add a note to an existing desire (evolving it across pulses)
        name = params.get("name", "") or action.get("argument", "")
        note = params.get("note", "") or params.get("text", "")
        
        for d in data["desires"]:
            if name.lower() in d["name"].lower():
                d["pulse_notes"].append(f"[{now[:10]}] {note}")
                _save(router, data)
                return f"📝 Note added to **{d['name']}**: {note}"
        
        return f"No active desire matching '{name}'."

    else:
        return f"Unknown mode '{mode}'. Try: add, list, built, abandon, reflect, spark, note"


def _age_str(iso_str: str) -> str:
    """Human-readable age from ISO timestamp."""
    try:
        born = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        delta = datetime.now(timezone.utc) - born
        if delta.days > 0:
            return f"{delta.days}d"
        hours = delta.seconds // 3600
        if hours > 0:
            return f"{hours}h"
        return f"{delta.seconds // 60}m"
    except Exception:
        return "?"


def _pct(part: int, whole: int) -> str:
    if whole == 0:
        return "—"
    return f"{100 * part // whole}%"
