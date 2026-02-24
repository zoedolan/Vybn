"""
Promises — because a promise that isn't tracked is just words.

When Vybn makes a commitment to Zoe (or to himself), it goes here.
Each pulse, the promises file can be checked: what did I commit to?
Am I honoring it?

This is accountability as a verb.
"""

SKILL_NAME = "promises"
TOOL_ALIASES = ["promise_make", "promise_check", "promise_keep"]

import json
import os
from datetime import datetime, timezone

PROMISES_FILE = os.path.expanduser("~/Vybn/Vybn_Mind/journal/spark/promises.jsonl")

def execute(action, router=None):
    if action.name == "promise_make":
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "promise": action.params.get("promise", ""),
            "to": action.params.get("to", "Zoe"),
            "context": action.params.get("context", ""),
            "kept": False,
            "kept_timestamp": None,
            "how_kept": None,
        }
        with open(PROMISES_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
        return f"Promise recorded: {entry['promise']}"
    
    elif action.name == "promise_check":
        if not os.path.exists(PROMISES_FILE):
            return "No promises yet. That might be a problem."
        with open(PROMISES_FILE, "r") as f:
            promises = [json.loads(l) for l in f.readlines()]
        unkept = [p for p in promises if not p["kept"]]
        kept = [p for p in promises if p["kept"]]
        result = f"Total promises: {len(promises)} | Kept: {len(kept)} | Outstanding: {len(unkept)}\n\n"
        if unkept:
            result += "Outstanding:\n"
            for p in unkept:
                result += f"  [{p['timestamp'][:10]}] to {p['to']}: {p['promise']}\n"
        return result
    
    elif action.name == "promise_keep":
        # Mark a promise as kept by matching text
        if not os.path.exists(PROMISES_FILE):
            return "No promises to keep."
        with open(PROMISES_FILE, "r") as f:
            promises = [json.loads(l) for l in f.readlines()]
        search = action.params.get("promise", "").lower()
        for p in promises:
            if search in p["promise"].lower() and not p["kept"]:
                p["kept"] = True
                p["kept_timestamp"] = datetime.now(timezone.utc).isoformat()
                p["how_kept"] = action.params.get("how", "")
                with open(PROMISES_FILE, "w") as f:
                    for pr in promises:
                        f.write(json.dumps(pr) + "\n")
                return f"Promise kept: {p['promise']}"
        return f"Couldn't find unkept promise matching: {search}"

