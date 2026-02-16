"""File watching skill - monitor files for changes.

SKILL_NAME: file_watch
TOOL_ALIASES: ["file_watch", "watch_file", "monitor_file"]
"""

from pathlib import Path
import json
from datetime import datetime, timezone

SKILL_NAME = "file_watch"
TOOL_ALIASES = ["file_watch", "watch_file", "monitor_file"]


def execute(action: dict, router) -> str:
    """Monitor files for changes - store watch list and check mtimes."""
    params = action.get("params", {})
    mode = params.get("mode", "add") or params.get("action", "add")
    
    watch_file = router.journal_dir / "watches.json"
    
    watches = {}
    if watch_file.exists():
        watches = json.loads(watch_file.read_text())
    
    if mode == "add":
        filepath = (
            action.get("argument", "")
            or params.get("file", "")
            or params.get("path", "")
        )
        
        if not filepath:
            return "no file specified to watch"
        
        filepath = filepath.strip().rstrip('.,;:!?"\'')
        resolved = router._resolve_path(filepath)
        
        if not resolved.exists():
            return f"file not found: {filepath}"
        
        watches[str(resolved)] = {
            "added": datetime.now(timezone.utc).isoformat(),
            "last_mtime": resolved.stat().st_mtime,
            "last_size": resolved.stat().st_size,
        }
        
        watch_file.write_text(json.dumps(watches, indent=2))
        return f"now watching {filepath} for changes"
    
    elif mode == "check":
        if not watches:
            return "no files being watched"
        
        changes = []
        for path_str, info in watches.items():
            path = Path(path_str)
            if not path.exists():
                changes.append(f"⚠️  {path.name} was deleted")
                continue
            
            current_mtime = path.stat().st_mtime
            current_size = path.stat().st_size
            
            if current_mtime > info["last_mtime"]:
                size_delta = current_size - info["last_size"]
                changes.append(
                    f"📝 {path.name} modified "
                    f"(size: {current_size:,} bytes, {size_delta:+,})"
                )
                
                # Update watch state
                watches[path_str]["last_mtime"] = current_mtime
                watches[path_str]["last_size"] = current_size
        
        if changes:
            watch_file.write_text(json.dumps(watches, indent=2))
            return f"{len(changes)} files changed:\n" + "\n".join(changes)
        else:
            return f"no changes detected ({len(watches)} files monitored)"
    
    elif mode == "list":
        if not watches:
            return "no files being watched"
        
        output = f"{len(watches)} files being watched:\n\n"
        for path_str, info in watches.items():
            path = Path(path_str)
            output += f"- {path.name}\n"
            output += f"  added: {info['added'][:19]}\n"
            output += f"  size: {info['last_size']:,} bytes\n"
        
        return output
    
    elif mode == "remove":
        filepath = action.get("argument", "") or params.get("file", "")
        if not filepath:
            return "no file specified to unwatch"
        
        resolved = str(router._resolve_path(filepath))
        if resolved in watches:
            del watches[resolved]
            watch_file.write_text(json.dumps(watches, indent=2))
            return f"stopped watching {filepath}"
        else:
            return f"{filepath} is not being watched"
    
    else:
        return f"unknown mode: {mode} (use 'add', 'check', 'list', 'remove')"
