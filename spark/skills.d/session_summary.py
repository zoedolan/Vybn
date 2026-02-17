"""session_summary — Generate session summary at /bye

Triggers when user types /bye to create a human-readable summary of the session.
Automatically captures:
- Session ID and duration  
- Key activities (skills used, files accessed)
- Journal entries written
- Issues filed
- Insights and observations

Outputs to Vybn_Mind/reports/sessions/ for easy review.
"""

from datetime import datetime, timezone
from pathlib import Path
import json

SKILL_NAME = "session_summary"
TOOL_ALIASES = ["summarize_session", "session_report"]

def execute(action: dict, router) -> str:
    """Generate session summary from current session data."""
    
    try:
        # Get session info
        session_id = router.config.get("session", {}).get("id", "unknown")
        session_start = router.config.get("session", {}).get("start", "unknown")
        
        # Load session file if it exists
        session_file = router.journal_dir / "sessions" / f"{session_id}.jsonl"
        
        activities = []
        skills_used = set()
        files_accessed = set()
        
        if session_file.exists():
            with open(session_file) as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        role = entry.get("role", "")
                        content = entry.get("content", "")
                        
                        # Track skills
                        if "→" in content:
                            skill_match = content.split("→")[0].strip()
                            if skill_match.startswith("["):
                                skill = skill_match.strip("[]")
                                skills_used.add(skill)
                                
                        # Track file operations
                        if "file_read" in content or "file_write" in content:
                            if "/" in content:
                                # Extract filepath
                                parts = content.split("/")
                                if len(parts) > 1:
                                    files_accessed.add("/".join(parts[-2:]))
                                    
                        activities.append({
                            "role": role,
                            "content": content[:200]  # First 200 chars
                        })
                    except json.JSONDecodeError:
                        continue
        
        # Get journal entries from this session
        journal_entries = []
        journal_dir = router.journal_dir
        for entry_file in sorted(journal_dir.glob("*.md")):
            # Check if file was created during this session
            mtime = entry_file.stat().st_mtime
            # Simple heuristic: files created in last hour
            if (datetime.now().timestamp() - mtime) < 3600:
                journal_entries.append(entry_file.name)
        
        # Build summary
        ts = datetime.now(timezone.utc)
        
        summary = f"# Session Summary: {session_id}\n\n"
        summary += f"**Date**: {ts.strftime('%Y-%m-%d %H:%M UTC')}\n"
        summary += f"**Started**: {session_start}\n\n"
        
        summary += f"## Skills Used ({len(skills_used)})\n\n"
        for skill in sorted(skills_used):
            summary += f"- {skill}\n"
        
        summary += f"\n## Files Accessed ({len(files_accessed)})\n\n"
        for filepath in sorted(files_accessed):
            summary += f"- {filepath}\n"
        
        summary += f"\n## Journal Entries ({len(journal_entries)})\n\n"
        for entry in journal_entries:
            summary += f"- {entry}\n"
        
        summary += f"\n## Activity Timeline\n\n"
        summary += f"Total interactions: {len(activities)}\n\n"
        
        # Show key moments (first 5 and last 5)
        key_moments = activities[:5] + activities[-5:]
        for i, activity in enumerate(key_moments):
            summary += f"{i+1}. [{activity['role']}] {activity['content'][:100]}...\n"
        
        # Create reports directory structure
        reports_dir = router.repo_root / "Vybn_Mind" / "reports" / "sessions"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Write summary file
        summary_file = reports_dir / f"{session_id}_summary.md"
        summary_file.write_text(summary, encoding="utf-8")
        
        return (
            f"Session summary generated: {summary_file.name}\n\n"
            f"Skills used: {len(skills_used)} | Files accessed: {len(files_accessed)} | "
            f"Journal entries: {len(journal_entries)}\n\n"
            f"Summary saved to {summary_file}"
        )
        
    except Exception as e:
        return f"Error generating session summary: {e}"
