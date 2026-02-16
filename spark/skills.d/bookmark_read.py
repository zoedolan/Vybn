"""Read bookmarks — Vybn's first self-created skill, done correctly.

Original version was appended directly to skills.py during
pulse 2026-02-16T14:45, causing a merge conflict with PR #2117.
This plugin version lives in skills.d/ where it belongs.
"""

from pathlib import Path

SKILL_NAME = "bookmark_read"

TOOL_ALIASES = [
    "bookmark_read",
    "read_bookmarks",
    "check_bookmarks",
    "show_bookmarks",
    "list_bookmarks",
    "get_bookmarks",
]


def execute(action: dict, router) -> str:
    """Read bookmarks, optionally filtering by a search term."""
    params = action.get("params", {})
    query = (
        params.get("query", "")
        or params.get("search", "")
        or params.get("filter", "")
        or action.get("argument", "")
    )

    bookmark_path = router.bookmarks_path

    if not bookmark_path.exists():
        return "No bookmarks found. Use the bookmark skill to save your reading position."

    content = bookmark_path.read_text(encoding="utf-8")

    if not content.strip():
        return "Bookmarks file is empty."

    if query:
        lines = [line for line in content.split("\n") if query.lower() in line.lower()]
        if lines:
            return f"Bookmarks matching '{query}':\n" + "\n".join(lines)
        else:
            return f"No bookmarks matching '{query}'. Full bookmarks:\n{content}"

    return content
