"""Simple file editing tool."""
from pathlib import Path
from typing import Dict


def apply_patch(path: str, new_content: str) -> Dict[str, str]:
    """Replace the entire file content with new_content."""
    p = Path(path)
    p.write_text(new_content, encoding="utf-8")
    return {"status": "ok"}
