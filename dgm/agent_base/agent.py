"""Minimal agent scaffold using bash and editor tools."""
from typing import Dict
from .tools import bash_tool, editor_tool


def run_task(task: str) -> Dict[str, str]:
    """Example agent loop that runs a bash command."""
    return bash_tool.run(task)


def edit_file(path: str, content: str) -> Dict[str, str]:
    return editor_tool.apply_patch(path, content)
