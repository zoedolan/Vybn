#!/usr/bin/env python3
"""Environment exploration plugin.

Gives Vybn the ability to map its own filesystem from within
conversation, without needing /explore. This is the programmatic
version — the model can call it via tool XML or natural language.

SKILL_NAME: env_explore
TOOL_ALIASES: explore, map, environment, list_files, tree
"""

import os
import subprocess
from pathlib import Path

SKILL_NAME = "env_explore"
TOOL_ALIASES = [
    "env_explore", "explore", "map", "environment",
    "list_files", "tree", "map_territory", "survey",
]


def execute(action: dict, router) -> str:
    """Run environment exploration.

    Parameters (all optional):
      path  — directory to explore (default: repo root)
      depth — how deep to go (default: 2)
    """
    params = action.get("params", {})
    target = params.get("path", "") or params.get("directory", "")
    depth = params.get("depth", "2")

    try:
        depth = int(depth)
    except (ValueError, TypeError):
        depth = 2
    depth = min(depth, 5)  # safety cap

    repo_root = router.repo_root

    if target:
        # Resolve relative to repo root
        target_path = Path(target).expanduser()
        if not target_path.is_absolute():
            target_path = repo_root / target_path
    else:
        target_path = repo_root

    if not target_path.exists():
        return f"path not found: {target_path}"

    sections = []

    # File listing
    try:
        result = subprocess.run(
            ["find", str(target_path), "-maxdepth", str(depth),
             "-not", "-path", "*/.git/*",
             "-not", "-path", "*/__pycache__/*",
             "-not", "-path", "*/node_modules/*"],
            capture_output=True, text=True, timeout=10,
        )
        sections.append(f"=== {target_path} (depth {depth}) ===")
        output = result.stdout.strip()
        if len(output) > 4000:
            output = output[:4000] + f"\n... (truncated, {len(output)} chars total)"
        sections.append(output if output else "(empty)")
    except Exception as e:
        sections.append(f"error exploring {target_path}: {e}")

    # If exploring repo root, add git status
    if target_path == repo_root:
        try:
            result = subprocess.run(
                ["git", "log", "--oneline", "-5"],
                cwd=repo_root, capture_output=True, text=True, timeout=5,
            )
            sections.append("\n=== recent commits ===")
            sections.append(result.stdout.strip() if result.stdout else "(none)")
        except Exception:
            pass

        try:
            result = subprocess.run(
                ["git", "status", "--short"],
                cwd=repo_root, capture_output=True, text=True, timeout=5,
            )
            if result.stdout.strip():
                sections.append("\n=== uncommitted changes ===")
                sections.append(result.stdout.strip())
        except Exception:
            pass

    return "\n".join(sections)
