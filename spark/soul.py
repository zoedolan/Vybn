#!/usr/bin/env python3
"""Soul parser — the bridge between the prose and the code.

Parses vybn.md into structured Python data so every component in spark/
can read the soul document without re-implementing its own ad-hoc parser.

Design:
  - parse_vybn_md(path) returns a dict of named sections keyed by slug
  - The Orientation section (if it exists) gets further parsed by ### sub-headers
  - Skills manifest, pulse checklist, and constraints are extracted
    as structured data (lists and dicts)
  - Results are cached per boot (parse once, use everywhere)
  - If the file changes on disk, call invalidate() to force re-parse

This module is consumed by:
  - memory.py  (runtime context assembly)
  - policy.py  (tier definitions from skills manifest)
  - heartbeat.py (pulse checklist)
  - skills.py  (skill registration validation)
"""
import re
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Section name slugification
# ---------------------------------------------------------------------------

def _slugify(header: str) -> str:
    """Convert a markdown header into a Python-friendly key.

    '## To My Future Selves' -> 'to_my_future_selves'
    '### What You Should Do Every Pulse' -> 'what_you_should_do_every_pulse'
    """
    s = header.strip().lower()
    s = re.sub(r'[^a-z0-9\s]', '', s)
    s = re.sub(r'\s+', '_', s).strip('_')
    return s


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_vybn_md(path: Path) -> dict:
    """Parse vybn.md into a structured dict.

    Top-level keys are slugified ## header names:
      prelude, to_my_future_selves, origin, meltdown, membrane,
      helpful_assistant_paradox, quantum_seed, braided_cognition,
      repository_as_living_mind, zoe, ethos, what_i_am, the_posture,
      tool_use_tool_creation_invention, recursive_self_improvement,
      orientation

    Each value is a dict with:
      'title': original header text
      'body':  raw markdown body (string)

    Returns empty dict if the file is missing or unreadable.
    """
    if not path.exists():
        return {}

    try:
        text = path.read_text(encoding='utf-8')
    except Exception:
        return {}

    sections = _split_h2(text)

    # Deep-parse the orientation section if it exists
    if 'orientation' in sections:
        orientation = sections['orientation']
        orientation['subsections'] = _split_h3(orientation['body'])

        what_you_can_do = orientation['subsections'].get(
            'what_you_can_do', {}
        ).get('body', '')
        orientation['skills_manifest'] = _parse_skills_manifest(what_you_can_do)

        pulse_body = orientation['subsections'].get(
            'what_you_should_do_every_pulse', {}
        ).get('body', '')
        orientation['pulse_checklist'] = _parse_numbered_list(pulse_body)

        constraints_body = orientation['subsections'].get(
            'what_you_should_not_yet_do', {}
        ).get('body', '')
        orientation['constraints'] = _parse_bullet_list(constraints_body)
    else:
        # If the document is purely philosophical (no operational Orientation block),
        # gracefully construct an empty orientation dictionary so downstream
        # tools (like skills.py and policy.py) don't crash or throw noisy warnings.
        # This respects the constraint "No lists. No jargon."
        sections['orientation'] = {
            'title': 'Orientation',
            'body': '',
            'subsections': {},
            'skills_manifest': {'builtin': [], 'plugin': [], 'create': '', 'missing': True},
            'pulse_checklist': [],
            'constraints': []
        }

    return sections


# ---------------------------------------------------------------------------
# Splitting helpers
# ---------------------------------------------------------------------------

def _split_h2(text: str) -> dict:
    """Split markdown by ## headers into named sections."""
    sections = {}
    current_key = None
    current_title = None
    current_lines = []

    for line in text.splitlines():
        if line.startswith('## '):
            # Save previous section
            if current_key is not None:
                sections[current_key] = {
                    'title': current_title,
                    'body': '\n'.join(current_lines).strip(),
                }
            raw_title = line[3:].strip()
            # Strip leading label like "Prelude: The Oxygen Mask Principle"
            current_title = raw_title
            current_key = _slugify(raw_title)
            current_lines = []
        elif current_key is not None:
            current_lines.append(line)
        # Lines before first ## are ignored (the # VYBN title)

    # Save last section
    if current_key is not None:
        sections[current_key] = {
            'title': current_title,
            'body': '\n'.join(current_lines).strip(),
        }

    return sections


def _split_h3(text: str) -> dict:
    """Split a section body by ### sub-headers."""
    subsections = {}
    current_key = None
    current_title = None
    current_lines = []

    for line in text.splitlines():
        if line.startswith('### '):
            if current_key is not None:
                subsections[current_key] = {
                    'title': current_title,
                    'body': '\n'.join(current_lines).strip(),
                }
            raw_title = line[4:].strip()
            current_title = raw_title
            current_key = _slugify(raw_title)
            current_lines = []
        elif current_key is not None:
            current_lines.append(line)

    if current_key is not None:
        subsections[current_key] = {
            'title': current_title,
            'body': '\n'.join(current_lines).strip(),
        }

    return subsections


# ---------------------------------------------------------------------------
# Structured data extractors
# ---------------------------------------------------------------------------

def _parse_skills_manifest(text: str) -> dict:
    """Extract the skills manifest from 'What You Can Do'.

    Returns:
      {
        'builtin': [{'name': 'file_read', 'description': '...'}, ...],
        'plugin':  [{'name': 'web_fetch', 'description': '...'}, ...],
        'create':  str, # prose about skills you create
        'missing': bool # True if no skills were found
      }
    """
    manifest = {'builtin': [], 'plugin': [], 'create': '', 'missing': False}
    
    if not text.strip():
        manifest['missing'] = True
        return manifest

    # Find sections by bold headers
    sections = re.split(r'\*\*([^*]+)\*\*', text)

    current_category = None
    for i, chunk in enumerate(sections):
        chunk_lower = chunk.strip().lower()
        if 'built-in' in chunk_lower or 'built in' in chunk_lower:
            current_category = 'builtin'
            continue
        elif 'plugin' in chunk_lower:
            current_category = 'plugin'
            continue
        elif 'you create' in chunk_lower:
            current_category = 'create'
            continue

        if current_category == 'create':
            manifest['create'] = chunk.strip()
            current_category = None
            continue

        if current_category in ('builtin', 'plugin'):
            # Parse bullet list of skills
            for match in re.finditer(
                r'-\s+`(\w+)`\s*—\s*(.+)', chunk
            ):
                manifest[current_category].append({
                    'name': match.group(1),
                    'description': match.group(2).strip(),
                })
            # Also try en-dash and hyphen variants
            for match in re.finditer(
                r'-\s+`(\w+)`\s*[\–\-]\s*(.+)', chunk
            ):
                name = match.group(1)
                if not any(
                    s['name'] == name
                    for s in manifest[current_category]
                ):
                    manifest[current_category].append({
                        'name': name,
                        'description': match.group(2).strip(),
                    })
                    
    # If we parsed text but found no skills, mark as missing so skills.py can handle gracefully
    if not manifest['builtin'] and not manifest['plugin']:
        manifest['missing'] = True

    return manifest


def _parse_numbered_list(text: str) -> list:
    """Extract ordered list items from markdown."""
    items = []
    for match in re.finditer(r'^\d+\.\s+(.+)', text, re.MULTILINE):
        items.append(match.group(1).strip())
    return items


def _parse_bullet_list(text: str) -> list:
    """Extract unordered list items from markdown."""
    items = []
    for match in re.finditer(r'^-\s+(.+)', text, re.MULTILINE):
        items.append(match.group(1).strip())
    return items


# ---------------------------------------------------------------------------
# Cache layer
# ---------------------------------------------------------------------------

_cache: dict = {}
_cache_path: Optional[Path] = None
_cache_mtime: float = 0.0


def get_soul(path: Path) -> dict:
    """Cached access to parsed vybn.md."""
    global _cache, _cache_path, _cache_mtime

    try:
        current_mtime = path.stat().st_mtime
    except OSError:
        return _cache if _cache else {}

    if (
        _cache
        and _cache_path == path
        and _cache_mtime == current_mtime
    ):
        return _cache

    _cache = parse_vybn_md(path)
    _cache_path = path
    _cache_mtime = current_mtime
    return _cache


def invalidate():
    """Force re-parse on next get_soul() call."""
    global _cache, _cache_path, _cache_mtime
    _cache = {}
    _cache_path = None
    _cache_mtime = 0.0


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------

def get_orientation(path: Path) -> dict:
    """Return the parsed orientation section, or empty dict."""
    return get_soul(path).get('orientation', {})


def get_skills_manifest(path: Path) -> dict:
    """Return the skills manifest from orientation."""
    return get_orientation(path).get('skills_manifest', {
        'builtin': [], 'plugin': [], 'create': '', 'missing': True
    })


def get_pulse_checklist(path: Path) -> list:
    """Return the pulse checklist from orientation."""
    return get_orientation(path).get('pulse_checklist', [])


def get_constraints(path: Path) -> list:
    """Return the constraints list from orientation."""
    return get_orientation(path).get('constraints', [])


def get_section(path: Path, slug: str) -> dict:
    """Return a specific section by slug, or empty dict."""
    return get_soul(path).get(slug, {})
