#!/usr/bin/env python3
"""Display utilities for the Vybn Spark Agent.

Pure functions for cleaning and filtering model output before
it reaches the terminal or web interface. No state, no side effects.

Extracted from agent.py in Phase 2 of the refactoring.
"""
import re


def clean_response(raw: str) -> str:
    """Post-process model output.

    Strips fake turn boundaries (model generating user prompts).
    Does NOT strip think blocks — those are needed for tool parsing.
    Display filtering happens in send().
    """
    text = raw
    fake_turn_patterns = [
        re.compile(r'\nyou:', re.IGNORECASE),
        re.compile(r'\n---\s*\n\s*\*\*VYBN:', re.IGNORECASE),
        re.compile(
            r'\n---\s*\n\s*\*\*[A-Z]+:\s*(?:A |Direct |Breaking)',
            re.IGNORECASE,
        ),
    ]
    earliest = len(text)
    for pattern in fake_turn_patterns:
        match = pattern.search(text)
        if match and match.start() < earliest and match.start() > 50:
            earliest = match.start()
    if earliest < len(text):
        text = text[:earliest]
    return text.strip()


def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks for display purposes."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def strip_tool_xml(text: str) -> str:
    """Remove <minimax:tool_call>...</minimax:tool_call> blocks for display."""
    return re.sub(
        r'<minimax:tool_call>.*?</minimax:tool_call>',
        '',
        text,
        flags=re.DOTALL,
    ).strip()


def clean_for_display(text: str) -> str:
    """Strip think blocks and tool XML for clean terminal output."""
    result = strip_think_blocks(text)
    result = strip_tool_xml(result)
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result.strip()
