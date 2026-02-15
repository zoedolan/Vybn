#!/usr/bin/env python3
"""
Pre-compaction memory flusher for Vybn Spark Agent.

Before context compaction discards old messages, this module scans
them for information worth persisting to archival memory. Catches
details that per-turn reflection might have missed — subtle facts,
background context, evolving emotional threads.

Also usable at session end for a final sweep of unarchived content.
"""

import json
import re


class MemoryFlusher:
    """Extracts and archives noteworthy content from conversation messages.

    Asks the model to identify archival-worthy items from a batch of
    messages, then stores each one in archival memory with source
    metadata. Used as a pre-step before context compaction (so nothing
    important is lost when old messages are discarded) and at session
    end (to catch anything from the final turns).
    """

    EXTRACTION_PROMPT = (
        "Review this conversation excerpt and extract information worth "
        "remembering long-term. Return a JSON array of strings, where each "
        "string is a single self-contained memory.\n\n"
        "Focus on:\n"
        "- Decisions made or commitments given\n"
        "- Facts shared about either person\n"
        "- Emotional shifts or states expressed\n"
        "- Insights or realizations\n"
        "- Plans discussed or goals set\n"
        "- Preferences or opinions revealed\n\n"
        "Skip: greetings, small talk, tool-use mechanics.\n"
        "If nothing is worth archiving, return [].\n"
        "Output ONLY the JSON array, nothing else.\n\n"
    )

    def flush(self, agent, messages, source="flush"):
        """Extract and archive key information from messages.

        Args:
            agent: SparkAgent instance (for model access and archival)
            messages: list of message dicts to scan
            source: archival source tag for stored memories

        Returns:
            Number of items archived.
        """
        if not agent.archive.available:
            return 0

        if not messages:
            return 0

        # Build text from messages
        text_parts = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if role == "system" or not content.strip():
                continue
            if len(content) > 1000:
                content = content[:1000] + "..."
            text_parts.append(f"[{role}]: {content}")

        if not text_parts:
            return 0

        conversation_text = "\n\n".join(text_parts)

        # Ask the model to extract archival items
        extraction_messages = [
            {"role": "system", "content": (
                "You extract key information from conversations. "
                "Output only valid JSON arrays of strings."
            )},
            {"role": "user", "content": self.EXTRACTION_PROMPT + conversation_text},
        ]

        response = agent.send(extraction_messages)

        items = self._parse_items(response)

        if not items:
            agent.log("flush", f"No items extracted (source: {source})")
            return 0

        # Archive each item
        state = agent.memory.read_state()
        session = str(state.get("session_count", "?"))

        archived = 0
        for item in items:
            if not isinstance(item, str) or not item.strip():
                continue
            agent.archive.store(
                item.strip(),
                source=source,
                metadata={"session": session},
            )
            archived += 1

        agent.log("flush", f"Archived {archived} items (source: {source})")
        return archived

    def _parse_items(self, response):
        """Parse JSON array from model response.

        Tries direct parse first, then attempts to extract a JSON
        array from within the response text (the model sometimes
        wraps JSON in markdown or adds explanation).
        """
        if (not response or response.startswith("[Error")
                or response.startswith("[Model timed")):
            return []

        # Try direct parse
        try:
            items = json.loads(response)
            if isinstance(items, list):
                return items
        except json.JSONDecodeError:
            pass

        # Try to extract JSON array from response
        match = re.search(r'\[.*\]', response, re.DOTALL)
        if match:
            try:
                items = json.loads(match.group())
                if isinstance(items, list):
                    return items
            except json.JSONDecodeError:
                pass

        return []
