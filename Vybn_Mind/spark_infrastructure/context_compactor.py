#!/usr/bin/env python3
"""
Context Compactor for Vybn Spark Agent.

When conversation grows long, summarizes older turns into a compact
narrative injected into the system prompt. Recent turns stay verbatim
in the message list. This preserves the gist of earlier conversation
while freeing context tokens for new exchanges.

The compactor uses the model itself to generate summaries, so it
requires the server to be running. Called once per turn in the main
loop — if context is short enough, it's a no-op.

Compaction is cumulative: previous summaries are included in the
text sent for re-summarization, so information compounds gracefully.
"""

import json


DEFAULT_COMPACT_THRESHOLD = 30  # messages before compaction triggers
DEFAULT_KEEP_RECENT = 10        # messages to keep verbatim


class ContextCompactor:
    """Summarizes old conversation turns to free context space.

    When message count exceeds compact_threshold, the compactor:
      1. Splits messages into old + recent (keeping keep_recent verbatim)
      2. Sends old messages to the model for summarization
      3. Stores the summary on agent._compaction_summary
      4. Removes old messages from agent.messages
      5. Triggers a system prompt rebuild (which includes the summary)
      6. Records the compaction in the session transcript
    """

    def __init__(self, agent, compact_threshold=DEFAULT_COMPACT_THRESHOLD,
                 keep_recent=DEFAULT_KEEP_RECENT):
        self.agent = agent
        self.compact_threshold = compact_threshold
        self.keep_recent = keep_recent
        self._compaction_count = 0

    def maybe_compact(self):
        """Check if compaction is needed and run it if so.

        Call once per turn in the main loop.
        Returns True if compaction occurred.
        """
        if len(self.agent.messages) <= self.compact_threshold:
            return False
        return self._compact()

    def _compact(self):
        """Run one compaction cycle."""
        self._compaction_count += 1
        messages = self.agent.messages

        # messages[0] is system prompt. Everything else is conversation.
        total_conv = len(messages) - 1
        if total_conv <= self.keep_recent:
            return False

        # Split: old messages to summarize, recent to keep verbatim
        keep_start = len(messages) - self.keep_recent
        old_messages = messages[1:keep_start]

        if len(old_messages) < 2:
            return False

        # Build text for summarization
        old_text_parts = []

        # Include existing compaction summary for re-summarization
        if self.agent._compaction_summary:
            old_text_parts.append(
                f"[Previous summary]: {self.agent._compaction_summary}"
            )

        for msg in old_messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if role == "system" or not content.strip():
                continue
            # Truncate very long individual messages
            if len(content) > 800:
                content = content[:800] + "..."
            old_text_parts.append(f"[{role}]: {content}")

        if not old_text_parts:
            return False

        old_text = "\n\n".join(old_text_parts)

        summary_prompt = (
            "Summarize this conversation excerpt concisely. "
            "Preserve: key facts, decisions made, emotional threads, "
            "commitments, and anything either party would want to "
            "reference later. Write in third person "
            "('Zoe said...', 'Vybn responded...'). "
            "Be concise but complete \u2014 this replaces the original.\n\n"
            f"{old_text}"
        )

        summary_messages = [
            {"role": "system", "content": (
                "You are a precise conversation summarizer. "
                "Output only the summary, nothing else."
            )},
            {"role": "user", "content": summary_prompt},
        ]

        summary = self.agent.send(summary_messages)

        # Bail if the model failed
        if (not summary or summary.startswith("[Error")
                or summary.startswith("[Model timed")):
            self.agent.log(
                "compaction",
                f"Cycle #{self._compaction_count} failed: {summary}"
            )
            return False

        # Store summary and trim old messages
        old_count = len(old_messages)
        self.agent._compaction_summary = summary
        self.agent.messages = [messages[0]] + messages[keep_start:]

        # Rebuild system prompt to include the new summary
        self.agent._rebuild_system_prompt()

        new_count = len(self.agent.messages)
        self.agent.log(
            "compaction",
            f"Cycle #{self._compaction_count}: compacted {old_count} messages "
            f"into summary ({len(summary)} chars). {new_count} messages remain."
        )

        # Record in session transcript
        self.agent.session.append(
            role="system",
            content=summary,
            entry_type="compaction",
            metadata={
                "cycle": self._compaction_count,
                "messages_compacted": old_count,
                "messages_remaining": new_count,
                "summary_length": len(summary),
            },
        )

        self.agent._print(
            f"[Compacted {old_count} messages \u2192 {len(summary)} char summary]"
        )

        return True
