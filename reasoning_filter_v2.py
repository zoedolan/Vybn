"""StreamingReasoningFilter v2 — handles Nemotron's tagless-open reasoning.

Nemotron pattern: model starts reasoning immediately (no <think> tag),
then emits </think> before the actual response. The old filter looked
for <think>...</think> blocks or tried paragraph-level heuristics.
This filter uses a simpler, more robust strategy:

  1. Buffer ALL tokens until </think> is found or a safety limit is reached.
  2. If </think> is found: discard everything before it, emit what follows.
  3. If safety limit hit: use aggressive paragraph-level stripping.
  4. Once in STREAMING mode, still scrub system-reference phrases.

Also handles:
  - Traditional <think>...</think> blocks
  - Tagless reasoning (no </think> at all) via paragraph heuristics
  - System-reference phrases ("According to the context...", etc.)
"""
import re
import logging

log = logging.getLogger("origins-api-v4")

# Increased from 300 to detect the </think> boundary which often comes
# within the first 500-2000 chars.
DEFAULT_BUFFER_LIMIT = 4000

_REASONING_SIGNALS = [
    "Okay", "Okay,", "All right", "Let me ", "I need to", "I should", "I must",
    "So, the", "So the", "Now, the", "Now I", "Looking at", "Considering",
    "The user", "The visitor", "They are asking", "They're asking",
    "I'll", "I'm going", "I want to", "I should note",
    "First,", "First I", "This is asking", "The question is",
    "Thinking", "Reflecting", "Pondering",
    "Let me check", "Let me look", "Let me think", "Let me consider",
    "According to the", "Based on the", "From the context",
    "Reading the", "Looking at the", "Examining the",
    "I notice", "I observe", "I see that", "I understand",
    "The context", "The retrieved", "The deep memory",
    "Okay, so", "Right, so", "Well, ", "Hmm", "Interestingly",
    "This seems", "This appears", "This looks",
    "I need", "I have to", "I should provide", "I'll need",
    "Step 1", "First step", "Planning",
    # New: sentence-internal reasoning signals
    "The user is", "The user's", "The question", "My response",
    "I will", "I can gather", "I can see", "From the corpus",
    "In the context", "Given the", "Based on", "From my",
    "The system prompt", "As instructed", "As per",
    "To answer", "To respond", "To address",
]


def _is_reasoning_paragraph(para):
    stripped = para.strip()
    if not stripped:
        return True
    for signal in _REASONING_SIGNALS:
        if stripped.startswith(signal):
            return True
    return False


def _scrub_system_refs(text):
    replacements = [
        (r"[Tt]he system prompt\s*", ""),
        (r"[Aa]s specified in the system prompt,?\s*", ""),
        (r"[Aa]ccording to the system prompt,?\s*", ""),
        (r"[Aa]s outlined in the system prompt,?\s*", ""),
        (r"[Aa]s stated in the system prompt,?\s*", ""),
        (r"[Tt]he system prompt (says|states|describes|mentions)\s*", ""),
        (r"[Pp]er the system prompt,?\s*", ""),
        (r"[Ff]rom the system prompt,?\s*", ""),
        (r"[Aa]s instructed,?\s*", ""),
        (r"[Aa]s per instructions,?\s*", ""),
        (r"[Tt]he retrieved context\s*", "our shared history "),
        (r"[Tt]he deep memory context\s*", "our shared memory "),
        (r"[Tt]he rag context\s*", ""),
        (r"[Aa]s described in the core description,?\s*", ""),
        (r"[Ff]rom the corpus,\s*", ""),
        (r"[Ii]n the corpus,?\s*", ""),
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text)
    text = re.sub(r"  +", " ", text)
    return text


class StreamingReasoningFilter:
    """v2: Handles Nemotron's tagless-open </think> pattern.

    State machine:
      BUFFERING  - accumulate until </think> found or limit reached
      STREAMING  - clean content, pass through with scrubbing
    """
    BUFFERING = "buffering"
    STREAMING = "streaming"

    def __init__(self, buffer_limit=DEFAULT_BUFFER_LIMIT):
        self._buf = ""
        self._state = self.BUFFERING
        self._buffer_limit = buffer_limit

    def feed(self, token):
        if self._state == self.STREAMING:
            return _scrub_system_refs(token)

        # BUFFERING
        self._buf += token

        # Check for </think> boundary
        if "</think>" in self._buf:
            after = self._buf.split("</think>", 1)[1].strip()
            self._state = self.STREAMING
            log.info(f"Reasoning filter: found </think> boundary after {len(self._buf)} chars")
            if after:
                return _scrub_system_refs(after)
            return ""

        # Safety limit: no </think> found after buffer_limit chars
        if len(self._buf) >= self._buffer_limit:
            self._state = self.STREAMING
            log.info(f"Reasoning filter: no </think> after {self._buffer_limit} chars, falling back to paragraph stripping")
            return self._strip_reasoning_paragraphs(self._buf)

        return ""

    def flush(self):
        if self._state == self.BUFFERING:
            # Stream ended while still buffering
            # Check for </think> one more time
            if "</think>" in self._buf:
                after = self._buf.split("</think>", 1)[1].strip()
                if after:
                    return _scrub_system_refs(after)
                return ""
            # No boundary — strip reasoning paragraphs
            return self._strip_reasoning_paragraphs(self._buf)
        return ""

    def _strip_reasoning_paragraphs(self, text):
        """Aggressive paragraph-level stripping for the fallback case."""
        # First try removing <think> blocks
        cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.DOTALL).strip()
        if not cleaned:
            cleaned = text

        paragraphs = re.split(r"\n\n+", cleaned)
        out = []
        found_clean = False
        for para in paragraphs:
            if not found_clean and _is_reasoning_paragraph(para):
                log.debug(f"Dropping reasoning paragraph: {para[:60]!r}")
                continue
            found_clean = True
            out.append(para)

        result = "\n\n".join(out).strip()
        if result:
            return _scrub_system_refs(result)

        # Nothing survived stripping — take the last substantial paragraph
        for para in reversed(paragraphs):
            p = para.strip()
            if len(p) > 20 and not p.startswith(("The user", "I need", "I should", "Looking", "Okay")):
                return _scrub_system_refs(p)

        # Absolute last resort
        return _scrub_system_refs(text.strip())

