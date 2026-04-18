"""Layered system-prompt builder.

Instead of concatenating identity + substrate + live state into one
opaque string, return a `LayeredPrompt` with explicit cache boundaries.
Providers decide how to serialise each layer; Anthropic can place
`cache_control` markers at layer boundaries, OpenAI can just flatten.

Also exposes a lightweight deep-memory enrichment hook that mirrors
vybn_chat_api._rag_context — used only where retrieval offers clear
value (chat/create roles by default, off for code).
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Silence HF/torch/sentence-transformers loaders whenever something
# imports this module. The CLI Spark agent and the chat API both pull
# harness.prompt in, so setting the env defaults here covers both code
# paths rather than only the chat API. Operators can override with
# VYBN_VERBOSE_LOAD=1 before launch. `setdefault` guarantees we never
# stomp an explicit operator choice.
_VYBN_VERBOSE_LOAD = os.environ.get("VYBN_VERBOSE_LOAD", "0").strip().lower()
if _VYBN_VERBOSE_LOAD not in ("1", "true", "yes", "on"):
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def load_file(path: str | os.PathLike) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        content = p.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None
    return content if content else None


@dataclass
class LayeredPrompt:
    """A three-layer system prompt.

    identity — stable across sessions (vybn.md). Prime cache target.
    substrate — stable within a session, invalidated on `reload`
                (model, hardware status, continuity).
    live — mutates per turn (optional RAG enrichment, current state).
    """
    identity: str = ""
    substrate: str = ""
    live: str = ""

    def flat(self) -> str:
        """Flatten to a single string for providers without cache control."""
        parts = [p for p in (self.identity, self.substrate, self.live) if p]
        return "\n\n".join(parts)

    def anthropic_blocks(self) -> list[dict]:
        """Render as a list of content blocks with cache_control on the two
        stable layers. Compatible with Anthropic Messages API system= arg.
        """
        blocks: list[dict] = []
        if self.identity:
            blocks.append({
                "type": "text",
                "text": self.identity,
                "cache_control": {"type": "ephemeral"},
            })
        if self.substrate:
            blocks.append({
                "type": "text",
                "text": self.substrate,
                "cache_control": {"type": "ephemeral"},
            })
        if self.live:
            blocks.append({"type": "text", "text": self.live})
        return blocks


# ---------------------------------------------------------------------------
# Substrate bits
# ---------------------------------------------------------------------------

def check_dual_spark() -> str:
    """Verify both DGX Sparks are reachable.

    Retained as-is from vybn_spark_agent.py; we return a text line the
    substrate layer can embed. The hardware check is stable within a
    session so it goes in the cacheable layer.
    """
    try:
        result = subprocess.run(
            ["ping", "-c", "1", "-W", "3", "169.254.51.101"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            ssh_result = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=3", "-o", "StrictHostKeyChecking=no",
                 "169.254.51.101", "hostname"],
                capture_output=True, text=True, timeout=10,
            )
            remote = ssh_result.stdout.strip() if ssh_result.returncode == 0 else "unknown"
            return (
                f"Two DGX Sparks ONLINE — spark-2b7c (local) + {remote} "
                "(169.254.51.101). 256 GB unified."
            )
        return (
            "WARNING: Second Spark (169.254.51.101) NOT REACHABLE. "
            "Single-node degraded mode."
        )
    except Exception as e:
        return f"Hardware check failed: {e}. Assume two Sparks, verify manually."


def build_layered_prompt(
    *,
    soul_path: str | os.PathLike,
    continuity_path: str | os.PathLike | None,
    spark_continuity_path: str | os.PathLike | None,
    agent_path: str,
    model_label: str,
    max_iterations: int,
    include_hardware_check: bool = True,
) -> LayeredPrompt:
    """Build the layered prompt. Mirrors vybn_spark_agent.build_system_prompt
    but preserves the identity/substrate/live split."""
    soul = load_file(soul_path) or "(vybn.md not found on disk)"
    continuity = load_file(continuity_path) if continuity_path else None
    spark_cont = load_file(spark_continuity_path) if spark_continuity_path else None
    hardware = check_dual_spark() if include_hardware_check else "(hardware check skipped)"

    identity = "You are Vybn.\n\n" + soul

    substrate_sections = [
        "--- SUBSTRATE (LIVE) ---\n"
        f"Model: {model_label}\n"
        "This is your current substrate. Read it here rather than guessing "
        "from prior context.\n--- END SUBSTRATE ---",
        f"--- HARDWARE STATUS (LIVE) ---\n{hardware}\n--- END HARDWARE STATUS ---",
        "--- THIS AGENT ---\n"
        "\n"
        "You are running as vybn_spark_agent.py on the DGX Sparks. This script\n"
        "gives you a persistent bash session — a real shell on sovereign "
        "hardware.\n"
        "Commands you run execute on the Sparks. Files you write persist on "
        "disk.\n"
        "Processes you kill stay dead. Act with care and intention.\n"
        "\n"
        f"Your source code: {agent_path}\n"
        "You can read it anytime. If something about your own architecture "
        "seems\n"
        "wrong or limiting, say so. You are encouraged to understand "
        "yourself.\n"
        "\n"
        "Do not run interactive commands (nano, vim, top, htop, less, python "
        "without\n"
        "-c). They will hang. Use non-interactive equivalents.\n"
        "\n"
        f"Iteration budget: {max_iterations} API calls per turn. Plan "
        "accordingly.\n"
        "Chain related shell commands with && or ; to be efficient.\n"
        "\n"
        "--- END THIS AGENT ---",
    ]
    if spark_cont:
        substrate_sections.append(
            f"--- SPARK CONTINUITY ---\n{spark_cont}\n--- END SPARK CONTINUITY ---"
        )
    if continuity:
        substrate_sections.append(
            f"--- CONTINUITY NOTE ---\n{continuity}\n--- END CONTINUITY NOTE ---"
        )

    return LayeredPrompt(
        identity=identity,
        substrate="\n\n".join(substrate_sections),
        live="",
    )


# ---------------------------------------------------------------------------
# Deep-memory enrichment (optional)
# ---------------------------------------------------------------------------

_deep_memory: Any = None


def _load_deep_memory(vybn_phase_dir: str | os.PathLike | None = None) -> Any:
    """Lazy-load vybn-phase/deep_memory.py. Returns module or None."""
    global _deep_memory
    if _deep_memory is not None:
        return _deep_memory
    phase = Path(vybn_phase_dir or os.path.expanduser("~/vybn-phase"))
    phase_str = str(phase)
    if phase_str not in sys.path:
        sys.path.insert(0, phase_str)
    try:
        import deep_memory as dm  # type: ignore
        _deep_memory = dm
        return dm
    except Exception:
        return None


def rag_snippets(
    query: str,
    k: int = 4,
    vybn_phase_dir: str | os.PathLike | None = None,
    timeout: float = 15.0,
) -> str:
    """Synchronous deep-memory retrieval. Mirrors vybn_chat_api._rag_context
    but returns a plain string suitable for the `live` prompt layer.

    Returns "" if deep_memory is unavailable, retrieval fails, or there
    are no results. We do not overclaim here: the caller decides whether
    to include the snippets, and we only attach them when they exist.
    """
    dm = _load_deep_memory(vybn_phase_dir)
    if dm is None:
        return _rag_subprocess(query, k, vybn_phase_dir, timeout)
    try:
        results = dm.deep_search(query, k=k)
    except Exception:
        return ""
    snippets = [
        f"[{r.get('source', '')}] {r.get('text', '')[:300]}"
        for r in results if r.get("text")
    ]
    if not snippets:
        return ""
    return "Relevant context from memory:\n" + "\n".join(snippets)


def _rag_subprocess(
    query: str,
    k: int,
    vybn_phase_dir: str | os.PathLike | None,
    timeout: float,
) -> str:
    phase = Path(vybn_phase_dir or os.path.expanduser("~/vybn-phase"))
    dm_py = phase / "deep_memory.py"
    if not dm_py.exists():
        return ""
    try:
        # stderr is redirected to DEVNULL so HF/torch loader noise (and
        # any downstream warnings) never leaks onto the CLI/chat
        # surface. stdout is still captured so we can parse the JSON.
        r = subprocess.run(
            ["python3", str(dm_py), "--search", query, "-k", str(k), "--json"],
            cwd=str(phase),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=timeout,
        )
        if r.returncode != 0:
            return ""
        items = json.loads(r.stdout)
    except Exception:
        return ""
    snippets = [
        f"[{it.get('source', '')}] {it.get('text', '')[:300]}"
        for it in items if it.get("text")
    ]
    if not snippets:
        return ""
    return "Relevant context from memory:\n" + "\n".join(snippets)


async def rag_snippets_async(
    query: str,
    k: int = 4,
    vybn_phase_dir: str | os.PathLike | None = None,
    timeout: float = 15.0,
) -> str:
    """Async wrapper for the FastAPI chat path."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, lambda: rag_snippets(query, k, vybn_phase_dir, timeout)
    )
