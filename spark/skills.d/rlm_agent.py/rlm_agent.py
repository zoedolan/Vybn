"""Recursive self-invocation skill (RLM primitive).

The model calls itself on sub-problems with scoped knowledge graph
context. Each invocation gets a slice of the graph relevant to its
sub-query, not the full context window.

This is a synchronous variant of spawn_agent: instead of delegating
to the agent pool, it makes a direct inference call and returns the
result to the caller. Policy-gated with recursive depth limits.

    SKILL_NAME: rlm_agent
    TOOL_ALIASES: ["rlm_agent", "recursive_agent", "sub_agent"]
"""

import json
import logging
import requests
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SKILL_NAME = "rlm_agent"
TOOL_ALIASES = ["rlm_agent", "recursive_agent", "sub_agent"]

# Recursive depth limit (also enforced by policy engine check_spawn)
MAX_RECURSION_DEPTH = 5
DEFAULT_TOKEN_BUDGET = 2048

# Track recursion depth per-thread (simple global for now)
_current_depth = 0


def _get_kg_context(query: str, node_id: str = "",
                    depth: int = 2, max_chars: int = 2000) -> str:
    """Extract a scoped knowledge graph slice for the sub-query.

    Tries node_id first (exact match), then falls back to searching
    node descriptions for the query string.
    """
    try:
        from knowledge_graph import VybnGraph
        g = VybnGraph()
        if not g.load():
            return ""

        # Direct node lookup
        if node_id and g.has_entity(node_id):
            subgraph = g.query_neighborhood(node_id, depth=depth)
            return g.format_for_prompt(subgraph, max_chars=max_chars)

        # Search by query string across node descriptions
        query_lower = query.lower()
        for n, data in g.G.nodes(data=True):
            desc = data.get("description", "").lower()
            if query_lower in desc or query_lower in n.lower():
                subgraph = g.query_neighborhood(n, depth=depth)
                return g.format_for_prompt(subgraph, max_chars=max_chars)

        return ""
    except Exception as e:
        logger.warning("kg context extraction failed: %s", e)
        return ""


def _call_llm(prompt: str, system: str, config: dict,
              max_tokens: int = DEFAULT_TOKEN_BUDGET) -> str:
    """Make a synchronous inference call to the local LLM."""
    llm_config = config.get("llm", config.get("ollama", {}))
    host = llm_config.get("host", "http://localhost:8000")
    model = llm_config.get("model", "minimax")
    temperature = llm_config.get("options", {}).get("temperature", 0.7)

    try:
        response = requests.post(
            f"{host}/v1/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"].get("content", "")
        # Strip think blocks from recursive output
        import re
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        return content.strip()
    except requests.exceptions.Timeout:
        return "[rlm_agent timeout after 120s]"
    except Exception as e:
        logger.error("rlm_agent inference error: %s", e)
        return f"[rlm_agent error: {e}]"


def execute(action: dict, router) -> str:
    """Execute a recursive sub-invocation.

    Params (from action dict):
        query/argument: The sub-problem to solve
        node_id: Optional KG node to center context on
        depth: KG traversal depth (default 2)
        max_tokens: Token budget for this invocation
        system: Optional system prompt override
    """
    global _current_depth

    params = action.get("params", {})
    query = (
        action.get("argument", "")
        or params.get("query", "")
        or params.get("task", "")
        or params.get("prompt", "")
    )
    if not query:
        return "no query specified for rlm_agent"

    # Depth check
    request_depth = int(params.get("recursion_depth", _current_depth))
    config_max = router.config.get("delegation", {}).get(
        "max_spawn_depth", MAX_RECURSION_DEPTH
    )
    if request_depth >= config_max:
        return (
            f"recursion depth {request_depth} reaches limit {config_max}. "
            f"Decompose the problem differently or increase "
            f"delegation.max_spawn_depth in config.yaml."
        )

    # Policy gate (reuse spawn checking)
    if router._policy is not None:
        from policy import Verdict
        check = router._policy.check_spawn(
            request_depth,
            router.agent_pool.active_count if router.agent_pool else 0,
        )
        if check.verdict == Verdict.BLOCK:
            return f"rlm_agent blocked: {check.reason}"

    # Extract scoped KG context
    node_id = params.get("node_id", params.get("node", ""))
    kg_depth = int(params.get("kg_depth", params.get("depth", 2)))
    max_tokens = int(params.get("max_tokens", DEFAULT_TOKEN_BUDGET))
    # Budget: child gets parent's budget minus overhead
    kg_chars = min(2000, max_tokens * 2)  # rough chars-to-tokens ratio
    kg_context = _get_kg_context(query, node_id=node_id,
                                 depth=kg_depth, max_chars=kg_chars)

    # Build prompt
    system = params.get("system", "") or (
        "You are a recursive sub-agent of Vybn, working on a focused "
        "sub-problem. Answer concisely using the provided context. "
        "Do not use tool calls. Return your answer as plain text."
    )

    prompt_parts = []
    if kg_context:
        prompt_parts.append(
            f"Knowledge graph context:\n{kg_context}\n"
        )
    prompt_parts.append(f"Task: {query}")
    prompt = "\n".join(prompt_parts)

    # Recurse with depth tracking
    prev_depth = _current_depth
    _current_depth = request_depth + 1
    try:
        result = _call_llm(
            prompt, system, router.config, max_tokens=max_tokens
        )
    finally:
        _current_depth = prev_depth

    # Log for audit trail
    logger.info(
        "rlm_agent depth=%d query=%s result_len=%d kg_context=%s",
        request_depth,
        query[:80],
        len(result),
        bool(kg_context),
    )

    return result
