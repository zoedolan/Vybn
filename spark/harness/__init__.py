"""Vybn multimodel harness.

Factors the Spark agent into (router, policy, provider, tools, logger,
prompt) so one file becomes the place role routing happens rather than
the place one model is called. See spark_harness_review.md for rationale.

Public surface:
    from spark.harness import (
        Router, Policy, EventLogger,
        Provider, AnthropicProvider, OpenAIProvider,
        ToolSpec, BashTool, absorb_gate, DANGEROUS_PATTERNS,
        build_layered_prompt, load_file,
    )
"""

from .constants import DANGEROUS_PATTERNS, TRACKED_REPOS
from .tools import (
    ToolSpec,
    BashTool,
    absorb_gate,
    validate_command,
)
from .prompt import (
    LayeredPrompt,
    build_layered_prompt,
    load_file,
)
from .policy import Policy, RoleConfig, load_policy
from .events import EventLogger, turn_event
from .providers import (
    Provider,
    AnthropicProvider,
    OpenAIProvider,
    ProviderRegistry,
    StreamHandle,
    NormalizedResponse,
    ToolCall,
)
from .router import Router, RouteDecision

__all__ = [
    "DANGEROUS_PATTERNS",
    "TRACKED_REPOS",
    "ToolSpec",
    "BashTool",
    "absorb_gate",
    "validate_command",
    "LayeredPrompt",
    "build_layered_prompt",
    "load_file",
    "Policy",
    "RoleConfig",
    "load_policy",
    "EventLogger",
    "turn_event",
    "Provider",
    "AnthropicProvider",
    "OpenAIProvider",
    "ProviderRegistry",
    "StreamHandle",
    "NormalizedResponse",
    "ToolCall",
    "Router",
    "RouteDecision",
]
