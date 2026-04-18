"""Router — classifies turns into roles.

Three tiers evaluated in order:
  1. Explicit directive: user typed /code, /chat, /plan, /create, /task, /local
  2. Heuristics: regex patterns from policy
  3. Default role: falls through to policy.default_role

The LLM classifier tier described in the review is deliberately not
wired in here. This agent runs unattended and we want routing to be
deterministic until a live session demonstrates the tail is wide
enough to justify a classifier call. Adding it later is one method.
"""

from __future__ import annotations

from dataclasses import dataclass

from .policy import Policy, RoleConfig


@dataclass
class RouteDecision:
    role: str
    config: RoleConfig
    cleaned_input: str
    reason: str
    forced: bool = False


class Router:
    def __init__(self, policy: Policy) -> None:
        self.policy = policy

    def classify(self, user_input: str, forced_role: str | None = None) -> RouteDecision:
        if forced_role and forced_role in self.policy.roles:
            return RouteDecision(
                role=forced_role,
                config=self.policy.role(forced_role),
                cleaned_input=user_input,
                reason=f"forced={forced_role}",
                forced=True,
            )

        text = user_input.strip()

        # 1. Directive
        for prefix, role in self.policy.directives.items():
            if text.startswith(prefix + " ") or text == prefix:
                cleaned = text[len(prefix):].lstrip()
                if role in self.policy.roles:
                    return RouteDecision(
                        role=role,
                        config=self.policy.role(role),
                        cleaned_input=cleaned or text,
                        reason=f"directive={prefix}",
                    )

        # 2. Heuristics
        for role_name, patterns in self.policy.heuristics.items():
            if role_name not in self.policy.roles:
                continue
            for rx in patterns:
                if rx.search(text):
                    return RouteDecision(
                        role=role_name,
                        config=self.policy.role(role_name),
                        cleaned_input=text,
                        reason=f"heuristic={rx.pattern}",
                    )

        # 3. Default
        default = self.policy.default_role
        return RouteDecision(
            role=default,
            config=self.policy.role(default),
            cleaned_input=text,
            reason="default",
        )
