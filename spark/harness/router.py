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

        # 2. Heuristics -- evaluate in an EXPLICIT priority order so
        # identity beats phatic beats chat beats task beats code.
        # Dict insertion order worked by accident; a future YAML
        # reorder would silently break routing. Pin it here.
        _HEURISTIC_PRIORITY = (
            "task",        # confirmations ("ok", "proceed") -- earliest
            "identity",    # "which model are you?" before greetings
            "phatic",      # bare greetings/closings
            "code",        # grounded code work
            "create",      # brainstorm/sketch
            "chat",        # how-are-you style
        )
        heur = self.policy.heuristics
        ranked = [r for r in _HEURISTIC_PRIORITY if r in heur]
        ranked += [r for r in heur if r not in ranked]
        for role_name in ranked:
            if role_name not in self.policy.roles:
                continue
            for rx in heur[role_name]:
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
