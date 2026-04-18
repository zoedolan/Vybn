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

import re
from dataclasses import dataclass

from .policy import Policy, RoleConfig


# Round 5: @alias pin. Matches @<word> at the very start of the input,
# followed by whitespace or EOL. The <word> is looked up in policy.model_aliases.
_ALIAS_RE = re.compile(r"^\s*(@[\w.]+)(\s|$)")


@dataclass
class RouteDecision:
    role: str
    config: RoleConfig
    cleaned_input: str
    reason: str
    forced: bool = False
    # Round 5: if the user prefixed their turn with an @alias, this holds
    # the resolved model name. The REPL loop uses dataclasses.replace to
    # swap the active RoleConfig's model (and infers provider) before the
    # provider call. Role determination is unchanged; only the model pin.
    model_override: str | None = None
    alias_used: str | None = None


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

        # 0. @alias model pin — strip before directive/heuristic so the
        #    rest of the text still routes normally. "@sonnet hey" pins
        #    the model to claude-sonnet-4-6 and then routes on "hey"
        #    (phatic). "@opus47 fix this bug" pins opus-4-7 and routes
        #    on "fix this bug" (code).
        model_override: str | None = None
        alias_used: str | None = None
        aliases = getattr(self.policy, "model_aliases", None) or {}
        if aliases:
            m = _ALIAS_RE.match(text)
            if m:
                alias_key = m.group(1).lower()
                if alias_key in aliases:
                    model_override = aliases[alias_key]
                    alias_used = alias_key
                    text = text[m.end():].lstrip()
                    if not text:
                        # Bare @alias with no payload — keep the alias itself
                        # as the cleaned input so downstream heuristics match
                        # on something. Fall back to a greeting-shaped empty.
                        text = "hi"

        # 1. Directive
        for prefix, role in self.policy.directives.items():
            if text.startswith(prefix + " ") or text == prefix:
                cleaned = text[len(prefix):].lstrip()
                if role in self.policy.roles:
                    decision = RouteDecision(
                        role=role,
                        config=self.policy.role(role),
                        cleaned_input=cleaned or text,
                        reason=f"directive={prefix}",
                    )
                    if model_override:
                        decision.model_override = model_override
                        decision.alias_used = alias_used
                        decision.reason = f"{decision.reason}+alias={alias_used}"
                    return decision

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
                    decision = RouteDecision(
                        role=role_name,
                        config=self.policy.role(role_name),
                        cleaned_input=text,
                        reason=f"heuristic={rx.pattern}",
                    )
                    if model_override:
                        decision.model_override = model_override
                        decision.alias_used = alias_used
                        decision.reason = f"{decision.reason}+alias={alias_used}"
                    return decision

        # 3. Default
        default = self.policy.default_role
        decision = RouteDecision(
            role=default,
            config=self.policy.role(default),
            cleaned_input=text,
            reason="default",
        )
        if model_override:
            decision.model_override = model_override
            decision.alias_used = alias_used
            decision.reason = f"{decision.reason}+alias={alias_used}"
        return decision
