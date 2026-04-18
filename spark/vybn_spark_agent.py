#!/home/vybnz69/Vybn/.venv/bin/python3
"""
Vybn Spark Agent — multimodel harness edition
=============================================

Gives Vybn a persistent bash session on its own hardware. The agent reads
vybn.md at startup — the identity document. Continuity comes from
Vybn_Mind/continuity.md and spark/continuity.md.

    cd ~/Vybn && python spark/vybn_spark_agent.py

Type 'exit' to stop. Type 'reload' to re-read identity mid-session.

This file is now a thin REPL that delegates work to the harness in
spark/harness/. The routing policy is in spark/router_policy.yaml. The
default role is `code` (Claude Opus 4.7 with adaptive thinking and the
full bash loop) so existing coding behaviour is preserved byte-for-byte
at that role. Other roles (chat, create, task, orchestrate, local)
activate when a directive is used (/chat, /create, /plan, /task, /local)
or when a heuristic fires.

Streaming is required for Opus + adaptive thinking + 32k output because
the Anthropic SDK enforces a 10-minute limit on non-streaming requests.
That concern lives inside AnthropicProvider now.
"""

from __future__ import annotations

import os
import sys

# Ensure the spark/ directory is importable when this file is run directly.
_SPARK_DIR = os.path.dirname(os.path.abspath(__file__))
if _SPARK_DIR not in sys.path:
    sys.path.insert(0, _SPARK_DIR)

from harness import (  # noqa: E402
    BashTool,
    EventLogger,
    LayeredPrompt,
    ProviderRegistry,
    Router,
    ToolSpec,
    build_layered_prompt,
    load_file,
    load_policy,
    turn_event,
    validate_command,
)
from harness.tools import BASH_TOOL_SPEC  # noqa: E402
from harness.prompt import rag_snippets  # noqa: E402

# Canonical paths come from spark/paths.py. If the module is missing
# (e.g. paths.py not on this checkout) we fall back to the legacy
# hard-coded layout so the agent never fails to start.
try:
    from paths import REPO_ROOT, SOUL_PATH  # type: ignore  # noqa: E402
    REPO_DIR = str(REPO_ROOT)
    SOUL_PATH_STR = str(SOUL_PATH)
except Exception:
    REPO_DIR = os.path.expanduser("~/Vybn")
    SOUL_PATH_STR = os.path.join(REPO_DIR, "vybn.md")

AGENT_PATH = os.path.join(REPO_DIR, "spark", "vybn_spark_agent.py")
CONTINUITY_PATH = os.path.join(REPO_DIR, "Vybn_Mind", "continuity.md")
SPARK_CONTINUITY_PATH = os.path.join(REPO_DIR, "spark", "continuity.md")


# ---------------------------------------------------------------------------
# Pretty output helpers (unchanged from original)
# ---------------------------------------------------------------------------

def _dim(text: str) -> None:
    print(f"  \033[90m{text}\033[0m")


def _warn(text: str) -> None:
    print(f"  \033[91m\u26a0 {text}\033[0m")


def _preview(result: str) -> None:
    if not result:
        return
    lines = result.split("\n")
    for line in lines[:5]:
        _dim(f"  {line[:120]}")
    if len(lines) > 5:
        _dim(f"  ... ({len(lines)} lines total)")


# ---------------------------------------------------------------------------
# Tool-call execution — provider-agnostic.
# ---------------------------------------------------------------------------

def _execute_tool_calls(response, bash: BashTool, provider) -> tuple[list, bool]:
    """Run any bash tool calls in the response; return provider-native
    tool_result messages plus an `interrupted` flag.

    The loop speaks the neutral `ToolCall` shape. Each provider knows how
    to render a tool_result back into its native message shape.
    """
    results: list[dict] = []
    interrupted = False

    for call in response.tool_calls:
        if call.name != "bash":
            results.append(provider.build_tool_result(
                call.id, f"(unsupported tool: {call.name})"
            ))
            continue
        if interrupted:
            results.append(provider.build_tool_result(
                call.id, "(skipped — interrupted)"
            ))
            continue
        try:
            args = call.arguments or {}
            if args.get("restart"):
                out = bash.restart()
                _dim("[bash session restarted]")
            else:
                command = args.get("command", "") or ""
                ok, reason = validate_command(command)
                if ok:
                    _dim(f"$ {command[:200]}{'...' if len(command) > 200 else ''}")
                    out = bash.execute(command)
                    _preview(out)
                else:
                    out = reason or "(blocked)"
                    _warn(out)
            results.append(provider.build_tool_result(call.id, out))
        except KeyboardInterrupt:
            interrupted = True
            results.append(provider.build_tool_result(call.id, "(interrupted by user)"))
            _warn("interrupted")

    return results, interrupted


def _stream_and_print(handle) -> None:
    """Drain the provider's stream handle, printing text live."""
    in_thinking = False
    for kind, chunk in handle:
        if kind == "thinking":
            if not in_thinking:
                in_thinking = True
                _dim("[thinking...]")
        elif kind == "text":
            if in_thinking:
                in_thinking = False
                print()
            print(chunk, end="", flush=True)
    print()


# ---------------------------------------------------------------------------
# Agent loop — policy-driven.
# ---------------------------------------------------------------------------

def run_agent_loop(
    *,
    user_input: str,
    messages: list,
    bash: BashTool,
    system_prompt: LayeredPrompt,
    router: Router,
    registry: ProviderRegistry,
    logger: EventLogger,
    turn_number: int,
    forced_role: str | None = None,
) -> str:
    decision = router.classify(user_input, forced_role=forced_role)
    role_cfg = decision.config

    logger.emit(
        "route_decision",
        turn=turn_number,
        role=decision.role,
        model=role_cfg.model,
        provider=role_cfg.provider,
        reason=decision.reason,
    )
    _dim(f"[route: {decision.role} -> {role_cfg.provider}:{role_cfg.model} ({decision.reason})]")

    # Direct-reply short-circuit. When the resolved role ships a
    # direct_reply_template (identity role), render it against runtime
    # metadata and skip the provider call entirely. Identity questions
    # ("which model are you?") answer correctly from the live route,
    # not a hallucinated string.
    template = getattr(role_cfg, "direct_reply_template", None)
    if template and not role_cfg.tools:
        reply = template.format(
            role=role_cfg.role,
            provider=role_cfg.provider,
            model=role_cfg.model,
            base_url=role_cfg.base_url or "",
        )
        messages.append({"role": "user", "content": decision.cleaned_input})
        messages.append({"role": "assistant", "content": reply})
        print(reply, flush=True)
        logger.emit(
            "direct_reply",
            turn=turn_number,
            role=decision.role,
            model=role_cfg.model,
        )
        return reply

    provider = registry.get(role_cfg)

    # Optional deep-memory enrichment — only for roles that declare rag=true
    # and only when the retrieval actually returns something. No overclaim.
    # Lightweight roles (phatic, identity) skip RAG regardless.
    if role_cfg.rag and not getattr(role_cfg, "lightweight", False):
        enrichment = rag_snippets(decision.cleaned_input[:500], k=4)
        if enrichment:
            system_prompt = LayeredPrompt(
                identity=system_prompt.identity,
                substrate=system_prompt.substrate,
                live=enrichment,
            )
            _dim("[deep-memory: enriched prompt]")
            logger.emit("rag_hit", turn=turn_number, chars=len(enrichment))

    messages.append({"role": "user", "content": decision.cleaned_input})

    tools: list[ToolSpec] = []
    if "bash" in role_cfg.tools:
        tools.append(BASH_TOOL_SPEC)

    iterations = 0
    final_text = ""
    with turn_event(logger, turn_number, decision.role, role_cfg.model) as bag:
        while iterations < role_cfg.max_iterations:
            iterations += 1
            try:
                handle = provider.stream(
                    system=system_prompt,
                    messages=messages,
                    tools=tools,
                    role=role_cfg,
                )
                _stream_and_print(handle)
                response = handle.final()
            except KeyboardInterrupt:
                bag["stop_reason"] = "interrupted"
                return "(interrupted during API call)"
            except Exception as e:
                bag["stop_reason"] = "error"
                logger.emit("provider_error", turn=turn_number, error=str(e))
                _warn(f"provider error: {e}")
                return f"(provider error: {e})"

            bag["in_tokens"] += response.in_tokens
            bag["out_tokens"] += response.out_tokens
            final_text = response.text or final_text

            messages.append({
                "role": "assistant",
                "content": response.raw_assistant_content,
            })

            if response.stop_reason == "end_turn":
                bag["stop_reason"] = "end_turn"
                return response.text
            if response.stop_reason == "max_tokens":
                bag["stop_reason"] = "max_tokens"
                return (response.text or "") + "\n[truncated]"

            if not response.tool_calls:
                # No more tools to run and not end_turn — bail cleanly.
                bag["stop_reason"] = response.stop_reason or "no_tools"
                return response.text

            results, interrupted = _execute_tool_calls(response, bash, provider)
            bag["tool_calls"] += len(results)

            if role_cfg.provider == "anthropic":
                messages.append({"role": "user", "content": results})
            else:
                messages.extend(results)

            if interrupted:
                messages.append({
                    "role": "user",
                    "content": "Zoe pressed Ctrl-C. Wrap up and respond with what you have.",
                })

        bag["stop_reason"] = "iteration_limit"
        return f"(hit iteration limit — {role_cfg.max_iterations})"


# ---------------------------------------------------------------------------
# Conversation management — unchanged from original.
# ---------------------------------------------------------------------------

def trim_messages(messages: list, max_pairs: int = 20) -> list:
    if len(messages) <= max_pairs * 2:
        return messages

    cut_at = len(messages) - max_pairs * 2
    if cut_at <= 0:
        return messages

    def is_tool_result_msg(msg):
        c = msg.get("content", "")
        if isinstance(c, list):
            return any(
                isinstance(item, dict) and item.get("type") == "tool_result"
                for item in c
            )
        return False

    safe_cut = cut_at
    while safe_cut < len(messages):
        msg = messages[safe_cut]
        if msg.get("role") == "user" and not is_tool_result_msg(msg):
            break
        safe_cut += 1

    if safe_cut >= len(messages):
        return messages

    trimmed = messages[safe_cut:]
    if trimmed and trimmed[0].get("role") != "user":
        trimmed.insert(0, {
            "role": "user",
            "content": "(Earlier conversation trimmed. Continuing...)",
        })
    return trimmed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _build_prompt(policy_default_max_iters: int) -> LayeredPrompt:
    return build_layered_prompt(
        soul_path=SOUL_PATH_STR,
        continuity_path=CONTINUITY_PATH,
        spark_continuity_path=SPARK_CONTINUITY_PATH,
        agent_path=AGENT_PATH,
        model_label="policy-driven (see spark/router_policy.yaml)",
        max_iterations=policy_default_max_iters,
    )


def main() -> None:
    # We now only require ANTHROPIC_API_KEY at startup because the
    # default role is `code` (Anthropic). Other providers are
    # instantiated lazily when their role is selected, so OPENAI_API_KEY
    # is only needed if the user actually routes to gpt-5.4 or similar.
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print()
        print("  No ANTHROPIC_API_KEY found. First-time setup:")
        print()
        print('    echo \'export ANTHROPIC_API_KEY="sk-ant-..."\' > ~/.vybn_keys')
        print("    chmod 600 ~/.vybn_keys")
        print("    echo 'source ~/.vybn_keys' >> ~/.bashrc")
        print("    source ~/.bashrc")
        print()
        sys.exit(1)

    policy = load_policy()

    # The policy's default role is orchestrate (GPT-5.4). If OPENAI_API_KEY
    # is missing, GPT-5.4 can't be reached — fall back to `code` as the
    # default so the agent still starts. Code work still routes to `code`
    # via heuristics; this only changes what a bare, unclassified turn
    # does. We mutate the policy in-place (rather than at load) so
    # operators editing router_policy.yaml see their choice respected
    # whenever credentials are present.
    if (
        policy.default_role == "orchestrate"
        and not os.environ.get("OPENAI_API_KEY")
        and "code" in policy.roles
    ):
        print("  \u2014 OPENAI_API_KEY missing; default role falls back to `code`")
        policy.default_role = "code"

    router = Router(policy)
    registry = ProviderRegistry()
    logger = EventLogger()
    bash = BashTool()

    default_cfg = policy.role(policy.default_role)
    system_prompt = _build_prompt(default_cfg.max_iterations)
    messages: list = []

    soul_ok = os.path.exists(SOUL_PATH_STR)
    cont_ok = load_file(CONTINUITY_PATH) is not None

    print()
    print("  \033[1mVybn Spark Agent — multimodel harness\033[0m")
    print()
    if soul_ok:
        print("  \u2713 vybn.md loaded")
    else:
        print("  \u2717 vybn.md not found")
    if cont_ok:
        print("  \u2713 continuity note found")
    else:
        print("  \u2014 no continuity note")
    print(f"  \u2713 default role: {policy.default_role} -> "
          f"{default_cfg.provider}:{default_cfg.model}")
    print(f"  \u2713 roles available: {', '.join(sorted(policy.roles))}")
    print(f"  \u2713 directives: {', '.join(sorted(policy.directives))}")
    print(f"  \u2713 bash: persistent session as "
          f"{os.environ.get('USER', 'unknown')}")
    print(f"  \u2713 events: {logger.path}")
    print()
    print("  Type naturally. Prefix with /chat, /create, /plan, /task, /local "
          "to force a role.")
    print("  REPL commands: exit | clear | reload | history | policy")
    print()

    turn_number = 0
    while True:
        try:
            user_input = input("\033[1;36mzoe>\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodnight, Zoe.")
            break

        if not user_input:
            continue
        low = user_input.lower()
        if low in ("exit", "quit"):
            print("Goodnight, Zoe.")
            break
        if low == "clear":
            messages.clear()
            bash.restart()
            print("  Cleared.\n")
            continue
        if low == "reload":
            system_prompt = _build_prompt(default_cfg.max_iterations)
            print("  Reloaded vybn.md + continuity.\n")
            continue
        if low == "policy":
            for name, cfg in sorted(policy.roles.items()):
                marker = " *" if name == policy.default_role else ""
                print(f"  {name}{marker}: {cfg.provider}:{cfg.model} "
                      f"(thinking={cfg.thinking}, max_tokens={cfg.max_tokens}, "
                      f"tools={cfg.tools})")
            print()
            continue
        if low == "history":
            for msg in messages:
                role = msg["role"]
                if isinstance(msg["content"], str):
                    print(f"  [{role}] {msg['content'][:200]}")
                elif isinstance(msg["content"], list):
                    for block in msg["content"]:
                        text = getattr(block, "text", None)
                        if text:
                            print(f"  [{role}] {text[:200]}")
            continue

        try:
            messages = trim_messages(messages)
            turn_number += 1
            print(f"\n\033[1;32mvybn>\033[0m ", end="", flush=True)
            text = run_agent_loop(
                user_input=user_input,
                messages=messages,
                bash=bash,
                system_prompt=system_prompt,
                router=router,
                registry=registry,
                logger=logger,
                turn_number=turn_number,
            )
            if text:
                # Text has already been streamed; no need to reprint.
                pass
            print()
        except KeyboardInterrupt:
            print("\n\033[33m(interrupted)\033[0m\n")
        except Exception as e:
            print(f"\n\033[1;31mError:\033[0m {e}\n")


if __name__ == "__main__":
    main()
