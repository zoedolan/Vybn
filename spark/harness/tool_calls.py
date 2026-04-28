"""Provider-agnostic tool-call execution for the Vybn REPL."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from .providers import execute_readonly, is_parallel_safe, validate_command

Printer = Callable[[str], None]



def default_introspect(spark_dir: str) -> str:
    """Live route/walk/deep-memory snapshot for the introspect tool."""
    import json
    import urllib.request
    from pathlib import Path

    lines: list[str] = []
    events_path = Path(spark_dir) / "agent_events.jsonl"
    try:
        events = [json.loads(l) for l in events_path.read_text().splitlines() if l.strip()]
        routes = [e for e in events if e.get("event") == "route_decision"][-5:]
        lines.append("=== last 5 route decisions ===")
        for r in routes:
            lines.append(f"  turn {r.get('turn')} -> {r.get('role')} via {r.get('model')} ({r.get('reason')})")
    except Exception as e:  # noqa: BLE001
        lines.append(f"  [events unavailable: {e}]")

    for name, url in (
        ("walk", "http://127.0.0.1:8101/health"),
        ("deep_memory", "http://127.0.0.1:8100/health"),
    ):
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                health = json.loads(resp.read())
            if name == "walk":
                lines.append(f"=== walk === step={health.get('walk_step')} alpha={health.get('walk_alpha','?')} chunks={health.get('chunks')}")
            else:
                lines.append(f"=== deep_memory === chunks={health.get('chunks')} walk_step={health.get('walk_step')}")
        except Exception as e:  # noqa: BLE001
            lines.append(f"  [{name} unavailable: {e}]")
    return "\n".join(lines)


def execute_tool_calls(
    response: Any,
    bash: Any,
    provider: Any,
    *,
    delegate_cb: Callable[[str, str], str] | None = None,
    dim: Printer = lambda text: None,
    warn: Printer = lambda text: None,
    preview: Printer = lambda text: None,
    introspect: Callable[[], str] | None = None,
) -> tuple[list, bool]:
    """Run provider-neutral ToolCall objects and return native tool results."""
    results: list[dict] = []
    interrupted = False

    bash_calls = [c for c in response.tool_calls if c.name == "bash"]
    parallel_candidates: list[tuple[Any, str]] = []
    if len(bash_calls) >= 2:
        ok = True
        for call in bash_calls:
            args = call.arguments or {}
            if args.get("restart") or "__parse_error__" in args:
                ok = False
                break
            cmd = args.get("command", "") or ""
            valid, _ = validate_command(cmd)
            if not valid or not is_parallel_safe(cmd):
                ok = False
                break
            parallel_candidates.append((call, cmd))
        if ok and parallel_candidates:
            dim(f"[parallel: {len(parallel_candidates)} read-only bash calls]")
            out_by_id: dict[str, str] = {}
            with ThreadPoolExecutor(max_workers=min(8, len(parallel_candidates))) as ex:
                future_to_call = {
                    ex.submit(execute_readonly, cmd): call
                    for call, cmd in parallel_candidates
                }
                for fut in future_to_call:
                    c = future_to_call[fut]
                    try:
                        out_by_id[c.id] = fut.result()
                    except Exception as e:  # noqa: BLE001
                        out_by_id[c.id] = f"(parallel exec error: {e})"
            first_cmd = parallel_candidates[0][1]
            dim(f"$ {first_cmd[:200]}{'...' if len(first_cmd) > 200 else ''}")
            preview(out_by_id[parallel_candidates[0][0].id])
            for call in response.tool_calls:
                if call.id in out_by_id:
                    results.append(provider.build_tool_result(call.id, out_by_id[call.id]))
                elif call.name != "bash":
                    results.append(provider.build_tool_result(
                        call.id, f"(unsupported tool: {call.name})"
                    ))
            return results, False

    for call in response.tool_calls:
        if call.name == "introspect":
            out = introspect() if introspect is not None else "(introspect unavailable)"
            results.append(provider.build_tool_result(call.id, out))
            continue

        if call.name == "delegate":
            if delegate_cb is None:
                results.append(provider.build_tool_result(
                    call.id,
                    "(delegate unavailable: specialists cannot themselves "
                    "delegate; only the orchestrator role may dispatch)",
                ))
                continue
            if interrupted:
                results.append(provider.build_tool_result(call.id, "(skipped — interrupted)"))
                continue
            try:
                args = call.arguments or {}
                if "__parse_error__" in args:
                    err = args["__parse_error__"]
                    raw = args.get("__raw_arguments__", "")
                    out = f"(delegate error: malformed JSON arguments — {err}; raw={raw!r})"
                    warn(out)
                    results.append(provider.build_tool_result(call.id, out))
                    continue
                sub_role = (args.get("role") or "").strip()
                sub_task = (args.get("task") or "").strip()
                if not sub_role or not sub_task:
                    out = "(delegate error: both `role` and `task` are required)"
                    warn(out)
                    results.append(provider.build_tool_result(call.id, out))
                    continue
                if sub_role not in ("code", "task", "create", "local", "chat"):
                    out = (
                        f"(delegate error: unknown role {sub_role!r}; must be "
                        "one of code/task/create/local/chat)"
                    )
                    warn(out)
                    results.append(provider.build_tool_result(call.id, out))
                    continue
                dim(f"[delegate -> {sub_role}] {sub_task[:160]}{'...' if len(sub_task) > 160 else ''}")
                try:
                    sub_out = delegate_cb(sub_role, sub_task)
                except KeyboardInterrupt:
                    interrupted = True
                    results.append(provider.build_tool_result(call.id, "(delegate interrupted by user)"))
                    continue
                except Exception as e:  # noqa: BLE001
                    sub_out = f"(delegate error: {e})"
                    warn(sub_out)
                results.append(provider.build_tool_result(call.id, sub_out or "(delegate returned no text)"))
            except KeyboardInterrupt:
                interrupted = True
                results.append(provider.build_tool_result(call.id, "(interrupted by user)"))
            continue

        if call.name != "bash":
            results.append(provider.build_tool_result(call.id, f"(unsupported tool: {call.name})"))
            continue
        if interrupted:
            results.append(provider.build_tool_result(call.id, "(skipped — interrupted)"))
            continue

        try:
            args = call.arguments or {}
            if "__parse_error__" in args:
                err = args["__parse_error__"]
                raw = args.get("__raw_arguments__", "")
                out = f"(tool-call error: malformed JSON arguments — {err}; raw={raw!r})"
                warn(out)
                results.append(provider.build_tool_result(call.id, out))
                continue
            if args.get("restart"):
                out = bash.restart()
                dim("[bash session restarted]")
            else:
                command = args.get("command", "") or ""
                ok, reason = validate_command(command)
                if ok:
                    dim(f"$ {command[:200]}{'...' if len(command) > 200 else ''}")
                    out = bash.execute(command)
                    preview(out)
                else:
                    out = reason or "(blocked)"
                    warn(out)
            results.append(provider.build_tool_result(call.id, out))
        except KeyboardInterrupt:
            interrupted = True
            results.append(provider.build_tool_result(call.id, "(interrupted by user)"))
            warn("interrupted")

    return results, interrupted
