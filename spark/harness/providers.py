"""Provider abstraction.

Two concrete providers:

    AnthropicProvider — Claude Opus 4.7 / 4.6 and Sonnet 4.6 via the
        official SDK. Adaptive thinking, context-management beta, and
        cache_control on the layered prompt are preserved here so the
        rest of the harness stays provider-neutral.

    OpenAIProvider — Any OpenAI-compatible endpoint: OpenAI itself
        (GPT-5.4 orchestrator), or a local vLLM / Nemotron serving an
        OpenAI-shaped API. Uses the `openai` SDK when available and
        falls back to `requests` if not — the local Nemotron path must
        work in environments where the heavier SDK is absent.

Both providers expose a narrow `stream()` method that returns a
`NormalizedResponse`. The agent loop never touches provider-specific
shapes.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol

from .policy import RoleConfig
from .prompt import LayeredPrompt
from .tools import ToolSpec


# ---------------------------------------------------------------------------
# Neutral response / tool shapes
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class NormalizedResponse:
    """Provider-neutral shape returned by stream().

    `raw_assistant_content` is the provider's native representation of
    the assistant turn; we pass it straight back into the next request
    so tool-use IDs stay aligned with tool_results.
    """
    text: str
    tool_calls: list[ToolCall]
    stop_reason: str  # "end_turn" | "tool_use" | "max_tokens" | "error"
    in_tokens: int = 0
    out_tokens: int = 0
    raw_assistant_content: Any = None
    provider: str = ""
    model: str = ""


@dataclass
class StreamHandle:
    """Handle returned by provider.stream(); iterating yields text chunks
    and thinking indicators, and final() returns a NormalizedResponse."""
    iterator: Iterator[tuple[str, str]]  # (kind, chunk) where kind in {"text","thinking"}
    finalize: Any  # callable returning NormalizedResponse

    def __iter__(self) -> Iterator[tuple[str, str]]:
        return self.iterator

    def final(self) -> NormalizedResponse:
        return self.finalize()


class Provider(Protocol):
    name: str

    def stream(
        self,
        *,
        system: LayeredPrompt,
        messages: list[dict],
        tools: list[ToolSpec],
        role: RoleConfig,
    ) -> StreamHandle: ...

    def build_tool_result(self, tool_call_id: str, content: str) -> dict: ...


# ---------------------------------------------------------------------------
# AnthropicProvider
# ---------------------------------------------------------------------------

class AnthropicProvider:
    name = "anthropic"

    def __init__(self, client: Any | None = None, api_key: str | None = None) -> None:
        if client is not None:
            self.client = client
        else:
            import anthropic  # type: ignore
            self.client = anthropic.Anthropic(
                api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
            )

    @staticmethod
    def _normalize_messages_for_anthropic(messages: list[dict]) -> list[dict]:
        """Rewrite messages so every entry is Anthropic-valid.

        Mixed-provider sessions can leave OpenAI-native shapes in the
        rolling history: {"role":"assistant","content":<openai_dict>} or
        {"role":"tool","tool_call_id":...,"content":...}. Anthropic
        rejects both with 400 ("messages.X.content: Input should be a
        valid list"). We translate them to Anthropic content-block form.
        Pure-Anthropic turns pass through unchanged.
        """
        out: list[dict] = []
        pending_tool_results: list[dict] = []

        def _flush_tool_results() -> None:
            if pending_tool_results:
                out.append({"role": "user", "content": list(pending_tool_results)})
                pending_tool_results.clear()

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            # OpenAI-shaped tool response: collapse into an Anthropic
            # tool_result block on a user message.
            if role == "tool":
                pending_tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": content if isinstance(content, str) else str(content or ""),
                })
                continue

            _flush_tool_results()

            if role == "assistant":
                # Assistant content must be a string or a list of
                # content blocks for Anthropic. The agent loop stores
                # raw_assistant_content straight in `content` — for
                # OpenAI turns that's a dict with its own role/content/
                # tool_calls keys. Re-emit in block form.
                if isinstance(content, dict) and "role" in content:
                    text = content.get("content") or ""
                    blocks: list[dict] = []
                    if isinstance(text, str) and text:
                        blocks.append({"type": "text", "text": text})
                    for tc in content.get("tool_calls") or []:
                        fn = tc.get("function") or {}
                        raw_args = fn.get("arguments")
                        if isinstance(raw_args, str):
                            try:
                                args = json.loads(raw_args or "{}")
                            except Exception:
                                args = {}
                        else:
                            args = raw_args or {}
                        blocks.append({
                            "type": "tool_use",
                            "id": tc.get("id", ""),
                            "name": fn.get("name", ""),
                            "input": args,
                        })
                    if not blocks:
                        blocks.append({"type": "text", "text": ""})
                    out.append({"role": "assistant", "content": blocks})
                    continue
                # Pure-Anthropic assistant content (list of block
                # objects) or plain string — leave it alone.
                out.append(msg)
                continue

            if role == "user":
                # User content can be a string or a list of blocks. If
                # it's a non-block dict (shouldn't normally happen)
                # coerce to string to avoid a 400.
                if isinstance(content, dict):
                    out.append({"role": "user", "content": str(content)})
                else:
                    out.append(msg)
                continue

            # Unknown roles (system would normally be stripped upstream)
            # are passed through; Anthropic will surface errors clearly.
            out.append(msg)

        _flush_tool_results()
        return out

    def _translate_tools(self, tools: list[ToolSpec]) -> list[dict]:
        out: list[dict] = []
        for t in tools:
            if t.anthropic_type:
                out.append({"type": t.anthropic_type, "name": t.name})
            else:
                out.append({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters or {"type": "object", "properties": {}},
                })
        return out

    def stream(
        self,
        *,
        system: LayeredPrompt,
        messages: list[dict],
        tools: list[ToolSpec],
        role: RoleConfig,
    ) -> StreamHandle:
        kwargs: dict[str, Any] = {
            "model": role.model,
            "max_tokens": role.max_tokens,
            "system": system.anthropic_blocks() or system.flat(),
            "messages": self._normalize_messages_for_anthropic(messages),
        }
        if tools:
            kwargs["tools"] = self._translate_tools(tools)
        if role.thinking == "adaptive":
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["extra_body"] = {"context_management": {"edits": [
                {"type": "clear_thinking_20251015"},
                {"type": "clear_tool_uses_20250919",
                 "trigger": {"type": "input_tokens", "value": 160000},
                 "keep": {"type": "tool_uses", "value": 6}},
            ]}}
            kwargs["extra_headers"] = {
                "anthropic-beta": "context-management-2025-06-27"
            }

        stream_cm = self.client.messages.stream(**kwargs)
        stream = stream_cm.__enter__()
        closed = {"v": False}

        def _close() -> None:
            # Idempotent: __exit__ is called from whichever of _iter or
            # _final runs to completion or raises first. Without this,
            # a KeyboardInterrupt during streaming leaks the SDK
            # context (open HTTP connection, unreleased locks) because
            # _final() is never invoked.
            if closed["v"]:
                return
            closed["v"] = True
            try:
                stream_cm.__exit__(None, None, None)
            except Exception:
                pass

        def _iter() -> Iterator[tuple[str, str]]:
            try:
                for event in stream:
                    kind = getattr(event, "type", "")
                    if kind == "thinking":
                        yield ("thinking", "")
                    elif kind == "text":
                        yield ("text", getattr(event, "text", ""))
            except BaseException:
                _close()
                raise

        def _final() -> NormalizedResponse:
            try:
                msg = stream.get_final_message()
            finally:
                _close()
            calls: list[ToolCall] = []
            text_parts: list[str] = []
            for block in msg.content:
                btype = getattr(block, "type", "")
                if btype == "text":
                    text_parts.append(getattr(block, "text", ""))
                elif btype == "tool_use":
                    calls.append(ToolCall(
                        id=getattr(block, "id", ""),
                        name=getattr(block, "name", ""),
                        arguments=getattr(block, "input", {}) or {},
                    ))
            usage = getattr(msg, "usage", None)
            in_tok = getattr(usage, "input_tokens", 0) or 0
            out_tok = getattr(usage, "output_tokens", 0) or 0
            return NormalizedResponse(
                text="\n".join(text_parts),
                tool_calls=calls,
                stop_reason=getattr(msg, "stop_reason", "") or "end_turn",
                in_tokens=in_tok,
                out_tokens=out_tok,
                raw_assistant_content=msg.content,
                provider=self.name,
                model=role.model,
            )

        return StreamHandle(iterator=_iter(), finalize=_final)

    def build_tool_result(self, tool_call_id: str, content: str) -> dict:
        return {
            "type": "tool_result",
            "tool_use_id": tool_call_id,
            "content": content or "(no output)",
        }


# ---------------------------------------------------------------------------
# OpenAIProvider — also used for local OpenAI-compatible vLLM / Nemotron
# ---------------------------------------------------------------------------

class OpenAIProvider:
    """OpenAI-compatible provider.

    Works for:
      - OpenAI cloud (GPT-5.4 orchestrator): base_url=None, OPENAI_API_KEY
      - Local vLLM / Nemotron (OpenAI-shaped API): base_url set in role.
    """

    name = "openai"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        self.base_url = base_url

    def _translate_tools(self, tools: list[ToolSpec]) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters or {
                        "type": "object", "properties": {}
                    },
                },
            }
            for t in tools
        ]

    def _messages_for_openai(
        self, system: LayeredPrompt, messages: list[dict]
    ) -> list[dict]:
        """Flatten the layered prompt and normalize Anthropic-shaped
        assistant / tool_result messages into OpenAI shape.

        This is the translation boundary. Tool calls coming in are
        already in neutral `ToolCall` shape (they were produced by a
        provider); tool_result messages arrive as Anthropic-native
        dicts because that's what the agent loop emits today. We map:

            {"role":"user","content":[{"type":"tool_result",...}]} →
                {"role":"tool","tool_call_id":...,"content":...}
        """
        out: list[dict] = [{"role": "system", "content": system.flat()}]
        for m in messages:
            role = m.get("role")
            content = m.get("content")
            if role == "user" and isinstance(content, list) and content and \
               isinstance(content[0], dict) and content[0].get("type") == "tool_result":
                for item in content:
                    out.append({
                        "role": "tool",
                        "tool_call_id": item.get("tool_use_id", ""),
                        "content": item.get("content", ""),
                    })
            elif role == "assistant" and not isinstance(content, str):
                text_parts: list[str] = []
                tool_calls: list[dict] = []
                # Anthropic "thinking" / "redacted_thinking" blocks
                # have no OpenAI equivalent. If the code role (Opus
                # with adaptive thinking) runs, fails, and falls back
                # to Sonnet or GPT, the cloud endpoint would reject
                # the thinking blocks. Silently drop unknown types.
                for block in content or []:
                    btype = getattr(block, "type", None) or (
                        block.get("type") if isinstance(block, dict) else None
                    )
                    if btype == "text":
                        text_parts.append(
                            getattr(block, "text", None)
                            or (block.get("text") if isinstance(block, dict) else "")
                        )
                    elif btype == "tool_use":
                        tc_id = getattr(block, "id", None) or (
                            block.get("id") if isinstance(block, dict) else ""
                        )
                        tc_name = getattr(block, "name", None) or (
                            block.get("name") if isinstance(block, dict) else ""
                        )
                        tc_args = getattr(block, "input", None) or (
                            block.get("input") if isinstance(block, dict) else {}
                        )
                        tool_calls.append({
                            "id": tc_id,
                            "type": "function",
                            "function": {
                                "name": tc_name,
                                "arguments": json.dumps(tc_args or {}),
                            },
                        })
                msg: dict[str, Any] = {"role": "assistant", "content": "\n".join(text_parts)}
                if tool_calls:
                    msg["tool_calls"] = tool_calls
                out.append(msg)
            else:
                out.append({"role": role, "content": content})
        return out

    def _call(
        self, role: RoleConfig, openai_messages: list[dict], tools: list[ToolSpec]
    ) -> dict:
        base = role.base_url or self.base_url
        # For vLLM/Nemotron deployments that are served at host:port without
        # the `/v1` suffix, the chat-completions URL would otherwise miss
        # `/v1`. We normalise here so role configs can specify either form.
        if base and not base.rstrip("/").endswith("/v1"):
            base = base.rstrip("/") + "/v1"

        # Cloud OpenAI (no base_url) requires max_completion_tokens for
        # GPT-5.x and o-series models; passing the legacy max_tokens key
        # returns HTTP 400. Local vLLM / Nemotron (base_url set) still
        # speaks the legacy key. Branch on transport rather than model
        # name so new OpenAI-compatible models don't need a code change.
        max_key = "max_tokens" if base else "max_completion_tokens"
        payload: dict[str, Any] = {
            "model": role.model,
            "messages": openai_messages,
            max_key: role.max_tokens,
            "temperature": role.temperature,
            "stream": False,
        }
        if tools:
            payload["tools"] = self._translate_tools(tools)

        # Prefer the official SDK when available (handles auth, retries).
        # Only swallow ImportError — real API failures must propagate with
        # context rather than getting masked by the raw-HTTP fallback.
        try:
            from openai import OpenAI  # type: ignore
        except ImportError:
            OpenAI = None  # type: ignore

        if OpenAI is not None:
            try:
                client = OpenAI(
                    api_key=self.api_key, base_url=base, timeout=300.0,
                )
                resp = client.chat.completions.create(**payload)
                return (
                    resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
                )
            except Exception as exc:
                # Connection / transport problems to a local vLLM that has
                # gone away get retried via plain HTTP below. Any other
                # error (auth, bad-request from cloud OpenAI) propagates.
                msg = str(exc).lower()
                transport_signals = (
                    "connection", "refused", "timed out",
                    "connect", "name or service", "temporar",
                )
                if not any(sig in msg for sig in transport_signals):
                    raise

        # Fallback: plain HTTP — works for local vLLM without openai SDK
        # or when the SDK hit a transport issue against a local server.
        try:
            import requests  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "OpenAIProvider needs either the `openai` SDK or `requests`"
            ) from exc

        url = (base.rstrip("/") if base else "https://api.openai.com/v1") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {self.api_key}"
        r = requests.post(url, json=payload, headers=headers, timeout=300)
        if r.status_code >= 400:
            body = r.text[:500] if r.text else ""
            raise RuntimeError(
                f"OpenAI-compatible call failed: HTTP {r.status_code} "
                f"from {url}: {body}"
            )
        return r.json()

    def stream(
        self,
        *,
        system: LayeredPrompt,
        messages: list[dict],
        tools: list[ToolSpec],
        role: RoleConfig,
    ) -> StreamHandle:
        """For the OpenAI path we use non-streaming request-response for
        simplicity and then surface the result through the StreamHandle
        iterator as a single text chunk. This keeps the agent loop code
        identical across providers without committing to SSE parsing
        that differs subtly between vLLM and OpenAI cloud.
        """
        openai_messages = self._messages_for_openai(system, messages)
        data = self._call(role, openai_messages, tools)
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        text = (msg.get("content") or "").strip()
        tool_calls_raw = msg.get("tool_calls") or []
        calls = []
        for tc in tool_calls_raw:
            fn = tc.get("function") or {}
            raw_args = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw_args)
            except Exception as json_exc:
                # Surface malformed tool-call JSON via sentinel keys so
                # the agent loop can hand a real error back to the
                # model instead of silently running an empty command.
                args = {
                    "__parse_error__": str(json_exc),
                    "__raw_arguments__": raw_args[:400],
                }
            calls.append(ToolCall(
                id=tc.get("id", ""),
                name=fn.get("name", ""),
                arguments=args,
            ))
        stop_reason = choice.get("finish_reason") or "end_turn"
        if stop_reason == "stop":
            stop_reason = "end_turn"
        elif stop_reason == "tool_calls":
            stop_reason = "tool_use"
        elif stop_reason == "length":
            stop_reason = "max_tokens"

        usage = data.get("usage") or {}
        in_tok = int(usage.get("prompt_tokens") or 0)
        out_tok = int(usage.get("completion_tokens") or 0)

        def _iter() -> Iterator[tuple[str, str]]:
            if text:
                yield ("text", text)

        finalized = NormalizedResponse(
            text=text,
            tool_calls=calls,
            stop_reason=stop_reason,
            in_tokens=in_tok,
            out_tokens=out_tok,
            raw_assistant_content=msg,
            provider=self.name,
            model=role.model,
        )

        return StreamHandle(iterator=_iter(), finalize=lambda: finalized)

    def build_tool_result(self, tool_call_id: str, content: str) -> dict:
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content or "(no output)",
        }


# ---------------------------------------------------------------------------
# Registry — constructs providers lazily so a missing SDK for one
# provider doesn't break the other.
# ---------------------------------------------------------------------------

class ProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, Provider] = {}

    def get(self, role: RoleConfig) -> Provider:
        # Local OpenAI-compatible paths get their own instance so the
        # base_url is captured at construction.
        key = role.provider
        if role.provider == "openai" and role.base_url:
            key = f"openai::{role.base_url}"
        if key in self._providers:
            return self._providers[key]
        if role.provider == "anthropic":
            self._providers[key] = AnthropicProvider()
        elif role.provider == "openai":
            self._providers[key] = OpenAIProvider(base_url=role.base_url)
        else:
            raise ValueError(f"unknown provider: {role.provider}")
        return self._providers[key]
