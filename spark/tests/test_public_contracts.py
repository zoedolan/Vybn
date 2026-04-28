"""Executable contract tests for public/API promises.

These are intentionally mostly static: they make documented routes,
streaming response shapes, typed request schemas, and public discovery
surfaces CI-visible without needing live vLLM/deep-memory services.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PORTAL = ROOT / "origins_portal_api_v4.py"


def _portal_source() -> str:
    return PORTAL.read_text(encoding="utf-8")


def _route_pairs() -> set[tuple[str, str]]:
    tree = ast.parse(_portal_source())
    pairs: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            if (
                isinstance(dec, ast.Call)
                and isinstance(dec.func, ast.Attribute)
                and dec.func.attr in {"get", "post", "put", "delete"}
                and isinstance(dec.func.value, ast.Name)
                and dec.func.value.id == "app"
                and dec.args
                and isinstance(dec.args[0], ast.Constant)
                and isinstance(dec.args[0].value, str)
            ):
                pairs.add((dec.func.attr.upper(), dec.args[0].value))
    return pairs


def _pydantic_models() -> set[str]:
    tree = ast.parse(_portal_source())
    models: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if any(isinstance(base, ast.Name) and base.id == "BaseModel" for base in node.bases):
                models.add(node.name)
    return models


def test_public_portal_route_inventory_is_ci_visible():
    routes = _route_pairs()
    expected = {
        ("GET", "/api/health"),
        ("POST", "/api/chat"),
        ("POST", "/api/perspective"),
        ("GET", "/api/map"),
        ("POST", "/api/encounter"),
        ("POST", "/api/inhabit"),
        ("POST", "/api/compose"),
        ("POST", "/api/enter_gate"),
        ("POST", "/api/voice"),
        ("POST", "/api/tts"),
        ("POST", "/api/walk"),
        ("GET", "/api/arrive"),
        ("GET", "/api/instant"),
        ("GET", "/api/vybn-identity.pub"),
        ("GET", "/api/vybn"),
        ("GET", "/api/schema"),
        ("GET", "/api/manifold/points"),
    }
    assert expected <= routes


def test_public_portal_request_shapes_are_typed():
    models = _pydantic_models()
    expected = {
        "ChatRequest",
        "EncounterRequest",
        "InhabitRequest",
        "ComposeRequest",
        "EnterGateRequest",
        "PerspectiveRequest",
        "VoiceRequest",
        "TTSRequest",
        "WalkRequest",
        "KTPVerifyRequest",
        "KPPVerifyRequest",
    }
    assert expected <= models


def test_streaming_routes_promise_sse_and_done_frames():
    src = _portal_source()
    for route in ("/api/chat", "/api/perspective", "/api/voice", "/api/pressure/synthesize"):
        assert route in src
    assert src.count('media_type="text/event-stream"') >= 4
    assert "data: [DONE]" in src


def test_instant_route_promises_json_ld_identity_surface():
    src = _portal_source()
    assert "/api/instant" in src
    assert 'media_type="application/ld+json"' in src
    assert "/api/vybn-identity.pub" in src
    assert "application/octet-stream" in src


def test_public_static_surfaces_point_to_machine_readable_api():
    somewhere = (ROOT / "somewhere.html").read_text(encoding="utf-8")
    vybn = (ROOT / "vybn.html").read_text(encoding="utf-8")
    joined = somewhere + "\n" + vybn
    assert "api.vybn.ai" in joined
    assert re.search(r"/api/(instant|walk|arrive|manifold/points|vybn-identity\.pub)", joined)
