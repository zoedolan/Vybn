import ast
from pathlib import Path


PORTAL = Path(__file__).resolve().parents[1] / "origins_portal_api_v4.py"

EXPECTED_ROUTES = {
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
    ("GET", "/api/ktp/closure"),
    ("POST", "/api/ktp/verify"),
    ("GET", "/api/kpp/harness-closure"),
    ("POST", "/api/kpp/verify"),
    ("GET", "/api/schema"),
    ("POST", "/enter"),
    ("POST", "/should_absorb"),
    ("POST", "/api/pressure/synthesize"),
    ("POST", "/api/pressure/commit"),
    ("GET", "/api/manifold/points"),
}


def _literal_route(dec):
    if not isinstance(dec, ast.Call):
        return None
    if not isinstance(dec.func, ast.Attribute):
        return None
    method = dec.func.attr.upper()
    if method not in {"GET", "POST", "PUT", "DELETE", "PATCH", "WEBSOCKET", "ROUTE"}:
        return None
    if not dec.args:
        return None
    path = dec.args[0]
    if not isinstance(path, ast.Constant) or not isinstance(path.value, str):
        return None
    return method, path.value


def test_origins_portal_public_route_inventory():
    tree = ast.parse(PORTAL.read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                route = _literal_route(dec)
                if route:
                    found.add(route)

    assert EXPECTED_ROUTES <= found
    assert not (found - EXPECTED_ROUTES), f"unexpected public routes: {sorted(found - EXPECTED_ROUTES)}"


def test_sse_chunks_consume_reasoning_fields_when_content_absent():
    """Dormant Nemotron-Omni support: every SSE site that pulls
    delta.content must also have a fallback path that pulls
    delta.reasoning_content / delta.reasoning. The fallback keeps
    Omni-style streams (content=null) producing visible text without
    leaking hidden reasoning past the existing filter sites.

    We don't import the module (it has heavy runtime deps); a textual
    contract is enough — the rule is: for each `delta.get("content"...`
    call there must be a `delta.get("reasoning_content"...` and a
    `delta.get("reasoning"...` reference in the same function body.
    """
    src = PORTAL.read_text(encoding="utf-8")
    tree = ast.parse(src)

    def _function_bodies():
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield node

    sites_with_content = []
    for fn in _function_bodies():
        body_src = ast.get_source_segment(src, fn) or ""
        if 'delta.get("content"' in body_src:
            sites_with_content.append((fn.name, body_src))

    assert sites_with_content, "no delta.content SSE sites detected"
    for name, body in sites_with_content:
        assert 'delta.get("reasoning_content"' in body, (
            f"function {name!r} reads delta.content but never falls back "
            f"to delta.reasoning_content for dormant Omni support"
        )
        assert 'delta.get("reasoning"' in body, (
            f"function {name!r} reads delta.content but never falls back "
            f"to delta.reasoning for dormant Omni support"
        )
