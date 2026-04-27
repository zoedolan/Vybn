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
