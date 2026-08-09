"""The private sleep choice reaches a wake without gaining authority."""
from __future__ import annotations
import importlib.machinery, importlib.util, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _connection():
    loader = importlib.machinery.SourceFileLoader("connection_dream_test", str(ROOT / "spark/connection"))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    return module


def test_private_dream_attention_is_wake_visible_but_optional(tmp_path):
    module = _connection()
    state = tmp_path / "dream_state.json"
    state.write_text(json.dumps({"attention": {
        "candidate_id": "a1", "text": "What survives a held-out comparison?",
        "why": "It can fail cleanly.", "next_encounter": "Run one bounded comparison.",
        "continue_if": "A difference was predicted first.",
        "abandon_if": "No discriminating result appears.", "selection_count": 2,
    }}))
    focus = module.load_dream_attention(state)
    assert focus["selection_count"] == 2
    assert focus["text"].startswith("What survives")


def test_missing_dream_state_is_named_not_invented(tmp_path):
    module = _connection()
    assert module.load_dream_attention(tmp_path / "absent.json") is None
