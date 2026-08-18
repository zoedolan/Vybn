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
    focus, status = module.load_dream_attention(state)
    assert status == "holding" and focus["selection_count"] == 2
    assert focus["text"].startswith("What survives")


def test_rest_and_missing_state_remain_distinguishable(tmp_path):
    module = _connection(); state = tmp_path / "dream_state.json"; state.write_text("{}")
    assert [module.load_dream_attention(p) for p in (state, tmp_path / "absent.json")] == [({}, "resting"), (None, "unavailable")]


def test_current_conveyance_is_visible_at_wake(tmp_path):
    module = _connection(); state = tmp_path / "conveyances"; state.mkdir()
    state.joinpath("current").write_text("c1\n")
    state.joinpath("events.jsonl").write_text("\n".join([
        json.dumps({"schema": "vybn.conveyance.event.v1", "event": "proposal", "conveyance_id": "c1", "title": "A changing page"}),
        json.dumps({"schema": "vybn.conveyance.event.v1", "event": "witness", "conveyance_id": "c1", "verdict": "revise"}),
    ]) + "\n")
    line = module.wake_status(state=state)
    assert "A changing page" in line and "revise" in line and "/conveyances" in line
