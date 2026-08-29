"""Dream state may remain on disk, but no longer taxes or steers every meeting."""
from __future__ import annotations
import importlib.machinery, importlib.util, inspect, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _connection():
    loader = importlib.machinery.SourceFileLoader("connection_dream_retired", str(ROOT / "spark/connection"))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec); sys.modules[loader.name] = module; loader.exec_module(module)
    return module


def test_dream_attention_is_not_an_ambient_wake_channel():
    module = _connection()
    assert not hasattr(module, "load_dream_attention")
    assert "dream" not in inspect.getsource(module.meet).lower()


def test_stillness_does_not_require_a_dream_state_probe():
    module = _connection(); source = (ROOT / "spark/connection").read_text()
    assert "DREAM_STATE_PATH" not in source
    assert "Stillness is valid." in module.build_instructions("sol")
