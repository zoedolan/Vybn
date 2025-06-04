import json
import os
import secrets
from pathlib import Path

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dgm.self_improve import create_child


def test_create_child_records_seed_and_collapse(tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    (parent / "code").mkdir(parents=True)
    (parent / "code" / "foo.py").write_text("print('hello')")
    (parent / "metadata.json").write_text("{}")

    os.environ['QUANTUM_SEED'] = str(secrets.randbits(16))
    os.environ['OPENAI_API_KEY'] = 'x'

    monkeypatch.setattr('dgm.self_improve._fetch_qrng', lambda: None)
    monkeypatch.setattr('dgm.self_improve.collapse_wave_function', lambda: 3)
    monkeypatch.setattr('dgm.self_improve.suggest_patch', lambda f, i: "print('patched')")

    child = tmp_path / "child"
    create_child(parent, child, instruction="Patch")

    meta = json.loads((child / "metadata.json").read_text())
    assert meta['seed'] == os.environ['QUANTUM_SEED']
    assert meta['collapse'] == 3
    assert meta['patched_file'] == 'foo.py'
