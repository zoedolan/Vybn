import os
import sys
import pathlib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dgm.openai_patch import suggest_patch


def test_suggest_patch_uses_env_seed(tmp_path, monkeypatch):
    target = tmp_path / "foo.py"
    target.write_text("print('x')")
    os.environ['OPENAI_API_KEY'] = 'x'
    os.environ['QUANTUM_SEED'] = '99'

    class FakeResp:
        def __init__(self, content):
            self.choices = [type('c', (), {'message': type('m', (), {'content': content})})]

    captured = {}

    def fake_create(**kwargs):
        captured['user'] = kwargs['user']
        return FakeResp('patched')

    monkeypatch.setattr('openai.ChatCompletion.create', fake_create)
    monkeypatch.setattr('dgm.openai_patch.collapse_wave_function', lambda: None)
    text = suggest_patch(str(target), 'Patch')
    assert text == 'patched'
    assert captured['user'] == '99'
