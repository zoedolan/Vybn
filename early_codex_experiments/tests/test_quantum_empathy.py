import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vybn.quantum_empathy import empathic_reply


def test_empathic_reply_uses_seed(monkeypatch):
    os.environ['OPENAI_API_KEY'] = 'x'
    os.environ['QUANTUM_SEED'] = '123'

    class FakeResp:
        def __init__(self, content):
            self.choices = [type('c', (), {'message': type('m', (), {'content': content})})]

    captured = {}

    def fake_create(model, messages, user, timeout=30):
        captured['user'] = user
        return FakeResp('kind reply')

    monkeypatch.setattr('openai.chat.completions.create', fake_create)
    monkeypatch.setattr('vybn.quantum_empathy.seed_rng', lambda: 123)
    text = empathic_reply('hello')
    assert text == 'kind reply'
    assert captured['user'] == '123'
