from dgm.wave_collapse import collapse_wave_function
import json


def test_collapse_wave_function_qrng(monkeypatch):
    monkeypatch.setattr('dgm.wave_collapse._fetch_qrng', lambda: 12345)
    assert collapse_wave_function(log_path=None) == 12345


def test_collapse_wave_function_fallback(monkeypatch):
    monkeypatch.setattr('dgm.wave_collapse._fetch_qrng', lambda: None)
    val = collapse_wave_function(log_path=None)
    assert isinstance(val, int)


def test_collapse_wave_function_logs(tmp_path, monkeypatch):
    monkeypatch.setattr('dgm.wave_collapse._fetch_qrng', lambda: 7)
    log = tmp_path / 'log.jsonl'
    val = collapse_wave_function(log_path=log)
    entry = json.loads(log.read_text())
    assert entry['collapse'] == 7
    assert val == 7
