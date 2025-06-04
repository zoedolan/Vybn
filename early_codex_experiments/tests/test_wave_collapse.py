from dgm.wave_collapse import collapse_wave_function


def test_collapse_wave_function_qrng(monkeypatch):
    monkeypatch.setattr('dgm.wave_collapse._fetch_qrng', lambda: 12345)
    assert collapse_wave_function() == 12345


def test_collapse_wave_function_fallback(monkeypatch):
    monkeypatch.setattr('dgm.wave_collapse._fetch_qrng', lambda: None)
    val = collapse_wave_function()
    assert isinstance(val, int)
