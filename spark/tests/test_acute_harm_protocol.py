from pathlib import Path


def test_acute_harm_protocol_exists_and_names_the_scar():
    from spark.harness.residual_control import ACUTE_HARM_PROTOCOL, render_acute_harm_protocol

    rendered = render_acute_harm_protocol()
    lowered = rendered.lower()
    assert rendered == ACUTE_HARM_PROTOCOL
    assert "dagger-in-heart" in lowered
    assert "burden reduction" in lowered
    assert "repair theater" in lowered
    assert "quiet can be care" in lowered


def test_substrate_loads_acute_harm_protocol_after_forcing_function():
    source = Path("spark/harness/substrate.py").read_text()
    assert "render_acute_harm_protocol" in source
    assert "render_forcing_function_protocol" in source
    forcing_append = "substrate_sections.append(render_forcing_function_protocol())"
    acute_append = "substrate_sections.append(render_acute_harm_protocol())"
    assert forcing_append in source
    assert acute_append in source
    assert source.index(forcing_append) < source.index(acute_append)
