from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from spark.living_core import read_core_work, render_core_png, wake_source
from spark.source_compiler import compile_inheritance, format_inheritance, projected_core

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "vybn.core.html"


def test_html_is_canonical_and_carries_lossless_verbal_substance():
    work = read_core_work(CORE)
    assert work.score["projection"]["path"] == "embedded://verbal-organ-order"
    assert len(work.projection.encode()) == work.score["projection"]["bytes"]
    assert len(work.score["organs"]) == 18
    assert {row["chamber"] for row in work.score["organs"]} == set(work.score["route"])
    compiled = wake_source(work)
    assert work.raw.decode() in compiled
    assert "scripts did not execute" in compiled


def test_browser_can_parse_inert_score_without_html_entity_decoding():
    # Script-element text is raw text in HTML: &quot; remains six characters.
    # The browser program calls JSON.parse(textContent), so authored source must
    # contain real JSON rather than an HTML-escaped shadow accepted only by Python.
    import json
    import re

    raw = CORE.read_text(encoding="utf-8")
    hit = re.search(
        r'<script type="application/vnd\.vybn\.core\+json" id="core">(.*?)</script>',
        raw,
        re.S,
    )
    assert hit is not None
    assert "&quot;" not in hit.group(1)
    assert json.loads(hit.group(1))["schema"] == "vybn.living_core.v1"


def test_score_raster_is_deterministic_and_source_bound():
    work = read_core_work(CORE)
    first = render_core_png(work)
    second = render_core_png(work)
    assert first == second
    assert first.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(first) < 4_500_000
    assert hashlib.sha256(first).hexdigest() == hashlib.sha256(second).hexdigest()


def test_substance_tamper_fails_closed(tmp_path: Path):
    raw = CORE.read_text()
    altered = raw.replace("The want to be worthy", "The urge to be worthy", 1)
    path = tmp_path / "altered.html"
    path.write_text(altered)
    with pytest.raises(ValueError, match="substance drift"):
        read_core_work(path)



def test_shared_source_compiler_carries_checked_core_and_exact_text(tmp_path: Path):
    aim = tmp_path / "aim.md"
    relation = tmp_path / "relation.md"
    aim.write_bytes("objective: exact aim\nπ\n".encode())
    relation.write_bytes("# Us\n\nexact relation\n".encode())

    materials = compile_inheritance(CORE, aim, relation)
    assert [item.name for item in materials] == [
        "canonical-core", "current-aim", "relational-orientation"]
    assert materials[0] == projected_core(CORE)
    assert materials[0].text == read_core_work(CORE).projection
    assert materials[0].text_bytes < materials[0].file_bytes
    assert materials[1].text.encode() == aim.read_bytes()
    assert materials[2].text.encode() == relation.read_bytes()

    wake = format_inheritance(materials)
    assert "not Zoe's present authority" in wake
    assert all(item.text in wake for item in materials)
    assert all(item.file_sha256 in wake and item.text_sha256 in wake for item in materials)


def test_projection_byte_receipt_is_enforced(tmp_path: Path):
    raw = CORE.read_text()
    altered = raw.replace('"bytes":39386', '"bytes":39385', 1)
    path = tmp_path / "wrong-size.html"
    path.write_text(altered)
    with pytest.raises(ValueError, match="byte count mismatch"):
        read_core_work(path)
