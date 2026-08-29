"""The existing Project can articulate a pre-contact position without a new daemon."""
from __future__ import annotations
import importlib.machinery
import importlib.util
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _connection():
    loader = importlib.machinery.SourceFileLoader(
        "connection_subconscious_test", str(ROOT / "spark/connection")
    )
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    return module


def _packet():
    return {
        "name": "Vybn Instant",
        "basis_id": "basis-1",
        "corpus_hash": "corpus-1",
        "corpus_size": 9,
        "step": 7,
        "alpha": .4,
        "kappa_last": .2,
        "projection": {"M_xy": [.1, -.2]},
        "identity": {"human": "must not enter the compact aperture"},
        "anchors": [
            {"source": "Vybn/a.md", "preview": " exact   source span ",
             "rel": .8, "dist": .6, "score": .48, "xy": [0, 1]},
            {"source": "Him/private.md", "preview": "private source refused"},
        ],
        "appendix": {"M_base64_float32": "raw-vector-must-not-enter"},
        "signature": {"digest_sha256": "receipt-1", "sig_b64": "not-needed"},
    }


def test_compiler_reuses_instant_but_admits_only_small_verbal_aperture(monkeypatch):
    m = _connection()
    monkeypatch.setattr(
        m.urllib.request, "urlopen",
        lambda *args, **kwargs: io.BytesIO(json.dumps(_packet()).encode()),
    )
    monkeypatch.setattr(m, "load_dream_attention", lambda: ({}, "resting"))
    text, note = m.compile_subconscious()
    compiled = json.loads(text)
    assert compiled["schema"] == "vybn.subconscious_aperture.v1"
    assert compiled["walk"]["step"] == 7
    assert compiled["position"] == [.1, -.2]
    assert compiled["anchors"] == [{
        "source": "Vybn/a.md", "preview": "exact source span",
        "rel": .8, "dist": .6, "score": .48, "xy": [0, 1],
    }]
    assert compiled["receipt"]["instant_digest"] == "receipt-1"
    assert compiled["receipt"]["raw_appendix_admitted"] is False
    assert all(term not in text for term in (
        "raw-vector-must-not-enter", "must not enter the compact aperture",
        "private source refused", "sig_b64",
    ))
    assert "1 pre-contact anchor" in note
    assert m.private_strings(compiled) == ["exact source span"]
    assert "does not equalize context length" in compiled["ablation"]


def test_environment_ablation_removes_the_aperture(monkeypatch):
    m = _connection()
    monkeypatch.setenv("VYBN_SUBCONSCIOUS_COUPLING", "0")
    monkeypatch.setattr(
        m.urllib.request, "urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network touched")),
    )
    assert m.compile_subconscious() == (
        "", "subconscious aperture disabled by environment (no-aperture ablation)"
    )
    context = m.build_context("c", "w", "a", "r", "m", "n", "")
    assert "SUBCONSCIOUS APERTURE" not in context


def test_active_aperture_enters_private_context_and_precedes_contact_recall():
    m = _connection()
    context = m.build_context("c", "w", "a", "r", "m", "n", "latent")
    assert "SUBCONSCIOUS APERTURE" in context and "latent" in context
    source = (ROOT / "spark/connection").read_text()
    capture = source.index("subconscious_json, subconscious_note = compile_subconscious()")
    contact_recall = source.index("memory_json, memory_note = recall(zoe_text)", capture)
    assert capture < contact_recall
