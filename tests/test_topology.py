"""Tests for the refactored topology tool."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "spark"))

from topology import (
    build_import_graph,
    build_coevolution_graph,
    build_keyword_graph,
    compute_surprise_scores,
    compute_shared_subspace,
    extract_module_text,
    parse_imports,
)


# ---------------------------------------------------------------------------
# Phase 1: Structural
# ---------------------------------------------------------------------------

class TestParseImports:
    def test_basic_imports(self, tmp_path):
        f = tmp_path / "test_mod.py"
        f.write_text("import os\nimport json\nfrom pathlib import Path\n")
        imports = parse_imports(f)
        assert "os" in imports
        assert "json" in imports
        assert "pathlib" in imports

    def test_syntax_error_handled(self, tmp_path):
        f = tmp_path / "bad.py"
        f.write_text("def broken(\n")
        imports = parse_imports(f)
        assert imports == []


class TestImportGraph:
    def test_only_local_imports(self, tmp_path):
        (tmp_path / "alpha.py").write_text("import beta\nimport os\n")
        (tmp_path / "beta.py").write_text("import json\n")
        files = list(tmp_path.glob("*.py"))
        graph = build_import_graph(files)
        assert "beta" in graph["alpha"]
        assert "os" not in graph["alpha"]  # stdlib, not local


# ---------------------------------------------------------------------------
# Phase 2: Temporal
# ---------------------------------------------------------------------------

class TestCoevolution:
    def test_basic_coevolution(self):
        groups = [
            ["alpha", "beta"],
            ["alpha", "beta"],
            ["alpha", "beta"],
            ["gamma", "delta"],
        ]
        coevo = build_coevolution_graph(groups, threshold=3)
        assert ("alpha", "beta") in coevo
        assert coevo[("alpha", "beta")] == 3
        assert ("gamma", "delta") not in coevo  # below threshold


# ---------------------------------------------------------------------------
# Phase 3: Semantic (keyword fallback)
# ---------------------------------------------------------------------------

class TestKeywordGraph:
    def test_keyword_detection(self, tmp_path):
        (tmp_path / "mem_mod.py").write_text(
            '"""Memory and persistence module."""\n'
            "# handles memory recall and persistence\n"
            "# also emergence patterns\n"
        )
        (tmp_path / "emerge_mod.py").write_text(
            '"""Emergence detection."""\n'
            "# emergence and memory patterns\n"
        )
        files = list(tmp_path.glob("*.py"))
        edges, concepts = build_keyword_graph(files, overlap_threshold=2)
        assert len(edges) >= 1
        assert "memory" in concepts.get("mem_mod", {})


class TestExtractModuleText:
    def test_extracts_docstring(self, tmp_path):
        f = tmp_path / "documented.py"
        f.write_text('"""This is the module docstring."""\n\ndef foo():\n    """Foo docs."""\n    pass\n')
        text = extract_module_text(f)
        assert "module docstring" in text
        assert "Foo docs" in text

    def test_extracts_comments(self, tmp_path):
        f = tmp_path / "commented.py"
        f.write_text("# This is an important comment\nx = 1\n# Another comment\n")
        text = extract_module_text(f)
        assert "important comment" in text


# ---------------------------------------------------------------------------
# Phase 4: Surprise
# ---------------------------------------------------------------------------

class TestSurpriseScores:
    def test_surprise_computation(self):
        import numpy as np

        # Create mock embeddings — one outlier, rest similar
        embeddings = {
            "normal_a": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "normal_b": np.array([0.9, 0.1, 0.0], dtype=np.float32),
            "outlier":  np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }
        # Normalize
        for k in embeddings:
            embeddings[k] = embeddings[k] / max(float(np.linalg.norm(embeddings[k])), 1e-9)

        scores = compute_surprise_scores(embeddings)
        assert scores["outlier"] > scores["normal_a"]
        assert scores["outlier"] > scores["normal_b"]

    def test_empty_embeddings(self):
        assert compute_surprise_scores({}) == {}


# ---------------------------------------------------------------------------
# Phase 5: Shared Subspace
# ---------------------------------------------------------------------------

class TestSharedSubspace:
    def test_subspace_computation(self):
        import numpy as np

        embeddings = {
            f"mod_{i}": np.random.randn(64).astype(np.float32)
            for i in range(10)
        }
        result = compute_shared_subspace(embeddings, n_components=3)
        assert result["n_components"] == 3
        assert len(result["variance_explained"]) == 3
        assert len(result["module_loadings"]) == 10
        assert len(result["clusters"]) > 0

    def test_too_few_modules(self):
        import numpy as np
        embeddings = {"a": np.zeros(10), "b": np.zeros(10)}
        result = compute_shared_subspace(embeddings)
        assert result["components"] == []
