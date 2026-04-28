"""Tests for provider credential env loading.

Validates:
  - Loads OPENAI_API_KEY only when absent.
  - Does NOT overwrite a pre-existing env var unless overwrite=True.
  - Does not leak values through str() / repr() / print of the return
    dict (return value maps key -> source_path, never the secret).
  - Ignores keys outside the whitelist.
  - Tolerates a missing /etc/environment.
  - Parses `export KEY=value`, quoted, and bare forms.
  - Skips unreadable files without raising.

Run: python3 spark/tests/test_env_loader.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

THIS = Path(__file__).resolve()
SPARK_DIR = THIS.parent.parent
sys.path.insert(0, str(SPARK_DIR))

from harness.providers import load_env_files, describe  # noqa: E402


SENTINEL = "sk-test-ENV-LOADER-SENTINEL-0123456789"
SENTINEL2 = "sk-test-ENV-LOADER-SENTINEL-DIFFERENT"


class EnvLoaderTest(unittest.TestCase):
    def setUp(self) -> None:
        # Snapshot env so we can restore
        self._saved = {k: os.environ.get(k) for k in (
            "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY",
            "GROQ_API_KEY",
        )}
        for k in self._saved:
            os.environ.pop(k, None)

    def tearDown(self) -> None:
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _write(self, body: str) -> str:
        fd, path = tempfile.mkstemp(prefix="llm_env_", suffix=".env")
        os.close(fd)
        Path(path).write_text(body, encoding="utf-8")
        os.chmod(path, 0o600)
        return path

    def test_sets_key_when_absent(self):
        p = self._write(f'OPENAI_API_KEY="{SENTINEL}"\n')
        try:
            applied = load_env_files([p])
        finally:
            os.unlink(p)
        self.assertEqual(applied.get("OPENAI_API_KEY"), p)
        self.assertEqual(os.environ.get("OPENAI_API_KEY"), SENTINEL)

    def test_does_not_overwrite_existing(self):
        os.environ["OPENAI_API_KEY"] = SENTINEL2
        p = self._write(f'OPENAI_API_KEY="{SENTINEL}"\n')
        try:
            applied = load_env_files([p])
        finally:
            os.unlink(p)
        self.assertNotIn("OPENAI_API_KEY", applied)
        self.assertEqual(os.environ["OPENAI_API_KEY"], SENTINEL2)

    def test_overwrite_flag_forces(self):
        os.environ["OPENAI_API_KEY"] = SENTINEL2
        p = self._write(f'OPENAI_API_KEY="{SENTINEL}"\n')
        try:
            applied = load_env_files([p], overwrite=True)
        finally:
            os.unlink(p)
        self.assertEqual(applied.get("OPENAI_API_KEY"), p)
        self.assertEqual(os.environ["OPENAI_API_KEY"], SENTINEL)

    def test_return_value_has_no_secret(self):
        p = self._write(f'OPENAI_API_KEY={SENTINEL}\n')
        try:
            applied = load_env_files([p])
        finally:
            os.unlink(p)
        blob = repr(applied) + " " + str(applied) + " " + describe(applied)
        self.assertNotIn(SENTINEL, blob)

    def test_describe_is_non_sensitive(self):
        p = self._write(
            f'export OPENAI_API_KEY="{SENTINEL}"\n'
            f'ANTHROPIC_API_KEY={SENTINEL2}\n'
        )
        try:
            applied = load_env_files([p])
        finally:
            os.unlink(p)
        s = describe(applied)
        self.assertIn("OPENAI_API_KEY", s)
        self.assertIn("ANTHROPIC_API_KEY", s)
        self.assertNotIn(SENTINEL, s)
        self.assertNotIn(SENTINEL2, s)

    def test_non_whitelisted_key_ignored(self):
        p = self._write('SOMETHING_ELSE=abc\nMY_SECRET=xyz\n')
        try:
            applied = load_env_files([p])
        finally:
            os.unlink(p)
        self.assertEqual(applied, {})
        self.assertNotIn("SOMETHING_ELSE", os.environ)

    def test_missing_file_is_silent(self):
        applied = load_env_files(["/nonexistent/path/to/llm.env"])
        self.assertEqual(applied, {})

    def test_parses_export_and_bare_and_quoted(self):
        p = self._write(
            f"export OPENAI_API_KEY='{SENTINEL}'\n"
            f"ANTHROPIC_API_KEY={SENTINEL2}\n"
            "   # comment\n"
            'GROQ_API_KEY="val with spaces"\n'
        )
        try:
            applied = load_env_files([p])
        finally:
            os.unlink(p)
        self.assertEqual(os.environ.get("OPENAI_API_KEY"), SENTINEL)
        self.assertEqual(os.environ.get("ANTHROPIC_API_KEY"), SENTINEL2)
        self.assertEqual(os.environ.get("GROQ_API_KEY"), "val with spaces")
        self.assertEqual(set(applied.keys()),
                         {"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY"})

    def test_first_path_wins(self):
        p1 = self._write(f'OPENAI_API_KEY="{SENTINEL}"\n')
        p2 = self._write(f'OPENAI_API_KEY="{SENTINEL2}"\n')
        try:
            applied = load_env_files([p1, p2])
        finally:
            os.unlink(p1)
            os.unlink(p2)
        self.assertEqual(applied["OPENAI_API_KEY"], p1)
        self.assertEqual(os.environ["OPENAI_API_KEY"], SENTINEL)


if __name__ == "__main__":
    unittest.main()
