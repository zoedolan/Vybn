#!/usr/bin/env python3
"""auto_debug.py - Core debugging engine for Vybn Spark.

Analyzes Python files for common errors and attempts automated fixes.
Designed to work with watchdog.py for continuous monitoring.
"""
import ast
import re
import sys
import logging
import subprocess
import traceback
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("auto_debug")


@dataclass
class DiagnosticResult:
  """Result of analyzing a single file."""
  path: str
  errors: list = field(default_factory=list)
  warnings: list = field(default_factory=list)
  fixes_applied: list = field(default_factory=list)
  fix_failed: list = field(default_factory=list)

  @property
  def ok(self):
    return not self.errors and not self.fix_failed

  def summary(self):
    parts = [f"  {self.path}:"]
    for e in self.errors:
      parts.append(f"    ERROR: {e}")
    for w in self.warnings:
      parts.append(f"    WARN:  {w}")
    for f in self.fixes_applied:
      parts.append(f"    FIXED: {f}")
    for f in self.fix_failed:
      parts.append(f"    FAIL:  {f}")
    return "\n".join(parts)


class AutoDebugger:
  """Analyzes and fixes Python source files."""

  def __init__(self, repo_root: Optional[str] = None):
    self.repo_root = Path(repo_root) if repo_root else Path.cwd()
    self.results: list[DiagnosticResult] = []

  # ---- public API ----

  def check_file(self, filepath: str, autofix: bool = True) -> DiagnosticResult:
    """Run all checks on a single file. Returns DiagnosticResult."""
    path = Path(filepath)
    result = DiagnosticResult(path=str(path))

    if not path.exists():
      result.errors.append("File not found")
      self.results.append(result)
      return result

    source = path.read_text(encoding="utf-8", errors="replace")

    # Phase 1: syntax check
    self._check_syntax(source, result)

    # Phase 2: import checks
    self._check_imports(source, result)

    # Phase 3: common anti-patterns
    self._check_patterns(source, result)

    # Phase 4: autofix if requested
    if autofix and (result.errors or result.warnings):
      source = self._apply_fixes(source, path, result)

    self.results.append(result)
    return result

  def check_directory(self, directory: str = ".", autofix: bool = True) -> list:
    """Check all .py files under *directory*."""
    root = Path(directory)
    results = []
    for py in sorted(root.rglob("*.py")):
      # skip venv / __pycache__
      parts = py.parts
      if any(p in (".venv", "venv", "__pycache__", ".git", "node_modules") for p in parts):
        continue
      results.append(self.check_file(str(py), autofix=autofix))
    return results

  # ---- internal checks ----

  def _check_syntax(self, source: str, result: DiagnosticResult):
    """Try to parse the file as Python AST."""
    try:
      ast.parse(source)
    except SyntaxError as exc:
      result.errors.append(
        f"SyntaxError at line {exc.lineno}: {exc.msg}"
      )

  def _check_imports(self, source: str, result: DiagnosticResult):
    """Detect imports that are likely to fail at runtime."""
    try:
      tree = ast.parse(source)
    except SyntaxError:
      return  # already reported

    for node in ast.walk(tree):
      if isinstance(node, ast.Import):
        for alias in node.names:
          self._probe_import(alias.name, result)
      elif isinstance(node, ast.ImportFrom):
        if node.module:
          self._probe_import(node.module, result)

  def _probe_import(self, module_name: str, result: DiagnosticResult):
    """Check if a top-level module can be imported."""
    top = module_name.split(".")[0]
    # skip stdlib / known-good
    skip = {
      "os", "sys", "re", "ast", "json", "time", "math",
      "logging", "pathlib", "subprocess", "typing",
      "dataclasses", "collections", "functools",
      "hashlib", "datetime", "threading", "socket",
      "traceback", "io", "abc", "enum", "copy",
      "argparse", "textwrap", "shutil", "tempfile",
      "unittest", "contextlib", "itertools", "signal",
      "uuid", "base64", "http", "urllib", "importlib",
    }
    if top in skip:
      return
    try:
      __import__(top)
    except ImportError:
      result.warnings.append(f"Module '{top}' not installed")

  def _check_patterns(self, source: str, result: DiagnosticResult):
    """Check for common anti-patterns."""
    lines = source.splitlines()
    for i, line in enumerate(lines, 1):
      stripped = line.rstrip()
      # mixed tabs and spaces
      if "\t" in line and "    " in line:
        result.warnings.append(f"Line {i}: mixed tabs and spaces")
      # bare except
      if re.match(r"^\s*except\s*:", stripped):
        result.warnings.append(f"Line {i}: bare except clause")
      # mutable default arg
      if re.match(r"^\s*def\s+\w+\(.*=\s*\[\]", stripped):
        result.warnings.append(f"Line {i}: mutable default argument []")
      if re.match(r"^\s*def\s+\w+\(.*=\s*\{\}", stripped):
        result.warnings.append(f"Line {i}: mutable default argument {{}}")

  # ---- autofix engine ----

  def _apply_fixes(self, source: str, path: Path, result: DiagnosticResult) -> str:
    """Attempt to fix common issues in-place."""
    original = source

    # Fix 1: tabs -> spaces
    if any("mixed tabs" in w for w in result.warnings):
      source = source.expandtabs(4)
      result.fixes_applied.append("Converted tabs to 4 spaces")

    # Fix 2: trailing whitespace
    cleaned_lines = [line.rstrip() for line in source.splitlines()]
    if cleaned_lines != source.splitlines():
      source = "\n".join(cleaned_lines) + "\n"
      result.fixes_applied.append("Removed trailing whitespace")

    # Fix 3: missing newline at EOF
    if source and not source.endswith("\n"):
      source += "\n"
      result.fixes_applied.append("Added newline at end of file")

    # Fix 4: attempt to install missing modules
    for w in list(result.warnings):
      m = re.match(r"Module '(\w+)' not installed", w)
      if m:
        mod = m.group(1)
        if self._try_install(mod):
          result.fixes_applied.append(f"Installed missing module '{mod}'")
          result.warnings.remove(w)
        else:
          result.fix_failed.append(f"Could not install '{mod}'")

    # Fix 5: indentation repair for simple cases
    if any("SyntaxError" in e and "indent" in e.lower() for e in result.errors):
      repaired = self._repair_indentation(source)
      if repaired != source:
        try:
          ast.parse(repaired)
          source = repaired
          result.fixes_applied.append("Repaired indentation")
          result.errors = [
            e for e in result.errors
            if "SyntaxError" not in e or "indent" not in e.lower()
          ]
        except SyntaxError:
          result.fix_failed.append("Indentation repair did not resolve syntax error")

    # write back if changed
    if source != original:
      path.write_text(source, encoding="utf-8")
      log.info("Wrote fixes to %s", path)

    return source

  def _try_install(self, module: str) -> bool:
    """Try pip-installing a module. Returns True on success."""
    # map common module names to pip packages
    pip_map = {
      "cv2": "opencv-python",
      "PIL": "Pillow",
      "sklearn": "scikit-learn",
      "yaml": "pyyaml",
      "bs4": "beautifulsoup4",
      "attr": "attrs",
      "dotenv": "python-dotenv",
      "websocket": "websocket-client",
    }
    pkg = pip_map.get(module, module)
    try:
      subprocess.check_call(
        [sys.executable, "-m", "pip", "install", pkg],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=120,
      )
      return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
      return False

  def _repair_indentation(self, source: str) -> str:
    """Best-effort indentation repair.

    Strategy: normalize all leading whitespace to multiples of 4 spaces,
    adjusting lines that don't align to the nearest valid indent level.
    """
    lines = source.splitlines(True)
    fixed = []
    indent_stack = [0]

    for line in lines:
      stripped = line.lstrip()
      if not stripped or stripped.startswith("#"):
        fixed.append(line)
        continue

      current_indent = len(line) - len(stripped)
      # round to nearest multiple of 4
      rounded = round(current_indent / 4) * 4

      # dedent keywords
      dedent_kw = ("else:", "elif ", "except", "except:",
                   "finally:", "except ")
      if any(stripped.startswith(kw) for kw in dedent_kw):
        if indent_stack and rounded >= indent_stack[-1] and len(indent_stack) > 1:
          rounded = indent_stack[-2] if len(indent_stack) >= 2 else 0

      fixed.append(" " * rounded + stripped)

      # track indent for next line
      if stripped.endswith(":") and not stripped.startswith("#"):
        indent_stack.append(rounded + 4)
      else:
        while indent_stack and indent_stack[-1] > rounded:
          indent_stack.pop()
        if not indent_stack or indent_stack[-1] != rounded:
          indent_stack.append(rounded)

    return "".join(fixed)


def run_check(path: str = ".", autofix: bool = True) -> list:
  """Convenience entry point."""
  debugger = AutoDebugger()
  if Path(path).is_file():
    return [debugger.check_file(path, autofix=autofix)]
  return debugger.check_directory(path, autofix=autofix)


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  import argparse
  parser = argparse.ArgumentParser(description="Vybn Auto-Debugger")
  parser.add_argument("path", nargs="?", default=".",
                      help="File or directory to check")
  parser.add_argument("--no-fix", action="store_true",
                      help="Report only, don't auto-fix")
  args = parser.parse_args()
  results = run_check(args.path, autofix=not args.no_fix)
  for r in results:
    if not r.ok or r.fixes_applied:
      print(r.summary())
  total_err = sum(len(r.errors) for r in results)
  total_fix = sum(len(r.fixes_applied) for r in results)
  total_fail = sum(len(r.fix_failed) for r in results)
  print(f"\n--- {len(results)} files checked | "
        f"{total_err} errors | {total_fix} fixed | {total_fail} unfixable ---")
  sys.exit(1 if total_err or total_fail else 0)
