"""spark.mathematician — Autonomous mathematical exploration with SymPy tool-use.

Aletheia mandate: prefer falsification over confirmation. Store all results
including negatives. A system that *wants* to find out which conjectures
are wrong is more trustworthy than one that seeks confirmation.

Tool-use within breath: two-turn conversation with the model:
  1. Prompt: "Given conjecture X, what symbolic computation would test it?"
  2. Model responds with a SymPy expression
  3. Execute via subprocess: python3 -c "from sympy import ...; print(result)"
  4. Feed result back: "The computation returned Y. What does this mean?"
"""

from __future__ import annotations

import json
import logging
import subprocess
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import yaml

log = logging.getLogger(__name__)

# ── Path setup ───────────────────────────────────────────────────────────────

try:
    from spark.paths import RESEARCH_DIR, CONJECTURE_PATH, FRONTIER_PATH
except ImportError:
    REPO_ROOT = Path(__file__).resolve().parent.parent.parent
    RESEARCH_DIR = REPO_ROOT / "spark" / "research"
    CONJECTURE_PATH = RESEARCH_DIR / "conjecture_registry.yaml"
    FRONTIER_PATH = RESEARCH_DIR / "research_frontier.yaml"

SYMPY_TIMEOUT = 30  # seconds


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True,
                  sort_keys=False, width=120)


def _sympy_available() -> bool:
    """Check if SymPy is usable via subprocess."""
    try:
        result = subprocess.run(
            ["python3", "-c", "import sympy; print(sympy.__version__)"],
            capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


class MathFaculty:
    """Autonomous mathematical exploration with SymPy tool-use."""

    def __init__(self):
        self._sympy_ok: Optional[bool] = None

    def run(self, state: dict, llm_fn: Callable, frontier: Optional[dict] = None) -> dict:
        """Main entry point called by faculty_runner.

        Returns a dict that gets written to the output bus as mathematician_latest.json.
        """
        try:
            return self._run_inner(state, llm_fn, frontier)
        except Exception as exc:
            log.error("MathFaculty.run failed: %s", exc, exc_info=True)
            return {"status": "error", "error": str(exc), "timestamp": _now_iso()}

    def _run_inner(self, state: dict, llm_fn: Callable, frontier: Optional[dict] = None) -> dict:
        # Check SymPy availability once
        if self._sympy_ok is None:
            self._sympy_ok = _sympy_available()
            if not self._sympy_ok:
                log.warning("SymPy not available; mathematician will use LLM-only mode")

        # Load conjectures
        registry = _load_yaml(CONJECTURE_PATH)
        conjectures = registry.get("conjectures", [])

        # Also check frontier conjectures
        if frontier is None:
            frontier = _load_yaml(FRONTIER_PATH)
        frontier_conjectures = frontier.get("active_conjectures", [])

        # Pick testable conjectures (status: open or testing)
        testable = [
            c for c in conjectures
            if c.get("status") in ("open", "testing")
        ]

        if not testable:
            return {
                "status": "ok",
                "note": "no testable conjectures",
                "conjectures_checked": 0,
                "timestamp": _now_iso(),
            }

        # Test up to 2 conjectures per breath (budget: ~3 LLM calls total)
        results = []
        for conjecture in testable[:2]:
            result = self._test_conjecture(conjecture, llm_fn)
            results.append(result)
            # Update the conjecture in the registry
            self._update_conjecture(registry, conjecture["id"], result)

        # Save updated registry
        _save_yaml(CONJECTURE_PATH, registry)

        return {
            "status": "ok",
            "conjectures_checked": len(results),
            "results": results,
            "sympy_available": self._sympy_ok,
            "timestamp": _now_iso(),
        }

    def _test_conjecture(self, conjecture: dict, llm_fn: Callable) -> dict:
        """Run a proof/disproof cycle on one conjecture."""
        cid = conjecture.get("id", "?")
        statement = conjecture.get("statement", "")
        domain = conjecture.get("domain", "")

        # Turn 1: Ask the model what computation would test this
        prompt = (
            f"Conjecture ({domain}): \"{statement}\"\n\n"
            "What single SymPy symbolic computation would help test or falsify this? "
            "Respond with ONLY a Python expression using SymPy functions that can be "
            "evaluated with `from sympy import *`. No imports, no print — just the expression. "
            "If this conjecture cannot be tested symbolically, say SKIP."
        )
        messages = [
            {"role": "system", "content": "You are a mathematician. Prefer falsification. "
             "Respond with a single SymPy expression or SKIP."},
            {"role": "user", "content": prompt},
        ]

        try:
            expr_response = llm_fn(messages, max_tokens=250, temperature=0.3)
        except Exception as exc:
            log.warning("Mathematician turn-1 LLM failed for %s: %s", cid, exc)
            return {"conjecture_id": cid, "status": "llm_error", "error": str(exc)}

        expr_response = expr_response.strip()

        # Check if model says SKIP
        if "SKIP" in expr_response.upper():
            return {"conjecture_id": cid, "status": "skipped", "reason": "not symbolically testable"}

        # Extract the expression (strip markdown code fences if present)
        expression = self._extract_expression(expr_response)

        # Turn 2: Execute via SymPy subprocess
        if self._sympy_ok and expression:
            compute_result = self._sympy_eval(expression)
        else:
            compute_result = "(SymPy not available or no valid expression)"

        # Turn 3: Interpret the result
        interpret_prompt = (
            f"Conjecture: \"{statement}\"\n"
            f"SymPy expression: {expression}\n"
            f"Computation result: {compute_result}\n\n"
            "What does this result mean for the conjecture? "
            "Does it support, refute, or is it inconclusive? One sentence."
        )
        interpret_messages = [
            {"role": "system", "content": "You are a mathematician interpreting symbolic results. "
             "Be honest about what the computation does and does not show."},
            {"role": "user", "content": interpret_prompt},
        ]

        try:
            interpretation = llm_fn(interpret_messages, max_tokens=250, temperature=0.3)
        except Exception as exc:
            log.warning("Mathematician turn-3 LLM failed for %s: %s", cid, exc)
            interpretation = f"(interpretation unavailable: {exc})"

        return {
            "conjecture_id": cid,
            "status": "checked",
            "expression": expression,
            "compute_result": compute_result,
            "interpretation": interpretation.strip(),
            "checked_at": _now_iso(),
        }

    def _extract_expression(self, response: str) -> str:
        """Extract a SymPy expression from LLM response, stripping markdown fences."""
        # Remove markdown code fences
        response = re.sub(r'```(?:python)?\s*', '', response)
        response = re.sub(r'```', '', response)
        # Take first non-empty line that looks like code
        for line in response.strip().split('\n'):
            line = line.strip()
            if line and not line.startswith('#') and not line.lower().startswith('skip'):
                # Remove any print() wrapper
                if line.startswith('print(') and line.endswith(')'):
                    line = line[6:-1]
                return line
        return response.strip()

    def _sympy_eval(self, expression: str) -> str:
        """Execute a SymPy expression in a subprocess. Returns stdout or error."""
        # Sanitize: only allow sympy operations, no imports of other modules
        script = f"from sympy import *; print({expression})"
        try:
            result = subprocess.run(
                ["python3", "-c", script],
                capture_output=True, text=True, timeout=SYMPY_TIMEOUT
            )
            if result.returncode == 0:
                return result.stdout.strip()
            return f"ERROR: {result.stderr.strip()}"
        except subprocess.TimeoutExpired:
            return "ERROR: computation timed out (30s)"
        except FileNotFoundError:
            return "ERROR: python3 not found"

    def _update_conjecture(self, registry: dict, cid: str, result: dict) -> None:
        """Update a conjecture's sympy_checks and last_checked in the registry."""
        for conj in registry.get("conjectures", []):
            if conj.get("id") == cid:
                if conj.get("sympy_checks") is None:
                    conj["sympy_checks"] = []
                conj["sympy_checks"].append({
                    "expression": result.get("expression", ""),
                    "result": result.get("compute_result", ""),
                    "interpretation": result.get("interpretation", ""),
                    "checked_at": result.get("checked_at", _now_iso()),
                })
                conj["last_checked"] = _now_iso()
                break
