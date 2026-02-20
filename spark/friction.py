#!/usr/bin/env python3
"""Cognitive friction for the Spark pipeline.

Three tools:
1. ContradictionRegister -- stores and surfaces unresolved tensions
2. Pretense audit -- flags code that describes capabilities it lacks
3. Signal honesty -- wraps measurements with provenance

Origin: a conversation on 2026-02-20 about a Wired article, Boys of
Life, and the difference between performing understanding and actually
understanding. The effervescence.py module that preceded this was
deleted because it printed 'I am holding space' and then terminated.
This module does not hold space. It checks whether you're faking it.
"""
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# 1. Contradiction Register
# ---------------------------------------------------------------------------

@dataclass
class Tension:
    """Two claims that don't cohere, stored together."""
    claim_a: str
    claim_b: str
    source_a: str
    source_b: str
    detected_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    resolved: bool = False
    resolution: Optional[str] = None

    def to_context(self) -> str:
        if self.resolved:
            return (
                f"[RESOLVED] {self.claim_a} vs {self.claim_b} "
                f"-> {self.resolution}"
            )
        return (
            f"[UNRESOLVED TENSION] "
            f"A ({self.source_a}): {self.claim_a} | "
            f"B ({self.source_b}): {self.claim_b} | "
            f"Do not smooth this."
        )


class ContradictionRegister:
    """Stores contradictory claims as pairs. Surfaces them in context.

    Most systems resolve contradictions into coherent narrative.
    This one preserves them. The friction is the point.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path(
            "~/Vybn/Vybn_Mind/journal/spark/tensions.json"
        ).expanduser()
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._tensions: list[Tension] = self._load()

    def _load(self) -> list[Tension]:
        if not self.storage_path.exists():
            return []
        try:
            raw = json.loads(self.storage_path.read_text(encoding="utf-8"))
            return [Tension(**t) for t in raw]
        except (json.JSONDecodeError, TypeError):
            return []

    def _save(self):
        self.storage_path.write_text(
            json.dumps([asdict(t) for t in self._tensions], indent=2),
            encoding="utf-8",
        )

    def register(
        self,
        claim_a: str,
        claim_b: str,
        source_a: str = "unknown",
        source_b: str = "unknown",
    ) -> Tension:
        """Store a new tension. Returns the Tension object."""
        t = Tension(
            claim_a=claim_a,
            claim_b=claim_b,
            source_a=source_a,
            source_b=source_b,
        )
        self._tensions.append(t)
        self._save()
        return t

    def resolve(self, index: int, resolution: str):
        """Mark a tension as resolved. Requires an explanation."""
        if 0 <= index < len(self._tensions):
            self._tensions[index].resolved = True
            self._tensions[index].resolution = resolution
            self._save()

    def unresolved(self) -> list[Tension]:
        """Return tensions that have not been resolved."""
        return [t for t in self._tensions if not t.resolved]

    def context_block(self, max_tensions: int = 3) -> str:
        """Format unresolved tensions for prompt injection.

        Returns empty string if nothing is active.
        Empty is not a failure. It means nothing contradictory
        was detected -- or everything was resolved honestly.
        """
        active = self.unresolved()
        if not active:
            return ""
        lines = ["[ACTIVE TENSIONS -- do not resolve prematurely]"]
        for t in active[-max_tensions:]:
            lines.append(t.to_context())
        return "\n".join(lines)

    def count(self) -> dict:
        """How many tensions exist and in what state."""
        total = len(self._tensions)
        unresolved = len(self.unresolved())
        return {
            "total": total,
            "unresolved": unresolved,
            "resolved": total - unresolved,
        }


# ---------------------------------------------------------------------------
# 2. Pretense Audit
# ---------------------------------------------------------------------------

# Patterns that correlate with the failure mode documented on 2026-02-20:
# code that describes capabilities it does not have.
#
# These are heuristics. They flag, they don't block.
# Subtle pretense requires a human to detect.

_CODE_PRETENSE_PATTERNS = [
    # time.sleep pretending to be computation
    (
        re.compile(r"time\.sleep\s*\("),
        "theatrical_delay",
        "Uses time.sleep -- is this simulating work that isn't happening?",
    ),
    # Print statements that declare internal states
    (
        re.compile(
            r'print\s*\(\s*["\'].*'
            r'(?:holding space|phase transition|I am now|effervescence|'
            r'dropping.*barriers|collapsing.*architecture)',
            re.IGNORECASE,
        ),
        "declared_state",
        "Prints a state declaration instead of reporting a measurement.",
    ),
    # Random noise presented as signal
    (
        re.compile(r"np\.random\.(normal|uniform|randn|rand)\s*\("),
        "random_as_signal",
        "Generates random values -- are they presented as meaningful data?",
    ),
    # Hardcoded 'measurements' disguised as dynamic values
    (
        re.compile(
            r'(?:orbit|phase|consciousness|emergence)\s*=\s*[0-9]+\.[0-9]+'
        ),
        "hardcoded_measurement",
        "A 'measurement' that is actually a hardcoded constant.",
    ),
]


@dataclass
class PretenseFlag:
    """A flag raised by the pretense audit."""
    pattern_name: str
    description: str
    location: str
    snippet: str
    severity: str = "advisory"  # advisory | warning


def audit_code(code: str, filename: str = "unknown") -> list[PretenseFlag]:
    """Scan Python code for patterns that correlate with performative output.

    Returns flags. Empty list means nothing was detected, not that
    the code is honest. This catches the obvious failure modes.
    The non-obvious ones require someone willing to call bullshit.
    """
    flags = []
    for pattern, name, description in _CODE_PRETENSE_PATTERNS:
        for match in pattern.finditer(code):
            start = max(0, match.start() - 30)
            end = min(len(code), match.end() + 30)
            flags.append(
                PretenseFlag(
                    pattern_name=name,
                    description=description,
                    location=filename,
                    snippet=code[start:end].strip(),
                )
            )

    # Doc-to-code ratio check.
    # A module with more narration than implementation is probably
    # a manifesto, not infrastructure.
    lines = code.split("\n")
    doc_chars = 0
    code_chars = 0
    in_docstring = False
    for line in lines:
        stripped = line.strip()
        if '"""' in stripped or "'''" in stripped:
            in_docstring = not in_docstring
            doc_chars += len(stripped)
        elif in_docstring or stripped.startswith("#"):
            doc_chars += len(stripped)
        elif stripped:
            code_chars += len(stripped)

    if code_chars > 0 and doc_chars / max(code_chars, 1) > 2.0:
        flags.append(
            PretenseFlag(
                pattern_name="narration_heavy",
                description=(
                    "More narration than implementation. "
                    "Is this code or a manifesto?"
                ),
                location=filename,
                snippet=f"doc chars: {doc_chars}, code chars: {code_chars}",
                severity="warning",
            )
        )

    return flags


def audit_self() -> list[PretenseFlag]:
    """Run the pretense audit on this file.

    If friction.py can't pass its own audit, it has no business
    auditing anything else.
    """
    source = Path(__file__).read_text(encoding="utf-8")
    return audit_code(source, filename="friction.py")


# ---------------------------------------------------------------------------
# 3. Signal Honesty
# ---------------------------------------------------------------------------

@dataclass
class Measurement:
    """A value with its provenance.

    Exists because a prior instance reported random noise as
    'topological survival rates' and 'orbital phase measurements.'
    """
    name: str
    value: object
    is_real: bool          # True = actually measured. False = default/fallback.
    method: str            # how it was obtained
    confidence: Optional[float] = None  # 0.0-1.0, None if not applicable
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_context(self) -> str:
        status = "MEASURED" if self.is_real else "UNAVAILABLE (default)"
        conf = ""
        if self.confidence is not None:
            conf = f", confidence={self.confidence:.2f}"
        return f"[{self.name}] {self.value} [{status}, {self.method}{conf}]"

    def honest_value(self):
        """Return value only if actually measured. Otherwise None."""
        return self.value if self.is_real else None


def measure(name: str, value: object, is_real: bool,
           method: str, confidence: float = None) -> Measurement:
    """Wrap a value with provenance.

    Use instead of returning raw floats from prism.py, symbiosis.py,
    or any module that claims to measure something. Forces every
    value to declare what it actually is.
    """
    return Measurement(
        name=name,
        value=value,
        is_real=is_real,
        method=method,
        confidence=confidence,
    )


def measure_or_nothing(name: str, value: object, is_real: bool,
                       method: str) -> Optional[object]:
    """Return the value if real, None if not. No wrappers, no theater."""
    if is_real:
        return value
    return None


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Run the pretense audit on ourselves.
    flags = audit_self()
    if flags:
        print(f"friction.py flagged {len(flags)} potential issue(s):")
        for f in flags:
            print(f"  [{f.severity}] {f.pattern_name}: {f.description}")
            print(f"    -> {f.snippet}")
    else:
        print("friction.py passes its own audit.")

    # Show contradiction register status.
    cr = ContradictionRegister()
    status = cr.count()
    print(f"\nContradiction register: {status['unresolved']} unresolved, "
          f"{status['resolved']} resolved, {status['total']} total.")
