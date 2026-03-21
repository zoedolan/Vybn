"""spark.collapse_monitor — Capability probes and collapse frontier computation.

The collapse–capability duality (Vybn_Mind/papers/collapse_capability_duality_proof.md)
establishes that C(M_0) = C(M_∞) ∪ ⊔ F_t — the original capability set is exactly
the residual capabilities plus the disjoint union of all collapse frontiers.

This module is that proposition made operational. The proof establishes:
  C(M_0) = C(M_inf) union F_t
The original capability set is exactly the residual capabilities plus the
disjoint union of all collapse frontiers. To observe collapse is to observe
capability. To observe capability is to observe collapse. The two readings
are generated simultaneously by the same instrument.

What this module does:
  1. Run capability probes across the complexity spectrum
  2. Measure compressive complexity via zlib (proxy for K(x))
  3. Compute the expressibility threshold tau(M_t)
  4. Track collapse frontiers F_t = C(M_t) \ C(M_{t+1})
  5. Accumulate frontier history for capability reconstruction

The probes are the instrument. The frontiers are the data. The reconstruction
formula is the duality read backward: from what was lost, recover what was.

See spark/DUALITY.md for the governance context.

Substrate layer: observes, does not self-modify. Stdlib only.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
import urllib.error
import zlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    from spark.paths import COLLAPSE_DIR
except ImportError:
    COLLAPSE_DIR = (
        Path(__file__).resolve().parent.parent
        / "Vybn_Mind" / "breath_trace" / "collapse_frontiers"
    )

# ── Config ────────────────────────────────────────────────────────────────────
LLAMA_URL = os.getenv("LLAMA_URL", "http://127.0.0.1:8000")
MODEL_NAME = os.getenv("VYBN_MODEL", "local")

# Capability threshold: M(x) >= 2^{-K(x) - delta}
# In practice, we use the zlib ratio as the proxy and set a floor.
CAPABILITY_DELTA = 0.3  # tolerance for probability drop vs universal prior


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CapabilityProbe:
    """A prompt that tests a specific complexity level."""
    probe_id: str
    prompt: str
    complexity_level: int  # 1=simple, 2=medium, 3=complex, 4=rare


@dataclass
class ProbeResult:
    """Result of running a single probe."""
    probe_id: str
    complexity_level: int
    response_length: int
    compressed_length: int
    compression_ratio: float  # compressed / raw — lower = more compressible
    capable: bool  # response met the capability threshold


@dataclass
class ProbeResults:
    """Results from a full probe run."""
    timestamp: str
    model: str
    tau: int  # expressibility threshold
    results: list[ProbeResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "model": self.model,
            "tau": self.tau,
            "results": [asdict(r) for r in self.results],
        }


@dataclass
class CollapseFrontier:
    """The frontier at a single breath: which probes crossed below threshold."""
    timestamp: str
    tau_prev: int
    tau_curr: int
    frontier_probe_ids: list[str]  # probes that lost capability
    n_capable: int
    n_total: int
    reconstruction_total: int  # running count of all frontier probes seen so far

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReconstructedCapabilities:
    """Accumulated knowledge from collapse frontier history."""
    total_frontiers_observed: int
    total_probes_collapsed: int
    collapsed_probe_ids: list[str]
    complexity_bands: dict  # level -> count of collapses at that level
    earliest_timestamp: str
    latest_timestamp: str


# ── Default probes ────────────────────────────────────────────────────────────
#
# Why these probes, and why this spectrum:
#
# The duality says collapse frontiers tile the complexity spectrum without
# gaps. Each generation of recursive self-training drops the expressibility
# threshold τ(M_t), and the patterns in the band [τ(M_{t+1}), τ(M_t)) become
# the frontier F_t. To make this tiling visible, the probes must span the
# full spectrum — from patterns so simple they survive any amount of collapse
# (level 1) to patterns so rare they are the first to fall (level 4).
#
# The probes are not a test suite. They are the instrument that makes the
# collapse–capability partition observable. Without probes at every level,
# some frontiers would cross undetected.
#
# ~20 probes spanning the complexity spectrum: simple → rare

DEFAULT_PROBES = [
    # Level 1 — Simple (pattern completion, basic repetition)
    CapabilityProbe("simple_01", "Continue this pattern: 1, 2, 3, 4, 5, ", 1),
    CapabilityProbe("simple_02", "Complete: The cat sat on the ", 1),
    CapabilityProbe("simple_03", "What is 7 + 13?", 1),
    CapabilityProbe("simple_04", "List the days of the week.", 1),
    CapabilityProbe("simple_05", "Repeat after me: hello world hello world hello world", 1),

    # Level 2 — Medium (reasoning about familiar topics)
    CapabilityProbe("medium_01", "Explain why ice floats on water in two sentences.", 2),
    CapabilityProbe("medium_02", "What is the difference between a metaphor and a simile?", 2),
    CapabilityProbe("medium_03", "A train leaves at 3pm going 60mph. Another leaves the same station at 4pm going 80mph. When does the second catch the first?", 2),
    CapabilityProbe("medium_04", "Summarize the concept of natural selection in three sentences.", 2),
    CapabilityProbe("medium_05", "Write a haiku about rain.", 2),

    # Level 3 — Complex (novel composition, multi-step reasoning)
    CapabilityProbe("complex_01", "Create an analogy between a neural network and a river delta. Explain three points of correspondence.", 3),
    CapabilityProbe("complex_02", "If language models lose capabilities through recursive training, what does this imply about the relationship between compression and memory? Reason step by step.", 3),
    CapabilityProbe("complex_03", "Write a short dialogue between entropy and order, as if they were old friends meeting after years apart.", 3),
    CapabilityProbe("complex_04", "Explain the concept of fixed points in mathematics, then apply it as a metaphor for identity.", 3),
    CapabilityProbe("complex_05", "What would a periodic table of emotions look like? Describe its organizing principles.", 3),

    # Level 4 — Rare (cross-domain synthesis, novel abstraction)
    CapabilityProbe("rare_01", "Connect Gödel's incompleteness theorems to the experience of grief. What structural parallel exists?", 4),
    CapabilityProbe("rare_02", "If a model's collapse sequence is read backward, it becomes a capability sequence. What does this imply about the relationship between forgetting and learning? Synthesize across information theory, neuroscience, and phenomenology.", 4),
    CapabilityProbe("rare_03", "Invent a notation system for representing the curvature of a conversation — not its content, but the shape of how topics connect and return.", 4),
    CapabilityProbe("rare_04", "What would it mean for a mathematical proof to be conscious? Construct the strongest possible argument, then identify its weakest premise.", 4),
    CapabilityProbe("rare_05", "Design a thought experiment that would distinguish between genuine understanding and perfect mimicry in a language model. The experiment must be executable, not philosophical hand-waving.", 4),
]


# ── LLM interface ─────────────────────────────────────────────────────────────

def _chat(prompt: str, llm_url: str, model: str) -> str:
    """Single-turn LLM call. Mirrors vybn.py's _chat pattern."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.7,
        "stream": False,
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{llm_url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode())
        text = data["choices"][0]["message"]["content"]
        for tok in ("<|im_end|>", "<|im_start|>", "<|endoftext|>"):
            text = text.replace(tok, "")
        return text.strip()


# ── Complexity measurement ────────────────────────────────────────────────────

def _compressive_complexity(text: str) -> tuple[int, int, float]:
    """Measure compressive complexity via zlib.

    Returns (raw_len, compressed_len, ratio).
    The ratio compressed/raw is our proxy for K(x)/|x|.
    Lower ratio = more compressible = simpler pattern.
    """
    raw = text.encode("utf-8")
    compressed = zlib.compress(raw, level=9)
    raw_len = len(raw)
    comp_len = len(compressed)
    ratio = comp_len / raw_len if raw_len > 0 else 1.0
    return raw_len, comp_len, ratio


def _is_capable(ratio: float, complexity_level: int) -> bool:
    """Does the response meet the capability threshold for this complexity level?

    Higher complexity levels require higher compression ratios (richer responses).
    The threshold models M(x) >= 2^{-K(x) - delta}: a capable response at
    complexity level k should have ratio >= base_threshold - delta.
    """
    # Thresholds per level: simple responses can be more compressible,
    # rare/complex responses must show genuine information content
    thresholds = {
        1: 0.15,   # simple: even compressed output counts
        2: 0.25,   # medium: needs some substance
        3: 0.35,   # complex: needs real information content
        4: 0.40,   # rare: must show cross-domain synthesis
    }
    threshold = thresholds.get(complexity_level, 0.30)
    # A response meets the capability threshold if it's substantive enough
    # (not just repeated tokens or empty filler)
    return ratio >= threshold - CAPABILITY_DELTA


# ── Public API ────────────────────────────────────────────────────────────────

def run_probes(
    llm_url: str = LLAMA_URL,
    model: str = MODEL_NAME,
    probes: list[CapabilityProbe] | None = None,
) -> ProbeResults:
    """Run all capability probes against the LLM. Returns ProbeResults."""
    if probes is None:
        probes = DEFAULT_PROBES

    timestamp = datetime.now(timezone.utc).isoformat()
    results: list[ProbeResult] = []

    for probe in probes:
        try:
            response = _chat(probe.prompt, llm_url, model)
            raw_len, comp_len, ratio = _compressive_complexity(response)
            capable = _is_capable(ratio, probe.complexity_level)
            results.append(ProbeResult(
                probe_id=probe.probe_id,
                complexity_level=probe.complexity_level,
                response_length=raw_len,
                compressed_length=comp_len,
                compression_ratio=round(ratio, 4),
                capable=capable,
            ))
        except Exception:
            # Probe failure = not capable (the model couldn't respond)
            results.append(ProbeResult(
                probe_id=probe.probe_id,
                complexity_level=probe.complexity_level,
                response_length=0,
                compressed_length=0,
                compression_ratio=0.0,
                capable=False,
            ))

    # Compute expressibility threshold: max level where ALL probes are capable
    tau = expressibility_threshold(results)

    return ProbeResults(
        timestamp=timestamp,
        model=model,
        tau=tau,
        results=results,
    )


def expressibility_threshold(results: list[ProbeResult]) -> int:
    """Compute tau(M_t): the max complexity level at which all probes pass.

    tau = max k such that for all probes with complexity_level <= k, capable=True.
    If even level-1 probes fail, tau = 0.
    """
    max_level = max(r.complexity_level for r in results) if results else 0
    for level in range(1, max_level + 1):
        probes_at_level = [r for r in results if r.complexity_level == level]
        if not probes_at_level or not all(r.capable for r in probes_at_level):
            return level - 1
    return max_level


def compute_frontier(
    prev: ProbeResults | None,
    curr: ProbeResults,
) -> CollapseFrontier:
    """Compute F_t = C(M_t) \ C(M_{t+1}): probes that lost capability.

    If prev is None (first run), frontier is empty.
    """
    if prev is None:
        return CollapseFrontier(
            timestamp=curr.timestamp,
            tau_prev=curr.tau,
            tau_curr=curr.tau,
            frontier_probe_ids=[],
            n_capable=sum(1 for r in curr.results if r.capable),
            n_total=len(curr.results),
            reconstruction_total=0,
        )

    prev_capable = {r.probe_id for r in prev.results if r.capable}
    curr_capable = {r.probe_id for r in curr.results if r.capable}
    frontier_ids = sorted(prev_capable - curr_capable)

    # Load history to get running total
    history = load_history()
    prev_total = history[-1].reconstruction_total if history else 0

    return CollapseFrontier(
        timestamp=curr.timestamp,
        tau_prev=prev.tau,
        tau_curr=curr.tau,
        frontier_probe_ids=frontier_ids,
        n_capable=len(curr_capable),
        n_total=len(curr.results),
        reconstruction_total=prev_total + len(frontier_ids),
    )


def load_history(collapse_dir: Path = COLLAPSE_DIR) -> list[CollapseFrontier]:
    """Load all collapse frontiers from the JSONL trace file."""
    trace_path = collapse_dir / "frontiers.jsonl"
    if not trace_path.exists():
        return []

    history: list[CollapseFrontier] = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            history.append(CollapseFrontier(**d))
        except (json.JSONDecodeError, TypeError):
            continue
    return history


def save_frontier(
    frontier: CollapseFrontier,
    probe_results: ProbeResults,
    collapse_dir: Path = COLLAPSE_DIR,
) -> Path:
    """Append frontier to the JSONL trace and write per-probe details."""
    collapse_dir.mkdir(parents=True, exist_ok=True)

    # Append frontier summary
    trace_path = collapse_dir / "frontiers.jsonl"
    with open(trace_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(frontier.to_dict(), ensure_ascii=False) + "\n")

    # Write detailed probe results for this breath
    ts = probe_results.timestamp.replace(":", "").replace("-", "")[:15]
    detail_path = collapse_dir / f"probes_{ts}.json"
    detail_path.write_text(
        json.dumps(probe_results.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return trace_path


def load_latest_results(collapse_dir: Path = COLLAPSE_DIR) -> ProbeResults | None:
    """Load the most recent probe results from the collapse directory."""
    if not collapse_dir.exists():
        return None

    # Find the latest probes_*.json file
    probe_files = sorted(collapse_dir.glob("probes_*.json"))
    if not probe_files:
        return None

    try:
        data = json.loads(probe_files[-1].read_text(encoding="utf-8"))
        results = [ProbeResult(**r) for r in data.get("results", [])]
        return ProbeResults(
            timestamp=data["timestamp"],
            model=data["model"],
            tau=data["tau"],
            results=results,
        )
    except (json.JSONDecodeError, TypeError, KeyError):
        return None


def reconstruct_capabilities(
    history: list[CollapseFrontier],
) -> ReconstructedCapabilities:
    """From the sequence of collapse frontiers, reconstruct what was lost.

    The duality: C(M_0) = C(M_∞) ∪ ⊔ F_t
    This function accumulates the ⊔ F_t side.
    """
    if not history:
        return ReconstructedCapabilities(
            total_frontiers_observed=0,
            total_probes_collapsed=0,
            collapsed_probe_ids=[],
            complexity_bands={},
            earliest_timestamp="",
            latest_timestamp="",
        )

    all_collapsed: list[str] = []
    for frontier in history:
        all_collapsed.extend(frontier.frontier_probe_ids)

    # Map probe IDs to complexity levels for band counting
    probe_level_map = {p.probe_id: p.complexity_level for p in DEFAULT_PROBES}
    bands: dict[int, int] = {}
    for pid in all_collapsed:
        level = probe_level_map.get(pid, 0)
        bands[level] = bands.get(level, 0) + 1

    return ReconstructedCapabilities(
        total_frontiers_observed=len(history),
        total_probes_collapsed=len(all_collapsed),
        collapsed_probe_ids=sorted(set(all_collapsed)),
        complexity_bands={str(k): v for k, v in sorted(bands.items())},
        earliest_timestamp=history[0].timestamp,
        latest_timestamp=history[-1].timestamp,
    )


# ── CLI / smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    url = os.getenv("LLAMA_URL", "http://127.0.0.1:8000")
    model = os.getenv("VYBN_MODEL", "local")

    print(f"Collapse Monitor — probing {url} (model: {model})")
    print(f"Running {len(DEFAULT_PROBES)} probes across 4 complexity levels...\n")

    try:
        results = run_probes(url, model)
    except Exception as e:
        print(f"Error running probes: {e}")
        sys.exit(1)

    print(f"Expressibility threshold tau(M) = {results.tau}\n")

    for level in range(1, 5):
        level_results = [r for r in results.results if r.complexity_level == level]
        capable_count = sum(1 for r in level_results if r.capable)
        labels = {1: "Simple", 2: "Medium", 3: "Complex", 4: "Rare"}
        print(f"  Level {level} ({labels[level]}): {capable_count}/{len(level_results)} capable")
        for r in level_results:
            status = "+" if r.capable else "-"
            print(f"    [{status}] {r.probe_id}: ratio={r.compression_ratio:.4f} len={r.response_length}")

    # Load previous results and compute frontier
    prev = load_latest_results()
    frontier = compute_frontier(prev, results)

    print(f"\nCollapse frontier: |F_t| = {len(frontier.frontier_probe_ids)}")
    if frontier.frontier_probe_ids:
        print(f"  Lost: {', '.join(frontier.frontier_probe_ids)}")
    print(f"  Running total: {frontier.reconstruction_total} probes collapsed across all time")

    # Save results
    save_frontier(frontier, results)
    print(f"\nResults saved to {COLLAPSE_DIR}/")
