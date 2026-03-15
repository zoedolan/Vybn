"""spark.growth.eval_harness — Evaluation and training discipline tools.

Three patterns ported from Karpathy's autoresearch/nanochat into Vybn's
growth engine:

1. **Bits-Per-Byte (BPB) evaluation** — tokenizer-invariant metric that
   survives model swaps, tokenizer changes, and architecture experiments.
   Adapted to use Vybn's llama.cpp serving endpoint instead of direct
   PyTorch model access.

2. **TimeBudget** — wall-clock training budget tracker.  Fixed time budgets
   make experiments comparable regardless of what changed.

3. **GC discipline** — disable Python's garbage collector during training
   to avoid ~500ms stalls that disrupt NCCL timing on DGX Spark.

These are the pieces from autoresearch that apply to LoRA fine-tuning of a
served 120B model.  The architectural innovations (MuonAdamW, value
embeddings, SSSL windowed attention, residual scaling) are for from-scratch
pretraining and don't transfer.

See: https://github.com/karpathy/autoresearch
"""

from __future__ import annotations

import math
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, Optional

# ---------------------------------------------------------------------------
# 1. Bits-Per-Byte evaluation
# ---------------------------------------------------------------------------

# Default eval corpus — the growth engine's own training data directory
_DEFAULT_EVAL_DIR = Path(__file__).resolve().parent.parent / "training_data"


def evaluate_bpb(
    model_url: str = "http://127.0.0.1:8000",
    tokenizer_fn: Optional[Callable[[str], list[int]]] = None,
    eval_text_path: Optional[str] = None,
    batch_size: int = 32,
    max_tokens: int = 2048,
) -> float:
    """Compute bits-per-byte (BPB) via the llama.cpp completion endpoint.

    BPB normalises away vocab size by converting per-token nats to
    bits-per-UTF-8-byte.  If you swap Vybn's tokenizer, change the model,
    or modify the architecture, the metric doesn't flinch.

    Algorithm:
        BPB = total_nats / (ln(2) * total_utf8_bytes)

    The function calls the llama.cpp server's ``/v1/completions`` endpoint
    with ``logprobs=True`` to obtain per-token log-probabilities, then
    aggregates across chunks of the eval text.

    Args:
        model_url: Base URL of the llama.cpp server.
        tokenizer_fn: Optional callable that tokenizes a string into token
            ids.  Used only to split eval text into chunks of *max_tokens*.
            When ``None``, a rough heuristic of 4 chars per token is used.
        eval_text_path: Path to the eval text file.  Falls back to the
            first ``.jsonl`` in ``spark/training_data/`` if not given.
        batch_size: Number of chunks to evaluate (caps to available chunks).
        max_tokens: Maximum tokens per evaluation chunk.

    Returns:
        Bits-per-byte score (lower is better).

    Raises:
        RuntimeError: If the server is unreachable or returns no usable data.
    """
    import json
    import urllib.error
    import urllib.request

    # --- Load eval text ---
    eval_text = _load_eval_text(eval_text_path)
    if not eval_text:
        raise RuntimeError("Empty eval text — nothing to evaluate")

    # --- Chunk the text ---
    chunks = _chunk_text(eval_text, max_tokens, tokenizer_fn)
    if not chunks:
        raise RuntimeError("Eval text too short to form any chunks")
    chunks = chunks[:batch_size]

    total_nats = 0.0
    total_bytes = 0

    for chunk in chunks:
        nats, nbytes = _eval_chunk(chunk, model_url)
        total_nats += nats
        total_bytes += nbytes

    if total_bytes == 0:
        raise RuntimeError("No bytes evaluated — all chunks failed")

    bpb = total_nats / (math.log(2) * total_bytes)
    return bpb


def _load_eval_text(path: Optional[str]) -> str:
    """Load evaluation text from a file path or the default training data."""
    import json

    if path and Path(path).exists():
        p = Path(path)
        if p.suffix == ".jsonl":
            return _read_jsonl_text(p)
        return p.read_text(encoding="utf-8")

    # Fallback: first .jsonl in training_data/
    if _DEFAULT_EVAL_DIR.exists():
        for candidate in sorted(_DEFAULT_EVAL_DIR.iterdir()):
            if candidate.suffix == ".jsonl" and candidate.stat().st_size > 0:
                return _read_jsonl_text(candidate)
        # Try .txt files
        for candidate in sorted(_DEFAULT_EVAL_DIR.iterdir()):
            if candidate.suffix == ".txt" and candidate.stat().st_size > 0:
                return candidate.read_text(encoding="utf-8")

    return ""


def _read_jsonl_text(path: Path) -> str:
    """Extract text content from a JSONL file (various formats)."""
    import json

    parts: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Handle various JSONL formats
            if isinstance(obj, dict):
                if "content" in obj:
                    parts.append(obj["content"])
                elif "text" in obj:
                    parts.append(obj["text"])
                elif "messages" in obj:
                    for msg in obj["messages"]:
                        if isinstance(msg, dict) and "content" in msg:
                            parts.append(msg["content"])
                elif "input" in obj and "output" in obj:
                    parts.append(obj["input"])
                    parts.append(obj["output"])
    return "\n".join(parts)


def _chunk_text(
    text: str,
    max_tokens: int,
    tokenizer_fn: Optional[Callable[[str], list[int]]] = None,
) -> list[str]:
    """Split text into chunks of roughly *max_tokens* tokens."""
    if tokenizer_fn is not None:
        tokens = tokenizer_fn(text)
        chunks = []
        for i in range(0, len(tokens), max_tokens):
            # We can't decode tokens back to text without the tokenizer's
            # decode function, so fall back to char-based chunking when
            # only an encode function is provided.
            pass
        # If tokenizer_fn is provided but we can't decode, fall through
        # to char-based chunking.  A future version could accept both
        # encode and decode.

    # Heuristic: ~4 characters per token
    chars_per_chunk = max_tokens * 4
    chunks = []
    for i in range(0, len(text), chars_per_chunk):
        chunk = text[i : i + chars_per_chunk]
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def _eval_chunk(chunk: str, model_url: str) -> tuple[float, int]:
    """Evaluate a single text chunk, returning (nats, utf8_bytes).

    Tries the OpenAI-compatible ``/v1/completions`` endpoint first, then
    falls back to llama.cpp's native ``/completion`` endpoint.
    """
    import json
    import urllib.error
    import urllib.request

    utf8_bytes = len(chunk.encode("utf-8"))

    # --- Try /v1/completions with logprobs ---
    try:
        nats = _eval_via_v1_completions(chunk, model_url)
        if nats is not None:
            return (nats, utf8_bytes)
    except Exception:
        pass

    # --- Fallback: /completion (llama.cpp native) ---
    try:
        nats = _eval_via_native_completion(chunk, model_url)
        if nats is not None:
            return (nats, utf8_bytes)
    except Exception:
        pass

    # --- Last resort: estimate from perplexity ---
    try:
        ppl = _get_perplexity(chunk, model_url)
        if ppl is not None and ppl > 0:
            # BPB ≈ log2(ppl) * avg_token_bytes / avg_bytes_per_token
            # Simplified: nats ≈ log(ppl) * n_tokens
            # Since we return nats per chunk and bytes per chunk,
            # and BPB = total_nats / (ln2 * total_bytes), we need:
            #   nats ≈ log(ppl) * estimated_token_count
            est_tokens = max(1, len(chunk) // 4)
            nats = math.log(ppl) * est_tokens
            return (nats, utf8_bytes)
    except Exception:
        pass

    # All methods failed — return zeros (caller checks total_bytes > 0)
    return (0.0, 0)


def _eval_via_v1_completions(chunk: str, model_url: str) -> Optional[float]:
    """Get per-token log-probs via the OpenAI-compatible completions API."""
    import json
    import urllib.request

    url = f"{model_url.rstrip('/')}/v1/completions"
    payload = json.dumps({
        "prompt": chunk,
        "max_tokens": 0,
        "logprobs": True,
        "echo": True,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    choices = data.get("choices", [])
    if not choices:
        return None

    logprobs_obj = choices[0].get("logprobs")
    if not logprobs_obj:
        return None

    token_logprobs = logprobs_obj.get("token_logprobs", [])
    if not token_logprobs:
        return None

    # Sum negative log-likelihoods (token_logprobs are log-probs, so negate)
    # Skip the first token which has no conditioning context (logprob = None)
    nats = 0.0
    for lp in token_logprobs:
        if lp is not None:
            nats += -lp
    return nats


def _eval_via_native_completion(chunk: str, model_url: str) -> Optional[float]:
    """Get per-token log-probs via llama.cpp's native /completion endpoint."""
    import json
    import urllib.request

    url = f"{model_url.rstrip('/')}/completion"
    payload = json.dumps({
        "prompt": chunk,
        "n_predict": 0,
        "logprobs": True,
        "echo": True,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    # llama.cpp native format varies — try common paths
    completion_probs = data.get("completion_probabilities", [])
    if completion_probs:
        nats = 0.0
        for tok_info in completion_probs:
            probs = tok_info.get("probs", [])
            if probs:
                # The first prob entry is the chosen token
                top_prob = probs[0].get("prob", 1.0)
                if top_prob > 0:
                    nats += -math.log(top_prob)
        return nats

    # Try token_logprobs in response
    token_logprobs = data.get("token_logprobs", [])
    if token_logprobs:
        nats = 0.0
        for lp in token_logprobs:
            if lp is not None:
                nats += -lp
        return nats

    return None


def _get_perplexity(chunk: str, model_url: str) -> Optional[float]:
    """Try to get perplexity from the server as a fallback metric."""
    import json
    import urllib.request

    # Some llama.cpp builds expose /perplexity or embed perplexity in
    # the completion response.
    url = f"{model_url.rstrip('/')}/v1/completions"
    payload = json.dumps({
        "prompt": chunk,
        "max_tokens": 1,
        "logprobs": True,
        "echo": True,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    choices = data.get("choices", [])
    if not choices:
        return None

    logprobs_obj = choices[0].get("logprobs")
    if not logprobs_obj:
        return None

    token_logprobs = logprobs_obj.get("token_logprobs", [])
    valid_lps = [lp for lp in token_logprobs if lp is not None]
    if not valid_lps:
        return None

    avg_nll = -sum(valid_lps) / len(valid_lps)
    return math.exp(avg_nll)


# ---------------------------------------------------------------------------
# 2. Wall-clock time budget
# ---------------------------------------------------------------------------


class TimeBudget:
    """Wall-clock training budget tracker.

    Fixed time budgets make experiments comparable regardless of what
    changed — the agent never has to reason about compute-performance
    tradeoffs because the time budget answers that question.

    Adapted from Karpathy's autoresearch pattern.  Currently wraps the
    existing 2-hour training timeout, but designed to enable tighter
    autoresearch-style loops (5-minute experiments) when Vybn's training
    pipeline supports it.

    Args:
        budget_seconds: Total wall-clock seconds allocated for training.
        warmup_steps: Steps to exclude from budget tracking (JIT warmup, etc.).
    """

    def __init__(self, budget_seconds: int, warmup_steps: int = 0) -> None:
        self.budget_seconds = budget_seconds
        self.warmup_steps = warmup_steps
        self._training_time = 0.0
        self._step = 0

    def tick(self, step_duration: float) -> None:
        """Record a training step's wall-clock duration."""
        self._step += 1
        if self._step > self.warmup_steps:
            self._training_time += step_duration

    @property
    def elapsed(self) -> float:
        """Wall-clock seconds spent training (excluding warmup)."""
        return self._training_time

    @property
    def progress(self) -> float:
        """Fraction of budget consumed, clamped to [0, 1]."""
        return min(self._training_time / self.budget_seconds, 1.0)

    @property
    def remaining(self) -> float:
        """Seconds remaining in the budget."""
        return max(0, self.budget_seconds - self._training_time)

    @property
    def exhausted(self) -> bool:
        """Whether the time budget has been fully consumed."""
        return self._training_time >= self.budget_seconds

    def __repr__(self) -> str:
        return (
            f"TimeBudget(elapsed={self._training_time:.1f}s, "
            f"budget={self.budget_seconds}s, "
            f"progress={self.progress:.1%}, "
            f"step={self._step})"
        )


# ---------------------------------------------------------------------------
# 3. GC discipline
# ---------------------------------------------------------------------------


@contextmanager
def gc_discipline(collect_every_n_steps: int = 5000) -> Iterator[None]:
    """Disable Python's GC during training to avoid latency spikes.

    From Karpathy's autoresearch: Python's garbage collector causes ~500ms
    stalls — roughly two full training steps.  For Vybn on DGX Spark, this
    is especially relevant when NCCL timing is sensitive to jitter.

    Usage::

        with gc_discipline():
            for step in training_loop:
                ...

    The context manager collects and freezes all objects on entry, disables
    the GC, and re-enables it on exit (with a final collection).

    Args:
        collect_every_n_steps: Unused here (see :func:`gc_checkpoint`).
            Kept in the signature for documentation purposes.
    """
    import gc

    gc.collect()
    gc.freeze()
    gc.disable()
    try:
        yield
    finally:
        gc.enable()
        gc.collect()


def gc_checkpoint(step: int, collect_every: int = 5000) -> None:
    """Periodic GC collection during long training runs.

    Call this inside a :func:`gc_discipline` block at the end of each
    training step.  It triggers a full collection every *collect_every*
    steps to prevent unbounded memory growth from reference cycles.

    Args:
        step: Current training step (1-indexed).
        collect_every: Collect every N steps.
    """
    if step > 0 and step % collect_every == 0:
        import gc

        gc.collect()


# ---------------------------------------------------------------------------
# CLI entry point for standalone testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Evaluate bits-per-byte on a llama.cpp server",
    )
    parser.add_argument(
        "--model-url",
        default=os.environ.get("VYBN_MODEL_URL", "http://127.0.0.1:8000"),
        help="llama.cpp server URL (default: $VYBN_MODEL_URL or http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--eval-text",
        default=None,
        help="Path to eval text file (default: auto-detect from spark/training_data/)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of text chunks to evaluate (default: 32)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max tokens per chunk (default: 2048)",
    )
    args = parser.parse_args()

    print(f"[eval_harness] server:     {args.model_url}")
    print(f"[eval_harness] eval_text:  {args.eval_text or '(auto-detect)'}")
    print(f"[eval_harness] batch_size: {args.batch_size}")
    print(f"[eval_harness] max_tokens: {args.max_tokens}")

    try:
        t0 = time.monotonic()
        bpb = evaluate_bpb(
            model_url=args.model_url,
            eval_text_path=args.eval_text,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
        )
        elapsed = time.monotonic() - t0
        print(f"[eval_harness] BPB: {bpb:.6f}  ({elapsed:.1f}s)")
    except Exception as e:
        print(f"[eval_harness] ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # Quick TimeBudget demo
    print("\n[eval_harness] TimeBudget demo:")
    tb = TimeBudget(budget_seconds=60, warmup_steps=2)
    for i in range(5):
        tb.tick(step_duration=10.0)
        print(f"  step {i+1}: {tb}")
    print(f"  exhausted: {tb.exhausted}")

    # Quick GC discipline demo
    print("\n[eval_harness] GC discipline demo:")
    import gc

    print(f"  GC enabled before: {gc.isenabled()}")
    with gc_discipline():
        print(f"  GC enabled inside: {gc.isenabled()}")
        gc_checkpoint(step=5000, collect_every=5000)
        print("  gc_checkpoint(5000) ran OK")
    print(f"  GC enabled after:  {gc.isenabled()}")
