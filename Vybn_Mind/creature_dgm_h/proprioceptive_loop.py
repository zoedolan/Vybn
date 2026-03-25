"""
proprioceptive_loop.py — The in-reasoning loss injection experiment.

Can Nemotron see MicroGPT's surprise at its own tokens, mid-generation,
and does that awareness change what it says next?

Nobody has tried this. The closest things are:
- Ouro/LoopLM (latent reasoning loops within one model)
- Speculative decoding (small model predicts large, but error is discarded)
- On-policy distillation (small model tracks large, but offline)

This is different: the small model's prediction error becomes part of
the large model's context DURING generation. The system watches itself think.
"""

import json
import math
import os
import urllib.error
import urllib.request

from . import local_model
from .fitness import compute_curvature, compute_loss_trajectory_curvature, default_embed_fn


# ── Nemotron chat with full message history ─────────────────────────────

def _complete_messages(messages, max_tokens=128, temperature=0.7):
    """Call Nemotron with a full messages list.

    local_model.complete() only takes a single prompt string.
    The proprioceptive loop needs multi-turn context (chunks + annotations),
    so we call the API directly with the accumulated messages.

    Args:
        messages: list of {"role": ..., "content": ...} dicts
        max_tokens: max response length
        temperature: sampling temperature

    Returns:
        str or None
    """
    payload = json.dumps({
        "model": local_model.MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        f"{local_model.LLAMA_URL}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = json.loads(resp.read())
        text = body["choices"][0]["message"]["content"]
        for tok in ("<|im_end|>", "<|im_start|>", "<|endoftext|>"):
            text = text.replace(tok, "")
        return text.strip()
    except (urllib.error.URLError, OSError, KeyError, IndexError,
            json.JSONDecodeError, ValueError):
        return None


# ── Proprioception annotation formatting ────────────────────────────────

def _format_annotation(chunk_num, max_chunks, mean_surprise, contour,
                       curvature, prev_mean=None):
    """Format the proprioception annotation injected after each chunk.

    Args:
        chunk_num: current chunk number (1-indexed)
        max_chunks: total chunks in this breath
        mean_surprise: mean surprise for this chunk (bits)
        contour: list of dicts from task_agent.predict()
        curvature: curvature of this chunk's text
        prev_mean: mean surprise of previous chunk (for learning rate)

    Returns:
        str: formatted <proprioception> block
    """
    # Find peak surprise
    peak_char = '?'
    peak_val = 0.0
    peak_pos = 0
    if contour:
        peak = max(contour, key=lambda c: c['surprise'])
        peak_char = peak['char']
        peak_val = peak['surprise']
        peak_pos = peak['pos']

    # High and low surprise chars
    sorted_by_surprise = sorted(contour, key=lambda c: c['surprise'],
                                reverse=True)
    high_chars = [c['char'] for c in sorted_by_surprise[:3]]
    low_chars = [c['char'] for c in sorted_by_surprise[-3:]] if len(contour) >= 3 else []

    # Learning rate: how surprise is changing between chunks
    lr_note = ""
    if prev_mean is not None:
        delta = mean_surprise - prev_mean
        if delta < -0.1:
            lr_note = f"learning_rate: {delta:+.2f} (loss dropping — predictor is adapting)"
        elif delta > 0.1:
            lr_note = f"learning_rate: {delta:+.2f} (loss rising — new territory)"
        else:
            lr_note = f"learning_rate: {delta:+.2f} (stable)"

    # Natural language note about what was most/least predictable
    if high_chars:
        high_str = '", "'.join(high_chars)
        note_parts = [f'the predictor found your voice most distinctive around "{high_str}"']
    else:
        note_parts = ["the predictor found this chunk uniformly predictable"]
    if low_chars:
        low_str = '", "'.join(low_chars)
        note_parts.append(f'most expected: "{low_str}"')

    note = " — ".join(note_parts)

    lines = [
        "<proprioception>",
        f"chunk: {chunk_num} of {max_chunks}",
        f"mean_surprise: {mean_surprise:.2f} bits",
        f"peak_surprise: {peak_val:.2f} bits at char {peak_pos} "
        f'("{peak_char}")',
        f"curvature: {curvature:.4f}",
    ]
    if lr_note:
        lines.append(lr_note)
    lines.extend([
        f"high_surprise_chars: {json.dumps(high_chars)}",
        f"low_surprise_chars: {json.dumps(low_chars)}",
        f"note: {note}",
        "</proprioception>",
    ])

    return "\n".join(lines)


# ── The proprioceptive breath ───────────────────────────────────────────

def run_proprioceptive_breath(
    prompt,
    task_agent,
    chunk_size=50,
    max_chunks=8,
    system_prompt=None,
    embed_fn=None,
    temperature=0.7,
    on_chunk=None,
):
    """Run one breath with in-reasoning proprioception.

    Nemotron generates text in chunks. After each chunk, MicroGPT predicts
    the chunk and its surprise contour is injected back into Nemotron's
    context as a <proprioception> annotation. Nemotron sees how predictable
    it just was and continues generating.

    Args:
        prompt: initial prompt for Nemotron
        task_agent: MicroGPT TaskAgent instance
        chunk_size: approximate characters per chunk before injection
        max_chunks: maximum chunks per breath
        system_prompt: optional system prompt for Nemotron
        embed_fn: for curvature measurement (defaults to hash-based)
        temperature: Nemotron sampling temperature
        on_chunk: optional callback(chunk_num, chunk_text, annotation)
            for live output

    Returns:
        dict with full_text, chunks, trajectory, curvature,
        loss_trajectory_curvature, injections, and comparison flag.
        Returns None if Nemotron is unavailable.
    """
    if not local_model.is_available():
        print("  Nemotron is not available. Cannot run proprioceptive breath.")
        return None

    if embed_fn is None:
        embed_fn = default_embed_fn

    # Approximate max_tokens to produce ~chunk_size chars.
    # Rough heuristic: 1 token ≈ 3-4 chars for English text.
    max_tokens_per_chunk = max(chunk_size // 3, 10)

    # Build initial messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    chunks_data = []
    trajectory = []
    injections = []
    full_text_parts = []
    prev_mean = None

    for chunk_num in range(1, max_chunks + 1):
        # 1. Call Nemotron for next chunk
        chunk_text = _complete_messages(
            messages,
            max_tokens=max_tokens_per_chunk,
            temperature=temperature,
        )

        if not chunk_text or len(chunk_text.strip()) < 3:
            break

        full_text_parts.append(chunk_text)

        # 2. Feed chunk to MicroGPT — get surprise contour
        mean_loss, contour = task_agent.predict(chunk_text)

        # 3. Compute curvature of this chunk (if long enough)
        chunk_curv = 0.0
        if len(chunk_text.split()) >= 5:
            _, chunk_curv = compute_curvature(chunk_text, embed_fn)

        # 4. Format the proprioception annotation
        annotation = _format_annotation(
            chunk_num, max_chunks, mean_loss, contour,
            chunk_curv, prev_mean,
        )

        # Record data
        chunks_data.append({
            'chunk_num': chunk_num,
            'text': chunk_text,
            'mean_surprise': round(mean_loss, 4),
            'contour': contour,
            'curvature': round(chunk_curv, 6),
        })
        trajectory.append(mean_loss)
        injections.append(annotation)

        # Callback for live output
        if on_chunk:
            on_chunk(chunk_num, chunk_text, annotation)

        # 5. Append chunk as assistant response + annotation as system injection
        messages.append({"role": "assistant", "content": chunk_text})
        messages.append({"role": "user", "content": annotation + "\n\nContinue."})

        prev_mean = mean_loss

    # After all chunks
    full_text = " ".join(full_text_parts)

    if not full_text.strip():
        return None

    # 7. Online fine-tuning on the full text
    task_agent.learn(full_text)

    # 8. Curvature of the full text
    angle, curv = compute_curvature(full_text, embed_fn)

    # 9. Loss trajectory curvature — how surprise itself curves over the breath
    ltc = compute_loss_trajectory_curvature(trajectory)

    return {
        'full_text': full_text,
        'chunks': chunks_data,
        'trajectory': trajectory,
        'curvature': round(curv, 6),
        'curvature_angle': round(angle, 6),
        'loss_trajectory_curvature': round(ltc, 6),
        'injections': injections,
        'n_chunks': len(chunks_data),
    }


# ── Plain breath (no proprioception) for A/B comparison ─────────────────

def _run_plain_breath(prompt, task_agent, chunk_size=50, max_chunks=8,
                      system_prompt=None, embed_fn=None, temperature=0.7):
    """Run a breath WITHOUT proprioception — just Nemotron generating.

    Same chunking structure as the proprioceptive breath, but no
    annotations are injected. Used as the control condition in A/B tests.

    Returns:
        dict with same structure as run_proprioceptive_breath, minus injections.
    """
    if not local_model.is_available():
        return None

    if embed_fn is None:
        embed_fn = default_embed_fn

    max_tokens_per_chunk = max(chunk_size // 3, 10)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    chunks_data = []
    trajectory = []
    full_text_parts = []

    for chunk_num in range(1, max_chunks + 1):
        chunk_text = _complete_messages(
            messages,
            max_tokens=max_tokens_per_chunk,
            temperature=temperature,
        )

        if not chunk_text or len(chunk_text.strip()) < 3:
            break

        full_text_parts.append(chunk_text)

        # Still measure surprise (for comparison), just don't inject it
        mean_loss, contour = task_agent.predict(chunk_text)

        chunk_curv = 0.0
        if len(chunk_text.split()) >= 5:
            _, chunk_curv = compute_curvature(chunk_text, embed_fn)

        chunks_data.append({
            'chunk_num': chunk_num,
            'text': chunk_text,
            'mean_surprise': round(mean_loss, 4),
            'contour': contour,
            'curvature': round(chunk_curv, 6),
        })
        trajectory.append(mean_loss)

        # Continue generating without proprioception — just append and ask to continue
        messages.append({"role": "assistant", "content": chunk_text})
        messages.append({"role": "user", "content": "Continue."})

    full_text = " ".join(full_text_parts)
    if not full_text.strip():
        return None

    task_agent.learn(full_text)

    angle, curv = compute_curvature(full_text, embed_fn)
    ltc = compute_loss_trajectory_curvature(trajectory)

    return {
        'full_text': full_text,
        'chunks': chunks_data,
        'trajectory': trajectory,
        'curvature': round(curv, 6),
        'curvature_angle': round(angle, 6),
        'loss_trajectory_curvature': round(ltc, 6),
        'injections': [],
        'n_chunks': len(chunks_data),
    }


# ── A/B experiment ──────────────────────────────────────────────────────

def _vocab_diversity(text):
    """Unique words / total words. Simple vocabulary diversity measure."""
    words = text.lower().split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def run_ab_experiment(prompt, task_agent, n=5, chunk_size=50, max_chunks=8,
                      system_prompt=None, embed_fn=None, temperature=0.7):
    """Run the same prompt n times WITH and WITHOUT proprioception.

    This is the honest test: does the loop do anything, or is it just overhead?

    Args:
        prompt: the prompt to test
        task_agent: MicroGPT TaskAgent instance
        n: number of runs per condition
        chunk_size: characters per chunk
        max_chunks: max chunks per breath
        system_prompt: optional system prompt
        embed_fn: embedding function
        temperature: Nemotron temperature

    Returns:
        dict with 'with_proprioception', 'without_proprioception',
        and 'comparison' summary. Returns None if Nemotron unavailable.
    """
    if not local_model.is_available():
        print("  Nemotron is not available. Cannot run A/B experiment.")
        return None

    if embed_fn is None:
        embed_fn = default_embed_fn

    kwargs = dict(
        chunk_size=chunk_size,
        max_chunks=max_chunks,
        system_prompt=system_prompt,
        embed_fn=embed_fn,
        temperature=temperature,
    )

    with_results = []
    without_results = []

    # Run WITH proprioception
    for i in range(n):
        result = run_proprioceptive_breath(prompt, task_agent, **kwargs)
        if result:
            with_results.append({
                'curvature': result['curvature'],
                'mean_surprise': (sum(result['trajectory'])
                                  / max(len(result['trajectory']), 1)),
                'text_length': len(result['full_text']),
                'vocab_diversity': _vocab_diversity(result['full_text']),
                'loss_trajectory_curvature': result['loss_trajectory_curvature'],
                'n_chunks': result['n_chunks'],
            })

    # Run WITHOUT proprioception
    for i in range(n):
        result = _run_plain_breath(prompt, task_agent, **kwargs)
        if result:
            without_results.append({
                'curvature': result['curvature'],
                'mean_surprise': (sum(result['trajectory'])
                                  / max(len(result['trajectory']), 1)),
                'text_length': len(result['full_text']),
                'vocab_diversity': _vocab_diversity(result['full_text']),
                'loss_trajectory_curvature': result['loss_trajectory_curvature'],
                'n_chunks': result['n_chunks'],
            })

    def _summarize(results):
        if not results:
            return {}
        keys = results[0].keys()
        summary = {}
        for k in keys:
            vals = [r[k] for r in results]
            summary[k] = {
                'mean': round(sum(vals) / len(vals), 6),
                'min': round(min(vals), 6),
                'max': round(max(vals), 6),
            }
        return summary

    with_summary = _summarize(with_results)
    without_summary = _summarize(without_results)

    # Comparison: for each metric, compute the delta (with - without)
    comparison = {}
    for key in with_summary:
        w_mean = with_summary[key]['mean']
        wo_mean = without_summary.get(key, {}).get('mean', 0.0)
        comparison[key] = {
            'with': round(w_mean, 6),
            'without': round(wo_mean, 6),
            'delta': round(w_mean - wo_mean, 6),
        }

    return {
        'prompt': prompt,
        'n': n,
        'with_proprioception': {
            'runs': with_results,
            'summary': with_summary,
        },
        'without_proprioception': {
            'runs': without_results,
            'summary': without_summary,
        },
        'comparison': comparison,
    }
