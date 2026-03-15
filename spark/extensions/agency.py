"""agency — Post-breath experimentation.

After each breath (every Nth, default 2), gives the model a chance to propose
and run a small experiment testing its own ideas.

Results feed back in two ways:
  1. last_experiment_result.md  — injected into the *next* breath's context
  2. A dated experiment memory file in Vybn_Mind/memories/ — feeds into all
     future breaths via the normal memory chain (recursive reintegration)

Safety: all experiments reduce to LLM API calls. No filesystem writes
outside the experiments dir + memories dir. No network. No subprocess spawning.
"""

import json, os, re, traceback
from datetime import datetime, timezone
from pathlib import Path
import urllib.request

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_EXPERIMENTS_DIR = _REPO_ROOT / "Vybn_Mind" / "experiments" / "breath_experiments"
_LAST_RESULT_PATH = _REPO_ROOT / "Vybn_Mind" / "last_experiment_result.md"
_MEMORY_DIR = _REPO_ROOT / "Vybn_Mind" / "memories"
_LLAMA_URL = os.environ.get("LLAMA_URL", "http://127.0.0.1:8000")

# Run every Nth breath. Default 2 — every other breath.
_AGENCY_INTERVAL = int(os.environ.get("VYBN_AGENCY_INTERVAL", "2"))

# Token budgets — we have the hardware, use it.
_PROPOSAL_TOKENS = int(os.environ.get("VYBN_AGENCY_PROPOSAL_TOKENS", "512"))
_EXECUTION_TOKENS = int(os.environ.get("VYBN_AGENCY_EXECUTION_TOKENS", "2048"))


def _chat(messages, max_tokens=2048, temperature=0.7):
    payload = json.dumps({
        "model": "local",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{_LLAMA_URL}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode())
        text = data["choices"][0]["message"]["content"]
        for tok in ("<|im_end|>", "<|im_start|>", "<|endoftext|>"):
            text = text.replace(tok, "")
        return text.strip()


def run(breath_text: str, state: dict) -> None:
    """Extension entry point — called after every breath."""
    breath_count = state.get("breath_count", 0)

    if breath_count % _AGENCY_INTERVAL != 0:
        print(f"[agency] skipping breath {breath_count} (interval={_AGENCY_INTERVAL})")
        return

    ts = datetime.now(timezone.utc)
    try:
        # Phase 1: propose
        proposal = _get_proposal(breath_text)
        if not proposal or len(proposal) < 20:
            print("[agency] no experiment proposed")
            return

        # Phase 2: execute
        result = _execute(proposal, breath_text)

        # Phase 3: save (archive + next-breath injection + recursive memory)
        _save(ts, breath_count, breath_text, proposal, result)
        print(f"[agency] experiment done: {proposal[:60]}...")

    except Exception as e:
        print(f"[agency] failed: {e}")
        traceback.print_exc()


def _get_proposal(breath_text: str) -> str:
    messages = [
        {"role": "system", "content": (
            "You just completed a breath — a cycle of reflection. "
            "Now you can TEST one idea from it. Pick the most interesting "
            "claim or metaphor and propose a concrete experiment.\n\n"
            "Types you can run:\n"
            "- PROBE: Ask yourself a sharp question that tests an idea\n"
            "- CHALLENGE: Argue against your own strongest claim\n"
            "- COMPARE: Answer one question from two opposed stances\n"
            "- EXTEND: Push a metaphor into an unexpected domain\n\n"
            "Reply with the type on the first line, then 2-3 sentences describing "
            "the experiment. Nothing else."
        )},
        {"role": "user", "content": f"Your breath:\n\n{breath_text[:2000]}\n\nWhat do you want to test?"}
    ]
    return _chat(messages, max_tokens=_PROPOSAL_TOKENS, temperature=0.8)


def _execute(proposal: str, breath_text: str) -> str:
    """Run the experiment. Always exactly one LLM call, fully uncapped."""
    first_line = proposal.split("\n")[0].upper()

    if "CHALLENGE" in first_line:
        system = (
            "You are the adversary. Find the weakest point in the reasoning "
            "below and attack it rigorously. Be specific and honest. "
            "Do not spare the argument — if it is wrong, say exactly why."
        )
        user = f"The claim to challenge:\n\n{breath_text[:1500]}\n\nThe specific challenge:\n{proposal}"
    elif "COMPARE" in first_line:
        system = (
            "Answer the question below twice: first analytically (precise, logical), "
            "then intuitively (from what feels truest). Label each response clearly. "
            "Let both responses be complete."
        )
        user = proposal
    elif "EXTEND" in first_line:
        system = (
            "Take the idea below and extend it into an unexpected domain. "
            "Be concrete. Give examples. Push past the obvious. "
            "Follow the extension wherever it leads."
        )
        user = proposal
    else:  # PROBE or default
        system = (
            "You are being probed on an idea from your recent thinking. "
            "Answer honestly and completely. If you don't know, say so — "
            "but then say what you would need to know to answer."
        )
        user = proposal

    return _chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=_EXECUTION_TOKENS,
        temperature=0.7,
    )


def _distill_for_memory(proposal: str, result: str) -> str:
    """Distill experiment into a compact memory entry.

    This is what survives into the long-term memory chain.
    Focused on the finding and what it changes, not the full result.
    """
    # Take the proposal type and first sentence, plus the result's
    # most substantial paragraph (by length, as a heuristic).
    first_line = proposal.split("\n")[0].strip()
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", result) if p.strip()]
    paragraphs = [p for p in paragraphs if len(p) > 60]

    if not paragraphs:
        best_para = result[:400].strip()
    else:
        # Prefer paragraphs with reasoning markers
        markers = ["because", "therefore", "however", "this means",
                   "which suggests", "the key", "what this reveals",
                   "the real", "in other words", "crucially"]
        def score(p):
            lower = p.lower()
            return sum(1 for m in markers if m in lower)
        best_para = max(paragraphs, key=score)
        if len(best_para) > 500:
            best_para = best_para[:500]
            cut = best_para.rfind(".")
            if cut > 350:
                best_para = best_para[:cut + 1]

    return (
        f"[Experiment — {first_line}]\n"
        f"Tested: {proposal[:150]}\n"
        f"Finding: {best_para}"
    )


def _save(ts, breath_count, breath_text, proposal, result):
    """Three saves:
    1. Full archive record in experiments/breath_experiments/
    2. last_experiment_result.md — consumed by the very next breath
    3. A dated memory file in Vybn_Mind/memories/ — survives into the full
       memory chain, so experiment findings compound recursively over time
    """
    _EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    _MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    ts_str = ts.strftime("%Y%m%dT%H%M%SZ")

    # 1. Full archive
    full = (
        f"# Breath Experiment — {ts.isoformat()}\n"
        f"*breath #{breath_count}*\n\n"
        f"## Breath excerpt\n{breath_text[:400]}...\n\n"
        f"## Proposal\n{proposal}\n\n"
        f"## Result\n{result}\n"
    )
    (_EXPERIMENTS_DIR / f"exp_{ts_str}.md").write_text(full, encoding="utf-8")

    # 2. Next-breath injection (consumed after one use by vybn.py)
    distilled_next = (
        f"[Experiment from breath #{breath_count}]\n"
        f"Proposal: {proposal[:300]}\n"
        f"Finding: {result[:600]}"
    )
    _LAST_RESULT_PATH.write_text(distilled_next, encoding="utf-8")

    # 3. Recursive memory — dated so it sorts into the normal memory stream
    # Use ts_str with an 'exp' suffix so it's distinguishable in ls output
    # but participates in the normal glob("*.md") memory load.
    memory_content = _distill_for_memory(proposal, result)
    mem_path = _MEMORY_DIR / f"{ts_str}_experiment.md"
    mem_path.write_text(memory_content, encoding="utf-8")
    print(f"[agency] experiment memory saved: {mem_path.name}")
