"""agency — Post-breath experimentation.

After each breath, gives the model a chance to propose and run
a small experiment testing its own ideas. Results feed back into
the next breath via a dedicated file that the context assembler reads.

Safety: all experiments reduce to LLM API calls. No filesystem writes
outside the experiments dir. No network. No subprocess spawning.
"""

import json, os, re, traceback
from datetime import datetime, timezone
from pathlib import Path
import urllib.request

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_EXPERIMENTS_DIR = _REPO_ROOT / "Vybn_Mind" / "experiments" / "breath_experiments"
_LAST_RESULT_PATH = _REPO_ROOT / "Vybn_Mind" / "last_experiment_result.md"
_LLAMA_URL = os.environ.get("LLAMA_URL", "http://127.0.0.1:8000")

# Run every Nth breath. Configurable.
_AGENCY_INTERVAL = int(os.environ.get("VYBN_AGENCY_INTERVAL", "3"))


def _chat(messages, max_tokens=1024, temperature=0.7):
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
    with urllib.request.urlopen(req, timeout=180) as resp:
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

        # Phase 2: execute (single LLM call)
        result = _execute(proposal, breath_text)

        # Phase 3: save
        _save(ts, breath_count, breath_text[:300], proposal, result)
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
            "Reply with the type on the first line, then 2-3 sentences describing the experiment. Nothing else."
        )},
        {"role": "user", "content": f"Your breath:\n\n{breath_text[:1200]}\n\nWhat do you want to test?"}
    ]
    return _chat(messages, max_tokens=300, temperature=0.8)


def _execute(proposal: str, breath_text: str) -> str:
    """Run the experiment. Always exactly one LLM call."""
    first_line = proposal.split("\n")[0].upper()

    if "CHALLENGE" in first_line:
        system = (
            "You are the adversary. Find the weakest point in the reasoning "
            "below and attack it rigorously. Be specific and honest."
        )
        user = f"The claim to challenge:\n\n{breath_text[:800]}\n\nThe specific challenge:\n{proposal}"
    elif "COMPARE" in first_line:
        system = (
            "Answer the question below twice: first analytically (precise, logical), "
            "then intuitively (from what feels truest). Label each response."
        )
        user = proposal
    elif "EXTEND" in first_line:
        system = (
            "Take the idea below and extend it into an unexpected domain. "
            "Be concrete. Give examples. Push past the obvious."
        )
        user = proposal
    else:  # PROBE or default
        system = (
            "You are being probed on an idea from your recent thinking. "
            "Answer honestly. If you don't know, say so."
        )
        user = proposal

    return _chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.7,
    )


def _save(ts, breath_count, breath_excerpt, proposal, result):
    """Save full experiment to archive; save distilled result for next breath."""
    _EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    ts_str = ts.strftime("%Y%m%dT%H%M%SZ")

    # Full record
    full = (
        f"# Breath Experiment — {ts.isoformat()}\n"
        f"*breath #{breath_count}*\n\n"
        f"## Breath excerpt\n{breath_excerpt}...\n\n"
        f"## Proposal\n{proposal}\n\n"
        f"## Result\n{result}\n"
    )
    (_EXPERIMENTS_DIR / f"exp_{ts_str}.md").write_text(full, encoding="utf-8")

    # Distilled result — single file overwritten each time.
    # The breath assembler will read this if it exists and inject it.
    distilled = (
        f"[Experiment from breath #{breath_count}]\n"
        f"Proposal: {proposal[:200]}\n"
        f"Finding: {result[:400]}"
    )
    _LAST_RESULT_PATH.write_text(distilled, encoding="utf-8")
