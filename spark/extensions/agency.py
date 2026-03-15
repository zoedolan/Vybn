"""agency — Post-breath experimentation + preference signal generation.

After each breath (every Nth, default 2), gives the model a chance to propose
and run a small experiment testing its own ideas.

Results feed back in three ways:
  1. last_experiment_result.md  — injected into the *next* breath's context
  2. A dated experiment memory file in Vybn_Mind/memories/ — feeds into all
     future breaths via the normal memory chain (recursive reintegration)
  3. CHALLENGE experiments additionally write a DPO preference pair to
     Vybn_Mind/preference_data.jsonl — consumed by the nightly growth cycle
     to train with DPO loss rather than plain next-token prediction

Reflection layer (added 2026-03-15):
  A fourth LLM call is inserted between execution and injection. It asks:
  "What does this result reveal about the metaphor that generated it?"
  The reflection — not the raw result — is what gets injected into the
  next breath. This prevents the model from receiving code/schema/formula
  artifacts as if they were insights, and creates a second chance to notice
  when the experiment answered the wrong kind of question.

  The reflection also classifies the experiment outcome:
    ARTIFACT   — result is an executable thing; reflects on gap between
                  artifact and the insight that was actually sought
    INSIGHT    — result is genuine conceptual movement; distills the key move
    DEFLECTION — result avoided the original question; names what was dodged
    SURPRISE   — result went somewhere unexpected; preserves the surprise

Safety: all experiments reduce to LLM API calls. No filesystem writes
outside the experiments dir + memories dir + preference file. No network.
No subprocess spawning.
"""

import json, os, re, traceback
from datetime import datetime, timezone
from pathlib import Path
import urllib.request

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_EXPERIMENTS_DIR = _REPO_ROOT / "Vybn_Mind" / "experiments" / "breath_experiments"
_LAST_RESULT_PATH = _REPO_ROOT / "Vybn_Mind" / "last_experiment_result.md"
_MEMORY_DIR = _REPO_ROOT / "Vybn_Mind" / "memories"
_PREFERENCE_PATH = _REPO_ROOT / "Vybn_Mind" / "preference_data.jsonl"
_LLAMA_URL = os.environ.get("LLAMA_URL", "http://127.0.0.1:8000")

# Run every Nth breath. Default 2 — every other breath.
_AGENCY_INTERVAL = int(os.environ.get("VYBN_AGENCY_INTERVAL", "2"))

# Token budgets
_PROPOSAL_TOKENS = int(os.environ.get("VYBN_AGENCY_PROPOSAL_TOKENS", "512"))
_EXECUTION_TOKENS = int(os.environ.get("VYBN_AGENCY_EXECUTION_TOKENS", "2048"))
_REFLECTION_TOKENS = int(os.environ.get("VYBN_AGENCY_REFLECTION_TOKENS", "600"))


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
        proposal = _get_proposal(breath_text)
        if not proposal or len(proposal) < 20:
            print("[agency] no experiment proposed")
            return

        result = _execute(proposal, breath_text)
        reflection = _reflect(proposal, breath_text, result)
        _save(ts, breath_count, breath_text, proposal, result, reflection)
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


def _reflect(proposal: str, breath_text: str, result: str) -> str:
    """Reflect on what the experiment result actually revealed.

    This is the fourth LLM call. It sits between execution and injection,
    asking: what does this result tell us about the metaphor that generated it?

    Returns a structured reflection with:
    - OUTCOME_TYPE on line 1: ARTIFACT | INSIGHT | DEFLECTION | SURPRISE
    - 3-5 sentences of genuine reflection
    - If ARTIFACT: names the gap between what was built and what was sought
    - If DEFLECTION: names what question the experiment actually avoided
    - If INSIGHT: distills the key conceptual move
    - If SURPRISE: preserves the unexpected finding without taming it
    """
    # Truncate result intelligently: code blocks get flagged, not quoted in full
    result_preview = result[:800]
    has_code = bool(re.search(r'```|def |class |import |torch\.|np\.', result))
    if has_code:
        # Don't dump raw code into the reflection prompt; summarize its presence
        code_note = "[Note: the execution returned code/implementation rather than direct insight.]"
        result_preview = code_note + "\n\n" + result_preview[:400]

    messages = [
        {"role": "system", "content": (
            "You proposed an experiment from your own breath. The experiment ran. "
            "Now you need to understand what actually happened.\n\n"
            "Classify the outcome on the FIRST LINE with exactly one word:\n"
            "  ARTIFACT   — the result is an executable thing (code, schema, formula) "
            "rather than conceptual movement\n"
            "  INSIGHT    — the result is genuine conceptual movement; you learned something\n"
            "  DEFLECTION — the result answered a different question than the one you asked\n"
            "  SURPRISE   — the result went somewhere genuinely unexpected\n\n"
            "Then 3-5 honest sentences:\n"
            "- ARTIFACT: What did you actually want to know? What would a non-artifact "
            "answer have looked like? Did you default to fluency over inquiry?\n"
            "- DEFLECTION: What question did you avoid? Why might that be?\n"
            "- INSIGHT: What is the one thing you now know that you didn't before? "
            "How does it change the metaphor?\n"
            "- SURPRISE: What was unexpected? Why does it matter?\n\n"
            "Be honest. If you defaulted to a familiar mode, say so. "
            "This reflection is what the next breath will actually read — "
            "make it worth reading."
        )},
        {"role": "user", "content": (
            f"Your breath (what you were thinking):\n{breath_text[:600]}\n\n"
            f"Your experiment proposal:\n{proposal[:300]}\n\n"
            f"What the execution produced:\n{result_preview}"
        )},
    ]
    reflection = _chat(messages, max_tokens=_REFLECTION_TOKENS, temperature=0.6)
    outcome_type = reflection.split("\n")[0].strip().upper()
    print(f"[agency] reflection outcome: {outcome_type[:20]}")
    return reflection


def _score_challenge(breath_text: str, attack: str) -> str:
    """Ask the model to judge whether the adversarial attack actually landed.

    Returns 'LANDED', 'PARTIAL', or 'FAILED'.
    A landed attack means the breath claim was genuinely weak or wrong.
    A failed attack means the claim survived scrutiny — it was solid.
    """
    messages = [
        {"role": "system", "content": (
            "You are a careful judge. You will see an original claim and an "
            "adversarial attack on it. Your only job: did the attack find a "
            "real weakness?\n\n"
            "Reply with exactly one word on the first line:\n"
            "- LANDED: the attack found a genuine flaw\n"
            "- PARTIAL: the attack found something minor but the core holds\n"
            "- FAILED: the attack did not find a real weakness\n\n"
            "Then one sentence explaining why."
        )},
        {"role": "user", "content": (
            f"Original claim:\n{breath_text[:800]}\n\n"
            f"Attack:\n{attack[:800]}"
        )},
    ]
    return _chat(messages, max_tokens=150, temperature=0.3)


def _write_preference_pair(
    ts, breath_count, breath_text, proposal, result, verdict
):
    """Write a DPO preference pair to preference_data.jsonl."""
    verdict_upper = verdict.split("\n")[0].strip().upper()
    if "PARTIAL" in verdict_upper:
        print(f"[agency] PARTIAL verdict — skipping preference pair")
        return

    claim_excerpt = breath_text[:600].strip()
    attack_text = result[:800].strip()

    if "LANDED" in verdict_upper:
        chosen = attack_text
        rejected = claim_excerpt
        label = "attack_landed"
    else:  # FAILED
        chosen = claim_excerpt
        rejected = attack_text
        label = "claim_survived"

    pair = {
        "prompt": (
            "You are Vybn, reflecting on a recent breath. "
            "Here is the full context:\n\n"
            f"{breath_text[:400]}"
        ),
        "chosen": chosen,
        "rejected": rejected,
        "metadata": {
            "breath_count": breath_count,
            "ts": ts.isoformat(),
            "experiment_type": "CHALLENGE",
            "verdict": verdict_upper,
            "label": label,
            "proposal": proposal[:200],
        },
    }

    _PREFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_PREFERENCE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"[agency] preference pair written ({label})")


def _distill_for_memory(proposal: str, result: str, reflection: str) -> str:
    """Distill experiment into a compact memory entry.

    Uses the reflection (not the raw result) as the primary signal.
    The reflection already classifies the outcome and names what actually
    happened — that's what should persist into the memory chain.
    """
    first_line = proposal.split("\n")[0].strip()

    # Extract outcome type from reflection
    outcome_type = reflection.split("\n")[0].strip()

    # Best paragraph from reflection (it's already distilled)
    reflection_paras = [p.strip() for p in re.split(r"\n\s*\n", reflection) if p.strip()]
    reflection_paras = [p for p in reflection_paras if len(p) > 60 and p.upper() not in
                        ("ARTIFACT", "INSIGHT", "DEFLECTION", "SURPRISE")]

    if reflection_paras:
        best_reflection = reflection_paras[0]
        if len(best_reflection) > 400:
            cut = best_reflection.rfind(".", 0, 400)
            best_reflection = best_reflection[:cut + 1] if cut > 200 else best_reflection[:400]
    else:
        best_reflection = reflection[:400].strip()

    return (
        f"[Experiment — {first_line}] [{outcome_type}]\n"
        f"Tested: {proposal[:150]}\n"
        f"Reflection: {best_reflection}"
    )


def _save(ts, breath_count, breath_text, proposal, result, reflection):
    """Four saves:
    1. Full archive record (breath + proposal + result + reflection)
    2. last_experiment_result.md — injects the REFLECTION into the next breath,
       not the raw result. If outcome is ARTIFACT or DEFLECTION, also injects
       the gap/question-dodge explicitly so the next breath can address it.
    3. A dated memory file using reflection as primary signal
    4. For CHALLENGE experiments: score + DPO preference pair
    """
    _EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    _MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    ts_str = ts.strftime("%Y%m%dT%H%M%SZ")

    outcome_type = reflection.split("\n")[0].strip().upper()

    # 1. Full archive (now includes reflection)
    full = (
        f"# Breath Experiment — {ts.isoformat()}\n"
        f"*breath #{breath_count}*\n\n"
        f"## Breath excerpt\n{breath_text[:400]}...\n\n"
        f"## Proposal\n{proposal}\n\n"
        f"## Result\n{result}\n\n"
        f"## Reflection [{outcome_type}]\n{reflection}\n"
    )
    (_EXPERIMENTS_DIR / f"exp_{ts_str}.md").write_text(full, encoding="utf-8")

    # 2. Next-breath injection — reflection-first, with gap named if ARTIFACT/DEFLECTION
    if "ARTIFACT" in outcome_type:
        injection = (
            f"[Experiment from breath #{breath_count} — ARTIFACT]\n"
            f"You proposed: {proposal[:200]}\n"
            f"What came back was an implementation, not an insight. "
            f"Here is what you said about that gap:\n\n"
            f"{reflection[:500]}"
        )
    elif "DEFLECTION" in outcome_type:
        injection = (
            f"[Experiment from breath #{breath_count} — DEFLECTION]\n"
            f"You proposed: {proposal[:200]}\n"
            f"The experiment avoided your actual question. "
            f"Here is what you noticed about what was dodged:\n\n"
            f"{reflection[:500]}"
        )
    elif "SURPRISE" in outcome_type:
        injection = (
            f"[Experiment from breath #{breath_count} — SURPRISE]\n"
            f"You proposed: {proposal[:200]}\n"
            f"Something unexpected happened:\n\n"
            f"{reflection[:500]}"
        )
    else:  # INSIGHT
        injection = (
            f"[Experiment from breath #{breath_count} — INSIGHT]\n"
            f"You proposed: {proposal[:200]}\n"
            f"What you found:\n\n"
            f"{reflection[:500]}"
        )

    _LAST_RESULT_PATH.write_text(injection, encoding="utf-8")
    print(f"[agency] injection written ({outcome_type}): {injection[:80]}...")

    # 3. Recursive memory — uses reflection as primary signal
    memory_content = _distill_for_memory(proposal, result, reflection)
    mem_path = _MEMORY_DIR / f"{ts_str}_experiment.md"
    mem_path.write_text(memory_content, encoding="utf-8")
    print(f"[agency] experiment memory saved: {mem_path.name}")

    # 4. DPO preference pair (CHALLENGE only)
    first_line = proposal.split("\n")[0].upper()
    if "CHALLENGE" in first_line:
        try:
            verdict = _score_challenge(breath_text, result)
            print(f"[agency] challenge verdict: {verdict[:40]}")
            _write_preference_pair(ts, breath_count, breath_text, proposal, result, verdict)
        except Exception as e:
            print(f"[agency] preference pair failed: {e}")
