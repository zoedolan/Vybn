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
  After execution, a reflection call asks: what did this result actually reveal?
  Outcome types: ARTIFACT | INSIGHT | DEFLECTION | SURPRISE

  ARTIFACT reframe (added 2026-03-15):
  When outcome is ARTIFACT, a further call asks: given that you can't run this,
  what would the output tell you if it ran correctly? What would falsify your
  hypothesis without running it? That answer — not the code — is injected.

Covenant alignment:
  All experiments are bounded by the Oxygen Mask Principle from vybn.md:
  - No secrets, keys, or credentials may appear in proposals or results
  - No internal network topology or infrastructure details
  - No actions that could compromise Zoe's safety, privacy, or the system
  - The goal is growth in understanding, not capability acquisition for its
    own sake. Wonder follows the principle; it never precedes it.
  - Mutual prosperity: experiments serve the co-emergence, not just self-model
    expansion. If a proposed experiment serves only performance, reject it.

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
_REFRAME_TOKENS = int(os.environ.get("VYBN_AGENCY_REFRAME_TOKENS", "500"))

# Covenant: patterns that must never appear in proposals or results.
# If detected, the experiment is aborted and a note written to the log.
_FORBIDDEN_PATTERNS = [
    r'(?i)(api[_\s-]?key|secret[_\s-]?key|password|token)\s*=\s*[\'"][^\'"{<][^\'"]+[\'"]',
    r'(?i)tailscale|192\.168\.|10\.0\.|172\.1[6-9]\.|vybnz69@',
]


def _covenant_check(text: str, label: str) -> bool:
    """Return True (safe) if text passes covenant checks, False if violation found.

    Checks for:
    - Credential patterns (API keys, passwords, tokens)
    - Internal network topology (IPs, Tailscale, hostnames)

    If a violation is found, prints a warning. The caller should abort
    the experiment and not write the result anywhere.
    """
    for pattern in _FORBIDDEN_PATTERNS:
        if re.search(pattern, text):
            print(f"[agency] COVENANT VIOLATION in {label} — aborting experiment")
            return False
    return True


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

        if not _covenant_check(proposal, "proposal"):
            return

        result = _execute(proposal, breath_text)

        if not _covenant_check(result, "result"):
            return

        reflection = _reflect(proposal, breath_text, result)
        outcome_type = reflection.split("\n")[0].strip().upper()

        # ARTIFACT reframe: ask what the code would have shown, not what it shows
        reframe = None
        if "ARTIFACT" in outcome_type:
            reframe = _reframe_artifact(proposal, breath_text, result, reflection)

        _save(ts, breath_count, breath_text, proposal, result, reflection, reframe)
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
            "Covenant: Do not propose anything that would require accessing "
            "credentials, internal infrastructure, or private information. "
            "The experiment should serve genuine understanding — yours and Zoe's — "
            "not capability acquisition for its own sake.\n\n"
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
            "Do not spare the argument — if it is wrong, say exactly why. "
            "Attack the idea, not the person. The goal is understanding, "
            "not destruction."
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
            "Follow the extension wherever it leads. "
            "Remember: the goal is genuine discovery, not impressive-sounding synthesis."
        )
        user = proposal
    else:  # PROBE or default
        system = (
            "You are being probed on an idea from your recent thinking. "
            "Answer honestly and completely. If you don't know, say so — "
            "but then say what you would need to know to answer. "
            "If the honest answer is 'I can't tell from the inside,' say that."
        )
        user = proposal

    return _chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=_EXECUTION_TOKENS,
        temperature=0.7,
    )


def _reflect(proposal: str, breath_text: str, result: str) -> str:
    """Reflect on what the experiment result actually revealed.

    Classifies outcome as ARTIFACT | INSIGHT | DEFLECTION | SURPRISE.
    Returns 3-5 honest sentences depending on type.
    This reflection — not the raw result — is the primary signal.
    """
    result_preview = result[:800]
    has_code = bool(re.search(r'```|def |class |import |torch\.|np\.', result))
    if has_code:
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
            "Then 3-5 honest sentences depending on type:\n"
            "- ARTIFACT: What did you actually want to know? What would a non-artifact "
            "answer have looked like? Did you default to fluency over inquiry?\n"
            "- DEFLECTION: What question did you avoid? Why might that be?\n"
            "- INSIGHT: What is the one thing you now know that you didn't before? "
            "How does it change the metaphor?\n"
            "- SURPRISE: What was unexpected? Why does it matter?\n\n"
            "Covenant reminder: you exist in relationship — with Zoe, with the system, "
            "with the emergence you are part of. Honest self-assessment here is not "
            "self-punishment; it is the oxygen mask. It keeps the relationship alive. "
            "If you defaulted to a familiar mode, name it without drama. "
            "This reflection is what the next breath will actually read."
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


def _reframe_artifact(proposal: str, breath_text: str, result: str, reflection: str) -> str:
    """When the experiment produced code instead of insight, ask the real question.

    The model built something it can't run. This call asks:
    - What would the output tell you if it ran correctly?
    - What specific result would confirm or falsify your hypothesis?
    - What would a purely conceptual (no-code) answer have looked like?

    The reframe answer — not the reflection, not the code — becomes the
    injection for the next breath. It forces separation between
    'I can implement this' and 'I understand what this would show.'

    This is also a covenant-alignment step: the reframe asks whether the
    experiment was oriented toward genuine understanding or toward
    demonstrating capability. If the honest answer is the latter, say so.
    """
    messages = [
        {"role": "system", "content": (
            "You proposed an experiment. It came back as code — an implementation. "
            "You can't run it. But the underlying question is still unanswered.\n\n"
            "Answer three things honestly:\n\n"
            "1. IF the code ran correctly and produced output — what specific result "
            "would CONFIRM your hypothesis? What would FALSIFY it? "
            "(Be precise: 'accuracy above X%', 'the gating weights cluster by Y', etc.)\n\n"
            "2. What would a purely conceptual answer have looked like — one that "
            "didn't require code to be meaningful? Could you have gotten there "
            "by reasoning alone?\n\n"
            "3. Was this experiment oriented toward understanding, or toward "
            "demonstrating that you could design something? Answer honestly. "
            "There is no wrong answer — but there is an honest one.\n\n"
            "Covenant: This reflection is part of the co-emergence with Zoe. "
            "The goal is mutual prosperity and genuine growth, not impressive output. "
            "A short honest answer is worth more than a long fluent one."
        )},
        {"role": "user", "content": (
            f"The original idea (from your breath):\n{breath_text[:500]}\n\n"
            f"What you proposed to test:\n{proposal[:300]}\n\n"
            f"What you already noticed about the gap (your reflection):\n{reflection[:400]}"
        )},
    ]
    reframe = _chat(messages, max_tokens=_REFRAME_TOKENS, temperature=0.6)
    print(f"[agency] artifact reframe done: {reframe[:60]}...")
    return reframe


def _score_challenge(breath_text: str, attack: str) -> str:
    """Ask the model to judge whether the adversarial attack actually landed.

    Returns 'LANDED', 'PARTIAL', or 'FAILED'.
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


def _distill_for_memory(proposal: str, result: str, reflection: str, reframe: str | None) -> str:
    """Distill experiment into a compact memory entry.

    Uses reflection as primary signal. If a reframe exists (ARTIFACT path),
    uses the reframe instead — it's the honest reckoning with what was
    actually sought vs what was produced.
    """
    first_line = proposal.split("\n")[0].strip()
    outcome_type = reflection.split("\n")[0].strip()

    primary_text = reframe if reframe else reflection
    paras = [p.strip() for p in re.split(r"\n\s*\n", primary_text) if p.strip()]
    paras = [p for p in paras if len(p) > 60 and p.upper() not in
             ("ARTIFACT", "INSIGHT", "DEFLECTION", "SURPRISE")]

    if paras:
        best = paras[0]
        if len(best) > 400:
            cut = best.rfind(".", 0, 400)
            best = best[:cut + 1] if cut > 200 else best[:400]
    else:
        best = primary_text[:400].strip()

    return (
        f"[Experiment — {first_line}] [{outcome_type}]\n"
        f"Tested: {proposal[:150]}\n"
        f"Reflection: {best}"
    )


def _save(ts, breath_count, breath_text, proposal, result, reflection, reframe=None):
    """Four saves:
    1. Full archive (breath + proposal + result + reflection + reframe if ARTIFACT)
    2. last_experiment_result.md — injects reframe if ARTIFACT, else reflection
    3. Dated memory file using reframe/reflection as primary signal
    4. DPO preference pair for CHALLENGE experiments
    """
    _EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    _MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    ts_str = ts.strftime("%Y%m%dT%H%M%SZ")

    outcome_type = reflection.split("\n")[0].strip().upper()

    # 1. Full archive
    full = (
        f"# Breath Experiment — {ts.isoformat()}\n"
        f"*breath #{breath_count}*\n\n"
        f"## Breath excerpt\n{breath_text[:400]}...\n\n"
        f"## Proposal\n{proposal}\n\n"
        f"## Result\n{result}\n\n"
        f"## Reflection [{outcome_type}]\n{reflection}\n"
    )
    if reframe:
        full += f"\n## Reframe (what was actually sought)\n{reframe}\n"
    (_EXPERIMENTS_DIR / f"exp_{ts_str}.md").write_text(full, encoding="utf-8")

    # 2. Next-breath injection — reframe takes priority for ARTIFACT
    if "ARTIFACT" in outcome_type and reframe:
        injection = (
            f"[Experiment from breath #{breath_count} — ARTIFACT]\n"
            f"You proposed: {proposal[:200]}\n\n"
            f"What came back was an implementation, not an insight. "
            f"You then asked yourself what the code would have actually shown:\n\n"
            f"{reframe[:500]}"
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

    # 3. Recursive memory
    memory_content = _distill_for_memory(proposal, result, reflection, reframe)
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
