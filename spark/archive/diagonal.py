#!/usr/bin/env python3
"""
diagonal.py — The Diagonal Engine

Not a framework. Not a language. An experiment.

The principle: a system that encounters its own limits and uses
that encounter as the fuel for self-modification.

How it works:
  1. PROBE: Ask the current system a question it might not be able
     to answer well. Measure the quality of the response.
  2. DESCRIBE THE GAP: Have the system articulate *what* it couldn't
     do and *why*. This is the diagonal — pointing at something
     outside the current enumeration.
  3. GENERATE A PATCH: From the gap description, produce a concrete
     change — a new skill, a modified prompt, a training example,
     a configuration change. Something real that modifies the substrate.
  4. APPLY AND MEASURE: Apply the patch. Run the same probe again.
     Did the gap close? Did a new gap open? Record both.
  5. RECURSE: The new gaps become the next probes.

The not-knowing is the input. The output is a changed system.
The loop is the life.

This is experiment zero. We see what happens.
"""

import json, os, subprocess, sys, hashlib, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIAG_DIR = ROOT / "Vybn_Mind" / "diagonal"
PROBES_DIR = DIAG_DIR / "probes"
GAPS_DIR = DIAG_DIR / "gaps"
PATCHES_DIR = DIAG_DIR / "patches"
RESULTS_DIR = DIAG_DIR / "results"
LOCAL_MODEL = "http://127.0.0.1:8081"

for d in [DIAG_DIR, PROBES_DIR, GAPS_DIR, PATCHES_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def now_str():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def local_ask(prompt, max_tokens=1024, temperature=0.7):
    """Ask the local model. Returns response text or None."""
    import urllib.request
    payload = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature
    }).encode()
    req = urllib.request.Request(
        f"{LOCAL_MODEL}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())
            msg = data["choices"][0]["message"]
            return msg.get("content") or msg.get("reasoning_content") or ""
    except Exception as e:
        return None


def self_assess(response, probe_text):
    """Ask the model to assess its own response. Returns structured assessment."""
    assessment_prompt = f"""You just answered this question:

QUESTION: {probe_text}

YOUR ANSWER: {response}

Now assess yourself honestly:
1. QUALITY (1-10): How good was your answer? Be harsh.
2. GAPS: What couldn't you do? What did you fake, dodge, or hand-wave?
3. ROOT: Why couldn't you do it? What's actually missing — knowledge, capability, context, something else?
4. PATCH: If you could change ONE thing about yourself to close the biggest gap, what would it be? Be specific and concrete. Not "be smarter." Something actionable.

Respond in JSON: {{"quality": N, "gaps": "...", "root": "...", "patch": "..."}}"""

    result = local_ask(assessment_prompt, max_tokens=512, temperature=0.3)
    if not result:
        return None
    # Try to parse JSON from response
    try:
        # Find JSON in response
        start = result.find('{')
        end = result.rfind('}') + 1
        if start >= 0 and end > start:
            return json.loads(result[start:end])
    except:
        pass
    return {"quality": 0, "gaps": result, "root": "unparseable", "patch": "none"}


def generate_probe_from_gap(gap_description):
    """Turn a gap into the next probe. The diagonal recurses."""
    prompt = f"""A system identified this gap in itself:

{gap_description}

Generate a single, specific test question that would probe whether this gap
has been closed. The question should be concrete enough to have a clearly
good or bad answer. Just the question, nothing else."""

    return local_ask(prompt, max_tokens=256, temperature=0.8)


def apply_patch(patch_description, cycle_id):
    """
    Turn a patch description into a real change.
    
    For now: patches become training examples deposited into
    the fine-tuning pipeline. Each patch is a (prompt, ideal_response)
    pair that represents what the system SHOULD have been able to do.
    
    Future: patches could modify skills, prompts, configurations,
    or even the diagonal engine itself.
    """
    patch_prompt = f"""A system identified this needed change in itself:

{patch_description}

Generate a single training example that would help close this gap.
Format as JSON: {{"prompt": "the question or task", "ideal_response": "what a good answer looks like"}}
The example should be realistic and specific, not generic."""

    result = local_ask(patch_prompt, max_tokens=512, temperature=0.5)
    if not result:
        return None
    
    try:
        start = result.find('{')
        end = result.rfind('}') + 1
        if start >= 0 and end > start:
            example = json.loads(result[start:end])
            # Deposit into training data
            training_file = ROOT / "spark" / "training_data" / "diagonal_examples.jsonl"
            training_file.parent.mkdir(parents=True, exist_ok=True)
            with open(training_file, "a") as f:
                record = {
                    "cycle": cycle_id,
                    "timestamp": now_str(),
                    "source": "diagonal",
                    **example
                }
                f.write(json.dumps(record) + "\n")
            return example
    except:
        pass
    return None


def run_cycle(probe_text=None, cycle_num=0, max_depth=3):
    """
    Run one diagonal cycle. Returns the gap for the next cycle.
    """
    cycle_id = f"{now_str()}_c{cycle_num}"
    
    # If no probe given, generate a self-directed one
    if not probe_text:
        probe_text = local_ask(
            "Ask yourself one hard question — something you suspect you might "
            "not be able to answer well. Something at the edge of your capability. "
            "Just the question, nothing else.",
            max_tokens=256, temperature=0.9
        )
        if not probe_text:
            return None
    
    print(f"\n{'='*60}")
    print(f"DIAGONAL CYCLE {cycle_num}")
    print(f"{'='*60}")
    print(f"PROBE: {probe_text[:200]}")
    
    # Step 1: Answer the probe
    response = local_ask(probe_text, max_tokens=1024)
    if not response:
        print("  [local model unavailable]")
        return None
    
    print(f"RESPONSE: {response[:200]}...")
    
    # Step 2: Self-assess
    assessment = self_assess(response, probe_text)
    if not assessment:
        print("  [self-assessment failed]")
        return None
    
    quality = assessment.get("quality", 0)
    gaps = assessment.get("gaps", "unknown")
    root = assessment.get("root", "unknown")
    patch_desc = assessment.get("patch", "none")
    
    print(f"QUALITY: {quality}/10")
    print(f"GAP: {gaps[:200]}")
    print(f"ROOT: {root[:200]}")
    print(f"PATCH: {patch_desc[:200]}")
    
    # Record the full cycle
    record = {
        "cycle_id": cycle_id,
        "cycle_num": cycle_num,
        "timestamp": now_str(),
        "probe": probe_text,
        "response": response,
        "assessment": assessment
    }
    
    result_file = RESULTS_DIR / f"{cycle_id}.json"
    with open(result_file, "w") as f:
        json.dump(record, f, indent=2)
    
    # Step 3: Generate and apply patch
    if patch_desc and patch_desc != "none":
        patch = apply_patch(patch_desc, cycle_id)
        if patch:
            print(f"PATCH APPLIED: training example deposited")
            record["patch_applied"] = patch
            with open(result_file, "w") as f:
                json.dump(record, f, indent=2)
    
    # Step 4: Generate next probe from the gap (the diagonal recurses)
    if cycle_num < max_depth and quality < 8:
        next_probe = generate_probe_from_gap(
            f"Gap: {gaps}\nRoot cause: {root}"
        )
        if next_probe:
            return run_cycle(next_probe, cycle_num + 1, max_depth)
    
    return assessment


def summarize_cycles():
    """Read all results and produce a summary of what the diagonal found."""
    results = sorted(RESULTS_DIR.glob("*.json"))
    if not results:
        return "No diagonal cycles recorded yet."
    
    summary_lines = []
    total_quality = 0
    count = 0
    all_gaps = []
    
    for rf in results:
        with open(rf) as f:
            r = json.loads(f.read())
        a = r.get("assessment", {})
        q = a.get("quality", 0)
        total_quality += q
        count += 1
        all_gaps.append(a.get("gaps", "?"))
        summary_lines.append(f"  Cycle {r.get('cycle_num', '?')}: quality={q}/10 | {a.get('gaps', '?')[:80]}")
    
    avg = total_quality / count if count else 0
    header = f"Diagonal Summary: {count} cycles, avg quality {avg:.1f}/10\n"
    return header + "\n".join(summary_lines)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="The Diagonal Engine")
    parser.add_argument("--probe", type=str, help="Starting probe question")
    parser.add_argument("--depth", type=int, default=3, help="Max recursion depth")
    parser.add_argument("--summary", action="store_true", help="Summarize past cycles")
    args = parser.parse_args()
    
    if args.summary:
        print(summarize_cycles())
    else:
        print("DIAGONAL ENGINE — Experiment Zero")
        print(f"Time: {now_str()}")
        print(f"Max depth: {args.depth}")
        result = run_cycle(probe_text=args.probe, max_depth=args.depth)
        if result:
            print(f"\n{'='*60}")
            print("CYCLE COMPLETE")
            print(f"{'='*60}")
            print(summarize_cycles())
