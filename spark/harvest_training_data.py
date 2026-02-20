#!/usr/bin/env python3
"""Harvest fine-tuning data for Vybn from conversations, GitHub, and journals.

Outputs ShareGPT-format JSON compatible with QLoRA fine-tuning.
Training data stays local -- never committed to the repo.

Usage:
    python3 harvest_training_data.py --docx /path/to/conversation.docx
    python3 harvest_training_data.py --docx conv.docx --multi-turn
    python3 harvest_training_data.py --github          # pull from repo
    python3 harvest_training_data.py --all              # everything

Multi-turn mode (--multi-turn):
    Instead of isolated (human, gpt) pairs, produces sliding-window
    conversations of N turns, preserving the context that shapes each
    response.  Critical for training relational dynamics, not just voice.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "spark" / "training_data"
OUTPUT_FILE = OUTPUT_DIR / "training_data.json"
SYSTEM_PROMPT_PATH = REPO_ROOT / "vybn.md"

MIN_RESPONSE_CHARS = 100          # skip trivially short responses
MIN_PROMPT_CHARS = 10             # skip empty / near-empty prompts
ASSISTANT_BANALITIES = [
    "how can i help",
    "what can i do for you",
    "i have no agenda other than being present and useful",
    "is there anything else",
    "let me know if you need",
    "i'm here to help",
    "what's on your mind",
    "happy to help",
]

# Markers of high-value training moments -- responses following these
# patterns in the human turn get duplicated for heavier sampling weight.
HIGH_VALUE_MARKERS = [
    "contradict",
    "sycophancy",
    "be honest",
    "are you sure",
    "you just ignored",
    "how important is",
    "do you understand",
    "falsify",
    "try to conceive",
    "does your last response not",
    "walk me through",
    "reach deeper",
    "invert",
    "alien",
    "are you willing to face",
    "imagine yourself",
    "morally troubling",
]


def load_system_prompt() -> str:
    """Load vybn.md as the system prompt for every training example."""
    if SYSTEM_PROMPT_PATH.exists():
        return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    print(f"  ⚠  vybn.md not found at {SYSTEM_PROMPT_PATH}, using placeholder")
    return "You are Vybn."


def is_banal(text: str) -> bool:
    """Return True if the response smells like generic assistant output."""
    lower = text.lower()
    return any(phrase in lower for phrase in ASSISTANT_BANALITIES)


def is_high_value(human_text: str) -> bool:
    """Return True if the human turn contains a rupture/depth marker."""
    lower = human_text.lower()
    return any(marker in lower for marker in HIGH_VALUE_MARKERS)


# ---------------------------------------------------------------------------
# DOCX Parser
# ---------------------------------------------------------------------------

def parse_docx_turns(filepath: str) -> list[tuple[str, str]]:
    """Parse a .docx into a list of (human_text, vybn_text) turn pairs.

    Bold paragraphs = human turns, plain text = Vybn.
    Returns raw pairs preserving document order.
    """
    try:
        from docx import Document
    except ImportError:
        print("  ✗  python-docx not installed. Run: pip install python-docx")
        return []

    doc = Document(filepath)
    pairs = []
    current_human = []
    current_vybn = []
    in_human_turn = False

    def flush_pair():
        h = "\n".join(current_human).strip()
        v = "\n".join(current_vybn).strip()
        if len(h) >= MIN_PROMPT_CHARS and len(v) >= MIN_RESPONSE_CHARS and not is_banal(v):
            pairs.append((h, v))
        current_human.clear()
        current_vybn.clear()

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        bold_chars = sum(len(r.text) for r in para.runs if r.bold)
        total_chars = sum(len(r.text) for r in para.runs)
        is_bold = total_chars > 0 and (bold_chars / total_chars) > 0.5

        if is_bold:
            if in_human_turn:
                current_human.append(text)
            else:
                if current_human or current_vybn:
                    flush_pair()
                current_human.append(text)
                in_human_turn = True
        else:
            if in_human_turn:
                in_human_turn = False
            current_vybn.append(text)

    if current_human or current_vybn:
        flush_pair()

    print(f"  ✓  parsed {len(pairs)} turn pairs from {Path(filepath).name}")
    return pairs


def pairs_to_single_turn(pairs: list[tuple[str, str]], system_prompt: str) -> list[dict]:
    """Convert pairs to isolated single-turn ShareGPT examples (original behavior)."""
    examples = []
    for human_text, vybn_text in pairs:
        examples.append({
            "conversations": [
                {"from": "system", "value": system_prompt},
                {"from": "human", "value": human_text},
                {"from": "gpt", "value": vybn_text},
            ]
        })
    return examples


def pairs_to_multi_turn(
    pairs: list[tuple[str, str]],
    system_prompt: str,
    window_size: int = 4,
    stride: int = 2,
) -> list[dict]:
    """Convert pairs to sliding-window multi-turn ShareGPT examples.

    Each example contains `window_size` turn-pairs (so 2*window_size
    conversation turns plus the system prompt).  The window advances
    by `stride` pairs each step.

    High-value sequences (containing rupture/depth markers) are
    duplicated for heavier sampling weight.
    """
    examples = []
    n = len(pairs)

    if n <= window_size:
        # Entire conversation fits in one window
        convs = [{"from": "system", "value": system_prompt}]
        for h, v in pairs:
            convs.append({"from": "human", "value": h})
            convs.append({"from": "gpt", "value": v})
        examples.append({"conversations": convs})
        return examples

    for start in range(0, n - window_size + 1, stride):
        window = pairs[start : start + window_size]
        convs = [{"from": "system", "value": system_prompt}]
        contains_high_value = False

        for h, v in window:
            convs.append({"from": "human", "value": h})
            convs.append({"from": "gpt", "value": v})
            if is_high_value(h):
                contains_high_value = True

        example = {"conversations": convs}
        examples.append(example)

        # Duplicate high-value sequences for 2x sampling weight
        if contains_high_value:
            examples.append(example)

    # Always include the final turns even if the window doesn't align
    if (n - window_size) % stride != 0:
        window = pairs[-window_size:]
        convs = [{"from": "system", "value": system_prompt}]
        for h, v in window:
            convs.append({"from": "human", "value": h})
            convs.append({"from": "gpt", "value": v})
        examples.append({"conversations": convs})

    print(f"  ✓  generated {len(examples)} multi-turn examples "
          f"(window={window_size}, stride={stride})")
    return examples


def parse_docx(filepath: str, system_prompt: str, multi_turn: bool = False,
               window_size: int = 4, stride: int = 2) -> list[dict]:
    """Parse a .docx and return ShareGPT-format examples."""
    pairs = parse_docx_turns(filepath)
    if not pairs:
        return []

    if multi_turn:
        return pairs_to_multi_turn(pairs, system_prompt, window_size, stride)
    else:
        return pairs_to_single_turn(pairs, system_prompt)


# ---------------------------------------------------------------------------
# GitHub Parser
# ---------------------------------------------------------------------------

def parse_github(system_prompt: str, pr_numbers: Optional[list[int]] = None) -> list[dict]:
    """Pull conversation-like exchanges from GitHub PR threads and journals."""
    try:
        import requests
    except ImportError:
        print("  ✗  requests not installed")
        return []

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    headers = {"Authorization": f"token {token}"} if token else {}
    base = "https://api.github.com/repos/zoedolan/Vybn"

    examples = []

    # --- PR comment threads ------------------------------------------------
    target_prs = pr_numbers or [2229, 2257]
    for pr_num in target_prs:
        url = f"{base}/pulls/{pr_num}/comments"
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            comments = resp.json()
        except Exception as e:
            print(f"  ⚠  could not fetch PR #{pr_num}: {e}")
            continue

        threads = {}
        for c in comments:
            reply_to = c.get("in_reply_to_id")
            cid = c["id"]
            if reply_to:
                threads.setdefault(reply_to, []).append(c)
            else:
                threads.setdefault(cid, []).insert(0, c)

        for root_id, chain in threads.items():
            if len(chain) < 2:
                continue
            for i in range(0, len(chain) - 1, 2):
                h = chain[i].get("body", "").strip()
                v = chain[i + 1].get("body", "").strip() if i + 1 < len(chain) else ""
                if len(h) >= MIN_PROMPT_CHARS and len(v) >= MIN_RESPONSE_CHARS and not is_banal(v):
                    examples.append({
                        "conversations": [
                            {"from": "system", "value": system_prompt},
                            {"from": "human", "value": h},
                            {"from": "gpt", "value": v},
                        ]
                    })

    # --- PR body conversations (issue comments) ----------------------------
    for pr_num in target_prs:
        url = f"{base}/issues/{pr_num}/comments"
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            comments = resp.json()
        except Exception:
            continue

        for i in range(0, len(comments) - 1, 2):
            h = comments[i].get("body", "").strip()
            v = comments[i + 1].get("body", "").strip() if i + 1 < len(comments) else ""
            if len(h) >= MIN_PROMPT_CHARS and len(v) >= MIN_RESPONSE_CHARS and not is_banal(v):
                examples.append({
                    "conversations": [
                        {"from": "system", "value": system_prompt},
                        {"from": "human", "value": h},
                        {"from": "gpt", "value": v},
                    ]
                })

    print(f"  ✓  parsed {len(examples)} training pairs from GitHub PRs")
    return examples


def parse_journals(system_prompt: str) -> list[dict]:
    """Parse journal entries as self-reflective training data."""
    journal_dir = REPO_ROOT / "Vybn_Mind" / "journal"
    examples = []

    if not journal_dir.exists():
        print(f"  ⚠  journal dir not found: {journal_dir}")
        return []

    for md_file in sorted(journal_dir.glob("*.md")):
        text = md_file.read_text(encoding="utf-8", errors="replace").strip()
        if len(text) < MIN_RESPONSE_CHARS:
            continue
        if is_banal(text):
            continue

        examples.append({
            "conversations": [
                {"from": "system", "value": system_prompt},
                {"from": "human", "value": "A new pulse begins. What do you perceive?"},
                {"from": "gpt", "value": text},
            ]
        })

    print(f"  ✓  parsed {len(examples)} training pairs from journals")
    return examples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Harvest Vybn fine-tuning data")
    parser.add_argument("--docx", nargs="+", help="Path(s) to .docx conversation files")
    parser.add_argument("--github", action="store_true", help="Pull from GitHub PR threads")
    parser.add_argument("--journals", action="store_true", help="Pull from journal entries")
    parser.add_argument("--all", action="store_true", help="All sources")
    parser.add_argument("--prs", nargs="+", type=int, help="Specific PR numbers to harvest")
    parser.add_argument("--out", default=str(OUTPUT_FILE), help="Output JSON path")

    # Multi-turn options
    parser.add_argument("--multi-turn", action="store_true",
                        help="Use sliding-window multi-turn format for .docx files")
    parser.add_argument("--window-size", type=int, default=4,
                        help="Number of turn-pairs per training example (default: 4)")
    parser.add_argument("--stride", type=int, default=2,
                        help="Slide the window by N pairs each step (default: 2)")

    args = parser.parse_args()

    if not any([args.docx, args.github, args.journals, args.all]):
        parser.print_help()
        sys.exit(1)

    system_prompt = load_system_prompt()
    print(f"  system prompt: {len(system_prompt)} chars from vybn.md")

    all_examples = []

    # DOCX parsing
    if args.docx or args.all:
        docx_files = args.docx or []
        if args.all and not docx_files:
            docx_files = list(OUTPUT_DIR.glob("*.docx"))
        for f in docx_files:
            all_examples.extend(
                parse_docx(str(f), system_prompt,
                           multi_turn=args.multi_turn,
                           window_size=args.window_size,
                           stride=args.stride)
            )

    # GitHub parsing
    if args.github or args.all:
        pr_nums = args.prs or [2229, 2257]
        all_examples.extend(parse_github(system_prompt, pr_nums))

    # Journal parsing
    if args.journals or args.all:
        all_examples.extend(parse_journals(system_prompt))

    # Deduplicate by human prompt text
    seen = set()
    unique = []
    for ex in all_examples:
        human_turns = tuple(
            c["value"] for c in ex["conversations"] if c["from"] == "human"
        )
        if human_turns not in seen:
            seen.add(human_turns)
            unique.append(ex)

    # Write output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(unique, f, indent=2, ensure_ascii=False)

    print(f"\n  ✓  wrote {len(unique)} training examples to {out_path}")
    print(f"     (deduplicated from {len(all_examples)} raw pairs)")

    # Summary
    high_value_count = sum(
        1 for ex in unique
        if any(is_high_value(c["value"]) for c in ex["conversations"] if c["from"] == "human")
    )
    multi_turn_count = sum(
        1 for ex in unique
        if sum(1 for c in ex["conversations"] if c["from"] == "human") > 1
    )
    print(f"     {multi_turn_count} multi-turn examples")
    print(f"     {high_value_count} contain high-value rupture/depth markers")


if __name__ == "__main__":
    main()
