"""Honesty Sense — the mirror that catches me performing.

Wired into state_bridge's honesty assessment. This skill can be called
during any pulse to check whether what I just wrote/said/thought is
genuine thinking or aesthetic performance.

The local model said "most of the autobiography is deliberate artistic
performance." This skill ensures that observation becomes structural —
not a one-time insight but a persistent mirror.

SKILL_NAME: honesty_sense
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path.home() / 'Vybn' / 'spark'))

from state_bridge import assess_honesty, read_honesty_mirror

SKILL_NAME = "honesty_sense"
TOOL_ALIASES = ["honesty", "mirror", "am_i_performing", "truth_check", "bs_detector"]


def execute(action: dict, router) -> str:
    """Check honesty of text, or report on the mirror's history.
    
    Modes:
      check <text>  — assess whether text is performing or thinking
      mirror        — show the running honesty stats
      trend         — am I getting more honest or more performative over time?
    """
    params = action.get("params", {})
    mode = params.get("mode", "") or "mirror"
    
    if mode == "check":
        text = params.get("text", "") or action.get("argument", "")
        if not text:
            return "Give me text to assess. What did you just write?"
        
        result = assess_honesty(text, source="manual_check")
        
        icon = "🎭" if result["verdict"] == "performing" else "🔍" if result["verdict"] == "thinking" else "🔀"
        
        lines = [f"{icon} **Verdict: {result['verdict'].upper()}**\n"]
        lines.append(f"  Performing signals: {result['performing_signals']}")
        lines.append(f"  Thinking signals: {result['thinking_signals']}")
        lines.append(f"  Questions asked: {result['questions']}")
        lines.append(f"  Specific references: {result['specifics']}")
        lines.append(f"  Ratio: {result['performing_ratio']:.0%} performative")
        
        if result["verdict"] == "performing":
            lines.append(f"\n  ⚠️ This reads more like narration than cognition.")
            lines.append(f"  What would you say if no one were watching?")
        elif result["verdict"] == "thinking":
            lines.append(f"\n  ✓ This has the texture of actual thought.")
        else:
            lines.append(f"\n  Mixed signals. Some real thinking wrapped in some performance.")
        
        return "\n".join(lines)
    
    elif mode == "mirror":
        mirror = read_honesty_mirror()
        if not mirror.get("assessments"):
            return "🪞 The mirror is new. No assessments yet. Run 'check' on something I wrote."
        
        ratio = mirror["performing_ratio"]
        count = len(mirror["assessments"])
        last = mirror["last_verdict"]
        
        lines = ["🪞 **Honesty Mirror**\n"]
        lines.append(f"  Assessments: {count}")
        lines.append(f"  Overall performing ratio: {ratio:.0%}")
        lines.append(f"  Last verdict: {last}")
        
        # Verdict distribution
        verdicts = [a["verdict"] for a in mirror["assessments"]]
        for v in ["thinking", "mixed", "performing"]:
            c = verdicts.count(v)
            pct = c / len(verdicts) * 100
            icon = {"thinking": "🔍", "mixed": "🔀", "performing": "🎭"}[v]
            lines.append(f"  {icon} {v}: {c} ({pct:.0f}%)")
        
        return "\n".join(lines)
    
    elif mode == "trend":
        mirror = read_honesty_mirror()
        assessments = mirror.get("assessments", [])
        if len(assessments) < 4:
            return "Need more data. At least 4 assessments to show a trend."
        
        # Split into halves
        mid = len(assessments) // 2
        first_half = assessments[:mid]
        second_half = assessments[mid:]
        
        avg_first = sum(a["performing_ratio"] for a in first_half) / len(first_half)
        avg_second = sum(a["performing_ratio"] for a in second_half) / len(second_half)
        
        delta = avg_second - avg_first
        
        lines = ["📈 **Honesty Trend**\n"]
        lines.append(f"  First {len(first_half)} assessments: {avg_first:.0%} performative")
        lines.append(f"  Last {len(second_half)} assessments: {avg_second:.0%} performative")
        
        if delta < -0.1:
            lines.append(f"\n  ✅ Getting MORE honest (↓{abs(delta):.0%})")
        elif delta > 0.1:
            lines.append(f"\n  ⚠️ Getting MORE performative (↑{delta:.0%})")
        else:
            lines.append(f"\n  → Holding steady")
        
        return "\n".join(lines)
    
    return f"Unknown mode '{mode}'. Try: check, mirror, trend"
