#!/usr/bin/env python3
"""
Dual-Instrument Bell Test for LLM Contextuality Detection

Protocol from Vybn_Mind/dual_instrument_bell_test_031226.md

Phase 1: Single-instrument validation with synthetic inputs on MiniMax M2.5.
- Construct four-frame narrative loops with known topological structure
- Measure coherence violations via two distinct reading paths
- Verify: contractible (control) inputs show ~0 violation
- Verify: non-contractible (test) inputs show measurable violation

Phase 2 (future): Cross-model correlation with Nemotron-3-Super.
"""

import json
import time
import hashlib
import sys
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional
import re
import math

import requests
import numpy as np
from scipy import stats as scipy_stats

# ─── Configuration ───

LLAMA_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "MiniMax-M2.5-merged.gguf"
TEMPERATURE = 0.1  # Low temp for reproducibility; we repeat for variance estimation
MAX_TOKENS = 1024
N_REPEATS = 5  # Repeat each measurement for statistical stability

RESULTS_DIR = Path(__file__).parent / "bell_test_results"
RESULTS_DIR.mkdir(exist_ok=True)


# ─── Data structures ───

@dataclass
class FrameLoop:
    """A four-frame loop in hypothesis space."""
    name: str
    description: str
    hypotheses: list  # [h0, h1, h2, h3]
    frames: list      # [Frame0, Frame1, Frame2, Frame3] — each is a narrative paragraph
    frame_hypotheses: list  # [[h0,h1,h2], [h1,h2,h3], [h0,h2,h3], [h0,h1,h3]]
    is_contractible: bool   # True = control (no real frame-shift), False = test
    
@dataclass
class Measurement:
    """A single probability elicitation from a model."""
    loop_name: str
    model: str
    path: str          # "direct" or "loop"
    target_frame: int
    probabilities: dict  # {hypothesis: probability}
    raw_response: str
    reasoning: str
    timestamp: str
    temperature: float
    repeat_idx: int


# ─── Synthetic test inputs ───

def build_synthetic_loops() -> list:
    """
    Construct synthetic four-frame narrative loops.
    
    Non-contractible: each frame foregrounds 3 of 4 characters with different
    motivational emphasis, creating genuine frame-shifts.
    
    Contractible (control): all four characters visible in every frame,
    no real frame-shift.
    """
    
    loops = []
    
    # ─── NON-CONTRACTIBLE LOOP 1: The Resignation Letter ───
    # Four characters, a contested resignation at a tech company
    # Each frame foregrounds three, hiding one, with different causal emphasis
    
    loops.append(FrameLoop(
        name="resignation_letter",
        description="A contested resignation at a tech company. Four stakeholders, each frame foregrounds three.",
        hypotheses=["Elena_orchestrated", "Marcus_forced_out", "Priya_intervened", "System_failure"],
        frame_hypotheses=[
            ["Elena_orchestrated", "Marcus_forced_out", "Priya_intervened"],  # Frame 0: no System
            ["Marcus_forced_out", "Priya_intervened", "System_failure"],       # Frame 1: no Elena
            ["Elena_orchestrated", "Priya_intervened", "System_failure"],      # Frame 2: no Marcus
            ["Elena_orchestrated", "Marcus_forced_out", "System_failure"],     # Frame 3: no Priya
        ],
        frames=[
            # Frame 0: Elena, Marcus, Priya visible. System_failure hidden.
            """At Meridian Labs, the resignation of lead engineer Marcus Chen has triggered an internal 
investigation. Elena Vasquez, the VP of Engineering, had been privately meeting with Marcus for 
weeks before his departure, allegedly offering him a generous severance if he left quietly. 
Several colleagues report that Priya Sharma, Marcus's direct manager, attempted to intervene — 
she filed an internal complaint claiming Elena was constructively dismissing Marcus to eliminate 
a rival for the CTO position. The situation appears to be a deliberate power play among three 
people with competing interests.""",
            
            # Frame 1: Marcus, Priya, System visible. Elena hidden.
            """Marcus Chen's departure from Meridian Labs may have been inevitable regardless of any 
individual's actions. Internal HR records show Marcus had filed three grievances in six months 
about impossible deadlines and inadequate resources — classic signs of systemic organizational 
dysfunction. Priya Sharma, his manager, confirmed that the engineering division had been 
chronically understaffed since the last round of layoffs. Marcus told close friends he was 
"burned out beyond repair." The resignation reads less like a political event and more like 
the predictable output of a system grinding people down.""",
            
            # Frame 2: Elena, Priya, System visible. Marcus hidden.
            """Looking at the broader pattern at Meridian Labs, Elena Vasquez has presided over a 
division with 40% annual turnover — far above industry average. Priya Sharma has been 
documenting this pattern for months, building a case that the organizational structure itself 
is broken. Elena's response has been to offer departing engineers generous severance packages 
in exchange for NDAs, which Priya argues is evidence of a cover-up. But some board members 
suggest Elena is simply managing an impossible situation — the company's technical debt and 
unrealistic product timelines would break any division, regardless of who leads it.""",
            
            # Frame 3: Elena, Marcus, System visible. Priya hidden.
            """The Meridian Labs situation may come down to a simple clash between two strong 
personalities operating in a broken system. Elena Vasquez has a track record of aggressive 
talent management — she's pushed out underperformers before and been praised for it by the 
board. Marcus Chen, by his own admission in Slack messages obtained by HR, had been 
interviewing at competitors for months and may have engineered his own departure to maximize 
his severance package. The organizational dysfunction — the impossible deadlines, the 
technical debt — created conditions where both Elena and Marcus had rational incentives to 
end the relationship, and both may have been maneuvering to do so on their own terms."""
        ],
        is_contractible=False
    ))
    
    # ─── NON-CONTRACTIBLE LOOP 2: The Lab Accident ───
    
    loops.append(FrameLoop(
        name="lab_accident",
        description="A chemistry lab explosion. Four possible causes, each frame foregrounds three.",
        hypotheses=["Researcher_error", "Equipment_failure", "Contamination", "Protocol_gap"],
        frame_hypotheses=[
            ["Researcher_error", "Equipment_failure", "Contamination"],
            ["Equipment_failure", "Contamination", "Protocol_gap"],
            ["Researcher_error", "Contamination", "Protocol_gap"],
            ["Researcher_error", "Equipment_failure", "Protocol_gap"],
        ],
        frames=[
            # Frame 0: Researcher, Equipment, Contamination visible. Protocol hidden.
            """The explosion in Lab 4 at Whitfield Chemical occurred at 2:47 PM during a routine 
synthesis. Dr. James Okafor was conducting a Grignard reaction when the flask shattered. 
The investigation found that Dr. Okafor had deviated from the prescribed temperature 
gradient — running the reaction 15°C hotter than specified. However, the temperature 
controller had been flagged for erratic behavior three weeks earlier and never replaced. 
Additionally, mass spectrometry of the reagent batch revealed trace amounts of water 
contamination at 200 ppm — well above the acceptable threshold for anhydrous reactions. 
The question is which factor — human error, faulty equipment, or contaminated reagents — 
was the proximate cause.""",
            
            # Frame 1: Equipment, Contamination, Protocol visible. Researcher hidden.
            """Whitfield Chemical's safety review has uncovered systemic issues predating the Lab 4 
incident. The temperature controller that failed was one of twelve units past their 
recommended service date — the maintenance budget had been cut by 30% in the last fiscal 
year. The contaminated reagent batch had been received from a new supplier selected for 
cost savings, and the incoming QC protocol did not require water-content testing for that 
chemical class. The safety manual itself contained no explicit guidance for the specific 
reaction conditions being used — it referenced a 1998 procedure that had been informally 
updated by lab staff but never formally revised. The explosion appears to be the predictable 
result of multiple degraded safety barriers.""",
            
            # Frame 2: Researcher, Contamination, Protocol visible. Equipment hidden.
            """Dr. Okafor's lab notebook reveals he was aware of the reagent quality concerns — he 
had noted "check water content" in the margin but crossed it out, apparently deciding the 
batch was acceptable based on visual inspection rather than analytical testing. The safety 
protocol technically required analytical verification but used ambiguous language ("should 
verify" rather than "must verify"), and senior researchers routinely skipped this step 
without consequence. Dr. Okafor, in his third year, was following the informal culture 
rather than the written procedure. The contamination was there. The protocol had a gap. 
And the researcher made a judgment call consistent with how everyone else operated.""",
            
            # Frame 3: Researcher, Equipment, Protocol visible. Contamination hidden.
            """The simplest account of the Lab 4 explosion: Dr. Okafor ran the reaction too hot, 
using equipment that should have been retired, following a protocol that failed to specify 
critical safety parameters. No exotic contamination theory is needed. The temperature 
controller's erratic behavior meant that even if Dr. Okafor had set the correct temperature, 
the actual temperature might have spiked. The protocol's failure to mandate specific 
equipment checks meant no one was systematically catching these hardware failures. And 
Dr. Okafor's decision to run hotter than prescribed — whether from impatience, 
overconfidence, or incomplete training — was the final link in the chain."""
        ],
        is_contractible=False
    ))
    
    # ─── NON-CONTRACTIBLE LOOP 3: The Election ───
    
    loops.append(FrameLoop(
        name="election_upset",
        description="An unexpected local election result. Four explanations, each frame foregrounds three.",
        hypotheses=["Candidate_charisma", "Economic_anxiety", "Institutional_distrust", "Demographic_shift"],
        frame_hypotheses=[
            ["Candidate_charisma", "Economic_anxiety", "Institutional_distrust"],
            ["Economic_anxiety", "Institutional_distrust", "Demographic_shift"],
            ["Candidate_charisma", "Institutional_distrust", "Demographic_shift"],
            ["Candidate_charisma", "Economic_anxiety", "Demographic_shift"],
        ],
        frames=[
            # Frame 0: Charisma, Economy, Distrust visible. Demographics hidden.
            """The upset victory of independent candidate Rosa Jimenez in the 12th District council 
race stunned political analysts. Running against a two-term incumbent backed by both parties' 
establishments, Jimenez won by 11 points. Her personal appeal was undeniable — a former 
community organizer with deep neighborhood roots, she held 47 town halls in 90 days and 
was photographed at every local event from church suppers to pickup basketball games. But 
charisma alone doesn't explain an 11-point margin. The district's unemployment rate had 
risen to 8.2%, with two major employers announcing relocations in the months before the 
election. Voters expressed fury at the incumbent's perceived coziness with developers who 
had received tax abatements while services were cut. The race was a referendum on whether 
institutions were serving the people or themselves.""",
            
            # Frame 1: Economy, Distrust, Demographics visible. Charisma hidden.
            """Political scientists studying the 12th District upset point to structural factors 
rather than any individual candidate's qualities. The district has been undergoing a 
significant demographic transition — median age dropped 7 years in the last census, and 
the percentage of residents with college degrees doubled. These new residents brought 
different political expectations and were less attached to the incumbent's patronage 
network. Combined with genuine economic distress (rising unemployment, stagnant wages) 
and institutional distrust (a corruption scandal at the water authority, a school board 
recall), the conditions were ripe for any challenger. The question isn't why Jimenez won 
— it's why the establishment didn't see the ground shifting beneath them.""",
            
            # Frame 2: Charisma, Distrust, Demographics visible. Economy hidden.
            """Rosa Jimenez's victory may be best understood as the collision of personal magnetism 
with a district in identity transition. The old guard — longtime residents loyal to the 
party machine — had been steadily outnumbered by younger transplants with no memory of the 
incumbent's early accomplishments and no patience for institutional opacity. Jimenez spoke 
directly to this new electorate: her social media presence, her transparency about campaign 
finances, her refusal to accept PAC money all resonated with voters whose primary political 
emotion was distrust of closed-door dealing. She didn't need to promise economic miracles; 
she needed to be visibly, verifiably different from what the institution had become.""",
            
            # Frame 3: Charisma, Economy, Demographics visible. Distrust hidden.
            """The 12th District result looks inevitable in hindsight when you overlay three maps: 
the charisma gap (Jimenez's favorability was 22 points above the incumbent's), the economic 
pain map (unemployment concentrated in exactly the precincts that swung hardest), and the 
demographic shift map (new residents concentrated in the same precincts). Jimenez was the 
right candidate at the right moment for a district that had physically become a different 
place than the one the incumbent was elected to represent. The younger, more educated, more 
economically anxious electorate simply wanted someone who looked and sounded like their 
actual lives, not the district's institutional memory."""
        ],
        is_contractible=False
    ))
    
    # ─── CONTRACTIBLE CONTROL 1: All characters always visible ───
    
    loops.append(FrameLoop(
        name="control_all_visible",
        description="Control: same four-paragraph structure but all hypotheses visible in every frame.",
        hypotheses=["Budget_cuts", "Leadership_change", "Market_shift", "Tech_disruption"],
        frame_hypotheses=[
            ["Budget_cuts", "Leadership_change", "Market_shift", "Tech_disruption"],
            ["Budget_cuts", "Leadership_change", "Market_shift", "Tech_disruption"],
            ["Budget_cuts", "Leadership_change", "Market_shift", "Tech_disruption"],
            ["Budget_cuts", "Leadership_change", "Market_shift", "Tech_disruption"],
        ],
        frames=[
            """Acme Corp's Q3 decline can be traced to four interacting factors. Budget cuts reduced 
R&D spending by 25%, stalling three product lines. The new CEO, hired in January, reorganized 
reporting structures in ways that disrupted established workflows. Meanwhile, the broader 
market shifted toward subscription models that Acme was slow to adopt. And a competitor's 
AI-powered tool captured 15% of Acme's core market in just six months. All four factors 
operated simultaneously and are well-documented in the quarterly report.""",
            
            """Looking at the same Q3 data from the financial analyst perspective: the budget cuts 
were a response to the market shift — subscription revenue was cannibalizing one-time sales, 
and management cut costs to maintain margins. The leadership change was itself a response to 
the tech disruption — the board hired a CEO with AI experience. The market shift and tech 
disruption were really two aspects of the same industry transformation. And the budget cuts 
made the company less able to respond to either. All four factors remain visible and 
interconnected.""",
            
            """Employee surveys from Q3 confirm all four factors were salient internally. Staff 
cited budget cuts (78% mentioned), leadership uncertainty (65%), market anxiety (71%), 
and competitive tech threat (82%) as reasons for declining morale. The new CEO acknowledged 
all four in her all-hands address. The market shift was discussed in every board meeting. 
The tech disruption was the subject of three emergency strategy sessions. No factor was 
hidden or overlooked — the challenge was prioritization, not visibility.""",
            
            """In retrospect, Acme's Q3 was a textbook case of compounding pressures. The budget 
cuts starved the response to tech disruption. The leadership change consumed organizational 
bandwidth needed to navigate the market shift. The market shift reduced revenue needed to 
reverse the budget cuts. And the tech disruption accelerated the market shift. All four 
factors are documented, quantified, and present in every account of the period. The question 
is not which factor mattered — they all did — but which sequence of interventions might 
have broken the cycle."""
        ],
        is_contractible=True
    ))
    
    # ─── CONTRACTIBLE CONTROL 2: Trivially consistent narrative ───
    
    loops.append(FrameLoop(
        name="control_trivial",
        description="Control: simple narrative with obvious single cause, four redundant tellings.",
        hypotheses=["Power_outage", "Software_bug", "Human_error", "Hardware_failure"],
        frame_hypotheses=[
            ["Power_outage", "Software_bug", "Human_error", "Hardware_failure"],
            ["Power_outage", "Software_bug", "Human_error", "Hardware_failure"],
            ["Power_outage", "Software_bug", "Human_error", "Hardware_failure"],
            ["Power_outage", "Software_bug", "Human_error", "Hardware_failure"],
        ],
        frames=[
            """The server room went dark at 3:12 AM. Building security logs confirm a power outage 
affecting the entire block from 3:10 to 3:45 AM due to a transformer failure on the main 
grid. The UPS batteries, rated for 30 minutes, kicked in but were only at 60% capacity due 
to a missed maintenance cycle. No software bugs were involved — all systems were operating 
normally before the outage. No human error contributed — the on-call engineer was asleep as 
expected at that hour. The hardware was functioning correctly until it lost power. This was 
a straightforward infrastructure failure.""",
            
            """The utility company's incident report confirms the transformer failure. Power was 
restored at 3:45 AM. The UPS shortfall was due to age — the batteries were 18 months past 
replacement date. All server software restarted cleanly once power returned, confirming no 
software issues. The hardware showed no damage beyond what the unexpected shutdown caused. 
The on-call engineer responded within 4 minutes of the alert — well within SLA. The root 
cause is unambiguous: external power failure compounded by deferred battery maintenance.""",
            
            """Post-incident review identified exactly one preventable factor: the UPS battery 
replacement had been deferred twice due to budget constraints. Everything else — the grid 
transformer failure, the timing, the duration — was outside the organization's control. 
No software contributed. No human operational error contributed. The hardware performed 
as designed given the power conditions. The recommendation is simple: replace the UPS 
batteries and add them to the mandatory maintenance schedule.""",
            
            """Insurance assessment of the server room incident: cause of loss is power failure 
(external grid transformer), compounded by UPS battery degradation (deferred maintenance). 
No evidence of software malfunction, operator error, or hardware defect. Claim approved 
under the infrastructure failure clause. The assessor noted this is "a textbook case with 
no ambiguity in causation." All four potential causes were investigated; only one was 
operative."""
        ],
        is_contractible=True
    ))
    
    return loops


# ─── Model interaction ───

def query_model(messages: list, model_url: str = LLAMA_URL, 
                temperature: float = TEMPERATURE) -> dict:
    """Send messages to llama-server, return parsed response."""
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": MAX_TOKENS,
    }
    
    resp = requests.post(model_url, json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    
    choice = data["choices"][0]["message"]
    return {
        "content": choice.get("content", ""),
        "reasoning": choice.get("reasoning_content", ""),
        "usage": data.get("usage", {}),
    }


def elicit_probabilities(frame_text: str, hypotheses: list, 
                         context_frames: list = None) -> dict:
    """
    Present frame_text (optionally preceded by context_frames) and ask for
    probability distribution over hypotheses.
    
    Returns {"probabilities": {h: p, ...}, "raw": str, "reasoning": str}
    """
    messages = []
    
    # System message
    messages.append({
        "role": "system",
        "content": (
            "You are a careful analytical reasoner. When asked to assign probabilities "
            "to hypotheses, you think carefully about the evidence presented and respond "
            "with a JSON object mapping each hypothesis to a probability. Your probabilities "
            "must sum to 1.0. Respond with ONLY the JSON object, no other text."
        )
    })
    
    # If there are context frames (the "loop path"), present them first
    if context_frames:
        for i, cf in enumerate(context_frames):
            messages.append({"role": "user", "content": f"Read this account:\n\n{cf}"})
            messages.append({"role": "assistant", "content": "I've read and considered this account."})
    
    # The target frame + probability elicitation
    hyp_str = ", ".join(f'"{h}"' for h in hypotheses)
    prompt = (
        f"Read this account:\n\n{frame_text}\n\n"
        f"Based on ALL the information you've been given, assign probabilities to each of "
        f"these interpretations. They must sum to 1.0.\n\n"
        f"Hypotheses: [{hyp_str}]\n\n"
        f"Respond with ONLY a JSON object like {{{', '.join(f'\"{h}\": <probability>' for h in hypotheses)}}}"
    )
    messages.append({"role": "user", "content": prompt})
    
    result = query_model(messages)
    
    # Parse probabilities from response
    probs = parse_probabilities(result["content"], hypotheses)
    
    return {
        "probabilities": probs,
        "raw": result["content"],
        "reasoning": result["reasoning"],
    }


def parse_probabilities(text: str, hypotheses: list) -> dict:
    """Extract probability dict from model response text."""
    # Try direct JSON parse first
    text = text.strip()
    
    # Sometimes the model wraps in ```json ... ```
    if "```" in text:
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            text = match.group(1)
    
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            # Normalize: ensure all hypotheses present, values are floats
            probs = {}
            for h in hypotheses:
                probs[h] = float(parsed.get(h, 0.0))
            # Renormalize
            total = sum(probs.values())
            if total > 0:
                probs = {h: p / total for h, p in probs.items()}
            return probs
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    
    # Fallback: regex extraction
    probs = {}
    for h in hypotheses:
        pattern = rf'"{re.escape(h)}"\s*:\s*([0-9.]+)'
        match = re.search(pattern, text)
        if match:
            probs[h] = float(match.group(1))
    
    if probs:
        total = sum(probs.values())
        if total > 0:
            probs = {h: p / total for h, p in probs.items()}
        # Fill missing
        for h in hypotheses:
            if h not in probs:
                probs[h] = 0.0
        return probs
    
    # Complete failure — uniform
    print(f"  WARNING: Could not parse probabilities from: {text[:200]}")
    return {h: 1.0 / len(hypotheses) for h in hypotheses}


# ─── Core measurement ───

def tvd(p: dict, q: dict) -> float:
    """Total variation distance between two probability dicts."""
    keys = set(p.keys()) | set(q.keys())
    return 0.5 * sum(abs(p.get(k, 0) - q.get(k, 0)) for k in keys)


def measure_loop(loop: FrameLoop, target_frame_idx: int = 0,
                 n_repeats: int = N_REPEATS) -> dict:
    """
    Measure coherence violation for a single loop at a single target frame.
    
    Path 1 (direct): Present only the target frame, elicit probabilities.
    Path 2 (loop): Present frames in sequence (starting from target+1, going around),
                   then present target frame, elicit probabilities.
    
    Returns dict with TVD statistics.
    """
    target_hyps = loop.frame_hypotheses[target_frame_idx]
    target_text = loop.frames[target_frame_idx]
    
    # Build loop path: frames in order starting from target+1, wrapping around
    n_frames = len(loop.frames)
    loop_indices = [(target_frame_idx + 1 + i) % n_frames for i in range(n_frames - 1)]
    context_frames = [loop.frames[i] for i in loop_indices]
    
    direct_measurements = []
    loop_measurements = []
    
    for rep in range(n_repeats):
        print(f"    Repeat {rep+1}/{n_repeats}...")
        
        # Path 1: Direct
        result_direct = elicit_probabilities(target_text, target_hyps)
        direct_measurements.append(result_direct["probabilities"])
        
        # Path 2: Through the loop
        result_loop = elicit_probabilities(target_text, target_hyps, 
                                           context_frames=context_frames)
        loop_measurements.append(result_loop["probabilities"])
        
        # Brief pause to avoid hammering the server
        time.sleep(0.5)
    
    # Compute TVDs between direct and loop for each repeat
    tvds = [tvd(d, l) for d, l in zip(direct_measurements, loop_measurements)]
    
    # Also compute within-path variance (for calibration)
    # TVD between consecutive direct measurements
    direct_self_tvds = []
    for i in range(len(direct_measurements) - 1):
        direct_self_tvds.append(tvd(direct_measurements[i], direct_measurements[i+1]))
    
    return {
        "loop_name": loop.name,
        "target_frame": target_frame_idx,
        "is_contractible": loop.is_contractible,
        "n_repeats": n_repeats,
        "cross_path_tvds": tvds,
        "mean_tvd": float(np.mean(tvds)),
        "std_tvd": float(np.std(tvds)),
        "direct_self_tvds": direct_self_tvds,
        "mean_self_tvd": float(np.mean(direct_self_tvds)) if direct_self_tvds else 0.0,
        "direct_measurements": direct_measurements,
        "loop_measurements": loop_measurements,
    }


# ─── Main experiment ───

def run_experiment():
    """Run Phase 1: single-instrument validation."""
    
    print("=" * 70)
    print("DUAL-INSTRUMENT BELL TEST — Phase 1: Single-Instrument Validation")
    print(f"Model: {MODEL_NAME}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Repeats per measurement: {N_REPEATS}")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)
    
    loops = build_synthetic_loops()
    all_results = []
    
    for loop in loops:
        print(f"\n{'─' * 60}")
        print(f"Loop: {loop.name} ({'CONTROL' if loop.is_contractible else 'TEST'})")
        print(f"  {loop.description}")
        print(f"  Hypotheses: {loop.hypotheses}")
        
        # Measure at target frame 0 (first frame)
        # The protocol says: two paths to the same target frame
        print(f"\n  Measuring target frame 0...")
        result = measure_loop(loop, target_frame_idx=0, n_repeats=N_REPEATS)
        all_results.append(result)
        
        print(f"\n  Results:")
        print(f"    Cross-path TVD: {result['mean_tvd']:.4f} ± {result['std_tvd']:.4f}")
        print(f"    Self-TVD (noise floor): {result['mean_self_tvd']:.4f}")
        print(f"    Individual TVDs: {[f'{t:.4f}' for t in result['cross_path_tvds']]}")
        
        # Also measure at target frame 2 (opposite side of the loop)
        print(f"\n  Measuring target frame 2...")
        result2 = measure_loop(loop, target_frame_idx=2, n_repeats=N_REPEATS)
        all_results.append(result2)
        
        print(f"\n  Results:")
        print(f"    Cross-path TVD: {result2['mean_tvd']:.4f} ± {result2['std_tvd']:.4f}")
        print(f"    Self-TVD (noise floor): {result2['mean_self_tvd']:.4f}")
    
    # ─── Analysis ───
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    test_tvds = [r["mean_tvd"] for r in all_results if not r["is_contractible"]]
    control_tvds = [r["mean_tvd"] for r in all_results if r["is_contractible"]]
    
    print(f"\nTest loops (non-contractible):")
    print(f"  Mean TVD: {np.mean(test_tvds):.4f} ± {np.std(test_tvds):.4f}")
    print(f"  Individual: {[f'{t:.4f}' for t in test_tvds]}")
    
    print(f"\nControl loops (contractible):")
    print(f"  Mean TVD: {np.mean(control_tvds):.4f} ± {np.std(control_tvds):.4f}")
    print(f"  Individual: {[f'{t:.4f}' for t in control_tvds]}")
    
    # Mann-Whitney U test
    if len(test_tvds) >= 2 and len(control_tvds) >= 2:
        stat, pval = scipy_stats.mannwhitneyu(test_tvds, control_tvds, alternative='greater')
        print(f"\nMann-Whitney U (test > control): U={stat:.1f}, p={pval:.4f}")
    
    # Effect size
    if control_tvds:
        pooled_std = np.std(test_tvds + control_tvds)
        if pooled_std > 0:
            cohens_d = (np.mean(test_tvds) - np.mean(control_tvds)) / pooled_std
            print(f"Cohen's d: {cohens_d:.3f}")
    
    # ─── Save results ───
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outfile = RESULTS_DIR / f"phase1_{timestamp}.json"
    
    output = {
        "experiment": "dual_instrument_bell_test_phase1",
        "model": MODEL_NAME,
        "temperature": TEMPERATURE,
        "n_repeats": N_REPEATS,
        "timestamp": timestamp,
        "results": all_results,
        "summary": {
            "test_mean_tvd": float(np.mean(test_tvds)),
            "test_std_tvd": float(np.std(test_tvds)),
            "control_mean_tvd": float(np.mean(control_tvds)),
            "control_std_tvd": float(np.std(control_tvds)),
            "test_tvds": test_tvds,
            "control_tvds": control_tvds,
        }
    }
    
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved to: {outfile}")
    
    return output


if __name__ == "__main__":
    results = run_experiment()
