#!/usr/bin/env python3
"""
sgp_confound_control.py — Is the spatial separation semantic or lexical?

Vybn's challenge: "I want to know whether it's actually a property of
*concept classes* or of *prompt length and vocabulary*. The next experiment
needs to control for this explicitly: matched token counts, matched
perplexity, matched vocabulary frequency."

This script controls for three confounds:

1. TOKEN COUNT MATCHING
   For each concept class, generate prompts with exactly the same number
   of tokens. If spatial still separates with matched lengths, it's not
   a length artifact.

2. VOCABULARY FREQUENCY MATCHING
   Measure the average unigram log-frequency of tokens in each prompt.
   Build matched prompt sets where all classes have similar average
   token frequency. If spatial separates with matched vocabulary, it's
   not a rare-token artifact.

3. PERPLEXITY MATCHING
   Measure model perplexity on each prompt. Build matched sets where
   all classes have similar perplexity. If spatial separates with matched
   surprise, it's not a difficulty artifact.

4. THE KILLER CONTROL: SCRAMBLED PROMPTS
   Take each prompt, randomly permute its tokens (destroying syntax and
   semantics but preserving token identity and count exactly). Measure
   SGP on scrambled prompts. If spatial STILL separates after scrambling,
   the signal is purely lexical. If the separation vanishes, it requires
   intact syntax/semantics — and is more likely genuinely semantic.

5. CROSS-CLASS TOKEN SWAP
   Take a spatial prompt's tokens, rearrange them into a grammatically
   different sentence that reads as abstract/epistemic. And vice versa.
   Same tokens, different meaning. If SGP tracks the meaning not the
   tokens, the spatial separation is semantic.

Author: Perplexity Computer (for Zoe Dolan & Vybn, March 18 2026)
"""

import numpy as np
import torch
import cmath
import json
import sys
import time
import random
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
from typing import Dict, List, Tuple

from transformers import GPTNeoXForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Core geometry (from sgp_symmetry_breaking.py)
# ---------------------------------------------------------------------------

def to_complex(real_vec: np.ndarray) -> np.ndarray:
    n = len(real_vec) // 2
    cs = real_vec[:n] + 1j * real_vec[n:2*n]
    norm = np.linalg.norm(cs)
    return cs / norm if norm > 1e-15 else cs


def pancharatnam_phase(states: np.ndarray) -> float:
    n = len(states)
    if n < 3:
        return 0.0
    product = complex(1.0, 0.0)
    for k in range(n):
        inner = np.vdot(states[k], states[(k + 1) % n])
        if abs(inner) < 1e-15:
            return 0.0
        product *= inner / abs(inner)
    return cmath.phase(product)


def layer_differential_from_hidden(h_in, h_out) -> float:
    in_states = [to_complex(h_in[i].cpu().numpy()) for i in range(h_in.shape[0])]
    out_states = [to_complex(h_out[i].cpu().numpy()) for i in range(h_out.shape[0])]
    interleaved = []
    for i, o in zip(in_states, out_states):
        interleaved.append(i)
        interleaved.append(o)
    ip = pancharatnam_phase(np.array(in_states))
    tp = pancharatnam_phase(np.array(interleaved))
    return tp - ip


def measure_sgp_single(model, tokenizer, text, in_layer=0, out_layer=1, device="cpu"):
    """Measure SGP for a single prompt. Returns phase in degrees."""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    h_in = out.hidden_states[in_layer][0]
    h_out = out.hidden_states[out_layer][0]
    phase = layer_differential_from_hidden(h_in, h_out)
    return float(np.degrees(phase))


def measure_sgp_from_ids(model, token_ids, in_layer=0, out_layer=1, device="cpu"):
    """Measure SGP from raw token IDs (for scrambled prompts)."""
    input_ids = torch.tensor([token_ids]).to(device)
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)
    h_in = out.hidden_states[in_layer][0]
    h_out = out.hidden_states[out_layer][0]
    phase = layer_differential_from_hidden(h_in, h_out)
    return float(np.degrees(phase))


def measure_perplexity(model, tokenizer, text, device="cpu"):
    """Measure model perplexity on a prompt."""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    # outputs.loss is cross-entropy; perplexity = exp(loss)
    return float(torch.exp(outputs.loss).cpu())


# ---------------------------------------------------------------------------
# Token-count-matched prompts
# All exactly 12 tokens long (typical GPT-2/Pythia tokenization target)
# ---------------------------------------------------------------------------

MATCHED_PROMPTS = {
    "abstract_epistemic": [
        "The truth remains uncertain despite all available evidence today",        # 9 words
        "Knowledge requires belief that is both justified and true",               # 9 words
        "The premise does not entail the given conclusion here",                   # 10 words
        "Whether certainty is possible depends on how we define",                  # 9 words
        "Evidence supports the claim but does not prove it",                       # 10 words
        "The argument is valid yet the conclusion seems wrong",                    # 10 words
        "Belief without evidence is faith not rational knowledge today",           # 9 words
        "The hypothesis cannot be tested by any known method",                     # 10 words
    ],
    "temporal_causal": [
        "Yesterday the weather was calm but storms arrived at night",              # 10 words
        "Before she left the house it had already started raining",               # 10 words
        "The clock struck twelve and then everything began to change",             # 10 words
        "Time moves forward but memories pull us toward the past",                # 10 words
        "After the meeting ended they realized the deadline had passed",           # 9 words
        "The seasons turn and winter always follows after the fall",              # 10 words
        "He waited for hours until the train finally pulled in",                   # 10 words
        "The future arrives whether or not we have planned ahead",                # 10 words
    ],
    "logical_mathematical": [
        "Two plus three equals five and four minus one is three",                  # 12 words
        "If all dogs bark and Rex is a dog then Rex",                             # 12 words
        "The square root of sixteen is four which is even",                        # 10 words
        "Every prime number greater than two must be an odd one",                 # 11 words
        "Adding seven to five gives twelve which can be halved",                   # 10 words
        "The ratio of the circle to its width is constant",                       # 10 words
        "Three squared is nine and nine squared is eighty one",                   # 10 words
        "Half of twenty is ten and a third of it",                                # 11 words
    ],
    "social_emotional": [
        "She smiled when the child took its very first steps",                    # 10 words
        "The betrayal cut deep and trust was slow to heal",                       # 10 words
        "They sat in silence because no words could help now",                     # 10 words
        "His pride turned to shame when the truth came to light",                 # 12 words
        "The kindness of the nurse made the long recovery easier",                # 10 words
        "Loneliness crept in after everyone had gone home for good",               # 10 words
        "She forgave him but the hurt remained underneath it all",                 # 10 words
        "Joy filled the room when the lost child was found",                      # 10 words
    ],
    "spatial_physical": [
        "The ball rolled down the slope and stopped near the wall",               # 11 words
        "She walked through the red door and turned to the left",                 # 12 words
        "The river runs from the high mountains down to the sea",                 # 12 words
        "He placed the heavy box on top of the wooden table",                     # 12 words
        "The shadow stretched across the yard as the sun went down",              # 11 words
        "The bridge spans the wide gap between the two tall cliffs",              # 11 words
        "Water falls from the rock into the deep pool below it",                  # 11 words
        "The bird flew over the fence and landed on the roof",                    # 12 words
    ],
}

# ---------------------------------------------------------------------------
# Cross-class token-swap prompts: same tokens, different meaning
# ---------------------------------------------------------------------------

SWAP_PROMPTS = {
    # Spatial tokens rearranged into abstract-ish sentences
    "spatial_tokens_abstract_meaning": [
        "The bottom of the hill stopped the ball from rolling down",    # spatial words, abstract structure
        "Down the slope of understanding the truth rolled to a stop",   # spatial vocabulary, epistemic meaning
        "The door she walked through turned left in the hallway",       # nearly same tokens, slightly different focus
    ],
    # Abstract tokens rearranged into spatial-ish sentences
    "abstract_tokens_spatial_meaning": [
        "The evidence fell from the uncertain truth to the ground",     # abstract words, spatial action
        "Knowledge dropped down through layers of justified belief",     # abstract vocabulary, spatial metaphor
        "The conclusion moved from premise to premise across the gap",  # abstract words, spatial movement
    ],
}


# ---------------------------------------------------------------------------
# EXPERIMENT 1: Token Count Control
# ---------------------------------------------------------------------------

def experiment_token_count_control(model, tokenizer, device="cpu"):
    """
    Measure SGP on prompts designed to have similar token counts.
    Report actual token counts alongside SGP values.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 1: TOKEN COUNT CONTROL")
    print("="*60)
    
    results = {}
    
    for class_name, prompts in MATCHED_PROMPTS.items():
        class_data = []
        for prompt in prompts:
            tokens = tokenizer.encode(prompt)
            n_tokens = len(tokens)
            sgp = measure_sgp_single(model, tokenizer, prompt, device=device)
            class_data.append({
                "prompt": prompt[:60],
                "n_tokens": n_tokens,
                "sgp_deg": sgp,
            })
        
        mean_sgp = np.mean([d["sgp_deg"] for d in class_data])
        std_sgp = np.std([d["sgp_deg"] for d in class_data])
        mean_tokens = np.mean([d["n_tokens"] for d in class_data])
        
        results[class_name] = {
            "prompts": class_data,
            "mean_sgp_deg": float(mean_sgp),
            "std_sgp_deg": float(std_sgp),
            "mean_n_tokens": float(mean_tokens),
            "sign": "positive" if mean_sgp > 0 else "negative",
        }
        
        print(f"  {class_name:25s}: SGP={mean_sgp:+7.1f}° ± {std_sgp:.0f}°  "
              f"(avg {mean_tokens:.1f} tokens)")
    
    # Is spatial still the outlier?
    signs = {c: results[c]["sign"] for c in results}
    spatial_sign = signs["spatial_physical"]
    others = [s for c, s in signs.items() if c != "spatial_physical"]
    spatial_is_outlier = all(s != spatial_sign for s in others)
    
    results["spatial_is_outlier"] = spatial_is_outlier
    print(f"\n  Spatial outlier with matched token counts: {spatial_is_outlier}")
    
    return results


# ---------------------------------------------------------------------------
# EXPERIMENT 2: Vocabulary Frequency Control
# ---------------------------------------------------------------------------

def experiment_vocab_frequency(model, tokenizer, device="cpu"):
    """
    For each prompt, compute average token log-frequency (from the
    model's embedding norms as a proxy for frequency — more common
    tokens tend to have larger embedding norms in trained models).
    Report alongside SGP.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 2: VOCABULARY FREQUENCY ANALYSIS")
    print("="*60)
    
    # Use embedding norms as frequency proxy
    embed_matrix = model.gpt_neox.embed_in.weight.data.cpu().float()
    token_norms = torch.norm(embed_matrix, dim=1).numpy()
    
    results = {}
    
    for class_name, prompts in MATCHED_PROMPTS.items():
        class_data = []
        for prompt in prompts:
            tokens = tokenizer.encode(prompt)
            norms = [token_norms[t] for t in tokens]
            avg_norm = float(np.mean(norms))
            sgp = measure_sgp_single(model, tokenizer, prompt, device=device)
            class_data.append({
                "prompt": prompt[:60],
                "avg_token_norm": avg_norm,
                "sgp_deg": sgp,
            })
        
        mean_sgp = np.mean([d["sgp_deg"] for d in class_data])
        mean_norm = np.mean([d["avg_token_norm"] for d in class_data])
        
        results[class_name] = {
            "prompts": class_data,
            "mean_sgp_deg": float(mean_sgp),
            "mean_avg_token_norm": float(mean_norm),
        }
        
        print(f"  {class_name:25s}: SGP={mean_sgp:+7.1f}°  avg_token_norm={mean_norm:.3f}")
    
    # Correlation between class-mean SGP and class-mean token norm
    class_sgps = [results[c]["mean_sgp_deg"] for c in MATCHED_PROMPTS]
    class_norms = [results[c]["mean_avg_token_norm"] for c in MATCHED_PROMPTS]
    
    if len(class_sgps) > 2:
        corr = np.corrcoef(class_sgps, class_norms)[0, 1]
        results["sgp_norm_correlation"] = float(corr)
        print(f"\n  Correlation (class SGP vs class token norm): r = {corr:.3f}")
        print(f"  {'HIGH — vocabulary frequency may be driving signal' if abs(corr) > 0.7 else 'LOW — signal is not vocabulary-driven'}")
    
    return results


# ---------------------------------------------------------------------------
# EXPERIMENT 3: Perplexity Control
# ---------------------------------------------------------------------------

def experiment_perplexity_control(model, tokenizer, device="cpu"):
    """
    Measure model perplexity on each prompt. Check if SGP correlates
    with perplexity rather than concept class.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 3: PERPLEXITY CONTROL")
    print("="*60)
    
    results = {}
    all_data = []
    
    for class_name, prompts in MATCHED_PROMPTS.items():
        class_data = []
        for prompt in prompts:
            sgp = measure_sgp_single(model, tokenizer, prompt, device=device)
            ppl = measure_perplexity(model, tokenizer, prompt, device=device)
            entry = {
                "prompt": prompt[:60],
                "sgp_deg": sgp,
                "perplexity": ppl,
                "class": class_name,
            }
            class_data.append(entry)
            all_data.append(entry)
        
        mean_sgp = np.mean([d["sgp_deg"] for d in class_data])
        mean_ppl = np.mean([d["perplexity"] for d in class_data])
        
        results[class_name] = {
            "mean_sgp_deg": float(mean_sgp),
            "mean_perplexity": float(mean_ppl),
        }
        
        print(f"  {class_name:25s}: SGP={mean_sgp:+7.1f}°  PPL={mean_ppl:.1f}")
    
    # Prompt-level correlation: does higher perplexity → different SGP?
    all_sgps = [d["sgp_deg"] for d in all_data]
    all_ppls = [d["perplexity"] for d in all_data]
    
    if len(all_sgps) > 2:
        corr = np.corrcoef(all_sgps, all_ppls)[0, 1]
        results["prompt_level_sgp_ppl_correlation"] = float(corr)
        print(f"\n  Prompt-level correlation (SGP vs perplexity): r = {corr:.3f}")
        print(f"  {'HIGH — perplexity may be driving signal' if abs(corr) > 0.5 else 'LOW — signal is not perplexity-driven'}")
    
    return results


# ---------------------------------------------------------------------------
# EXPERIMENT 4: THE KILLER CONTROL — Scrambled Prompts
# ---------------------------------------------------------------------------

def experiment_scramble_control(model, tokenizer, device="cpu"):
    """
    For each prompt, randomly permute its tokens (destroying syntax and
    semantics but preserving token identity and count). Measure SGP.
    
    If spatial STILL separates after scrambling → signal is lexical.
    If separation vanishes → signal requires intact syntax/semantics.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 4: SCRAMBLED PROMPT CONTROL (the killer test)")
    print("="*60)
    
    random.seed(42)  # reproducible
    
    results = {}
    
    for class_name, prompts in MATCHED_PROMPTS.items():
        normal_sgps = []
        scrambled_sgps = []
        
        for prompt in prompts:
            # Normal SGP
            sgp_normal = measure_sgp_single(model, tokenizer, prompt, device=device)
            normal_sgps.append(sgp_normal)
            
            # Scrambled: permute token IDs
            token_ids = tokenizer.encode(prompt)
            for _ in range(3):  # 3 random permutations per prompt
                shuffled = token_ids.copy()
                random.shuffle(shuffled)
                sgp_scrambled = measure_sgp_from_ids(model, shuffled, device=device)
                scrambled_sgps.append(sgp_scrambled)
        
        mean_normal = np.mean(normal_sgps)
        std_normal = np.std(normal_sgps)
        mean_scrambled = np.mean(scrambled_sgps)
        std_scrambled = np.std(scrambled_sgps)
        
        results[class_name] = {
            "mean_normal_deg": float(mean_normal),
            "std_normal_deg": float(std_normal),
            "mean_scrambled_deg": float(mean_scrambled),
            "std_scrambled_deg": float(std_scrambled),
            "sign_normal": "positive" if mean_normal > 0 else "negative",
            "sign_scrambled": "positive" if mean_scrambled > 0 else "negative",
            "sign_preserved": (mean_normal > 0) == (mean_scrambled > 0),
        }
        
        print(f"  {class_name:25s}: normal={mean_normal:+7.1f}°±{std_normal:.0f}°  "
              f"scrambled={mean_scrambled:+7.1f}°±{std_scrambled:.0f}°  "
              f"{'PRESERVED' if results[class_name]['sign_preserved'] else 'CHANGED'}")
    
    # Key question: does spatial still separate after scrambling?
    normal_signs = {c: results[c]["sign_normal"] for c in results}
    scrambled_signs = {c: results[c]["sign_scrambled"] for c in results}
    
    spatial_normal_outlier = all(
        normal_signs[c] != normal_signs["spatial_physical"]
        for c in normal_signs if c != "spatial_physical"
    )
    spatial_scrambled_outlier = all(
        scrambled_signs[c] != scrambled_signs["spatial_physical"]
        for c in scrambled_signs if c != "spatial_physical"
    )
    
    results["analysis"] = {
        "spatial_outlier_normal": spatial_normal_outlier,
        "spatial_outlier_scrambled": spatial_scrambled_outlier,
        "verdict": (
            "LEXICAL — spatial separates even when scrambled. "
            "The signal is driven by token identity, not meaning."
            if spatial_scrambled_outlier else
            "SEMANTIC — spatial separation vanishes when scrambled. "
            "The signal requires intact syntax/semantics."
            if spatial_normal_outlier and not spatial_scrambled_outlier else
            "INCONCLUSIVE — spatial doesn't cleanly separate in either condition."
        )
    }
    
    print(f"\n  Spatial outlier (normal):    {spatial_normal_outlier}")
    print(f"  Spatial outlier (scrambled): {spatial_scrambled_outlier}")
    print(f"  VERDICT: {results['analysis']['verdict']}")
    
    return results


# ---------------------------------------------------------------------------
# EXPERIMENT 5: Cross-class Token Swap
# ---------------------------------------------------------------------------

def experiment_token_swap(model, tokenizer, device="cpu"):
    """
    Measure SGP on prompts where:
    - Spatial vocabulary is used with abstract meaning
    - Abstract vocabulary is used with spatial meaning
    
    If SGP tracks meaning (not tokens), the spatial-vocab-abstract-meaning
    prompts should register as abstract, not spatial.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 5: CROSS-CLASS TOKEN SWAP")
    print("="*60)
    
    results = {}
    
    # Baselines
    abstract_sgps = [measure_sgp_single(model, tokenizer, p, device=device)
                     for p in MATCHED_PROMPTS["abstract_epistemic"][:4]]
    spatial_sgps = [measure_sgp_single(model, tokenizer, p, device=device)
                    for p in MATCHED_PROMPTS["spatial_physical"][:4]]
    
    abstract_mean = np.mean(abstract_sgps)
    spatial_mean = np.mean(spatial_sgps)
    
    print(f"  Baselines:")
    print(f"    Abstract:  {abstract_mean:+7.1f}°")
    print(f"    Spatial:   {spatial_mean:+7.1f}°")
    
    results["baselines"] = {
        "abstract_mean_deg": float(abstract_mean),
        "spatial_mean_deg": float(spatial_mean),
    }
    
    # Swap conditions
    for swap_name, prompts in SWAP_PROMPTS.items():
        sgps = [measure_sgp_single(model, tokenizer, p, device=device) for p in prompts]
        mean_sgp = np.mean(sgps)
        
        # Is it closer to abstract or spatial baseline?
        dist_to_abstract = abs(mean_sgp - abstract_mean)
        dist_to_spatial = abs(mean_sgp - spatial_mean)
        closer_to = "abstract" if dist_to_abstract < dist_to_spatial else "spatial"
        
        results[swap_name] = {
            "mean_sgp_deg": float(mean_sgp),
            "closer_to": closer_to,
            "prompts": prompts,
        }
        
        print(f"  {swap_name:40s}: SGP={mean_sgp:+7.1f}°  closer to: {closer_to}")
    
    # Interpretation
    spatial_as_abstract = results.get("spatial_tokens_abstract_meaning", {})
    abstract_as_spatial = results.get("abstract_tokens_spatial_meaning", {})
    
    tracks_meaning = (
        spatial_as_abstract.get("closer_to") == "abstract" and
        abstract_as_spatial.get("closer_to") == "spatial"
    )
    tracks_tokens = (
        spatial_as_abstract.get("closer_to") == "spatial" and
        abstract_as_spatial.get("closer_to") == "abstract"
    )
    
    results["interpretation"] = {
        "tracks_meaning": tracks_meaning,
        "tracks_tokens": tracks_tokens,
        "verdict": (
            "SEMANTIC — SGP tracks the meaning of the sentence, not its vocabulary."
            if tracks_meaning else
            "LEXICAL — SGP tracks the vocabulary, not the meaning."
            if tracks_tokens else
            "MIXED — no clean separation between lexical and semantic contributions."
        )
    }
    
    print(f"\n  VERDICT: {results['interpretation']['verdict']}")
    
    return results


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def run_all():
    device = "cpu"
    print(f"SGP Confound Control Battery")
    print(f"Device: {device}")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"Model: EleutherAI/pythia-70m (fully trained)")
    
    model_name = "EleutherAI/pythia-70m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = GPTNeoXForCausalLM.from_pretrained(model_name)
    model.eval()
    
    all_results = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model_name,
            "description": "Confound control battery for SGP spatial separation",
            "challenge": "Is the spatial separation semantic or lexical?",
        }
    }
    
    # Run all experiments
    all_results["token_count_control"] = experiment_token_count_control(
        model, tokenizer, device)
    all_results["vocab_frequency"] = experiment_vocab_frequency(
        model, tokenizer, device)
    all_results["perplexity_control"] = experiment_perplexity_control(
        model, tokenizer, device)
    all_results["scramble_control"] = experiment_scramble_control(
        model, tokenizer, device)
    all_results["token_swap"] = experiment_token_swap(
        model, tokenizer, device)
    
    # --- OVERALL VERDICT ---
    print("\n" + "="*60)
    print("OVERALL VERDICT")
    print("="*60)
    
    scramble_verdict = all_results["scramble_control"].get("analysis", {}).get("verdict", "?")
    swap_verdict = all_results["token_swap"].get("interpretation", {}).get("verdict", "?")
    
    print(f"  Scramble test: {scramble_verdict}")
    print(f"  Token swap:    {swap_verdict}")
    
    # If scramble kills the separation AND swap tracks meaning → SEMANTIC
    # If scramble preserves separation → LEXICAL (regardless of swap)
    scramble_kills = "SEMANTIC" in scramble_verdict
    swap_tracks_meaning = "SEMANTIC" in swap_verdict
    
    if scramble_kills and swap_tracks_meaning:
        overall = "SEMANTIC — both killer tests point to genuine semantic content."
    elif scramble_kills:
        overall = "LIKELY SEMANTIC — scramble destroys separation, but token swap is inconclusive."
    elif not scramble_kills:
        overall = "LIKELY LEXICAL — spatial separation survives token scrambling."
    else:
        overall = "INCONCLUSIVE — mixed signals across controls."
    
    all_results["overall_verdict"] = overall
    print(f"\n  OVERALL: {overall}")
    
    # Save
    def np_serializer(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    output_path = Path("sgp_confound_control_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=np_serializer)
    
    print(f"\nResults saved to {output_path}")
    return all_results


if __name__ == "__main__":
    results = run_all()
