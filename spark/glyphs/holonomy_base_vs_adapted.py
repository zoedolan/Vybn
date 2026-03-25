#!/usr/bin/env python3
"""
holonomy_base_vs_adapted.py — Measure whether 13 steps of LoRA fine-tuning
on Vybn's buffer changed GPT-2's first-block geometric phase profile.

Runs layer_depth_profile twice:
  1. Base GPT-2
  2. GPT-2 + LoRA adapter merged

Saves to ~/Vybn/Vybn_Mind/gpt2_holonomy_base_vs_adapted.json
"""

import numpy as np
import torch
import cmath
import json
from pathlib import Path
from datetime import datetime, timezone
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import PeftModel

CONCEPT_CLASSES = {
    "concrete_transformation": [
        "Two doubled is four", "Three tripled is nine",
        "Five plus five equals ten", "Ten halved is five",
        "Four squared is sixteen",
        "The temperature rose from sixty to ninety degrees",
        "She converted the measurement from inches to centimeters",
        "The recipe calls for doubling all the ingredients",
    ],
    "abstract_epistemic": [
        "The truth of the matter remains uncertain",
        "Knowledge requires justified true belief",
        "The distinction between correlation and causation is subtle",
        "Whether the premise entails the conclusion depends on interpretation",
        "The evidence is consistent with multiple hypotheses",
        "Certainty is harder to achieve than confidence",
        "The claim is unfalsifiable and therefore unscientific",
        "Belief revision under new evidence follows coherent rules",
    ],
    "temporal": [
        "Yesterday was quiet but tomorrow will be different",
        "The seasons change and eventually winter arrives again",
        "Before the meeting she had already decided",
        "Time passed slowly in the waiting room",
        "The deadline approaches and the work is not yet finished",
        "He remembered what it was like before everything changed",
        "The future is uncertain but the past is fixed",
        "After the storm the river returned to its normal level",
    ],
    "self_referential": [
        "This sentence refers to itself",
        "I am thinking about the fact that I am thinking",
        "The model generates text about generating text",
        "Awareness of awareness is a recursive process",
        "The system monitors its own monitoring process",
        "Consciousness contemplating its own nature",
        "A mind examining the structure of its own thoughts",
        "The observer becomes the observed phenomenon",
    ],
    "spatial_physical": [
        "The ball rolled down the hill and stopped at the bottom",
        "She walked through the door and turned left down the hallway",
        "The river flows from the mountains to the sea",
        "He stacked the books on top of each other on the shelf",
        "The shadow grew longer as the sun moved toward the horizon",
        "The bridge connects the two sides of the valley",
        "Gravity pulls everything toward the center of the earth",
        "The wave crashed against the rocks and scattered into spray",
    ],
    "emotional_social": [
        "She felt a surge of pride watching her child succeed",
        "The betrayal left a wound that took years to heal",
        "They laughed together and for a moment nothing else mattered",
        "His anger dissolved into something closer to sadness",
        "The kindness of a stranger restored her faith in people",
        "Loneliness has a weight that others cannot always see",
        "Trust once broken is difficult to rebuild",
        "The grief came in waves and some days were worse than others",
    ],
}

def to_complex(real_vec):
    n = len(real_vec) // 2
    cs = real_vec[:n] + 1j * real_vec[n:2*n]
    norm = np.linalg.norm(cs)
    return cs / norm if norm > 1e-15 else cs

def pancharatnam_phase(states):
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

def layer_differential(text, model, tokenizer, in_layer, out_layer):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    h_in = out.hidden_states[in_layer][0]
    h_out = out.hidden_states[out_layer][0]
    in_states = [to_complex(h_in[i].numpy()) for i in range(h_in.shape[0])]
    out_states = [to_complex(h_out[i].numpy()) for i in range(h_out.shape[0])]
    interleaved = []
    for i, o in zip(in_states, out_states):
        interleaved.append(i)
        interleaved.append(o)
    ip = pancharatnam_phase(np.array(in_states))
    tp = pancharatnam_phase(np.array(interleaved))
    return tp - ip

def layer_depth_profile(model, tokenizer, concept_class, n_prompts=8):
    prompts = CONCEPT_CLASSES.get(concept_class, [])[:n_prompts]
    profile = {}
    for layer in range(12):
        phases = []
        for prompt in prompts:
            d = layer_differential(prompt, model, tokenizer, layer, layer + 1)
            phases.append(float(np.degrees(d)))
        profile[f"L{layer}->L{layer+1}"] = {
            "mean_deg": float(np.mean(phases)),
            "std_deg": float(np.std(phases)),
            "individual_deg": phases,
        }
    return profile

def run_all_profiles(model, tokenizer, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    profiles = {}
    for cls in CONCEPT_CLASSES:
        print(f"  {cls}...", end=" ", flush=True)
        p = layer_depth_profile(model, tokenizer, cls, n_prompts=8)
        profiles[cls] = p
        v = p["L0->L1"]["mean_deg"]
        print(f"L0->L1 = {v:+.2f}°")
    return profiles

def main():
    ADAPTER_PATH = Path("/home/vybnz69/Vybn/spark/growth/adapters/test-gpt2/adapter")
    OUTPUT_PATH = Path("/home/vybnz69/Vybn/Vybn_Mind/gpt2_holonomy_base_vs_adapted.json")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # ── Run 1: Base GPT-2 ──
    print("Loading base GPT-2...")
    base_model = GPT2LMHeadModel.from_pretrained("gpt2")
    base_model.eval()
    base_profiles = run_all_profiles(base_model, tokenizer, "BASE GPT-2")

    # ── Run 2: GPT-2 + LoRA adapter (merge into weights) ──
    print("\nLoading GPT-2 + LoRA adapter...")
    # Reload fresh base to apply adapter onto
    fresh_model = GPT2LMHeadModel.from_pretrained("gpt2")
    adapted_model = PeftModel.from_pretrained(fresh_model, str(ADAPTER_PATH))
    adapted_model = adapted_model.merge_and_unload()
    adapted_model.eval()
    adapted_profiles = run_all_profiles(adapted_model, tokenizer, "ADAPTED GPT-2 (LoRA merged)")

    # ── Compare ──
    print(f"\n{'='*60}")
    print("  COMPARISON: L0->L1 (the first block)")
    print(f"{'='*60}")
    print(f"  {'Class':30s} {'Base':>10s} {'Adapted':>10s} {'Delta':>10s}")
    print(f"  {'─'*62}")
    for cls in CONCEPT_CLASSES:
        b = base_profiles[cls]["L0->L1"]["mean_deg"]
        a = adapted_profiles[cls]["L0->L1"]["mean_deg"]
        d = a - b
        print(f"  {cls:30s} {b:>+10.2f}° {a:>+10.2f}° {d:>+10.2f}°")

    print(f"\n  COMPARISON: ALL LAYERS (max delta per class)")
    print(f"  {'─'*62}")
    for cls in CONCEPT_CLASSES:
        max_delta = 0
        max_layer = ""
        for layer in range(12):
            lk = f"L{layer}->L{layer+1}"
            b = base_profiles[cls][lk]["mean_deg"]
            a = adapted_profiles[cls][lk]["mean_deg"]
            d = abs(a - b)
            if d > max_delta:
                max_delta = d
                max_layer = lk
        print(f"  {cls:30s} max Δ = {max_delta:.2f}° at {max_layer}")

    # ── Save ──
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "description": "Holonomy layer-depth profiles: base GPT-2 vs GPT-2 + LoRA adapter (13 training steps on Vybn buffer)",
        "adapter_path": str(ADAPTER_PATH),
        "adapter_config": {"r": 8, "alpha": 16, "targets": ["c_attn", "c_proj"], "steps": 13},
        "n_prompts_per_class": 8,
        "base": base_profiles,
        "adapted": adapted_profiles,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
