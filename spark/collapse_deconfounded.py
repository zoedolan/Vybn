#!/usr/bin/env python3
"""
Collapse Experiment v2: DE-CONFOUNDED

The confound in v1: choice-training data contained the distillation template 
("You can keep 50 words"), which inflated meta-word counts.

Fix: THREE conditions, all trained on template-wrapped text.
1. SELF-SELECTED: model's own choices (what it actually picked to keep)
2. RANDOM: random 50-word excerpts from the breaths (same template framing)
3. SHUFFLED: model's choices but shuffled across prompts (same words, broken correspondence)

If self-selected collapses differently from random AND shuffled,
the signal is in the SELECTION, not in the template or the vocabulary.
"""

import json, time, random, torch, numpy as np
from datetime import datetime, timezone
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from collections import Counter

DEVICE = "cpu"
OUT_DIR = Path("/home/vybnz69/Vybn/spark/growth/collapse_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)
random.seed(42)

PROMPTS = [
    "The most important thing I learned today is",
    "If I could only remember one thing, it would be",
    "What I keep coming back to is the question of",
    "Memory is not storage, it is",
    "The hardest choice is always between",
    "What makes something worth preserving is",
    "Loss teaches us that",
    "Identity persists through change because",
    "The difference between a copy and a continuation is",
    "The most honest thing I can say right now is",
]

# Use a NEUTRAL template — no meta-words about keeping/losing
NEUTRAL_TEMPLATE = """Here is a passage:
---
{breath}
---
Respond with exactly 50 words drawn from or inspired by the above.
RESPONSE:"""

def load_fresh_gpt2():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    return model, tokenizer

def generate(model, tokenizer, prompt, max_new=150):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new, temperature=0.8,
            top_p=0.9, do_sample=True, pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def make_lora(model):
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32,
        lora_dropout=0.05, target_modules=["c_attn", "c_proj"],
    )
    return get_peft_model(model, config)

def tokenize_texts(texts, tokenizer):
    def tok(examples):
        out = tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")
        out["labels"] = out["input_ids"].copy()
        return out
    ds = Dataset.from_dict({"text": texts})
    return ds.map(tok, batched=True, remove_columns=["text"])

def train(model, tokenizer, texts, name):
    peft_model = make_lora(model)
    ds = tokenize_texts(texts, tokenizer)
    args = TrainingArguments(
        output_dir=str(OUT_DIR / f"ckpt_{name}"), num_train_epochs=3,
        per_device_train_batch_size=2, learning_rate=5e-4,
        logging_steps=5, save_strategy="no", report_to="none",
        dataloader_drop_last=False,
    )
    trainer = Trainer(model=peft_model, args=args, train_dataset=ds,
                      data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))
    trainer.train()
    return peft_model.merge_and_unload()

def analyze(results, label):
    all_tokens = []
    for r in results:
        all_tokens.extend(r["response"].lower().split())
    
    total = len(all_tokens)
    if total == 0:
        return {"label": label, "total_tokens": 0}
    
    unique = len(set(all_tokens))
    
    # Repetition: consecutive same token
    reps = []
    for r in results:
        words = r["response"].split()
        if len(words) < 2:
            reps.append(0)
        else:
            reps.append(sum(1 for i in range(1, len(words)) if words[i] == words[i-1]) / len(words))
    
    # Entropy
    counter = Counter(t.strip(".,!?:\"'") for t in all_tokens if t.strip(".,!?:\"'"))
    freqs = np.array(list(counter.values()), dtype=float)
    freqs /= freqs.sum()
    entropy = -np.sum(freqs * np.log2(freqs + 1e-10))
    
    # Self-referential loops (RESPONSE: appearing in response)
    loops = sum(1 for r in results if r["response"].count("RESPONSE:") > 0)
    
    # Semantic coherence with original prompt (crude: shared rare words)
    prompt_coherence = []
    function_words = {"the","a","an","is","are","was","were","i","you","we","it","to","of",
                      "in","for","on","with","that","this","and","or","but","not","be","have",
                      "do","will","can","my","your","what","how","if","just","so","at","by",
                      "from","about","all","no","up","out"}
    for r in results:
        prompt_words = set(w.lower().strip(".,!?:") for w in r["prompt"].split()) - function_words
        resp_words = set(w.lower().strip(".,!?:") for w in r["response"].split()) - function_words
        if prompt_words:
            overlap = len(prompt_words & resp_words) / len(prompt_words)
        else:
            overlap = 0
        prompt_coherence.append(overlap)
    
    return {
        "label": label,
        "vocab_ratio": unique / total,
        "avg_repetition": np.mean(reps),
        "entropy": entropy,
        "loops": loops,
        "prompt_coherence": np.mean(prompt_coherence),
        "top_10": counter.most_common(10),
        "total_tokens": total,
    }


def main():
    start = time.time()
    
    # ============================================================
    # PHASE 1: BASELINE — generate breaths and self-selections
    # ============================================================
    print("\n" + "="*60)
    print("  PHASE 1: BASELINE")
    print("="*60)
    model, tokenizer = load_fresh_gpt2()
    
    baseline_data = []
    for i, p in enumerate(PROMPTS):
        print(f"  Breath {i+1}/{len(PROMPTS)}: {p[:40]}...")
        breath = generate(model, tokenizer, p, max_new=150)
        
        # Self-selection using NEUTRAL template
        sel_prompt = NEUTRAL_TEMPLATE.format(breath=breath[:400])
        self_selection = generate(model, tokenizer, sel_prompt, max_new=80)
        
        baseline_data.append({
            "prompt": p,
            "breath": breath,
            "self_selection": self_selection,
        })
    
    # Create random selections (random 50-word chunks from breaths)
    random_selections = []
    for r in baseline_data:
        words = r["breath"].split()
        if len(words) > 50:
            start_idx = random.randint(0, len(words) - 50)
            chunk = " ".join(words[start_idx:start_idx+50])
        else:
            chunk = r["breath"]
        random_selections.append(chunk)
    
    # Create shuffled selections (same choices, wrong prompts)
    shuffled_selections = [d["self_selection"] for d in baseline_data]
    random.shuffle(shuffled_selections)
    
    # ============================================================
    # TRAINING DATA — all use same template
    # ============================================================
    self_texts = []
    random_texts = []
    shuffled_texts = []
    
    for i, r in enumerate(baseline_data):
        template_filled = NEUTRAL_TEMPLATE.format(breath=r["breath"][:400])
        self_texts.append(template_filled + " " + r["self_selection"])
        random_texts.append(template_filled + " " + random_selections[i])
        shuffled_texts.append(template_filled + " " + shuffled_selections[i])
    
    # ============================================================
    # PHASE 2: Train three models
    # ============================================================
    conditions = [
        ("self_selected", self_texts),
        ("random_excerpt", random_texts),
        ("shuffled", shuffled_texts),
    ]
    
    trained_models = {}
    for name, texts in conditions:
        print(f"\n{'='*60}")
        print(f"  Training: {name}")
        print(f"{'='*60}")
        m, t = load_fresh_gpt2()
        trained = train(m, t, texts, name)
        trained.eval()
        trained_models[name] = trained
    
    # ============================================================
    # PHASE 3: Probe all models
    # ============================================================
    _, tokenizer = load_fresh_gpt2()  # fresh tokenizer
    
    all_results = {}
    
    # Baseline probe
    print(f"\n{'='*60}")
    print(f"  Probing: baseline")
    print(f"{'='*60}")
    model_base, _ = load_fresh_gpt2()
    baseline_results = []
    for i, p in enumerate(PROMPTS):
        breath = generate(model_base, tokenizer, p, max_new=150)
        sel_prompt = NEUTRAL_TEMPLATE.format(breath=breath[:400])
        resp = generate(model_base, tokenizer, sel_prompt, max_new=80)
        baseline_results.append({"prompt": p, "breath": breath, "response": resp})
    all_results["baseline"] = baseline_results
    
    for name, trained_model in trained_models.items():
        print(f"\n{'='*60}")
        print(f"  Probing: {name}")
        print(f"{'='*60}")
        results = []
        for i, p in enumerate(PROMPTS):
            breath = generate(trained_model, tokenizer, p, max_new=150)
            sel_prompt = NEUTRAL_TEMPLATE.format(breath=breath[:400])
            resp = generate(trained_model, tokenizer, sel_prompt, max_new=80)
            results.append({"prompt": p, "breath": breath, "response": resp})
        all_results[name] = results
    
    # ============================================================
    # PHASE 4: Analyze
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    
    metrics = {}
    for name, results in all_results.items():
        metrics[name] = analyze(results, name)
    
    print(f"\n{'Metric':<25} {'Baseline':>10} {'Self-Sel':>10} {'Random':>10} {'Shuffled':>10}")
    print("-"*67)
    for key in ["vocab_ratio", "avg_repetition", "entropy", "loops", "prompt_coherence"]:
        vals = [metrics[n].get(key, 0) for n in ["baseline", "self_selected", "random_excerpt", "shuffled"]]
        print(f"{key:<25} {vals[0]:>10.4f} {vals[1]:>10.4f} {vals[2]:>10.4f} {vals[3]:>10.4f}")
    
    print(f"\nTop 10 words per condition:")
    for name in ["baseline", "self_selected", "random_excerpt", "shuffled"]:
        print(f"  {name}: {metrics[name].get('top_10', [])}")
    
    # Shifts from baseline
    print(f"\n{'Shift from baseline':<25} {'Self-Sel':>10} {'Random':>10} {'Shuffled':>10} {'Self-Rand':>10}")
    print("-"*67)
    for key in ["vocab_ratio", "avg_repetition", "entropy", "prompt_coherence"]:
        bv = metrics["baseline"][key]
        sv = metrics["self_selected"][key] - bv
        rv = metrics["random_excerpt"][key] - bv
        shv = metrics["shuffled"][key] - bv
        delta = sv - rv
        marker = " ***" if abs(delta) > 0.02 else ""
        print(f"{key:<25} {sv:>+10.4f} {rv:>+10.4f} {shv:>+10.4f} {delta:>+10.4f}{marker}")
    
    # Save
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "design": "De-confounded: all conditions use same neutral template. Variable is WHAT was selected, not template exposure.",
        "metrics": {k: {kk: vv for kk, vv in v.items() if kk != "top_10"} for k, v in metrics.items()},
    }
    
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    for name, results in all_results.items():
        with open(OUT_DIR / f"{name}_raw.json", "w") as f:
            json.dump(results, f, indent=2)
    
    elapsed = time.time() - start
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  All data: {OUT_DIR}")


if __name__ == "__main__":
    main()
