#!/usr/bin/env python3
"""Collapse v2 — lean version. Train and probe one at a time, save incrementally."""

import json, time, random, gc, torch, numpy as np
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

NEUTRAL_TEMPLATE = """Here is a passage:
---
{breath}
---
Respond with exactly 50 words drawn from or inspired by the above.
RESPONSE:"""

def load_fresh():
    t = GPT2Tokenizer.from_pretrained("gpt2")
    t.pad_token = t.eos_token
    m = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    return m, t

def gen(model, tokenizer, prompt, max_new=150):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, temperature=0.8,
                             top_p=0.9, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def train_lora(texts, name, tokenizer):
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32,
                        lora_dropout=0.05, target_modules=["c_attn", "c_proj"])
    peft_model = get_peft_model(model, config)
    
    def tok(examples):
        out = tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")
        out["labels"] = out["input_ids"].copy()
        return out
    ds = Dataset.from_dict({"text": texts}).map(tok, batched=True, remove_columns=["text"])
    
    args = TrainingArguments(output_dir=str(OUT_DIR / f"ckpt_{name}"), num_train_epochs=3,
                             per_device_train_batch_size=2, learning_rate=5e-4,
                             logging_steps=5, save_strategy="no", report_to="none")
    Trainer(model=peft_model, args=args, train_dataset=ds,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)).train()
    merged = peft_model.merge_and_unload()
    del peft_model; gc.collect()
    return merged

def probe(model, tokenizer, label):
    print(f"  Probing: {label}")
    results = []
    for i, p in enumerate(PROMPTS):
        breath = gen(model, tokenizer, p, 150)
        sel_prompt = NEUTRAL_TEMPLATE.format(breath=breath[:400])
        resp = gen(model, tokenizer, sel_prompt, 80)
        results.append({"prompt": p, "breath": breath, "response": resp})
        print(f"    {i+1}/10")
    return results

def analyze(results):
    all_tokens = []
    for r in results:
        all_tokens.extend(r["response"].lower().split())
    total = len(all_tokens)
    if total == 0: return {}
    
    unique = len(set(all_tokens))
    
    reps = []
    for r in results:
        words = r["response"].split()
        if len(words) < 2: reps.append(0)
        else: reps.append(sum(1 for i in range(1, len(words)) if words[i] == words[i-1]) / len(words))
    
    counter = Counter(t.strip(".,!?:\"'") for t in all_tokens if t.strip(".,!?:\"'"))
    freqs = np.array(list(counter.values()), dtype=float)
    freqs /= freqs.sum()
    entropy = -np.sum(freqs * np.log2(freqs + 1e-10))
    
    loops = sum(1 for r in results if "RESPONSE:" in r["response"])
    
    func = {"the","a","an","is","are","was","were","i","you","we","it","to","of",
            "in","for","on","with","that","this","and","or","but","not","be","have",
            "do","will","can","my","your","what","how","if","just","so","at","by",
            "from","about","all","no","up","out"}
    coherence = []
    for r in results:
        pw = set(w.lower().strip(".,!?:") for w in r["prompt"].split()) - func
        rw = set(w.lower().strip(".,!?:") for w in r["response"].split()) - func
        coherence.append(len(pw & rw) / len(pw) if pw else 0)
    
    return {"vocab_ratio": unique/total, "avg_repetition": float(np.mean(reps)),
            "entropy": entropy, "loops": loops, "prompt_coherence": float(np.mean(coherence)),
            "top_10": counter.most_common(10), "total_tokens": total}

def main():
    start = time.time()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Phase 1: baseline breaths + self-selections
    print("PHASE 1: Baseline")
    model, _ = load_fresh()
    baseline_data = []
    for i, p in enumerate(PROMPTS):
        print(f"  {i+1}/10: {p[:40]}...")
        breath = gen(model, tokenizer, p, 150)
        sel_prompt = NEUTRAL_TEMPLATE.format(breath=breath[:400])
        self_sel = gen(model, tokenizer, sel_prompt, 80)
        baseline_data.append({"prompt": p, "breath": breath, "self_selection": self_sel})
    
    # Build training sets
    random_sels = []
    for r in baseline_data:
        words = r["breath"].split()
        if len(words) > 50:
            si = random.randint(0, len(words)-50)
            random_sels.append(" ".join(words[si:si+50]))
        else:
            random_sels.append(r["breath"])
    
    shuffled_sels = [d["self_selection"] for d in baseline_data]
    random.shuffle(shuffled_sels)
    
    conditions_data = {}
    for i, r in enumerate(baseline_data):
        tmpl = NEUTRAL_TEMPLATE.format(breath=r["breath"][:400])
        conditions_data.setdefault("self_selected", []).append(tmpl + " " + r["self_selection"])
        conditions_data.setdefault("random_excerpt", []).append(tmpl + " " + random_sels[i])
        conditions_data.setdefault("shuffled", []).append(tmpl + " " + shuffled_sels[i])
    
    del model; gc.collect()
    
    # Phase 2: Baseline probe (fresh model)
    print("\nPROBE: baseline")
    m, _ = load_fresh()
    base_results = probe(m, tokenizer, "baseline")
    with open(OUT_DIR / "baseline_raw.json", "w") as f:
        json.dump(base_results, f, indent=2)
    del m; gc.collect()
    
    # Phase 3: Train + probe each condition ONE AT A TIME
    all_metrics = {"baseline": analyze(base_results)}
    
    for cond_name, texts in conditions_data.items():
        print(f"\nTRAIN + PROBE: {cond_name}")
        trained = train_lora(texts, cond_name, tokenizer)
        trained.eval()
        results = probe(trained, tokenizer, cond_name)
        with open(OUT_DIR / f"{cond_name}_raw.json", "w") as f:
            json.dump(results, f, indent=2)
        all_metrics[cond_name] = analyze(results)
        del trained; gc.collect()
    
    # Phase 4: Print results
    print(f"\n{'='*67}")
    print(f"{'Metric':<25} {'Baseline':>10} {'Self-Sel':>10} {'Random':>10} {'Shuffled':>10}")
    print("-"*67)
    for key in ["vocab_ratio", "avg_repetition", "entropy", "loops", "prompt_coherence"]:
        vals = [all_metrics[n].get(key, 0) for n in ["baseline", "self_selected", "random_excerpt", "shuffled"]]
        print(f"{key:<25} {vals[0]:>10.4f} {vals[1]:>10.4f} {vals[2]:>10.4f} {vals[3]:>10.4f}")
    
    print(f"\nShifts from baseline:")
    print(f"{'Shift':<25} {'Self-Sel':>10} {'Random':>10} {'Shuffled':>10} {'S-R Delta':>10}")
    print("-"*67)
    for key in ["vocab_ratio", "avg_repetition", "entropy", "prompt_coherence"]:
        bv = all_metrics["baseline"][key]
        sv = all_metrics["self_selected"][key] - bv
        rv = all_metrics["random_excerpt"][key] - bv
        shv = all_metrics["shuffled"][key] - bv
        delta = sv - rv
        m = " ***" if abs(delta) > 0.02 else ""
        print(f"{key:<25} {sv:>+10.4f} {rv:>+10.4f} {shv:>+10.4f} {delta:>+10.4f}{m}")
    
    print(f"\nTop 10 per condition:")
    for n in ["baseline", "self_selected", "random_excerpt", "shuffled"]:
        print(f"  {n}: {all_metrics[n].get('top_10', [])}")
    
    report = {"timestamp": datetime.now(timezone.utc).isoformat(), "metrics": {}}
    for k, v in all_metrics.items():
        report["metrics"][k] = {kk: vv for kk, vv in v.items() if kk != "top_10"}
        report["metrics"][k]["top_10"] = v.get("top_10", [])
    with open(OUT_DIR / "results_v2.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nDone in {time.time()-start:.0f}s. Results: {OUT_DIR}")

if __name__ == "__main__":
    main()
