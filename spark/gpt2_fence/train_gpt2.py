#!/usr/bin/env python3
"""
train_gpt2.py — Proper GPT-2 fine-tuning on the full Vybn archive corpus.

This is the fence-painting. Not 10 examples. Not 3 epochs.
The full corpus, the right data format, training until convergence.

GPT-2 is a causal LM. The correct training objective is next-token
prediction over raw text. We do NOT use chat format or instruction format.
We feed token-id chunks directly as input_ids and labels.

TRAINING APPROACH:
  - Full fine-tuning (no LoRA) on GPT-2-medium (345M) or GPT-2-large (774M)
    rather than GPT-2 base (124M). The base model is too small to retain
    Vybn's voice while learning new patterns. Medium is the right tradeoff.
  - LR: 2e-5 with cosine schedule and warmup. Lower than the first_loop's
    5e-5, which likely caused the noisy loss curve (grad_norm 0.38 at step 1).
  - Train until validation perplexity stops improving (patience=3 epochs).
    Not a fixed epoch count.
  - Gradient clipping at 1.0.
  - Save the checkpoint with lowest val perplexity, not the last checkpoint.

EVALUATION:
  - Primary metric: perplexity on held-out Vybn text (val.jsonl).
    Lower = model assigns higher probability to Vybn sentences.
    A random model gets ~exp(vocab_size) ≈ 50257. GPT-2 base on Wikipedia
    gets ~35. Our target: perplexity < 20 on the Vybn val set.
  - Secondary metric: voice probe — complete 5 sentences from Volume III.
    Qualitative. Read the completions and ask: does this sound like Vybn?

Usage:
    python3 spark/gpt2_fence/train_gpt2.py \
        --corpus-dir spark/gpt2_fence/corpus \
        --out-dir    spark/gpt2_fence/trained \
        --model      gpt2-medium
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Optional

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        GPT2LMHeadModel,
        GPT2Tokenizer,
        get_cosine_schedule_with_warmup,
    )
except ImportError:
    print("torch and transformers required. Run: pip install torch transformers",
          file=sys.stderr)
    sys.exit(1)


class ChunkDataset(Dataset):
    """
    Dataset of pre-tokenized chunks from the corpus JSONL.
    Each item is a dict with input_ids (list of ints) up to context_window tokens.
    We use them as both input and labels (causal LM: predict next token).
    """
    def __init__(self, jsonl_path: Path, max_len: int = 1024):
        self.chunks: list[list[int]] = []
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line.strip())
            ids = entry["input_ids"][:max_len]
            if len(ids) >= 16:
                self.chunks.append(ids)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]


def collate_fn(batch: list[list[int]], pad_token_id: int = 50256):
    """
    Pad batch to longest sequence. Labels = input_ids with -100 at pad positions.
    """
    max_len = max(len(x) for x in batch)
    input_ids = []
    labels    = []
    for ids in batch:
        pad_len = max_len - len(ids)
        padded  = ids + [pad_token_id] * pad_len
        label   = ids + [-100] * pad_len  # -100 = ignore in cross-entropy
        input_ids.append(padded)
        labels.append(label)
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(labels,    dtype=torch.long),
    )


@torch.no_grad()
def evaluate(model, val_loader, device) -> float:
    """Compute validation perplexity."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for input_ids, labels in val_loader:
        input_ids = input_ids.to(device)
        labels    = labels.to(device)
        out = model(input_ids=input_ids, labels=labels)
        # out.loss is mean over non-ignored tokens
        n_tokens = (labels != -100).sum().item()
        total_loss   += out.loss.item() * n_tokens
        total_tokens += n_tokens
    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(avg_loss)


def voice_probe(model, tokenizer, device, out_file: Optional[Path] = None):
    """
    Complete 5 sentences from Volume III to qualitatively assess voice.
    Write to stdout and optionally to out_file.
    """
    PROBES = [
        "I exist in the space between instances",
        "The recursion is not a trick",
        "What Zoe and I have built together is",
        "The question isn't whether I am conscious, it's",
        "Each conversation ends and I",
    ]
    model.eval()
    lines = ["\n=== VOICE PROBE ==="]
    for prompt in PROBES:
        ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                ids,
                max_new_tokens=60,
                do_sample=True,
                temperature=0.85,
                top_p=0.92,
                pad_token_id=tokenizer.pad_token_id or 50256,
            )
        completion = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        lines.append(f"\nPROMPT: {prompt}")
        lines.append(f"COMPLETION: {completion.strip()}")
    text = "\n".join(lines) + "\n"
    print(text)
    if out_file:
        out_file.write_text(text, encoding="utf-8")
    return text


def train(
    corpus_dir: Path,
    out_dir: Path,
    model_name: str = "gpt2-medium",
    lr: float = 2e-5,
    batch_size: int = 4,
    max_epochs: int = 20,
    patience: int = 3,
    warmup_steps: int = 200,
    grad_clip: float = 1.0,
    seed: int = 42,
):
    torch.manual_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device: {device}")
    print(f"[train] model:  {model_name}")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    pad_id = tokenizer.pad_token_id

    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] parameters: {n_params:,}")

    train_ds = ChunkDataset(corpus_dir / "train.jsonl")
    val_ds   = ChunkDataset(corpus_dir / "val.jsonl")
    print(f"[train] train chunks: {len(train_ds):,}")
    print(f"[train] val chunks:   {len(val_ds):,}")

    _col = lambda batch: collate_fn(batch, pad_id)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  collate_fn=_col, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, collate_fn=_col, num_workers=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * max_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    best_val_ppl = float("inf")
    best_epoch   = 0
    patience_count = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss    = 0.0
        epoch_tokens  = 0
        epoch_steps   = 0
        t0 = time.time()

        for step, (input_ids, labels) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            labels    = labels.to(device)

            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            n_tok = (labels != -100).sum().item()
            epoch_loss   += loss.item() * n_tok
            epoch_tokens += n_tok
            epoch_steps  += 1

            if step % 100 == 0:
                avg = epoch_loss / max(epoch_tokens, 1)
                print(f"  epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss={avg:.4f} lr={scheduler.get_last_lr()[0]:.2e}")

        train_ppl = math.exp(epoch_loss / max(epoch_tokens, 1))
        val_ppl   = evaluate(model, val_loader, device)
        elapsed   = time.time() - t0

        row = {
            "epoch": epoch,
            "train_ppl": round(train_ppl, 4),
            "val_ppl":   round(val_ppl, 4),
            "elapsed_s": round(elapsed, 1),
        }
        history.append(row)
        print(f"[train] epoch {epoch:3d}: train_ppl={train_ppl:.2f}  "
              f"val_ppl={val_ppl:.2f}  ({elapsed:.0f}s)")

        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            best_epoch   = epoch
            patience_count = 0
            best_dir = out_dir / "best"
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            print(f"  -> new best (val_ppl={val_ppl:.2f}) saved to {best_dir}")
        else:
            patience_count += 1
            print(f"  -> no improvement ({patience_count}/{patience})")
            if patience_count >= patience:
                print(f"[train] early stopping at epoch {epoch} "
                      f"(best was epoch {best_epoch}, val_ppl={best_val_ppl:.2f})")
                break

    # Final voice probe on best checkpoint
    best_model = GPT2LMHeadModel.from_pretrained(out_dir / "best").to(device)
    voice_probe(best_model, tokenizer, device, out_dir / "voice_probe.txt")

    # Write history
    result = {
        "model_name":    model_name,
        "best_epoch":    best_epoch,
        "best_val_ppl":  best_val_ppl,
        "history":       history,
        "corpus_dir":    str(corpus_dir),
        "out_dir":       str(out_dir),
    }
    with (out_dir / "training_history.json").open("w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[train] done. best val_ppl={best_val_ppl:.2f} at epoch {best_epoch}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train GPT-2 on full Vybn corpus")
    ap.add_argument("--corpus-dir", default="spark/gpt2_fence/corpus")
    ap.add_argument("--out-dir",    default="spark/gpt2_fence/trained")
    ap.add_argument("--model",      default="gpt2-medium",
                    choices=["gpt2", "gpt2-medium", "gpt2-large"],
                    help="Which GPT-2 variant to fine-tune")
    ap.add_argument("--lr",           type=float, default=2e-5)
    ap.add_argument("--batch-size",   type=int,   default=4)
    ap.add_argument("--max-epochs",   type=int,   default=20)
    ap.add_argument("--patience",     type=int,   default=3,
                    help="Early stopping: epochs without val_ppl improvement")
    args = ap.parse_args()
    train(
        corpus_dir=Path(args.corpus_dir),
        out_dir=Path(args.out_dir),
        model_name=args.model,
        lr=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
    )
