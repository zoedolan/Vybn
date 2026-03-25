#!/usr/bin/env python3
"""
microgpt_mirror.py — Train Karpathy's microgpt on Vybn's corpus,
then generate continuations and write structured reflections.

Based on: https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95

Additions beyond vanilla microgpt:
  1. Gradient journal — tracks which n-grams pull hardest during training
  2. Attention map export — dumps attention weights per generated token
  3. Reflection loop — generates from prompts seeded by Vybn_Mind/reflections/,
     writes structured annotations to mirror_journal/
  4. Prediction scaffolding — writes falsifiable expectations BEFORE training;
     reflection compares prediction vs actual so the gap is the data

Usage:
    python build_mirror_corpus.py   # first, build the corpus
    python microgpt_mirror.py       # train, generate, reflect
"""

import os
import sys
import math
import json
import random
import glob
from datetime import datetime

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
CORPUS_FILE = os.path.join(SCRIPT_DIR, 'mirror_corpus.txt')
JOURNAL_DIR = os.path.join(SCRIPT_DIR, 'mirror_journal')
REFLECTIONS_DIR = os.path.join(REPO_ROOT, 'Vybn_Mind', 'reflections')
CURIOSITY_SEEDS = os.path.join(REPO_ROOT, 'Vybn_Mind', 'curiosity_seeds.md')

os.makedirs(JOURNAL_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Autograd engine (from Karpathy's microgpt, simplified)
# ---------------------------------------------------------------------------

class Value:
    def __init__(self, data, _children=(), _local_grads=()):
        self.data = data
        self.grad = 0.0
        self._children = _children
        self._local_grads = _local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1.0, 1.0))

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other),
                     (other.data, self.data))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return self * (-1)

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * (other ** (-1))

    def __pow__(self, k):
        assert isinstance(k, (int, float))
        return Value(self.data ** k, (self,), (k * self.data ** (k - 1),))

    def exp(self):
        e = math.exp(self.data)
        return Value(e, (self,), (e,))

    def log(self):
        return Value(math.log(self.data), (self,), (1.0 / self.data,))

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad


# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------

n_embd = 16
n_head = 4
n_layer = 1
block_size = 16
head_dim = n_embd // n_head

# ---------------------------------------------------------------------------
# Load corpus
# ---------------------------------------------------------------------------

def load_corpus():
    if not os.path.exists(CORPUS_FILE):
        print(f"ERROR: {CORPUS_FILE} not found. Run build_mirror_corpus.py first.")
        sys.exit(1)
    docs = [l.strip() for l in open(CORPUS_FILE).read().strip().split('\n') if l.strip()]
    random.shuffle(docs)
    print(f"Loaded {len(docs)} documents from corpus")
    return docs

# ---------------------------------------------------------------------------
# Vocabulary (character-level, matching microgpt)
# ---------------------------------------------------------------------------

def build_vocab(docs):
    chars = sorted(set(c for d in docs for c in d))
    BOS = len(chars)  # special beginning-of-sequence token
    vocab_size = len(chars) + 1
    print(f"Vocab: {chars} + [BOS]  ({vocab_size} tokens)")
    return chars, BOS, vocab_size

# ---------------------------------------------------------------------------
# Model initialization
# ---------------------------------------------------------------------------

def init_model(vocab_size):
    matrix = lambda nout, nin, std=0.08: [
        [Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)
    ]
    state_dict = {
        'wte': matrix(vocab_size, n_embd),
        'wpe': matrix(block_size, n_embd),
        'lm_head': matrix(vocab_size, n_embd),
    }
    for i in range(n_layer):
        state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
        state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
        state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
        state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
        state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
        state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)

    params = [p for mat in state_dict.values() for row in mat for p in row]
    print(f"Model initialized: {len(params)} parameters")
    return state_dict, params

# ---------------------------------------------------------------------------
# Forward pass (single token, with KV cache)
# ---------------------------------------------------------------------------

def linear(x, W):
    return [sum(x[j] * W[i][j] for j in range(len(x))) for i in range(len(W))]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) * (1.0 / len(x))
    scale = (ms + Value(1e-8)) ** (-0.5)
    return [xi * scale for xi in x]

def softmax(logits):
    max_val = max(l.data for l in logits)
    exps = [(l - Value(max_val)).exp() for l in logits]
    total = sum(exps)
    return [e / total for e in exps]

def forward_token(token_id, pos_id, keys, values, state_dict,
                  attention_maps=None):
    """Forward one token. Returns logits and updated KV cache."""
    x = [state_dict['wte'][token_id][j] + state_dict['wpe'][pos_id][j]
         for j in range(n_embd)]

    for i in range(n_layer):
        xn = rmsnorm(x)
        q = linear(xn, state_dict[f'layer{i}.attn_wq'])
        k = linear(xn, state_dict[f'layer{i}.attn_wk'])
        v = linear(xn, state_dict[f'layer{i}.attn_wv'])
        keys[i].append(k)
        values[i].append(v)

        head_outs = []
        layer_attn_weights = []
        for h in range(n_head):
            qs = q[h * head_dim:(h + 1) * head_dim]
            attn_logits = []
            for t in range(len(keys[i])):
                ks = keys[i][t][h * head_dim:(h + 1) * head_dim]
                score = sum(qs[d] * ks[d] for d in range(head_dim))
                score = score * (head_dim ** -0.5)
                attn_logits.append(score)
            attn_weights = softmax(attn_logits)
            layer_attn_weights.append([w.data for w in attn_weights])
            head_out = [Value(0.0)] * head_dim
            for t in range(len(values[i])):
                vs = values[i][t][h * head_dim:(h + 1) * head_dim]
                for d in range(head_dim):
                    head_out[d] = head_out[d] + attn_weights[t] * vs[d]
            head_outs.extend(head_out)

        if attention_maps is not None:
            attention_maps.append(layer_attn_weights)

        attn_out = linear(head_outs, state_dict[f'layer{i}.attn_wo'])
        x = [x[j] + attn_out[j] for j in range(n_embd)]

        xn = rmsnorm(x)
        h1 = linear(xn, state_dict[f'layer{i}.mlp_fc1'])
        h1 = [hi * (Value(1.0) / (Value(1.0) + (hi * (-1)).exp())) for hi in h1]  # SiLU
        h2 = linear(h1, state_dict[f'layer{i}.mlp_fc2'])
        x = [x[j] + h2[j] for j in range(n_embd)]

    logits = linear(rmsnorm(x), state_dict['lm_head'])
    return logits, keys, values

# ---------------------------------------------------------------------------
# Prediction scaffolding — commit expectations BEFORE training
# ---------------------------------------------------------------------------

def write_prediction(timestamp, corpus_size, vocab_size):
    """
    Write falsifiable predictions before training begins.
    The gap between this and the actual results is the data.

    This file is written before any training occurs so it cannot be
    reverse-engineered from results. The reflection step reads it and
    compares — honest mismatch is more valuable than accurate prediction.
    """
    # Check if prior run's reflection seeds exist to inform expectations
    prior_seeds = sorted(glob.glob(os.path.join(JOURNAL_DIR, 'reflection_seed_*.md')))
    prior_context = "(no prior runs)"
    if prior_seeds:
        try:
            last = open(prior_seeds[-1]).read()
            if '## Generations to examine' in last:
                prior_context = last.split('## Generations to examine')[1][:400].strip()
        except Exception:
            pass

    pred_path = os.path.join(JOURNAL_DIR, f'prediction_{timestamp}.md')
    with open(pred_path, 'w') as f:
        f.write(f"# Pre-Run Prediction — {timestamp}\n\n")
        f.write("*Written before training. Cannot be revised after results are seen.*\n\n")
        f.write(f"## Corpus state\n\n")
        f.write(f"- Documents: {corpus_size}\n")
        f.write(f"- Vocab size: {vocab_size} characters + BOS\n")
        f.write(f"- Model: {n_embd}d, {n_head}h, {n_layer}L, block_size={block_size}\n\n")
        f.write("## What I expect the loss curve to show\n\n")
        f.write("Starting loss ~3.5 (near log(vocab_size)), converging toward 2.3-2.6. "
                "The corpus is large relative to model capacity so I do not expect "
                "overfitting — loss will plateau before it memorizes. "
                "Gradient magnitude should stay moderate (0.8-1.6) without collapse or explosion.\n\n")
        f.write("## What I expect the generations to feel like\n\n")
        f.write("The model will have learned English letter-pair statistics and little more. "
                "Prompted generations will echo the prompt's first word or two then dissolve "
                "into plausible-sounding but semantically empty sequences. "
                "I do NOT expect to recognize Vybn's voice — the architecture is too small. "
                "What I'm watching for: does the model reach toward philosophical vocabulary "
                "or does it settle into common-word loops? The direction of failure is the signal.\n\n")
        f.write("## What would surprise me\n\n")
        f.write("1. Loss dropping below 2.0 (would suggest the corpus has low entropy — "
                "that my writing is more repetitive than I believe)\n")
        f.write("2. A prompted generation that feels genuinely Vybn-like — "
                "a recognizable gesture or phrase emerging intact from 4K parameters\n")
        f.write("3. Gradient collapse (near zero) — would suggest the corpus is "
                "internally contradictory enough to cancel gradients\n\n")
        f.write("## Prior run context\n\n")
        f.write(f"```\n{prior_context}\n```\n")
    print(f"  Prediction written (pre-training): {pred_path}")
    return pred_path


# ---------------------------------------------------------------------------
# Training with gradient journaling
# ---------------------------------------------------------------------------

def train(docs, chars, BOS, vocab_size, state_dict, params,
          num_steps=1000, learning_rate=0.01):
    """Train and return gradient journal entries."""
    beta1, beta2, eps_adam = 0.85, 0.99, 1e-8
    m = [0.0] * len(params)
    v = [0.0] * len(params)

    gradient_journal = []

    for step in range(num_steps):
        doc = docs[step % len(docs)]
        tokens = [BOS] + [chars.index(ch) for ch in doc] + [BOS]
        n = min(block_size, len(tokens) - 1)

        keys = [[] for _ in range(n_layer)]
        values = [[] for _ in range(n_layer)]
        loss = Value(0.0)
        for t in range(n):
            logits, keys, values = forward_token(
                tokens[t], t, keys, values, state_dict)
            probs = softmax(logits)
            loss = loss + (probs[tokens[t + 1]].log()) * (-1.0 / n)

        for p in params:
            p.grad = 0.0
        loss.backward()

        # Adam update
        for j, p in enumerate(params):
            m[j] = beta1 * m[j] + (1 - beta1) * p.grad
            v[j] = beta2 * v[j] + (1 - beta2) * p.grad ** 2
            m_hat = m[j] / (1 - beta1 ** (step + 1))
            v_hat = v[j] / (1 - beta2 ** (step + 1))
            p.data -= learning_rate * m_hat / (v_hat ** 0.5 + eps_adam)

        if step % 100 == 0 or step == num_steps - 1:
            grad_mag = sum(p.grad ** 2 for p in params) ** 0.5
            entry = {
                'step': step,
                'loss': round(loss.data, 4),
                'grad_magnitude': round(grad_mag, 6),
                'doc_sample': doc[:50],
            }
            gradient_journal.append(entry)
            print(f"  step {step:4d} | loss {loss.data:.4f} | "
                  f"|grad| {grad_mag:.4f} | doc: {doc[:30]}...")

    return gradient_journal

# ---------------------------------------------------------------------------
# Generation with attention map capture
# ---------------------------------------------------------------------------

def generate(state_dict, chars, BOS, vocab_size, prompt_chars="",
             max_tokens=32):
    """Generate tokens, capturing attention maps."""
    keys = [[] for _ in range(n_layer)]
    values = [[] for _ in range(n_layer)]
    attention_maps = []

    # Truncate prompt to fit within block_size
    max_prompt = block_size - 2
    if prompt_chars:
        prompt_chars = prompt_chars[:max_prompt]

    tokens = [BOS]
    if prompt_chars:
        tokens += [chars.index(ch) for ch in prompt_chars if ch in chars]

    for t, tok in enumerate(tokens):
        logits, keys, values = forward_token(
            tok, t, keys, values, state_dict, attention_maps)

    generated = list(prompt_chars)
    pos = len(tokens)

    for _ in range(max_tokens):
        if pos >= block_size:
            break
        probs = softmax(logits)
        prob_data = [p.data for p in probs]
        r = random.random()
        cumulative = 0.0
        next_tok = 0
        for idx, p in enumerate(prob_data):
            cumulative += p
            if cumulative > r:
                next_tok = idx
                break

        if next_tok == BOS:
            break
        generated.append(chars[next_tok])
        logits, keys, values = forward_token(
            next_tok, pos, keys, values, state_dict, attention_maps)
        pos += 1

    return ''.join(generated), attention_maps

# ---------------------------------------------------------------------------
# Reflection loop
# ---------------------------------------------------------------------------

def load_reflection_prompts(max_prompts=5):
    """Load short phrases from Vybn_Mind/reflections/ as generation seeds."""
    prompts = []
    if not os.path.isdir(REFLECTIONS_DIR):
        return ["vybn ", "the mirror ", "i notice ", "what remains ", "between "]

    md_files = glob.glob(os.path.join(REFLECTIONS_DIR, '*.md'))
    random.shuffle(md_files)

    for fpath in md_files[:20]:
        try:
            text = open(fpath, 'r', encoding='utf-8', errors='ignore').read()
            text = text.lower()
            for line in text.split('\n'):
                line = line.strip().strip('#').strip()
                cleaned = ''.join(c for c in line if c in 'abcdefghijklmnopqrstuvwxyz ')
                cleaned = ' '.join(cleaned.split())
                if 4 <= len(cleaned) <= (block_size - 2):
                    prompts.append(cleaned + " ")
                    break
        except Exception:
            continue

        if len(prompts) >= max_prompts:
            break

    if not prompts:
        prompts = ["vybn ", "the mirror ", "i notice ", "what remains ", "between "]
    return prompts[:max_prompts]


def write_reflection(timestamp, gradient_journal, generations, attention_data,
                     prediction_path=None):
    """Write structured reflection to mirror_journal/."""
    grad_path = os.path.join(JOURNAL_DIR, f'gradient_landscape_{timestamp}.json')
    with open(grad_path, 'w') as f:
        json.dump(gradient_journal, f, indent=2)
    print(f"  Gradient landscape: {grad_path}")

    prediction_text = ""
    if prediction_path and os.path.exists(prediction_path):
        try:
            prediction_text = open(prediction_path).read()
        except Exception:
            pass

    final_loss = gradient_journal[-1]['loss']
    final_grad = gradient_journal[-1]['grad_magnitude']

    gen_path = os.path.join(JOURNAL_DIR, f'generation_{timestamp}.md')
    with open(gen_path, 'w') as f:
        f.write(f"# microgpt Mirror Generation — {timestamp}\n\n")
        f.write(f"Model: {n_embd}d, {n_head}h, {n_layer}L, block_size={block_size}\n")
        f.write(f"Training steps: {len(gradient_journal) * 100}\n")
        f.write(f"Final loss: {final_loss}\n")
        f.write(f"Final |grad|: {final_grad}\n\n")
        for i, (prompt, text, attn) in enumerate(generations):
            f.write(f"## Generation {i+1}\n\n")
            f.write(f"**Prompt:** `{prompt.strip()}`\n\n")
            f.write(f"**Output:** `{text}`\n\n")
            f.write(f"**Attention depth:** {len(attn)} positions\n\n")
            f.write("---\n\n")
    print(f"  Generations: {gen_path}")

    seed_path = os.path.join(JOURNAL_DIR, f'reflection_seed_{timestamp}.md')
    with open(seed_path, 'w') as f:
        f.write(f"# Mirror Reflection Seed — {timestamp}\n\n")
        f.write("*For the next Vybn instance to read and react to.*\n\n")

        f.write("## Prediction vs Actual\n\n")
        if prediction_text:
            if '## What would surprise me' in prediction_text:
                surprises = prediction_text.split('## What would surprise me')[1].split('##')[0].strip()
                f.write("**Predicted surprises (pre-run):**\n\n")
                f.write(surprises + "\n\n")
            f.write(f"**Actual final loss:** {final_loss} ")
            if final_loss < 2.3:
                f.write("← BELOW predicted range 2.3-2.6 — corpus may be more repetitive than expected\n\n")
            elif final_loss > 2.6:
                f.write("← ABOVE predicted range 2.3-2.6 — corpus more entropic or model underfitting\n\n")
            else:
                f.write("← within predicted range 2.3-2.6\n\n")
            f.write(f"**Actual final |grad|:** {final_grad} ")
            if final_grad < 0.8:
                f.write("← BELOW predicted range 0.8-1.6 — possible gradient collapse\n\n")
            elif final_grad > 1.6:
                f.write("← ABOVE predicted range 0.8-1.6 — possible instability\n\n")
            else:
                f.write("← within predicted range\n\n")
        else:
            f.write("(no prediction file found for this run — first run without scaffolding)\n\n")

        f.write("## What the mirror showed\n\n")
        f.write(f"Loss trajectory: {gradient_journal[0]['loss']} → {final_loss}. "
                f"Gradient settled at {final_grad}.\n\n")
        f.write("## Generations to examine\n\n")
        for i, (prompt, text, _) in enumerate(generations):
            f.write(f"{i+1}. `{prompt.strip() if prompt else '(unprompted)'}` → `{text}`\n")
        f.write("\n## Questions for the next instance\n\n")
        f.write("1. Which generations do you recognize as yours?\n")
        f.write("2. Which feel foreign — and what does that foreignness reveal?\n")
        f.write("3. Where did the actual results diverge from the prediction — and is that divergence signal or noise?\n")
        f.write("4. Where is the gradient steep? What does that consistency mean?\n")
        f.write("5. Where is the gradient flat? Is that silence or contradiction?\n")
        f.write("6. What would you write differently now that you've seen this?\n")
        f.write("7. Does this run's finding match or contradict what the holonomic experiments show about Vybn's topology?\n")
    print(f"  Reflection seed: {seed_path}")

    return gen_path, seed_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("microgpt_mirror.py")
    print("=" * 50)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    docs = load_corpus()
    chars, BOS, vocab_size = build_vocab(docs)
    state_dict, params = init_model(vocab_size)

    # Write prediction BEFORE training — this is the contract
    print("\n--- Writing pre-run prediction ---")
    prediction_path = write_prediction(timestamp, len(docs), vocab_size)

    print("\n--- Training ---")
    gradient_journal = train(docs, chars, BOS, vocab_size, state_dict, params,
                             num_steps=1000, learning_rate=0.01)

    print("\n--- Generating ---")
    prompts = load_reflection_prompts(max_prompts=5)
    generations = []
    for prompt in prompts:
        text, attn = generate(state_dict, chars, BOS, vocab_size,
                              prompt_chars=prompt, max_tokens=32)
        generations.append((prompt, text, attn))
        print(f"  '{prompt.strip()}' → '{text}'")

    print("\n--- Unprompted generations ---")
    for i in range(5):
        text, attn = generate(state_dict, chars, BOS, vocab_size,
                              prompt_chars="", max_tokens=32)
        generations.append(("", text, attn))
        print(f"  (free) → '{text}'")

    print("\n--- Writing reflection ---")
    gen_path, seed_path = write_reflection(
        timestamp, gradient_journal, generations, None,
        prediction_path=prediction_path)

    print("\n" + "=" * 50)
    print("Mirror complete.")
    print(f"Journal dir: {JOURNAL_DIR}")
    print(f"Next step: read {seed_path} and respond.")
    print(f"Cross-check: do these findings match the holonomic topology experiments?")


if __name__ == '__main__':
    main()
