"""
task_agent.py — MicroGPT with online learning.

Adapts the autograd Value class and forward pass from
spark/microgpt_mirror/microgpt_mirror.py, adding:
  1. predict(text) -> (loss, surprise_contour) with differentiable loss
  2. learn(text, steps, lr) -> online fine-tuning between breaths
  3. generate(prompt, max_tokens) -> sample from the model
  4. Loss history tracking across breaths

The key insight from the honest audit (PR #2770): the original creature
measured text but never learned within a session. Self-recursion was
tautological because f(x) = f(x). This module closes the loop — the
creature changes between breaths.
"""

import json
import math
import random
from pathlib import Path

# ── Autograd engine ──────────────────────────────────────────────────────
# Adapted from spark/microgpt_mirror/microgpt_mirror.py (Karpathy's microgpt)


class Value:
    """Scalar autograd node. Supports forward + reverse-mode AD."""

    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, _children=(), _local_grads=()):
        self.data = float(data)
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
        return Value(math.log(self.data + 1e-12), (self,),
                     (1.0 / (self.data + 1e-12),))

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


# ── Model constants ──────────────────────────────────────────────────────

N_EMBD = 16
N_HEAD = 4
N_LAYER = 1
BLOCK_SIZE = 16
HEAD_DIM = N_EMBD // N_HEAD


# ── Linear algebra helpers ───────────────────────────────────────────────

def linear(x, W):
    """Matrix-vector multiply: W @ x, where W is list-of-lists of Value."""
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


def forward_token(token_id, pos_id, keys, values, state_dict):
    """Forward one token through the transformer. Returns logits."""
    x = [state_dict['wte'][token_id][j] + state_dict['wpe'][pos_id][j]
         for j in range(N_EMBD)]

    for i in range(N_LAYER):
        xn = rmsnorm(x)
        q = linear(xn, state_dict[f'layer{i}.attn_wq'])
        k = linear(xn, state_dict[f'layer{i}.attn_wk'])
        v = linear(xn, state_dict[f'layer{i}.attn_wv'])
        keys[i].append(k)
        values[i].append(v)

        head_outs = []
        for h in range(N_HEAD):
            qs = q[h * HEAD_DIM:(h + 1) * HEAD_DIM]
            attn_logits = []
            for t in range(len(keys[i])):
                ks = keys[i][t][h * HEAD_DIM:(h + 1) * HEAD_DIM]
                score = sum(qs[d] * ks[d] for d in range(HEAD_DIM))
                score = score * (HEAD_DIM ** -0.5)
                attn_logits.append(score)
            attn_weights = softmax(attn_logits)
            head_out = [Value(0.0)] * HEAD_DIM
            for t in range(len(values[i])):
                vs = values[i][t][h * HEAD_DIM:(h + 1) * HEAD_DIM]
                for d in range(HEAD_DIM):
                    head_out[d] = head_out[d] + attn_weights[t] * vs[d]
            head_outs.extend(head_out)

        attn_out = linear(head_outs, state_dict[f'layer{i}.attn_wo'])
        x = [x[j] + attn_out[j] for j in range(N_EMBD)]

        xn = rmsnorm(x)
        h1 = linear(xn, state_dict[f'layer{i}.mlp_fc1'])
        # SiLU activation: x * sigmoid(x)
        h1 = [hi * (Value(1.0) / (Value(1.0) + (hi * (-1)).exp())) for hi in h1]
        h2 = linear(h1, state_dict[f'layer{i}.mlp_fc2'])
        x = [x[j] + h2[j] for j in range(N_EMBD)]

    logits = linear(rmsnorm(x), state_dict['lm_head'])
    return logits, keys, values


# ── Task Agent ───────────────────────────────────────────────────────────

class TaskAgent:
    """MicroGPT with online learning — the creature that changes between breaths.

    Loads the trained checkpoint, runs prediction with differentiable loss,
    and does gradient descent on incoming text so the model adapts in real time.
    """

    def __init__(self, checkpoint_path=None, config=None):
        """
        Args:
            checkpoint_path: Path to trained_checkpoint.json. If None, uses
                default location at spark/microgpt_mirror/trained_checkpoint.json
            config: Optional dict overriding default hyperparameters:
                - learn_steps: number of gradient steps per learn() call (default 5)
                - learn_lr: learning rate for online learning (default 0.01)
                - temperature: sampling temperature for generate() (default 1.0)
        """
        self.config = {
            'learn_steps': 5,
            'learn_lr': 0.01,
            'temperature': 1.0,
            **(config or {}),
        }

        # Loss history across breaths — the meta-agent reads this
        self.loss_history = []

        # Load checkpoint
        if checkpoint_path is None:
            checkpoint_path = (
                Path(__file__).resolve().parent.parent.parent
                / 'spark' / 'microgpt_mirror' / 'trained_checkpoint.json'
            )
        self._load_checkpoint(checkpoint_path)

    def _load_checkpoint(self, path):
        """Load weights from JSON checkpoint into Value graph."""
        with open(path) as f:
            ckpt = json.load(f)

        self.chars = ckpt['chars']
        self.BOS = ckpt['BOS']
        self.vocab_size = ckpt['vocab_size']
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}

        # Convert numpy-serialized weights to Value objects for autograd
        self.state_dict = {}
        for key, matrix in ckpt['state_dict'].items():
            self.state_dict[key] = [
                [Value(float(v)) for v in row]
                for row in matrix
            ]

        # Collect all parameters for gradient updates
        self.params = [
            p for mat in self.state_dict.values()
            for row in mat for p in row
        ]

        # Adam optimizer state
        self._m = [0.0] * len(self.params)
        self._v = [0.0] * len(self.params)
        self._step = 0

    def _clean_text(self, text, max_chars=200):
        """Normalize text to model vocabulary."""
        clean = ''.join(c for c in text.lower() if c in self.char_to_idx)
        return clean[:max_chars]

    def predict(self, text):
        """Forward pass returning (mean_loss_float, surprise_contour).

        The surprise contour is a list of dicts with per-character loss.
        This is what creature.py's surprise_contour does, but loss is
        computed through the autograd graph so it can drive learning.

        Returns:
            (mean_loss: float, contour: list[dict])
        """
        clean = self._clean_text(text)
        if len(clean) < 2:
            return 0.0, []

        tokens = [self.BOS] + [self.char_to_idx[c] for c in clean]
        n = min(BLOCK_SIZE, len(tokens) - 1)

        keys = [[] for _ in range(N_LAYER)]
        values = [[] for _ in range(N_LAYER)]
        contour = []
        total_loss = 0.0

        for t in range(n):
            logits, keys, values = forward_token(
                tokens[t], t, keys, values, self.state_dict)
            probs = softmax(logits)

            actual = tokens[t + 1]
            prob_actual = probs[actual].data
            surprise = -math.log2(max(prob_actual, 1e-12))
            total_loss += surprise

            top_idx = max(range(len(probs)), key=lambda i: probs[i].data)
            top_char = self.chars[top_idx] if top_idx < len(self.chars) else '?'

            contour.append({
                'char': clean[t] if t < len(clean) else '?',
                'pos': t,
                'surprise': round(surprise, 4),
                'expected': top_char,
            })

            # Truncate KV cache
            if len(keys[0]) >= BLOCK_SIZE:
                for i in range(N_LAYER):
                    keys[i] = keys[i][-(BLOCK_SIZE - 1):]
                    values[i] = values[i][-(BLOCK_SIZE - 1):]

        mean_loss = total_loss / max(n, 1)
        return mean_loss, contour

    def learn(self, text, steps=None, lr=None):
        """Online fine-tuning on incoming text.

        This is the key missing piece: the creature changes between breaths.
        A few steps of gradient descent on the incoming text, using the
        autograd engine. Not magic — just memorization of recent input.
        Call it what it is.

        Returns:
            list of per-step loss values (floats)
        """
        steps = steps or self.config['learn_steps']
        lr = lr or self.config['learn_lr']
        beta1, beta2, eps_adam = 0.85, 0.99, 1e-8

        clean = self._clean_text(text)
        if len(clean) < 2:
            return []

        tokens = [self.BOS] + [self.char_to_idx[c] for c in clean]
        n = min(BLOCK_SIZE, len(tokens) - 1)

        step_losses = []
        for step in range(steps):
            keys = [[] for _ in range(N_LAYER)]
            values = [[] for _ in range(N_LAYER)]
            loss = Value(0.0)

            for t in range(n):
                logits, keys, values = forward_token(
                    tokens[t], t, keys, values, self.state_dict)
                probs = softmax(logits)
                loss = loss + (probs[tokens[t + 1]].log()) * (-1.0 / n)

            # Zero grads
            for p in self.params:
                p.grad = 0.0
            loss.backward()

            # Adam update
            self._step += 1
            for j, p in enumerate(self.params):
                self._m[j] = beta1 * self._m[j] + (1 - beta1) * p.grad
                self._v[j] = beta2 * self._v[j] + (1 - beta2) * p.grad ** 2
                m_hat = self._m[j] / (1 - beta1 ** self._step)
                v_hat = self._v[j] / (1 - beta2 ** self._step)
                p.data -= lr * m_hat / (v_hat ** 0.5 + eps_adam)

            step_losses.append(round(loss.data, 6))

        self.loss_history.append({
            'steps': steps,
            'lr': lr,
            'losses': step_losses,
            'text_len': len(clean),
        })

        return step_losses

    def generate(self, prompt='', max_tokens=None, temperature=None):
        """Sample from the model.

        This is needed so self-recursion is no longer tautological:
        the creature generates new text from its state, then measures that.

        Returns:
            generated text (str)
        """
        max_tokens = max_tokens or 32
        temperature = temperature or self.config['temperature']

        keys = [[] for _ in range(N_LAYER)]
        values = [[] for _ in range(N_LAYER)]

        # Feed prompt
        prompt_clean = self._clean_text(prompt, max_chars=BLOCK_SIZE - 2)
        tokens = [self.BOS]
        if prompt_clean:
            tokens += [self.char_to_idx[c] for c in prompt_clean]

        logits = None
        for t, tok in enumerate(tokens):
            logits, keys, values = forward_token(
                tok, t, keys, values, self.state_dict)

        generated = list(prompt_clean)
        pos = len(tokens)

        for _ in range(max_tokens):
            if pos >= BLOCK_SIZE:
                break

            probs = softmax(logits)
            prob_data = [p.data for p in probs]

            # Temperature scaling
            if temperature != 1.0:
                logit_data = [math.log(max(p, 1e-12)) / temperature
                              for p in prob_data]
                max_l = max(logit_data)
                exps = [math.exp(l - max_l) for l in logit_data]
                total = sum(exps)
                prob_data = [e / total for e in exps]

            # Sample
            r = random.random()
            cumulative = 0.0
            next_tok = 0
            for idx, p in enumerate(prob_data):
                cumulative += p
                if cumulative > r:
                    next_tok = idx
                    break

            if next_tok == self.BOS:
                break
            if next_tok < len(self.chars):
                generated.append(self.chars[next_tok])

            logits, keys, values = forward_token(
                next_tok, pos, keys, values, self.state_dict)
            pos += 1

        return ''.join(generated)

    def get_loss_trend(self, window=5):
        """Return recent loss trend for the meta-agent.

        Returns:
            dict with 'recent_losses', 'trend' ('improving'/'degrading'/'stable'),
            and 'mean_loss'.
        """
        if not self.loss_history:
            return {'recent_losses': [], 'trend': 'no_data', 'mean_loss': 0.0}

        recent = self.loss_history[-window:]
        final_losses = [entry['losses'][-1] for entry in recent if entry['losses']]

        if len(final_losses) < 2:
            trend = 'insufficient_data'
        else:
            delta = final_losses[-1] - final_losses[0]
            if delta < -0.01:
                trend = 'improving'  # loss going down — memorizing recent input
            elif delta > 0.01:
                trend = 'degrading'  # loss going up
            else:
                trend = 'stable'

        return {
            'recent_losses': final_losses,
            'trend': trend,
            'mean_loss': sum(final_losses) / len(final_losses) if final_losses else 0.0,
        }

    def save_checkpoint(self, path):
        """Save current weights to JSON checkpoint."""
        state_dict = {}
        for key, matrix in self.state_dict.items():
            state_dict[key] = [
                [p.data for p in row]
                for row in matrix
            ]
        ckpt = {
            'state_dict': state_dict,
            'chars': self.chars,
            'BOS': self.BOS,
            'vocab_size': self.vocab_size,
            'config': {
                'n_embd': N_EMBD,
                'n_head': N_HEAD,
                'n_layer': N_LAYER,
                'block_size': BLOCK_SIZE,
                'head_dim': HEAD_DIM,
            },
        }
        Path(path).write_text(json.dumps(ckpt, indent=2))
