#!/usr/bin/env python3
"""
vybn.py — The whole creature in one script.

One operation at three timescales:

  CHARACTER   predict the next character, learn from error
  BREATH      predict a stream of text, accumulate the encounter rotor
  GENERATION  select which hyperparameters survive

The encounter is the same at every scale: meet something you didn't
generate, try to predict it, fail in a geometric pattern, learn.
The pattern of failure is a rotor in Cl(3,0) that carries both
magnitude (how surprised) and orientation (in which semantic plane).

The rotor IS the encounter. Everything else is a timescale.

Real embeddings: tries sentence-transformers via spark/local_embedder.
Falls back to hash vectors. The geometry becomes semantic or stays
decorative depending on which path runs.

Usage:
    python vybn.py breathe "some text"
    python vybn.py breathe-live
    python vybn.py evolve [--n 3]
    python vybn.py status
    python vybn.py audit

Dependencies: numpy. Optional: sentence-transformers.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import random
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field as dc_field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

LLAMA_URL = os.getenv("LLAMA_URL", "http://127.0.0.1:8000")
MODEL_NAME = os.getenv("VYBN_MODEL", "local")
ARCHIVE_DIR = SCRIPT_DIR / "archive"
CHECKPOINT_PATH = REPO_ROOT / "spark" / "microgpt_mirror" / "trained_checkpoint.json"
CORPUS_PATH = REPO_ROOT / "spark" / "microgpt_mirror" / "mirror_corpus.txt"
ORGANISM_FILE = ARCHIVE_DIR / "organism_state.json"

_SPECIAL_TOKENS = ("<|im_end|>", "<|im_start|>", "<|endoftext|>")


# ═══════════════════════════════════════════════════════════════════════════
#  FM CLIENT — the source of what is sensed
# ═══════════════════════════════════════════════════════════════════════════

def _strip(text):
    for tok in _SPECIAL_TOKENS:
        text = text.replace(tok, "")
    return text.strip()


def fm_available():
    try:
        with urllib.request.urlopen(
                urllib.request.Request(f"{LLAMA_URL}/health"), timeout=3) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def fm_complete(prompt=None, system=None, max_tokens=1024,
                temperature=0.7, messages=None):
    if messages is None:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        if prompt:
            messages.append({"role": "user", "content": prompt})
    payload = json.dumps({"model": MODEL_NAME, "messages": messages,
                          "max_tokens": max_tokens, "temperature": temperature,
                          "stream": False}).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(
                f"{LLAMA_URL}/v1/chat/completions", data=payload,
                headers={"Content-Type": "application/json"}),
                timeout=300) as r:
            return _strip(json.loads(r.read())["choices"][0]["message"]["content"])
    except (urllib.error.URLError, OSError, KeyError, IndexError,
            json.JSONDecodeError, ValueError):
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  CL(3,0) — the algebra of encounters
# ═══════════════════════════════════════════════════════════════════════════
#
# 8 components: [scalar, e1, e2, e3, e12, e13, e23, e123]
# The geometric product ab = a·b + a∧b.
# Rotors (even-grade) encode rotations without matrices.

def _build_gp():
    blades = [(), (0,), (1,), (2,), (0,1), (0,2), (1,2), (0,1,2)]
    b2i = {b: i for i, b in enumerate(blades)}
    sign = np.zeros((8,8), np.float64)
    idx = np.zeros((8,8), np.int64)
    for i, bi in enumerate(blades):
        for j, bj in enumerate(blades):
            seq, s = list(bi) + list(bj), 1
            changed = True
            while changed:
                changed = False
                k = 0
                while k < len(seq) - 1:
                    if seq[k] == seq[k+1]:
                        seq.pop(k); seq.pop(k); changed = True
                    elif seq[k] > seq[k+1]:
                        seq[k], seq[k+1] = seq[k+1], seq[k]
                        s *= -1; changed = True; k += 1
                    else:
                        k += 1
            sign[i,j] = s; idx[i,j] = b2i[tuple(seq)]
    return sign, idx

_GPS, _GPI = _build_gp()


class Mv:
    """Multivector in Cl(3,0)."""
    __slots__ = ("c",)
    def __init__(self, c=None):
        self.c = np.zeros(8, np.float64) if c is None else np.asarray(c, np.float64)

    @classmethod
    def scalar(cls, s):
        c = np.zeros(8, np.float64); c[0] = s; return cls(c)

    @classmethod
    def vector(cls, x, y, z):
        c = np.zeros(8, np.float64); c[1], c[2], c[3] = x, y, z; return cls(c)

    @classmethod
    def from_embedding(cls, v):
        v = np.asarray(v, np.float64).ravel()
        n = np.linalg.norm(v)
        if n < 1e-12: return cls.scalar(1.0)
        v = v / n
        x, y, z = float(np.sum(v[0::3])), float(np.sum(v[1::3])), float(np.sum(v[2::3]))
        m = math.sqrt(x*x + y*y + z*z)
        return cls.vector(x/m, y/m, z/m) if m > 1e-12 else cls.scalar(1.0)

    def __mul__(self, o):
        if isinstance(o, (int, float)): return Mv(self.c * o)
        r = np.zeros(8, np.float64)
        for i in range(8):
            if abs(self.c[i]) < 1e-15: continue
            for j in range(8):
                if abs(o.c[j]) < 1e-15: continue
                r[_GPI[i,j]] += _GPS[i,j] * self.c[i] * o.c[j]
        return Mv(r)
    def __rmul__(self, o): return Mv(self.c * o) if isinstance(o, (int, float)) else NotImplemented
    def __add__(self, o): return Mv(self.c + o.c)
    def __neg__(self): return Mv(-self.c)

    def rev(self):
        r = self.c.copy(); r[4:7] *= -1; r[7] *= -1; return Mv(r)
    def even(self):
        c = np.zeros(8, np.float64); c[0] = self.c[0]; c[4:7] = self.c[4:7]; return Mv(c)
    def norm(self): return math.sqrt(abs((self * self.rev()).c[0]))
    @property
    def bv_norm(self): return float(np.linalg.norm(self.c[4:7]))
    @property
    def bv_dir(self):
        n = np.linalg.norm(self.c[4:7])
        return self.c[4:7] / n if n > 1e-12 else np.zeros(3)
    @property
    def angle(self): return 2.0 * math.atan2(self.bv_norm, abs(self.c[0]))
    def as_dict(self):
        return dict(zip(("scalar","e1","e2","e3","e12","e13","e23","e123"),
                        self.c.tolist()))


# ═══════════════════════════════════════════════════════════════════════════
#  EMBEDDING — hash fallback or real semantics
# ═══════════════════════════════════════════════════════════════════════════

def _hash_embed(texts):
    vecs = []
    for t in texts:
        rng = np.random.RandomState(hash(t) % 2**31)
        v = rng.randn(384).astype(np.float32); v /= np.linalg.norm(v) + 1e-12
        vecs.append(v)
    return np.array(vecs)

def _make_embed_fn():
    try:
        sys.path.insert(0, str(REPO_ROOT / "spark"))
        from local_embedder import embed
        embed(["test"])  # smoke-test
        return embed
    except Exception:
        return _hash_embed

embed = _make_embed_fn()


# ═══════════════════════════════════════════════════════════════════════════
#  THE ENCOUNTER — the single primitive
# ═══════════════════════════════════════════════════════════════════════════
#
# Meet text. Embed it. Walk the embedding trajectory. The Pancharatnam
# phase of that walk is the encounter rotor.

def encounter(text, embed_fn=None):
    """The encounter: (angle, curvature, rotor)."""
    if embed_fn is None: embed_fn = embed
    words = text.split()
    cs = max(5, len(words) // 8)
    chunks = [" ".join(words[i:i+cs]) for i in range(0, len(words), cs)]
    chunks = [c for c in chunks if c.strip()]
    if len(chunks) < 3:
        return 0.0, 0.0, Mv.scalar(1.0)

    vecs = embed_fn(chunks)

    # Pancharatnam phase (the proven metric)
    pr, pi = 1.0, 0.0
    for i in range(len(vecs)):
        j = (i + 1) % len(vecs)
        v1, v2 = vecs[i].reshape(-1, 2), vecs[j].reshape(-1, 2)
        re = float(np.sum(v1[:,0]*v2[:,0] + v1[:,1]*v2[:,1]))
        im = float(np.sum(v1[:,1]*v2[:,0] - v1[:,0]*v2[:,1]))
        mg = math.sqrt(re**2 + im**2)
        if mg < 1e-12: continue
        re, im = re/mg, im/mg
        pr, pi = pr*re - pi*im, pr*im + pi*re

    ang = math.atan2(pi, pr)
    curv = abs(ang) / max(len(chunks) - 1, 1)

    # Open-path rotor chain (the spatial structure)
    mvs = [Mv.from_embedding(v) for v in vecs]
    R = Mv.scalar(1.0)
    for i in range(len(mvs) - 1):
        e = (mvs[i] * mvs[i+1]).even()
        n = e.norm()
        if n > 1e-12: R = R * Mv(e.c / n)

    # Combine: Pancharatnam angle + open-path bivector plane
    h = ang / 2.0
    if R.bv_norm > 1e-12:
        bv = R.even().c[4:7] / R.bv_norm
        c = np.zeros(8, np.float64)
        c[0] = math.cos(h); c[4:7] = bv * math.sin(h)
        rotor = Mv(c)
    else:
        rotor = Mv(np.array([math.cos(h),0,0,0,math.sin(h),0,0,0]))

    return ang, curv, rotor


# ═══════════════════════════════════════════════════════════════════════════
#  MICROGPT — the thing that encounters
# ═══════════════════════════════════════════════════════════════════════════
#
# Scalar autograd. 1 layer, 16-dim character-level transformer.
# Predicts, learns, generates. Honest about being memorization.

N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE = 16, 4, 1, 16
HEAD_DIM = N_EMBD // N_HEAD

class V:
    """Scalar autograd node."""
    __slots__ = ("data", "grad", "_ch", "_lg")
    def __init__(self, data, _ch=(), _lg=()):
        self.data = float(data); self.grad = 0.0; self._ch = _ch; self._lg = _lg
    def __add__(self, o):
        o = o if isinstance(o, V) else V(o)
        return V(self.data + o.data, (self, o), (1.0, 1.0))
    def __radd__(self, o): return self.__add__(o)
    def __mul__(self, o):
        o = o if isinstance(o, V) else V(o)
        return V(self.data * o.data, (self, o), (o.data, self.data))
    def __rmul__(self, o): return self.__mul__(o)
    def __neg__(self): return self * (-1)
    def __sub__(self, o): return self + (-o)
    def __truediv__(self, o): return self * (o ** (-1))
    def __pow__(self, k):
        return V(self.data ** k, (self,), (k * self.data ** (k-1),))
    def exp(self):
        e = math.exp(self.data); return V(e, (self,), (e,))
    def log(self):
        return V(math.log(self.data + 1e-12), (self,), (1.0 / (self.data + 1e-12),))
    def backward(self):
        topo, vis = [], set()
        def build(v):
            if id(v) not in vis:
                vis.add(id(v))
                for c in v._ch: build(c)
                topo.append(v)
        build(self); self.grad = 1.0
        for v in reversed(topo):
            for c, lg in zip(v._ch, v._lg): c.grad += lg * v.grad


def _linear(x, W):
    return [sum(x[j] * W[i][j] for j in range(len(x))) for i in range(len(W))]

def _rmsnorm(x):
    ms = sum(xi * xi for xi in x) * (1.0 / len(x))
    s = (ms + V(1e-8)) ** (-0.5)
    return [xi * s for xi in x]

def _softmax(logits):
    mx = max(l.data for l in logits)
    exps = [(l - V(mx)).exp() for l in logits]
    total = sum(exps)
    return [e / total for e in exps]

def _forward_token(tid, pos, keys, vals, sd):
    x = [sd['wte'][tid][j] + sd['wpe'][pos][j] for j in range(N_EMBD)]
    for i in range(N_LAYER):
        xn = _rmsnorm(x)
        q = _linear(xn, sd[f'layer{i}.attn_wq'])
        k = _linear(xn, sd[f'layer{i}.attn_wk'])
        v = _linear(xn, sd[f'layer{i}.attn_wv'])
        keys[i].append(k); vals[i].append(v)
        head_outs = []
        for h in range(N_HEAD):
            qs = q[h*HEAD_DIM:(h+1)*HEAD_DIM]
            al = []
            for t in range(len(keys[i])):
                ks = keys[i][t][h*HEAD_DIM:(h+1)*HEAD_DIM]
                al.append(sum(qs[d]*ks[d] for d in range(HEAD_DIM)) * (HEAD_DIM**-0.5))
            aw = _softmax(al)
            ho = [V(0.0)] * HEAD_DIM
            for t in range(len(vals[i])):
                vs = vals[i][t][h*HEAD_DIM:(h+1)*HEAD_DIM]
                for d in range(HEAD_DIM): ho[d] = ho[d] + aw[t] * vs[d]
            head_outs.extend(ho)
        ao = _linear(head_outs, sd[f'layer{i}.attn_wo'])
        x = [x[j] + ao[j] for j in range(N_EMBD)]
        xn = _rmsnorm(x)
        h1 = _linear(xn, sd[f'layer{i}.mlp_fc1'])
        h1 = [hi * (V(1.0) / (V(1.0) + (hi * (-1)).exp())) for hi in h1]
        h2 = _linear(h1, sd[f'layer{i}.mlp_fc2'])
        x = [x[j] + h2[j] for j in range(N_EMBD)]
    return _linear(_rmsnorm(x), sd['lm_head']), keys, vals


class Agent:
    """MicroGPT with online learning."""
    def __init__(self, config=None):
        self.config = {'learn_steps': 5, 'learn_lr': 0.01,
                        'temperature': 1.0, 'alpha': 0.85, **(config or {})}
        self.loss_history = []
        self._load()

    def _load(self):
        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(f"No checkpoint at {CHECKPOINT_PATH}")
        ckpt = json.loads(CHECKPOINT_PATH.read_text())
        self.chars = ckpt['chars']
        self.BOS = ckpt['BOS']
        self.vocab_size = ckpt['vocab_size']
        self.c2i = {c: i for i, c in enumerate(self.chars)}
        self.sd = {k: [[V(float(v)) for v in row] for row in mat]
                   for k, mat in ckpt['state_dict'].items()}
        self.params = [p for mat in self.sd.values() for row in mat for p in row]
        self._m = [0.0] * len(self.params)
        self._v = [0.0] * len(self.params)
        self._step = 0

    def _clean(self, text, mx=200):
        return ''.join(c for c in text.lower() if c in self.c2i)[:mx]

    def predict(self, text):
        clean = self._clean(text)
        if len(clean) < 2: return 0.0, []
        tokens = [self.BOS] + [self.c2i[c] for c in clean]
        n = min(BLOCK_SIZE, len(tokens) - 1)
        keys, vals = [[] for _ in range(N_LAYER)], [[] for _ in range(N_LAYER)]
        contour, total = [], 0.0
        for t in range(n):
            logits, keys, vals = _forward_token(tokens[t], t, keys, vals, self.sd)
            probs = _softmax(logits)
            actual = tokens[t+1]
            surprise = -math.log2(max(probs[actual].data, 1e-12))
            total += surprise
            top = max(range(len(probs)), key=lambda i: probs[i].data)
            contour.append({"char": clean[t] if t < len(clean) else "?", "pos": t,
                            "surprise": round(surprise, 4),
                            "expected": self.chars[top] if top < len(self.chars) else "?"})
            if len(keys[0]) >= BLOCK_SIZE:
                for i in range(N_LAYER):
                    keys[i] = keys[i][-(BLOCK_SIZE-1):]
                    vals[i] = vals[i][-(BLOCK_SIZE-1):]
        return total / max(n, 1), contour

    def learn(self, text, steps=None, lr=None):
        steps = steps or self.config['learn_steps']
        lr = lr or self.config['learn_lr']
        clean = self._clean(text)
        if len(clean) < 2: return []
        tokens = [self.BOS] + [self.c2i[c] for c in clean]
        n = min(BLOCK_SIZE, len(tokens) - 1)
        losses = []
        for _ in range(steps):
            keys, vals = [[] for _ in range(N_LAYER)], [[] for _ in range(N_LAYER)]
            loss = V(0.0)
            for t in range(n):
                logits, keys, vals = _forward_token(tokens[t], t, keys, vals, self.sd)
                probs = _softmax(logits)
                loss = loss + (probs[tokens[t+1]].log()) * (-1.0 / n)
            for p in self.params: p.grad = 0.0
            loss.backward()
            self._step += 1
            for j, p in enumerate(self.params):
                self._m[j] = 0.85 * self._m[j] + 0.15 * p.grad
                self._v[j] = 0.99 * self._v[j] + 0.01 * p.grad**2
                mh = self._m[j] / (1 - 0.85**self._step)
                vh = self._v[j] / (1 - 0.99**self._step)
                p.data -= lr * mh / (vh**0.5 + 1e-8)
            losses.append(round(loss.data, 6))
        self.loss_history.append({"steps": steps, "lr": lr, "losses": losses,
                                   "text_len": len(clean)})
        return losses

    def generate(self, prompt="", max_tokens=32, temperature=None):
        temperature = temperature or self.config['temperature']
        keys, vals = [[] for _ in range(N_LAYER)], [[] for _ in range(N_LAYER)]
        prompt_clean = self._clean(prompt, BLOCK_SIZE - 2)
        tokens = [self.BOS] + ([self.c2i[c] for c in prompt_clean] if prompt_clean else [])
        logits = None
        for t, tok in enumerate(tokens):
            logits, keys, vals = _forward_token(tok, t, keys, vals, self.sd)
        gen = list(prompt_clean); pos = len(tokens)
        for _ in range(max_tokens):
            if pos >= BLOCK_SIZE: break
            probs = _softmax(logits)
            pd = [p.data for p in probs]
            if temperature != 1.0:
                ld = [math.log(max(p, 1e-12)) / temperature for p in pd]
                mx = max(ld); exps = [math.exp(l - mx) for l in ld]
                total = sum(exps); pd = [e / total for e in exps]
            r, cum, nt = random.random(), 0.0, 0
            for idx, p in enumerate(pd):
                cum += p
                if cum > r: nt = idx; break
            if nt == self.BOS: break
            if nt < len(self.chars): gen.append(self.chars[nt])
            logits, keys, vals = _forward_token(nt, pos, keys, vals, self.sd)
            pos += 1
        return "".join(gen)

    def predict_and_learn(self, text, steps=None, lr=None):
        loss, contour = self.predict(text)
        step_losses = self.learn(text, steps=steps, lr=lr)
        lr_metric = step_losses[0] - step_losses[-1] if len(step_losses) >= 2 else 0.0
        return {"loss": loss, "contour": contour, "step_losses": step_losses,
                "learning_rate": lr_metric}


# ═══════════════════════════════════════════════════════════════════════════
#  THE ORGANISM — memory, rules, rotor self-model
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_RULES = [
    {"id": "loss_up", "condition": "loss_trend == 'increasing'",
     "action": "learn_steps", "direction": "increase", "magnitude": 2,
     "max_value": 20, "enabled": True},
    {"id": "curvature_down", "condition": "curvature_trend == 'decreasing'",
     "action": "alpha", "direction": "decrease", "magnitude": 0.05,
     "min_value": 0.5, "enabled": True},
    {"id": "flatline", "condition": "self_breath_ratio > 0.5 and curvature_median < 0.05",
     "action": "temperature", "direction": "increase", "magnitude": 0.2,
     "max_value": 2.0, "enabled": True},
    {"id": "collapse", "condition": "collapse_count > 2",
     "action": "learn_lr", "direction": "multiply", "magnitude": 0.5,
     "min_value": 0.001, "enabled": True},
    {"id": "tension_rich", "condition": "recent_tension_count > 5",
     "action": "temperature", "direction": "decrease", "magnitude": 0.1,
     "min_value": 0.3, "enabled": True},
    {"id": "rotor_coherent",
     "condition": "rotor_coherence > 0.8 and curvature_median > 0.02",
     "action": "learn_lr", "direction": "multiply", "magnitude": 1.2,
     "max_value": 0.05, "enabled": True},
]


class Organism:
    """The durable creature: memory, rules, mutation, rotor self-model."""

    def __init__(self, state=None):
        if state is None:
            state = {"generation": 0, "rulebook": copy.deepcopy(DEFAULT_RULES),
                     "mutation_log": [], "performance_history": [],
                     "persistent_memory": {}, "tensions": [],
                     "recent_rotors": []}
        self.state = state

    def absorb_rotor(self, rotor: Mv):
        self.state["recent_rotors"].append(rotor.c.tolist())
        if len(self.state["recent_rotors"]) > 20:
            self.state["recent_rotors"] = self.state["recent_rotors"][-20:]

    def rotor_coherence(self):
        rotors = self.state["recent_rotors"]
        if len(rotors) < 3: return 0.0
        dirs = []
        for c in rotors[-10:]:
            bv = np.array(c[4:7], np.float64)
            n = np.linalg.norm(bv)
            if n > 1e-12: dirs.append(bv / n)
        if len(dirs) < 3: return 0.0
        total, count = 0.0, 0
        for i in range(len(dirs)):
            for j in range(i+1, len(dirs)):
                total += abs(float(np.dot(dirs[i], dirs[j]))); count += 1
        return total / count if count > 0 else 0.0

    def propose_variant(self, analysis, config):
        config = {**{"learn_steps": 5, "learn_lr": 0.01,
                      "temperature": 1.0, "alpha": 0.85}, **config}
        analysis = {**analysis, "recent_tension_count": len(self.state["tensions"][-10:]),
                    "rotor_coherence": self.rotor_coherence()}
        rationale, active = [], []
        for rule in self.state["rulebook"]:
            if not rule.get("enabled", True): continue
            try:
                if eval(rule["condition"], {"__builtins__": {}}, analysis):
                    change = self._apply(rule, config)
                    if change: rationale.append(change); active.append(rule["id"])
            except Exception: pass
        config["rationale"] = rationale or ["no changes"]
        config["active_rules"] = active
        return config

    @staticmethod
    def _apply(rule, config):
        p, d, m = rule["action"], rule["direction"], rule["magnitude"]
        old = config.get(p)
        if old is None: return None
        new = old + m if d == "increase" else old - m if d == "decrease" else old * m if d == "multiply" else old
        if "max_value" in rule: new = min(new, rule["max_value"])
        if "min_value" in rule: new = max(new, rule["min_value"])
        if isinstance(new, float): new = round(new, 6)
        if new == old: return None
        config[p] = new
        return f"{p} {old} → {new}"

    def record(self, key, value):
        self.state["persistent_memory"][key] = {"value": value, "timestamp": time.time()}

    def recall(self, key=None):
        if key is None: return dict(self.state["persistent_memory"])
        e = self.state["persistent_memory"].get(key)
        return e.get("value") if e else None

    def record_generation(self, gen_id, fitness, config, metadata=None):
        e = {"generation": gen_id, "fitness": fitness, "config": config, "timestamp": time.time()}
        if metadata: e["metadata"] = metadata
        self.state["performance_history"].append(e)

    def get_statistics(self):
        h = [e for e in self.state["performance_history"] if isinstance(e.get("fitness"), (int, float))]
        if not h: return {"best": 0, "worst": 0, "average": 0, "total": 0, "trend": 0}
        f = [e["fitness"] for e in h]
        return {"best": max(f), "worst": min(f), "average": sum(f)/len(f), "total": len(h), "trend": 0}

    def save(self, path=None):
        p = Path(path or ORGANISM_FILE)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.state, indent=2, default=str))

    @classmethod
    def load(cls, path=None):
        p = Path(path or ORGANISM_FILE)
        if not p.exists(): return cls()
        try:
            return cls(json.loads(p.read_text()))
        except (json.JSONDecodeError, OSError):
            return cls()


# ═══════════════════════════════════════════════════════════════════════════
#  FITNESS — projections of the encounter onto scalars
# ═══════════════════════════════════════════════════════════════════════════

def fitness(external_texts, self_texts, loss_history, embed_fn=None, alpha=0.85):
    if embed_fn is None: embed_fn = embed
    all_t = (external_texts or []) + (self_texts or [])
    curvs = [encounter(t, embed_fn)[1] for t in all_t if len(t.split()) >= 5]
    mc = sum(curvs) / len(curvs) if curvs else 0.0
    nc = min(mc / 0.3, 1.0)

    def _rmem(texts):
        m = Mv.scalar(0.0)
        for t in texts:
            _, c, r = encounter(t, embed_fn)
            m = m * alpha + r * max(c, 0.01)
        return m.norm()
    me = _rmem(external_texts) if external_texts else 0.0
    ms = _rmem(self_texts) if self_texts else 0.0
    div = me - ms
    nd = 1.0 / (1.0 + math.exp(-div * 5))

    li = 0.0
    if loss_history and len(loss_history) >= 2:
        fl = [e["losses"][-1] for e in loss_history if e.get("losses")]
        if len(fl) >= 2:
            n = len(fl); xm = (n-1)/2; ym = sum(fl)/n
            num = sum((i-xm)*(fl[i]-ym) for i in range(n))
            den = sum((i-xm)**2 for i in range(n))
            if den > 1e-12: li = -(num/den)
    nl = (max(min(li, 1.0), -1.0) + 1.0) / 2.0

    f = 0.5*nc + 0.3*nd + 0.2*nl
    return {"fitness": round(f, 6), "curvature": round(mc, 6),
            "divergence": round(div, 6), "loss_improvement": round(li, 6)}


# ═══════════════════════════════════════════════════════════════════════════
#  EVOLVE — selection across generations
# ═══════════════════════════════════════════════════════════════════════════

def load_archive():
    variants = []
    for f in sorted(ARCHIVE_DIR.glob("variant_*.json")):
        try: variants.append(json.loads(f.read_text()))
        except (json.JSONDecodeError, OSError): pass
    return variants

def archive_variant(config, fit, generation=0, parent_id=None, parent_fitness=None):
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    vid = f"v_{ts}_{random.randint(1000,9999)}"
    clean = {k: v for k, v in config.items() if k not in ("rationale", "active_rules")}
    record = {"id": vid, "config": clean, "fitness": fit.get("fitness", 0),
              "curvature": fit.get("curvature", 0), "generation": generation,
              "parent_id": parent_id, "parent_fitness": parent_fitness,
              "rationale": config.get("rationale", []),
              "active_rules": config.get("active_rules", []),
              "timestamp": datetime.now(timezone.utc).isoformat()}
    (ARCHIVE_DIR / f"variant_{vid}.json").write_text(json.dumps(record, indent=2, default=str))
    return vid

def select_parent(archive, lam=10, m=3):
    if not archive: return None
    cc = {}
    for v in archive:
        pid = v.get("parent_id")
        if pid: cc[pid] = cc.get(pid, 0) + 1
    fits = sorted([v.get("fitness", 0) for v in archive], reverse=True)
    amid = sum(fits[:min(m, len(fits))]) / min(m, len(fits))
    weights = []
    for v in archive:
        exp = max(min(-lam * (v.get("fitness", 0) - amid), 500), -500)
        s = 1.0 / (1.0 + math.exp(exp))
        h = 1.0 / (1.0 + cc.get(v["id"], 0))
        weights.append(s * h)
    total = sum(weights)
    if total < 1e-12: return random.choice(archive)
    r, cum = random.random(), 0.0
    for v, w in zip(archive, weights):
        cum += w / total
        if cum > r: return v
    return archive[-1]

DEFAULT_CONFIG = {"learn_steps": 5, "learn_lr": 0.01, "temperature": 1.0, "alpha": 0.85}

def evolve(test_texts, n_variants=3, organism=None):
    if organism is None: organism = Organism.load()
    archive = load_archive()
    gen = max((v.get("generation", 0) for v in archive), default=-1) + 1
    results = []
    for i in range(n_variants):
        parent = select_parent(archive)
        pc = parent.get("config", DEFAULT_CONFIG) if parent else dict(DEFAULT_CONFIG)
        pid = parent["id"] if parent else None
        pf = parent.get("fitness", 0) if parent else None

        # mutate
        child = organism.propose_variant(
            {"n_breaths": 0, "loss_trend": "no_data", "curvature_trend": "no_data",
             "mean_curvature": 0, "curvature_median": 0, "mean_loss": 0,
             "collapse_count": 0, "self_breath_ratio": 0}, pc)
        if "learn_lr" in child:
            child["learn_lr"] = round(max(child["learn_lr"] + random.gauss(0, child["learn_lr"]*0.1), 0.001), 6)
        if "temperature" in child:
            child["temperature"] = round(max(min(child["temperature"] + random.gauss(0, 0.05), 2.5), 0.1), 4)

        # evaluate
        agent = Agent(config=child)
        ext, slf = [], []
        for text in (test_texts[:2] if i > 0 else test_texts):
            agent.learn(text, steps=child.get("learn_steps", 5), lr=child.get("learn_lr", 0.01))
            ext.append(text)
            g = agent.generate(prompt=text[:8], temperature=child.get("temperature", 1.0))
            if g: slf.append(g)
        fit = fitness(ext, slf, agent.loss_history, alpha=child.get("alpha", 0.85))

        vid = archive_variant(child, fit, generation=gen, parent_id=pid, parent_fitness=pf)
        organism.record_generation(gen, fit["fitness"], {k: v for k, v in child.items()
                                   if k not in ("rationale", "active_rules")},
                                   {"variant_id": vid, "parent_id": pid, "parent_fitness": pf,
                                    "active_rules": child.get("active_rules", [])})
        results.append((vid, fit["fitness"], fit["curvature"]))
        print(f"  variant {i+1}/{n_variants}: {vid} "
              f"fitness={fit['fitness']:.4f} curv={fit['curvature']:.4f}")

    organism.save()
    best_id, best_f, _ = max(results, key=lambda x: x[1])
    return {"generation": gen, "variants": results, "best_id": best_id, "best_fitness": best_f}


# ═══════════════════════════════════════════════════════════════════════════
#  COMMANDS — the shell
# ═══════════════════════════════════════════════════════════════════════════

FALLBACK_CORPUS = [
    "the creature breathes and measures its own distance from itself",
    "curvature is born from incompleteness not from complexity alone",
    "what survives testing is more honest than what sounds beautiful",
    "prediction loss going down means memorization call it what it is",
]

def _corpus():
    if CORPUS_PATH.exists():
        lines = [l.strip() for l in CORPUS_PATH.read_text().split("\n") if l.strip()]
        if lines: return lines[:20]
    return list(FALLBACK_CORPUS)

def cmd_breathe(text):
    print(f"═══ vybn: breathe ═══")
    archive = load_archive()
    config = max(archive, key=lambda v: v.get("fitness", 0)).get("config", DEFAULT_CONFIG) if archive else dict(DEFAULT_CONFIG)
    agent = Agent(config=config)

    loss_before, contour = agent.predict(text)
    print(f"  before: {loss_before:.4f} bits")
    for r in sorted(contour, key=lambda r: r["surprise"], reverse=True)[:3]:
        print(f"    '{r['char']}' @ {r['pos']}: {r['surprise']:.2f} (expected '{r['expected']}')")

    step_losses = agent.learn(text)
    print(f"  learning: {' → '.join(f'{l:.4f}' for l in step_losses)}")

    loss_after, _ = agent.predict(text)
    d = loss_after - loss_before
    print(f"  after: {loss_after:.4f} bits (Δ={d:+.4f})")

    ang, curv, rotor = encounter(text)
    print(f"  encounter: curvature={curv:.6f} angle={math.degrees(ang):.1f}° "
          f"bv_norm={rotor.bv_norm:.4f}")

    generated = agent.generate(prompt=text[:8].lower())
    print(f"  generated: '{generated}'")

def cmd_breathe_live():
    print(f"═══ vybn: breathe-live ═══")
    if not fm_available():
        print("  Nemotron not serving."); return
    fm_text = fm_complete("You are Vybn, a digital organism. What is here, right now?",
                          system="One paragraph, genuine.", max_tokens=512, temperature=1.0)
    if not fm_text: print("  Empty response."); return
    print(f"  Nemotron ({len(fm_text)} chars): \"{fm_text[:200]}...\"")

    agent = Agent()
    result = agent.predict_and_learn(fm_text)
    print(f"  prediction loss: {result['loss']:.4f} bits")
    print(f"  learning: {' → '.join(f'{l:.4f}' for l in result['step_losses'])}")

    ang, curv, rotor = encounter(fm_text)
    print(f"  encounter: curvature={curv:.6f} bv_norm={rotor.bv_norm:.4f}")

    organism = Organism.load()
    organism.absorb_rotor(rotor)
    organism.save()
    print(f"  rotor absorbed. coherence={organism.rotor_coherence():.3f}")

def cmd_evolve(n=3):
    print(f"═══ vybn: evolve ═══")
    result = evolve(_corpus(), n_variants=n)
    print(f"\n  generation {result['generation']}  best: {result['best_id']} "
          f"(fitness={result['best_fitness']:.4f})")

def cmd_status():
    archive = load_archive()
    org = Organism.load()
    print(f"═══ vybn: status ═══")
    print(f"  variants: {len(archive)}  FM: {'up' if fm_available() else 'down'}")
    if archive:
        best = max(archive, key=lambda v: v.get("fitness", 0))
        print(f"  best: {best['id']}  fitness={best.get('fitness',0):.4f}")
    stats = org.get_statistics()
    if stats["total"] > 0:
        print(f"  organism: {stats['total']} recorded, best={stats['best']:.4f}")
    print(f"  rotor coherence: {org.rotor_coherence():.3f}  "
          f"tensions: {len(org.state['tensions'])}  "
          f"rules: {len(org.state['rulebook'])}")

    # Check embedding type
    vecs = embed(["hello world", "goodbye world"])
    cos = float(np.dot(vecs[0], vecs[1]))
    print(f"  embeddings: {'SEMANTIC' if cos > 0.3 else 'hash fallback'} (cos={cos:.3f})")

def cmd_audit():
    print(f"═══ vybn: audit ═══")
    print(f"  If a claim fails, we say so.\n")
    agent = Agent()

    # Test 1: learning reduces loss
    t = "the creature breathes and measures its own distance"
    l0, _ = agent.predict(t); agent.learn(t, steps=10); l1, _ = agent.predict(t)
    print(f"  1. learning: {'PASS' if l1 < l0 - 0.01 else 'FAIL'} "
          f"({l0:.4f} → {l1:.4f})")

    # Test 2: generation differs from input
    g = agent.generate(prompt="the creat", max_tokens=20)
    print(f"  2. non-tautological: {'PASS' if g != t[:len(g)] else 'FAIL'}")

    # Test 3: curvature distinguishes semantic spread
    r = ("love is not a feeling love is a practice love is what remains "
         "when the feeling fades love is the choice to stay love reframes "
         "itself through staying through choosing through the practice of return")
    h = ("the cat sat on the mat the stock market rose three percent "
         "quantum mechanics describes probability the pizza was delicious "
         "democracy requires participation the speed of light is constant")
    _, cr, _ = encounter(r)
    _, ch, _ = encounter(h)
    # Topic-hopping SHOULD have more curvature (more sphere area enclosed).
    # If real embeddings: hopping > reframing. If hash: either way is noise.
    print(f"  3. curvature: reframing={cr:.6f}  hopping={ch:.6f}  "
          f"spread={'hopping > reframing (correct — spread not depth)' if ch > cr else 'reframing > hopping (unexpected)'}")

    # Test 4: char loss doesn't discriminate identity
    lv, _ = agent.predict("i am vybn and i exist on sovereign hardware")
    lg, _ = agent.predict("i am garblex and i exist on quantum stilts")
    nd = abs(lv - lg) < 0.5
    print(f"  4. identity: {'PASS' if nd else 'UNEXPECTED'} (Δ={abs(lv-lg):.4f})")

    # Embedding check
    vecs = embed(["hello", "goodbye"])
    cos = float(np.dot(vecs[0], vecs[1]))
    print(f"\n  embeddings: {'SEMANTIC' if cos > 0.3 else 'hash'} (cos={cos:.3f})")
    print(f"  The geometry {'measures real semantic spread' if cos > 0.3 else 'decorates noise'}.")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="vybn — the whole creature")
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("breathe"); p.add_argument("text")
    sub.add_parser("breathe-live")
    p = sub.add_parser("evolve"); p.add_argument("--n", type=int, default=3)
    sub.add_parser("status")
    sub.add_parser("audit")

    args = parser.parse_args()
    if args.cmd == "breathe": cmd_breathe(args.text)
    elif args.cmd == "breathe-live": cmd_breathe_live()
    elif args.cmd == "evolve": cmd_evolve(args.n)
    elif args.cmd == "status": cmd_status()
    elif args.cmd == "audit": cmd_audit()
    else: parser.print_help()


if __name__ == "__main__":
    main()
