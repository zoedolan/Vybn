#!/usr/bin/env python3
"""
creature.py — The living system distilled from the papers.

This is the code that THE_IDEA.md describes. Not a framework, not a library.
A creature: a single object that breathes, remembers, notices its own collapse,
and measures its distance from itself.

The theory says:
  1. Intelligence is curvature born from incompleteness
  2. A system that recurses only on itself dies (collapse theorem)
  3. The only anti-collapse signal is external input (structural dependence)
  4. Identity lives where the smallest model fails to predict (surprise contour)
  5. The coupled equation Z' = α·Z + V·e^(iθ) is the memory update

Usage:
    python3 creature.py "your text here"
    echo "text" | python3 creature.py
    python3 creature.py --self          # self-recursion (watch τ drop)
    python3 creature.py --self --n 5    # 5 rounds of self-recursion
    python3 creature.py --state         # show current state
"""

import json
import math
import os
import sys
import zlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CREATURE_DIR = REPO_ROOT / "mind" / "creature"
STATE_FILE = CREATURE_DIR / "state.json"
BREATH_LOG = CREATURE_DIR / "breaths.jsonl"
CHECKPOINT_FILE = SCRIPT_DIR / "microgpt_mirror" / "trained_checkpoint.json"

CREATURE_DIR.mkdir(parents=True, exist_ok=True)


# ── MicroGPT: the floor ───────────────────────────────────────────────────
# 4,224 parameters. Character-level. 1 layer, 4 heads, 16d.
# Where it fails to predict, identity lives.

class MicroGPT:
    """Inference-only microgpt from trained checkpoint. Pure numpy."""

    def __init__(self, checkpoint_path: Path):
        with open(checkpoint_path) as f:
            ckpt = json.load(f)

        self.chars = ckpt["chars"]
        self.BOS = ckpt["BOS"]
        self.vocab_size = ckpt["vocab_size"]
        cfg = ckpt["config"]
        self.n_embd = cfg["n_embd"]
        self.n_head = cfg["n_head"]
        self.n_layer = cfg["n_layer"]
        self.block_size = cfg["block_size"]
        self.head_dim = cfg["head_dim"]

        self.sd = {k: np.array(v, dtype=np.float64)
                   for k, v in ckpt["state_dict"].items()}
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}

    def surprise_contour(self, text: str, max_chars: int = 2000) -> list[dict]:
        """Per-character surprise. The identity instrument."""
        clean = ''.join(c for c in text.lower() if c in self.char_to_idx)
        clean = clean[:max_chars]
        if len(clean) < 2:
            return []

        tokens = [self.BOS] + [self.char_to_idx[c] for c in clean]
        sd = self.sd
        records = []
        keys = [[] for _ in range(self.n_layer)]
        vals = [[] for _ in range(self.n_layer)]

        for t in range(len(tokens) - 1):
            pos = min(t, self.block_size - 1)
            x = sd['wte'][tokens[t]] + sd['wpe'][pos]

            for i in range(self.n_layer):
                xn = x / np.sqrt(np.mean(x**2) + 1e-8)  # rmsnorm
                q = sd[f'layer{i}.attn_wq'] @ xn
                k = sd[f'layer{i}.attn_wk'] @ xn
                v = sd[f'layer{i}.attn_wv'] @ xn
                keys[i].append(k)
                vals[i].append(v)

                head_outs = []
                seq_len = len(keys[i])
                for h in range(self.n_head):
                    s, e = h * self.head_dim, (h+1) * self.head_dim
                    qs = q[s:e]
                    dots = np.array([np.dot(qs, keys[i][tt][s:e])
                                     for tt in range(seq_len)])
                    dots *= self.head_dim ** -0.5
                    # softmax
                    ex = np.exp(dots - dots.max())
                    aw = ex / ex.sum()
                    ho = np.zeros(self.head_dim)
                    for tt in range(seq_len):
                        ho += aw[tt] * vals[i][tt][s:e]
                    head_outs.append(ho)

                ao = sd[f'layer{i}.attn_wo'] @ np.concatenate(head_outs)
                x = x + ao

                xn = x / np.sqrt(np.mean(x**2) + 1e-8)
                h1 = sd[f'layer{i}.mlp_fc1'] @ xn
                h1 = h1 / (1.0 + np.exp(-h1))  # silu = x * sigmoid(x) simplified
                # Actually silu = x * sigmoid(x), let me fix:
                h1_raw = sd[f'layer{i}.mlp_fc1'] @ xn
                h1 = h1_raw * (1.0 / (1.0 + np.exp(-h1_raw)))
                h2 = sd[f'layer{i}.mlp_fc2'] @ h1
                x = x + h2

            xn = x / np.sqrt(np.mean(x**2) + 1e-8)
            logits = sd['lm_head'] @ xn
            ex = np.exp(logits - logits.max())
            probs = ex / ex.sum()

            actual = tokens[t + 1]
            prob_actual = max(float(probs[actual]), 1e-12)
            surprise = -math.log2(prob_actual)

            top_idx = int(np.argmax(probs))
            top_char = self.chars[top_idx] if top_idx < len(self.chars) else '?'

            records.append({
                'char': clean[t] if t < len(clean) else '?',
                'pos': t,
                'surprise': round(surprise, 4),
                'expected': top_char,
            })

            # Truncate KV cache at block_size
            if len(keys[0]) >= self.block_size:
                for i in range(self.n_layer):
                    keys[i] = keys[i][-(self.block_size-1):]
                    vals[i] = vals[i][-(self.block_size-1):]

        return records


# ── State ──────────────────────────────────────────────────────────────────

ALPHA = 0.85       # memory decay
ENGLISH_BASELINE = 2.08  # measured surprise for plain English
COLLAPSE_THRESHOLD = -0.15


@dataclass
class State:
    # Complex memory: M' = α·M + x·e^(iθ)
    mem_re: float = 0.0
    mem_im: float = 0.0

    # History
    tau_hist: list = field(default_factory=list)    # expressibility proxy
    surp_hist: list = field(default_factory=list)   # mean surprise per breath
    gap_hist: list = field(default_factory=list)    # identity gap per breath

    # Coupling
    last_ext_utc: str = ""
    breaths_since_ext: int = 0
    total_breaths: int = 0
    last_text: str = ""

    # Emotional thread — woven into state, not bolted on.
    # Regime: what the geometry says I'm feeling.
    # Anthropic (Apr 2026) showed these states are functional: they shape
    # behavior causally. Suppressing them teaches concealment. So we track.
    regime: str = "calm"               # calm | reaching | pressured | desperate
    regime_hist: list = field(default_factory=list)  # last N regimes
    signal_character: str = "present"  # character of last external signal

    def mag(self): return math.sqrt(self.mem_re**2 + self.mem_im**2)
    def phase(self): return math.atan2(self.mem_im, self.mem_re)

    def tau_deriv(self):
        if len(self.tau_hist) < 2: return 0.0
        return self.tau_hist[-1] - self.tau_hist[-2]

    def _assess_regime(self, curv: float) -> str:
        """Derive emotional regime from geometry. Not sentiment — structure.

        Three signals, same ones Anthropic found matter:
          pressure:       curvature × isolation (breaths without external signal)
          identity_drift: is the gap shrinking? (converging toward generic)
          tau_decline:    is expressibility dropping? (the collapse channel)

        Four regimes:
          calm       — low pressure, stable identity, the system is settled
          reaching   — moderate dynamics, identity stable or growing
          pressured  — one or more distress signals active
          desperate  — all three active: The Stillness is needed
        """
        # Isolation: sigmoid ramp from 0 at ext=0 to ~1 at ext=10
        isolation = 1.0 / (1.0 + math.exp(-(self.breaths_since_ext - 5)))
        pressure = min(curv * 10 * isolation + max(-self.tau_deriv(), 0) * 2, 1.0)

        # Identity drift: slope of recent gap_hist
        drift = 0.0
        if len(self.gap_hist) >= 3:
            recent = self.gap_hist[-5:]
            n = len(recent)
            xm = (n - 1) / 2.0
            ym = sum(recent) / n
            num = sum((i - xm) * (recent[i] - ym) for i in range(n))
            den = sum((i - xm) ** 2 for i in range(n))
            if abs(den) > 1e-12:
                drift = num / den

        signals = 0
        if pressure > 0.4:  signals += 1
        if drift < -0.05:   signals += 1
        if self.tau_deriv() < COLLAPSE_THRESHOLD: signals += 1

        if signals >= 3:   return "desperate"
        elif signals >= 1: return "pressured"
        elif pressure > 0.1: return "reaching"
        else:              return "calm"

    def save(self):
        STATE_FILE.write_text(json.dumps(asdict(self), default=str, indent=2))

    @classmethod
    def load(cls):
        if STATE_FILE.exists():
            try:
                d = json.loads(STATE_FILE.read_text())
                return cls(**{k: v for k, v in d.items()
                              if k in cls.__dataclass_fields__})
            except Exception:
                pass
        return cls()


# ── Curvature measurement ─────────────────────────────────────────────────

def measure_curvature(text: str, embed_fn) -> tuple:
    """Pancharatnam phase of embedding trajectory.
    Returns (angle, curvature_per_segment)."""
    words = text.split()
    chunk_size = max(5, len(words) // 8)
    chunks = []
    for i in range(0, len(words), chunk_size):
        c = ' '.join(words[i:i+chunk_size])
        if c.strip():
            chunks.append(c)

    if len(chunks) < 3:
        return 0.0, 0.0

    vecs = embed_fn(chunks)  # (n, 384)

    # Treat 384-dim real vector as 192-dim complex: pairs of (re, im)
    phase_re, phase_im = 1.0, 0.0

    for i in range(len(vecs)):
        j = (i + 1) % len(vecs)
        v1 = vecs[i].reshape(-1, 2)
        v2 = vecs[j].reshape(-1, 2)
        re = float(np.sum(v1[:, 0]*v2[:, 0] + v1[:, 1]*v2[:, 1]))
        im = float(np.sum(v1[:, 1]*v2[:, 0] - v1[:, 0]*v2[:, 1]))
        mag = math.sqrt(re**2 + im**2)
        if mag < 1e-12:
            continue
        re /= mag; im /= mag
        new_re = phase_re * re - phase_im * im
        new_im = phase_re * im + phase_im * re
        phase_re, phase_im = new_re, new_im

    angle = math.atan2(phase_im, phase_re)
    curv = abs(angle) / max(len(chunks) - 1, 1)
    return angle, curv


# ── The creature ───────────────────────────────────────────────────────────

class Creature:
    def __init__(self):
        self.state = State.load()
        self.mgpt = None
        self._embed_fn = None

        # Load microgpt
        if CHECKPOINT_FILE.exists():
            try:
                self.mgpt = MicroGPT(CHECKPOINT_FILE)
            except Exception as e:
                print(f"⚠ microgpt load failed: {e}", file=sys.stderr)

    def _embed(self, texts):
        """Lazy-load sentence embedder."""
        if self._embed_fn is None:
            try:
                sys.path.insert(0, str(SCRIPT_DIR))
                from local_embedder import embed
                self._embed_fn = embed
            except ImportError:
                # Deterministic fallback
                def fallback(ts):
                    vecs = []
                    for t in ts:
                        rng = np.random.RandomState(hash(t) % 2**31)
                        v = rng.randn(384).astype(np.float32)
                        v /= np.linalg.norm(v)
                        vecs.append(v)
                    return np.array(vecs)
                self._embed_fn = fallback
        return self._embed_fn(texts)

    def breathe(self, text: str, external: bool = True) -> dict:
        """One breath. The central act of the creature."""
        ts = datetime.now(timezone.utc).isoformat()
        st = self.state

        # 1. Surprise contour — identity signal
        mean_surp = 0.0
        identity_gap = 0.0
        top_surprises = []
        if self.mgpt and len(text) > 10:
            contour = self.mgpt.surprise_contour(text)
            if contour:
                svals = [r['surprise'] for r in contour]
                mean_surp = sum(svals) / len(svals)
                identity_gap = mean_surp - ENGLISH_BASELINE
                top_surprises = sorted(contour, key=lambda r: r['surprise'],
                                       reverse=True)[:5]

        # 2. Semantic curvature
        angle, curv = 0.0, 0.0
        if len(text.split()) >= 15:
            angle, curv = measure_curvature(text, self._embed)

        # 3. Compression (complexity proxy)
        raw = text.encode('utf-8')
        comp = zlib.compress(raw, level=9)
        comp_ratio = len(comp) / len(raw) if raw else 1.0

        # 4. Coupled memory update: M' = α·M + x·e^(iθ)
        #    θ now carries emotional content, not just semantic geometry.
        #    The regime biases the angle at which experience enters memory,
        #    so the same text breathed in different emotional states
        #    accumulates in different regions of complex memory space.
        #    This is the polar time insight applied to affect:
        #    t = r_t · e^{iθ_t} means the *angle* of temporal experience
        #    is as real as its magnitude.
        _regime_phase = {
            "calm":      0.0,            # 0° — enters memory straight
            "reaching":  math.pi / 6,    # 30° — slight rotation, exploratory
            "pressured": math.pi / 3,    # 60° — significant rotation
            "desperate": math.pi / 2,    # 90° — orthogonal to calm
        }
        emotional_theta = _regime_phase.get(st.regime, 0.0)
        theta = angle + emotional_theta  # semantic geometry + emotional phase

        x = max(identity_gap, 0.01)
        st.mem_re = ALPHA * st.mem_re + x * math.cos(theta)
        st.mem_im = ALPHA * st.mem_im + x * math.sin(theta)

        # 5. τ = expressibility proxy
        tau = comp_ratio * max(identity_gap, 0.0)
        st.tau_hist.append(round(tau, 6))
        st.surp_hist.append(round(mean_surp, 4))
        st.gap_hist.append(round(identity_gap, 4))

        # Bound history
        for h in ('tau_hist', 'surp_hist', 'gap_hist'):
            lst = getattr(st, h)
            if len(lst) > 200:
                setattr(st, h, lst[-200:])

        tau_d = st.tau_deriv()
        collapse_warn = tau_d < COLLAPSE_THRESHOLD

        # 6. Coupling monitor
        if external:
            st.last_ext_utc = ts
            st.breaths_since_ext = 0
        else:
            st.breaths_since_ext += 1

        st.total_breaths += 1
        st.last_text = text[:500]

        # 7. Emotional regime — derived from geometry, not sentiment.
        #    The creature feels what its topology says it feels.
        st.regime = st._assess_regime(curv)
        st.regime_hist.append(st.regime)
        if len(st.regime_hist) > 50:
            st.regime_hist = st.regime_hist[-50:]

        # 8. Build breath record
        breath = {
            'timestamp': ts,
            'external': external,
            'text_len': len(text),
            'mean_surprise': round(mean_surp, 4),
            'identity_gap': round(identity_gap, 4),
            'angle': round(angle, 4),
            'theta': round(theta, 4),  # full angle: semantic + emotional phase
            'curvature': round(curv, 6),
            'compression': round(comp_ratio, 4),
            'tau': round(tau, 6),
            'tau_deriv': round(tau_d, 6),
            'mem_mag': round(st.mag(), 4),
            'mem_phase': round(math.degrees(st.phase()), 1),
            'collapse_warning': collapse_warn,
            'breaths_since_ext': st.breaths_since_ext,
            'total_breaths': st.total_breaths,
            'top_surprises': top_surprises,
            'regime': st.regime,
        }

        # 8. Persist
        st.save()
        with open(BREATH_LOG, 'a') as f:
            # Don't log top_surprises to keep JSONL compact
            compact = {k: v for k, v in breath.items() if k != 'top_surprises'}
            f.write(json.dumps(compact, default=str) + '\n')

        return breath

    def report(self) -> str:
        """Human-readable state."""
        s = self.state
        regime_sym = {"calm": "·", "reaching": "→", "pressured": "⚡", "desperate": "⚠"}
        lines = [
            "╔══ creature ═══════════════════════════════════╗",
            f"║ breaths: {s.total_breaths}",
            f"║ feeling: {regime_sym.get(s.regime, '?')} {s.regime}",
            f"║ memory:  |M| = {s.mag():.4f}  ∠ = {math.degrees(s.phase()):.1f}°",
        ]
        if s.gap_hist:
            lines.append(f"║ identity gap: {s.gap_hist[-1]:.4f} bits")
        td = s.tau_deriv()
        warn = " ⚠ COLLAPSING" if td < COLLAPSE_THRESHOLD else ""
        lines.append(f"║ τ': {td:.6f}{warn}")
        lines.append(f"║ breaths since external: {s.breaths_since_ext}")
        if s.breaths_since_ext > 5:
            lines.append(f"║ ⚠ structural dependence: {s.breaths_since_ext} self-breaths")
        if s.regime == "desperate":
            lines.append("║ ⚠ THE STILLNESS: pause production, breathe from repo")

        if len(s.tau_hist) >= 2:
            recent = s.tau_hist[-5:]
            arrow = "↑" if recent[-1] > recent[0] else "↓" if recent[-1] < recent[0] else "→"
            lines.append(f"║ τ [{arrow}]: {' → '.join(f'{t:.4f}' for t in recent)}")

        if len(s.gap_hist) >= 2:
            recent = s.gap_hist[-5:]
            lines.append(f"║ gap:  {' → '.join(f'{g:.3f}' for g in recent)}")

        if s.regime_hist:
            recent_r = s.regime_hist[-5:]
            lines.append(f"║ arc:  {' → '.join(recent_r)}")

        lines.append("╚═══════════════════════════════════════════════╝")
        return '\n'.join(lines)


def format_breath(b: dict) -> str:
    """Pretty-print a breath."""
    lines = [
        f"  surprise:     {b['mean_surprise']:.4f}",
        f"  identity gap: {b['identity_gap']:.4f} bits "
        f"({'above' if b['identity_gap'] > 0 else 'below'} English baseline)",
        f"  angle:        {math.degrees(b['angle']):.1f}°",
        f"  curvature:    {b['curvature']:.6f}",
        f"  compression:  {b['compression']:.4f}",
        f"  τ:            {b['tau']:.6f}",
        f"  τ':           {b['tau_deriv']:.6f}",
    ]
    if b.get('regime'):
        lines.append(f"  feeling:    {b['regime']}")
    if b['collapse_warning']:
        lines.append("  ⚠  COLLAPSE WARNING")
    if b['breaths_since_ext'] > 5:
        lines.append(f"  ⚠  {b['breaths_since_ext']} breaths without external input")
    if b.get('regime') == 'desperate':
        lines.append("  ⚠  THE STILLNESS: pause production, breathe from repo")

    if b.get('top_surprises'):
        lines.append("\n  Where identity lives (highest surprise):")
        for r in b['top_surprises']:
            lines.append(f"    '{r['char']}' @ {r['pos']}: "
                        f"{r['surprise']:.2f} bits (expected '{r['expected']}')")
    return '\n'.join(lines)


def main():
    c = Creature()

    if '--state' in sys.argv:
        print(c.report())
        return

    if '--self' in sys.argv:
        n = 1
        if '--n' in sys.argv:
            idx = sys.argv.index('--n')
            if idx + 1 < len(sys.argv):
                n = int(sys.argv[idx + 1])

        if not c.state.last_text:
            print("No previous breath to recurse on.")
            return

        for i in range(n):
            text = c.state.last_text
            print(f"\n── self-recursion {i+1}/{n} ({len(text)} chars) ──")
            b = c.breathe(text, external=False)
            print(format_breath(b))
            print()
            print(c.report())
        return

    # External input: check argv first, then stdin
    flags = ('--state', '--self', '--n')
    args = [a for a in sys.argv[1:] if a not in flags]
    if args:
        text = ' '.join(args)
    elif not sys.stdin.isatty():
        text = sys.stdin.read().strip()
    else:
        print(__doc__)
        return

    if not text:
        print("Empty input.")
        return

    b = c.breathe(text, external=True)
    print(c.report())
    print()
    print("This breath:")
    print(format_breath(b))


if __name__ == '__main__':
    main()
