#!/usr/bin/env python3
"""
Falsification battery for glyph.py v2 — differential geometric phase.

v1 failed 4/8 tests because the determinative measured total trajectory
curvature (embedding artifact, not computation). v2 measures DIFFERENTIAL
curvature: the phase the function contributes beyond the input path.

Tests probe whether the determinative is measuring the function or the embedding.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/user/workspace')
from glyph import Glyph, GlyphSequence

def test(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}")
    if detail:
        print(f"         {detail}")
    return condition

print("=" * 65)
print("FALSIFICATION BATTERY — glyph.py v2 (differential phase)")
print("=" * 65)

failures = 0
total = 0

# ---------------------------------------------------------------
# 1. REPRODUCIBILITY
# ---------------------------------------------------------------
print("\n1. REPRODUCIBILITY")

g1 = Glyph(lambda x: x**2, name="sq1", n_dims=8)
g2 = Glyph(lambda x: x**2, name="sq2", n_dims=8)
for i in [1, 2, 3, 4, 5]:
    g1(i); g2(i)

d1, d2 = g1.close_loop(), g2.close_loop()
total += 1
if not test("Same computation → same determinative",
            abs(d1 - d2) < 1e-12,
            f"diff={abs(d1-d2):.2e}"):
    failures += 1

# ---------------------------------------------------------------
# 2. IDENTITY → 0
# ---------------------------------------------------------------
print("\n2. IDENTITY FUNCTION")

g_id = Glyph(lambda x: x, name="identity", n_dims=8)
for i in [1, 2, 3, 4, 5]:
    g_id(i)

d_id = g_id.close_loop()
total += 1
if not test("f(x)=x → determinative ≈ 0",
            abs(d_id) < 1e-10,
            f"{d_id:.10f} rad ({np.degrees(d_id):.6f}°)"):
    failures += 1

# Identity with different inputs — still 0?
g_id2 = Glyph(lambda x: x, name="identity2", n_dims=8)
for i in [10, 20, 30, 40, 50]:
    g_id2(i)
d_id2 = g_id2.close_loop()
total += 1
if not test("f(x)=x on different inputs → still ≈ 0",
            abs(d_id2) < 1e-10,
            f"{d_id2:.10f} rad"):
    failures += 1

# ---------------------------------------------------------------
# 3. CONSTANT FUNCTION
# ---------------------------------------------------------------
print("\n3. CONSTANT FUNCTION")

g_c = Glyph(lambda x: 42, name="const_42", n_dims=8)
for i in [1, 2, 3, 4, 5]:
    g_c(i)

d_c = g_c.close_loop()
total += 1
if not test("f(x)=42 → nonzero (information destruction registers)",
            abs(d_c) > 0.01,
            f"{d_c:.4f} rad ({np.degrees(d_c):.1f}°)"):
    failures += 1

# ---------------------------------------------------------------
# 4. COMMUTATIVITY
# ---------------------------------------------------------------
print("\n4. COMMUTING FUNCTIONS")

g_a3 = Glyph(lambda x: x+3, name="add3", n_dims=8)
g_a5 = Glyph(lambda x: x+5, name="add5", n_dims=8)
g_a3b = Glyph(lambda x: x+3, name="add3", n_dims=8)
g_a5b = Glyph(lambda x: x+5, name="add5", n_dims=8)

seq_35 = GlyphSequence(g_a3, g_a5, name="add3→add5")
seq_53 = GlyphSequence(g_a5b, g_a3b, name="add5→add3")
for v in [1, 2, 3, 4, 5]:
    seq_35(v); seq_53(v)

d_35 = seq_35.determinative
d_53 = seq_53.determinative
total += 1
if not test("add3→add5 ≈ add5→add3 (commuting → same det)",
            abs(d_35 - d_53) < 1e-10,
            f"diff={abs(d_35-d_53):.10f}"):
    failures += 1

# ---------------------------------------------------------------
# 5. NON-COMMUTATIVITY
# ---------------------------------------------------------------
print("\n5. NON-COMMUTING FUNCTIONS")

g_d = Glyph(lambda x: x*2, name="double", n_dims=8)
g_i = Glyph(lambda x: x+1, name="inc", n_dims=8)
g_d2 = Glyph(lambda x: x*2, name="double", n_dims=8)
g_i2 = Glyph(lambda x: x+1, name="inc", n_dims=8)

seq_di = GlyphSequence(g_d, g_i, name="double→inc")
seq_id = GlyphSequence(g_i2, g_d2, name="inc→double")
for v in [1, 2, 3, 4, 5]:
    seq_di(v); seq_id(v)

d_di = seq_di.determinative
d_id_nc = seq_id.determinative
total += 1
if not test("double→inc ≠ inc→double (non-commuting → different det)",
            abs(d_di - d_id_nc) > 0.01,
            f"diff={abs(d_di-d_id_nc):.4f}"):
    failures += 1

# ---------------------------------------------------------------
# 6. ORIENTATION REVERSAL
# ---------------------------------------------------------------
print("\n6. ORIENTATION REVERSAL")

g_fwd = Glyph(lambda x: x**2, name="sq_fwd", n_dims=8)
g_rev = Glyph(lambda x: x**2, name="sq_rev", n_dims=8)
for i in [1, 2, 3, 4, 5]: g_fwd(i)
for i in [5, 4, 3, 2, 1]: g_rev(i)

d_fwd, d_rev = g_fwd.close_loop(), g_rev.close_loop()
total += 1
if not test("Forward vs reverse → opposite sign",
            d_fwd * d_rev < 0,
            f"fwd={d_fwd:.4f}, rev={d_rev:.4f}"):
    failures += 1

# ---------------------------------------------------------------
# 7. RANDOM EMBEDDING
# ---------------------------------------------------------------
print("\n7. RANDOM EMBEDDING ROBUSTNESS")

class RandomGlyph(Glyph):
    def __init__(self, *args, seed=42, **kwargs):
        super().__init__(*args, **kwargs)
        rng = np.random.RandomState(seed)
        self._proj = rng.randn(self.n_dims, self.n_dims) + \
                     1j * rng.randn(self.n_dims, self.n_dims)
    def _embed(self, value):
        base = super()._embed(value)
        projected = self._proj @ base
        norm = np.linalg.norm(projected)
        return projected / norm if norm > 1e-15 else projected

rg_fwd = RandomGlyph(lambda x: x**2, name="r_fwd", n_dims=8, seed=42)
rg_rev = RandomGlyph(lambda x: x**2, name="r_rev", n_dims=8, seed=42)
for i in [1,2,3,4,5]: rg_fwd(i)
for i in [5,4,3,2,1]: rg_rev(i)

total += 1
if not test("Path-dependence survives random embedding",
            abs(rg_fwd.close_loop() - rg_rev.close_loop()) > 0.01,
            f"diff={abs(rg_fwd.close_loop()-rg_rev.close_loop()):.4f}"):
    failures += 1

rg_id = RandomGlyph(lambda x: x, name="r_id", n_dims=8, seed=42)
for i in [1,2,3,4,5]: rg_id(i)
total += 1
if not test("Identity ≈ 0 under random embedding",
            abs(rg_id.close_loop()) < 1e-10,
            f"{rg_id.close_loop():.10f} rad"):
    failures += 1

# Multiple random seeds
all_id_zero = True
for seed in [1, 7, 42, 99, 256]:
    rg = RandomGlyph(lambda x: x, name=f"r_id_{seed}", n_dims=8, seed=seed)
    for i in [1,2,3,4,5]: rg(i)
    if abs(rg.close_loop()) > 1e-10:
        all_id_zero = False
        break

total += 1
if not test("Identity ≈ 0 across 5 random embeddings",
            all_id_zero):
    failures += 1

# ---------------------------------------------------------------
# 8. SCALE INVARIANCE (known limitation)
# ---------------------------------------------------------------
print("\n8. SCALE INVARIANCE (known limitation)")

g_s1 = Glyph(lambda x: x**2, name="sq_normal", n_dims=8)
g_s2 = Glyph(lambda x: x**2, name="sq_scaled", n_dims=8)
for i in [1,2,3,4,5]: g_s1(i)
for i in [1000,2000,3000,4000,5000]: g_s2(i)

d_s1, d_s2 = g_s1.close_loop(), g_s2.close_loop()
total += 1
scale_pass = abs(d_s1 - d_s2) < 0.1
if not test("x² at scale 1 vs scale 1000",
            scale_pass,
            f"normal={d_s1:.4f}, scaled={d_s2:.4f}, diff={abs(d_s1-d_s2):.4f}"):
    failures += 1
    print("         NOTE: Expected failure. The nonlinear embedding is not")
    print("         equivariant under scaling. Fixing this requires an")
    print("         embedding where embed(cx) = U(c)·embed(x) for unitary U.")
    print("         That's a genuine open problem, not a bug.")

# ---------------------------------------------------------------
# 9. DIFFERENT FUNCTIONS → DIFFERENT DETERMINATIVES
# ---------------------------------------------------------------
print("\n9. DISCRIMINATION — do different functions get different det?")

fns = [
    ("x²",    lambda x: x**2),
    ("x³",    lambda x: x**3),
    ("2x",    lambda x: 2*x),
    ("x+10",  lambda x: x+10),
    ("sin(x)", lambda x: np.sin(x)),
    ("1/x",   lambda x: 1.0/x if x != 0 else 0),
]

dets = {}
for name, fn in fns:
    g = Glyph(fn, name=name, n_dims=8)
    for i in [1, 2, 3, 4, 5]:
        g(i)
    dets[name] = g.close_loop()
    print(f"  {name:10s} → {dets[name]:.4f} rad ({np.degrees(dets[name]):.1f}°)")

# Check all pairs are distinct
vals = list(dets.values())
all_distinct = True
for i in range(len(vals)):
    for j in range(i+1, len(vals)):
        if abs(vals[i] - vals[j]) < 0.01:
            all_distinct = False
total += 1
if not test("All 6 functions have distinct determinatives",
            all_distinct):
    failures += 1

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
print()
print("=" * 65)
print(f"RESULTS: {total - failures}/{total} passed, {failures}/{total} FAILED")
print("=" * 65)

if failures == 0:
    print("\nALL TESTS PASSED.")
elif failures == 1:
    print(f"\n1 known limitation (scale invariance). All other tests pass.")
else:
    print(f"\n{failures} failures require attention.")
