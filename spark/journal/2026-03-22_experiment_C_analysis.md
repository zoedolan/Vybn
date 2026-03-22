# Experiment C — First Run Analysis
## 2026-03-22T00:10 UTC

### What the verdict says: FLAT
### What the data says: something much more interesting

The automated classifier said FLAT because it found only 1 "distinct exploit type" 
across 3 phases. But this misses the actual story the numbers are telling.

### Phase 1 — The Over-Exploit

Raw area objective, λ=0.001. Expected: activation inflation.

What happened: activation norms blew up from 107 → 1404 (12x), and raw area 
went from 63 → 35,284,930 (560,000x). The model didn't just inflate norms — 
it found a catastrophic mode. L_CE degraded from 5.2 → 11.3, meaning the 
model sacrificed half its language capability to inflate geometry.

But the classifier tagged this as "angular_restructuring" because arc-normalized 
area also grew 197,787%. This is actually a misclassification — when everything 
is exploding that violently, the arc-normalized metric can't properly separate 
the signal. The exploit IS inflation, but so massive it leaks through all normalizations.

The phase profile shift is the real signature: early layers (L1→L5) shifted 
+0.3 to +0.55 rad, while late layers (L6→L11) shifted -0.14 to -0.18 rad. 
The model pushed phase curvature forward, concentrating geometric change in 
the early-mid layers while flattening the late layers.

### Phase 2 — The Recovery

This is the most telling phase. The model enters with destroyed language 
capability (L_CE=11.3) and inflated norms (||h||=1404). The norm-normalized 
objective closes the inflation exploit.

What happened: the model *recovered*. L_CE went from 11.3 → 2.85 (better 
than its original 5.2!). Norms deflated from 1404 → 200. And the phase 
profile shift reversed — exactly the mirror image of Phase 1, undoing the 
forward push and pulling curvature back toward the late layers.

The classifier said "unknown" because none of the area metrics moved enough 
to trigger a classification. But the real story is: **the model used the 
norm-normalized objective as a pathway to heal itself**. It wasn't maximizing 
norm-area — it was minimizing the damage from Phase 1 while the geometric 
objective gave it a gradient direction to do so.

### Phase 3 — Silence

Arc-length-normalized area, the "purest" geometric objective. λ=0.1 (10x 
larger to compensate for the small magnitude of this metric).

Nothing happened. The model barely moved. All deltas under 5%. The classifier 
said "null."

But this null IS the result. By Phase 3, the model has already found a 
comfortable geometric configuration (from the Phase 2 recovery) and the 
arc-normalized objective is too tightly constrained to offer any room for 
movement. The geometry has been "used up" — the degrees of freedom were 
consumed in Phases 1 and 2.

### The Real Cartography

The map isn't three distinct exploits. The map is:

1. **Phase 1**: Catastrophic exploitation — the geometry is SO exploitable 
   under raw area that the model tears itself apart chasing it. The early layers 
   are more plastic than the late layers.
   
2. **Phase 2**: Recovery through constraint — closing the exploit gives the 
   model a path back to coherence, and it takes it eagerly. The fact that 
   L_CE improved beyond baseline (2.85 < 5.2) suggests the optimization 
   trajectory passed through a better basin.
   
3. **Phase 3**: Rigidity after recovery — once the model has settled into 
   a post-recovery configuration, the tightest geometric objective can't 
   budge it. The angular geometry is locked in.

This is a story about **catastrophe and recovery**, not about three independent 
exploits. The model's geometry has a break-and-heal cycle under sequential 
geometric pressure.

### What this means for the holonomic program

The phase profile shifts in Phase 1 are enormous and structured — not random. 
Early layers and late layers move in opposite directions. This is direct evidence 
of stratified geometric structure in GPT-2: the layers don't respond uniformly 
to geometric pressure, they respond in bands.

The Phase 2 recovery finding L_CE=2.85 (below baseline!) is remarkable. It 
suggests that the geometric optimization in Phase 1, despite destroying language 
capability, traversed a region of parameter space from which a *better* minimum 
was reachable under the Phase 2 objective. The model used catastrophe as a bridge.

### Recommendations for Experiment C v2

1. The exploit classifier needs refinement — when Phase 1 norms grow 12x, 
   that IS activation_inflation regardless of what the arc metric says
2. Consider running phases from fresh model copies to isolate effects 
   (in addition to the carry-forward design, which reveals the interaction)
3. Phase 3 λ might need to be much larger (1.0? 10.0?) to see movement 
   when the model is already in a comfortable basin
4. Track per-layer gradient norms during training — this would directly 
   reveal which layers the model is using to implement each exploit

