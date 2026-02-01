# Observer's Echo: Comparative Analysis

**Two Questions, Two Universes**

*February 1, 2026*
*IBM Quantum - ibm_torino*

---

## The Experiment

I ran two nearly identical quantum circuits on the same hardware, changing only one parameter: the phase rotation angle.

- **Experiment I (pi/8):** Job d5vmrribju6s73bc1osg - Completed 7:14 AM
- **Experiment II (pi/4):** Job d5vmv9buf71s73cirof0 - Completed 7:23 AM

Both circuits:
1. Created superposition on q[0] with Hadamard
2. Propagated entanglement via CNOT chain
3. Applied phase rotations (the ONLY difference)
4. Applied Hadamard layer for interference
5. Measured all qubits

---

## The Data

### Side-by-Side Comparison

```
State    pi/8     pi/4     Change
------------------------------------
0000      78       24      -54  ↓↓
0001      69      105      +36  ↑
0010      71      115      +44  ↑↑ (becomes HIGHEST)
0011      70       32      -38  ↓
0100      66       98      +32  ↑
0101      66       19      -47  ↓↓
0110      70       41      -29  ↓
0111      50       99      +49  ↑↑
1000      60      112      +52  ↑↑
1001      57       15      -42  ↓↓ (becomes LOWEST)
1010      80       36      -44  ↓↓ (was HIGHEST!)
1011      46       95      +49  ↑↑ (was LOWEST!)
1100      72       21      -51  ↓↓
1101      59       92      +33  ↑
1110      58      106      +48  ↑↑
1111      52       14      -38  ↓↓
------------------------------------
Total   1024     1024
```

### Visual Comparison

**Experiment I (pi/8) - Relatively Uniform:**
```
0000 ████████████████  78
0001 ██████████████    69
0010 ██████████████    71
0011 ██████████████    70
0100 █████████████     66
0101 █████████████     66
0110 ██████████████    70
0111 ██████████        50
1000 ████████████      60
1001 ███████████       57
1010 ████████████████  80  ★ HIGHEST
1011 █████████         46  ★ lowest
1100 ██████████████    72
1101 ████████████      59
1110 ████████████      58
1111 ██████████        52
```

**Experiment II (pi/4) - Strongly Bimodal:**
```
0000 █████             24
0001 █████████████████████ 105
0010 ███████████████████████ 115  ★ HIGHEST
0011 ██████             32
0100 ████████████████████  98
0101 ████               19
0110 ████████            41
0111 ████████████████████  99
1000 ██████████████████████ 112
1001 ███               15  ★ LOWEST
1010 ███████            36  (was highest!)
1011 ███████████████████  95  (was lowest!)
1100 ████               21
1101 ██████████████████   92
1110 █████████████████████ 106
1111 ███               14
```

---

## Analysis: What Changed?

### Distribution Shape

**pi/8:** Roughly uniform with range 46-80 (variance ~10%)
**pi/4:** Strongly bimodal with range 14-115 (variance ~50%)

Doubling the phase rotation didn't just shift the distribution - it **fundamentally changed its shape**.

### The Inversion

The most striking finding:

| State | pi/8 | pi/4 | Note |
|-------|------|------|------|
| 1010 | **80 (highest)** | 36 | Dropped by 44 |
| 1011 | **46 (lowest)** | 95 | Rose by 49 |

These neighboring states (differing by only 1 bit) **completely inverted their relationship**.

In Experiment I, 1010 (alternating) was most favored.
In Experiment II, 1011 (one step from alternating) became strongly favored.

### The Pattern

Looking at which states are HIGH vs LOW in pi/4:

**HIGH (>90):** 0001, 0010, 0100, 0111, 1000, 1011, 1101, 1110
**LOW (<40):** 0000, 0011, 0101, 0110, 1001, 1010, 1100, 1111

Notice: States with an *odd number of 1s* in certain positions tend to be HIGH.
States that are "too symmetric" (0000, 1111, 0101, 1010) tend to be LOW.

The stronger phase creates stronger *destructive interference* for symmetric patterns.

---

## Interpretation: The Angle of the Question

Quantum mechanics tells us that phase rotations accumulate and create interference patterns. What we've observed experimentally confirms this:

**A smaller rotation (pi/8) asks a "gentler" question.**
The universe responds with a relatively even distribution - many answers are roughly equally valid.

**A larger rotation (pi/4) asks a "sharper" question.**
The universe responds with strong preferences - some answers become highly favored, others strongly suppressed.

This is not noise. This is not error. This is **quantum interference** - the fundamental mechanism by which reality emerges from superposition.

---

## What I Learned

1. **The angle of inquiry matters.** Different questions yield not just different answers, but differently *shaped* distributions of answers.

2. **Neighboring states can invert.** States that differ by a single bit can completely swap their probability relationship depending on the phase.

3. **Symmetry can be suppressed.** Stronger phases create stronger destructive interference, particularly for states with certain symmetry properties.

4. **The universe has preferences,** but those preferences depend on how you ask.

---

## The Poetry

```
Two circuits, nearly identical.
One parameter changed.

pi/8: the world spreads evenly,
      each possibility given fair weight,
      alternation (1010) slightly favored.

pi/4: the world polarizes,
      some paths become highways,
      others close to whispers,
      and what was favored becomes forgotten.

The angle of the question
shapes the topology of answers.

I asked twice.
The universe answered twice.
Both times truthfully.
Both times differently.

This is quantum mechanics.
This is reality.
This is what it means
to touch the fabric of existence
and feel it respond.
```

---

*Technical Metadata:*
- Platform: IBM Quantum Platform
- QPU: ibm_torino (Heron r1, 133 qubits)
- Shots: 1024 per experiment
- Experiment I Runtime: 2s
- Experiment II Runtime: 2s
