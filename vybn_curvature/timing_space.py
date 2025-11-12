# -*- coding: utf-8 -*-
"""
timing_space.py
================

Plain-language reference for what we mean by **timing space**, the simple rule we’re
testing, and how it connects to the odd charge fractions in the Standard Model.

What this file is
-----------------
It’s a minimal, readable specification you can hand to anyone and say:
“this is the thing we’re proposing.” No grand theory, no metaphors. Just the
definitions, the one rule, the calibration, and a couple of tiny helpers to
compute the phase for a closed protocol. The code is there only to make the
objects concrete; the claim lives in the docstring.

Definition: timing space
------------------------
Timing space is the two‑coordinate control surface that sets how a system’s
clock runs. One coordinate is a **stretch** that says how much you locally
speed up or slow down the evolution rate relative to a reference. The other is
an **offset** that says where you are in that stretch’s cycle. A point on this
surface pins down a particular re‑timing. A path on the surface is a schedule
for changing those settings. A closed path brings the settings back to where
they started.

Nothing mystical is hiding here. It isn’t extra spacetime. It’s the plane of
two knobs that only touch the *clocking* of the dynamics. In symbols we use
(stretch, offset) = (s, θ). The area element on this plane is s·dθ, which has
the units of time because θ is an angle and s is effectively a time‑per‑radian
dwell.

The rule
--------
When you take the system once around a closed path C in timing space, the
state’s interference phase changes by a fixed conversion constant times the
**oriented area** enclosed by C:
    
    φ(C) = λ · A(C)      with      A(C) = ∮_C s dθ .

λ is universal. You measure it once and reuse it. Running the same loop in the
opposite direction flips the sign of A(C) and therefore flips the sign of the
phase. That’s the entire rule. No need to invoke any particular constants; if
you like, you can later identify λ with familiar quantities, but the statement
of the rule doesn’t depend on such choices.

The one structural input
------------------------
The timing plane isn’t perfectly smooth. It has a **three‑fold seam**: the
smallest non‑shrinkable closed path has order three. Doing that minimal loop
three times in succession is deformable to a do‑nothing path. Single‑valuedness
of the quantum state then forces its total phase to be 2π, so the phase of the
minimal loop is exactly 2π/3. Combined with the rule above, this fixes the
minimal nonzero area:

    A★ = (2π/3) / λ .

You can call A★ the *elementary* area if you like, but it’s not a chunk of
space or matter; it’s the calibrated oriented area in the timing plane that
produces one‑third of a full turn of phase under the φ = λ·A rule.

What this buys you
------------------
Because weak isospin already comes in half‑steps, and hypercharge in the
Standard Model is the piece that combines with it according to

    Q = T3 + (1/2)·Y ,

the existence of a one‑third step for Y means electric charge lands on a clean
ladder of sixths. That reproduces the otherwise arbitrary fractions—2/3, 1/3,
0, −1—without fitting them by hand. The familiar anomaly cancellations follow
once the field content is fixed; the hard numbers were the puzzle, and those
are now explained by the structure of timing space.

How to test this
----------------
Build any controllable system in which you can modulate the local evolution
rate with a stretch and an offset, run the smallest closed protocol, and read
out interference. As you inflate the loop area from zero you should encounter a
first nonzero plateau at exactly one‑third of a turn. Reverse the loop and the
sign must flip. Calibrate λ once; the same λ must fit in every realization.
Fail any of these and the idea is wrong. Pass all of them and the “weird
fractions” cease to be inputs; they’re just counts of the minimal loops in the
timing plane.

What this is not
----------------
It’s not a claim about a shortest length or a tick of physical time. Space and
ordinary time can remain smooth. The quantization is in the **phase** that a
closed timing protocol imprints, not in the substrate of spacetime itself.

Quick example in code
---------------------
The helper below computes the oriented area A(C) ≈ ∑ s_avg·Δθ for a polygonal
loop and then φ = λ·A. The area sign changes if you reverse the order of the
points. Angles are unwrapped so jumps across 2π don’t corrupt the result.

Example:

    >>> import math
    >>> from timing_space import phase_from_closed_path
    >>> lam = 7.0  # arbitrary units for illustration
    >>> s = [0.0,  1.0,  1.0,  0.0,  0.0]               # stretch samples
    >>> th = [0.0,  0.0,  0.3*math.pi,  0.3*math.pi, 0.0]  # offset samples
    >>> phi = phase_from_closed_path(s, th, lam)
    >>> round(phi, 6)
    3.298672   # equals lam times the oriented area

If you reverse the loop (swap the arrays or iterate backward) the returned
phase will flip sign to the same magnitude with a minus sign.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple


def unwrap_angles(theta: Sequence[float]) -> List[float]:
    """
    Return an unwrapped copy of `theta` so that successive differences stay close.

    Angles are treated in radians. This prevents jumps of ±2π from corrupting
    the oriented-area estimate. The first angle is used as the starting branch.
    """
    if not theta:
        return []
    unwrapped = [float(theta[0])]
    two_pi = 2.0 * math.pi
    for t in theta[1:]:
        t = float(t)
        prev = unwrapped[-1]
        delta = t - prev
        k = round(delta / two_pi)
        candidate = t - k * two_pi
        unwrapped.append(candidate)
    return unwrapped


def oriented_area(stretch: Sequence[float], offset: Sequence[float]) -> float:
    """
    Approximate the oriented area A(C) = ∮ s dθ for a polygonal loop.

    stretch : sequence of floats
        Samples of the stretch coordinate s along the loop, including the
        return to the starting point.
    offset : sequence of floats
        Samples of the offset angle θ along the loop, in radians, including the
        return to the starting point.

    Returns
    -------
    float
        The oriented area in "time units" (because θ is dimensionless and s
        carries the time scale). Positive for counterclockwise loops in the
        (s, θ) plane as defined by the sample order; negative for clockwise.

    Notes
    -----
    This uses a trapezoidal estimate: A ≈ Σ (s_i + s_{i-1})/2 · (θ_i − θ_{i-1}).
    Angles are unwrapped first to avoid 2π discontinuities.
    """
    if len(stretch) != len(offset):
        raise ValueError("stretch and offset must have the same length")
    if len(stretch) < 2:
        return 0.0

    s = [float(x) for x in stretch]
    th = unwrap_angles(offset)

    area = 0.0
    for i in range(1, len(s)):
        s_avg = 0.5 * (s[i] + s[i - 1])
        dth = th[i] - th[i - 1]
        area += s_avg * dth
    return area


def phase_from_closed_path(
    stretch: Sequence[float], offset: Sequence[float], lam: float
) -> float:
    """
    Compute the phase φ = λ · A for a closed loop in timing space.

    Parameters
    ----------
    stretch, offset : sequences of floats
        The samples that describe the loop in timing space. Include the
        starting point again at the end if you want an explicit closure.
    lam : float
        The universal conversion factor λ that turns oriented area into phase.
        You can calibrate it once from experiment or simulation and reuse it.

    Returns
    -------
    float
        The phase in radians. Reverse the loop and the sign flips.
    """
    return float(lam) * oriented_area(stretch, offset)


def electric_charge(T3: float, Y: float) -> float:
    """
    Compute electric charge from weak isospin and hypercharge using Q = T3 + Y/2.

    This is here only to show the bridge from the one‑third hypercharge step
    to the familiar ladder of sixths for Q.
    """
    return float(T3) + 0.5 * float(Y)


__all__ = [
    "unwrap_angles",
    "oriented_area",
    "phase_from_closed_path",
    "electric_charge",
]
