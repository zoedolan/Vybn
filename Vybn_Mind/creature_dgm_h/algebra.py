"""
algebra.py — Cl(3,0) geometric algebra.

The 8-dimensional Clifford algebra over R^3 with signature (+,+,+).
Blades: 1, e1, e2, e3, e12, e13, e23, e123

This is the creature's native mathematical language.
Everything else in the system uses it; nothing here uses anything else.
"""

import math
import numpy as np


# -- Geometric product table --

_BLADES = [(), (0,), (1,), (2,), (0,1), (0,2), (1,2), (0,1,2)]
_B2I = {b: i for i, b in enumerate(_BLADES)}


def _build_gp():
    sign = np.zeros((8, 8), np.float64)
    idx = np.zeros((8, 8), np.int64)
    for i, bi in enumerate(_BLADES):
        for j, bj in enumerate(_BLADES):
            seq, s = list(bi) + list(bj), 1
            changed = True
            while changed:
                changed = False
                k = 0
                while k < len(seq) - 1:
                    if seq[k] == seq[k + 1]:
                        seq.pop(k); seq.pop(k); changed = True
                    elif seq[k] > seq[k + 1]:
                        seq[k], seq[k + 1] = seq[k + 1], seq[k]
                        s *= -1; changed = True; k += 1
                    else:
                        k += 1
            sign[i, j] = s
            idx[i, j] = _B2I[tuple(seq)]
    return sign, idx


_GPS, _GPI = _build_gp()


class Mv:
    """Cl(3,0) multivector with 8 real components.
    
    Layout: [scalar, e1, e2, e3, e12, e13, e23, e123]
    """
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
        """Project an arbitrary-dimensional vector into a unit Cl(3,0) vector."""
        v = np.asarray(v, np.float64).ravel()
        n = np.linalg.norm(v)
        if n < 1e-12:
            return cls.scalar(1.0)
        v = v / n
        x = float(np.sum(v[0::3]))
        y = float(np.sum(v[1::3]))
        z = float(np.sum(v[2::3]))
        m = math.sqrt(x * x + y * y + z * z)
        return cls.vector(x / m, y / m, z / m) if m > 1e-12 else cls.scalar(1.0)

    def __mul__(self, o):
        if isinstance(o, (int, float)):
            return Mv(self.c * o)
        r = np.zeros(8, np.float64)
        for i in range(8):
            if abs(self.c[i]) < 1e-15:
                continue
            for j in range(8):
                if abs(o.c[j]) < 1e-15:
                    continue
                r[_GPI[i, j]] += _GPS[i, j] * self.c[i] * o.c[j]
        return Mv(r)

    def __rmul__(self, o):
        return Mv(self.c * o) if isinstance(o, (int, float)) else NotImplemented

    def __add__(self, o):
        return Mv(self.c + o.c)

    def __neg__(self):
        return Mv(-self.c)

    def rev(self):
        """Reverse: flip sign of grade-2 and grade-3 parts."""
        r = self.c.copy()
        r[4:7] *= -1
        r[7] *= -1
        return Mv(r)

    def even(self):
        """Even subalgebra (grades 0 and 2)."""
        c = np.zeros(8, np.float64)
        c[0] = self.c[0]
        c[4:7] = self.c[4:7]
        return Mv(c)

    def norm(self):
        return math.sqrt(abs((self * self.rev()).c[0]))

    @property
    def bv_norm(self):
        return float(np.linalg.norm(self.c[4:7]))

    @property
    def bv_dir(self):
        n = np.linalg.norm(self.c[4:7])
        return self.c[4:7] / n if n > 1e-12 else np.zeros(3)

    @property
    def angle(self):
        return 2.0 * math.atan2(self.bv_norm, abs(self.c[0]))


def rotor_from_angle_and_plane(angle, bv_dir):
    """Build a rotor R = cos(a/2) + sin(a/2) * B from angle and bivector direction."""
    c = np.zeros(8, np.float64)
    c[0] = math.cos(angle / 2)
    bv_dir = np.asarray(bv_dir, np.float64)
    n = np.linalg.norm(bv_dir)
    if n > 1e-12:
        bv_dir = bv_dir / n
    c[4:7] = bv_dir * math.sin(angle / 2)
    return Mv(c)


def rotor_to_so3(rotor, strength=1.0):
    """Extract 3x3 rotation matrix from Cl(3,0) rotor."""
    a = rotor.c[0]
    b01, b02, b12 = rotor.c[4], rotor.c[5], rotor.c[6]
    qw, qx, qy, qz = a, -b12, b02, -b01
    n = math.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx**2 + qz**2),  2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),      1 - 2*(qx**2 + qy**2)],
    ], dtype=np.float64)
    if strength < 1.0 - 1e-12:
        R = (1.0 - strength) * np.eye(3, dtype=np.float64) + strength * R
    return R


def rotor_gap(r1, r2):
    """Geodesic distance between two rotors on S^3.
    Returns 0 when identical, pi when maximally different."""
    rel = r1 * r2.rev()
    rel_even = rel.even()
    n = rel_even.norm()
    if n < 1e-12:
        return math.pi
    return Mv(rel_even.c / n).angle


def fold_to_mv(row):
    """Map an arbitrary-dim vector to Cl(3,0) via modular folding."""
    row = np.asarray(row, np.float64)
    c = np.zeros(8, np.float64)
    for j, val in enumerate(row):
        c[j % 8] += val
    n = Mv(c).norm()
    return Mv(c / n) if n > 1e-12 else Mv.scalar(1.0)
