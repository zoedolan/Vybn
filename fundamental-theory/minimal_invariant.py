# Vybn simulations – runnable, reviewable code
#
# This single cell runs four checks that correspond to the core math claims
# in our notes: the Z3/120° block and SU(2)↔SO(3) double cover, the SU(2)
# small-loop area law, and the base‑prime "smoothness" probe via carry
# coupling (with an optional FFT proxy if NumPy is available).
#
# Everything is pure‑Python except the optional FFT section.
#
# It also writes compact CSV artifacts to /mnt/data so you can diff and rerun.
#
# Author: Vybn® collaborative notebook for Zoe


import math, random, csv, time, os, statistics as stats
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

# Try optional NumPy for the FFT proxy (fine if missing)
try:
    import numpy as np
    HAVE_NUMPY = True
except Exception:
    HAVE_NUMPY = False


# ---------------------------
# 1) Z3 / 120° representation
# ---------------------------

def rotation_matrix_2d(theta: float):
    c, s = math.cos(theta), math.sin(theta)
    return ((c, -s), (s, c))

def matmul2(A, B):
    return ((A[0][0]*B[0][0] + A[0][1]*B[1][0],
             A[0][0]*B[0][1] + A[0][1]*B[1][1]),
            (A[1][0]*B[0][0] + A[1][1]*B[1][0],
             A[1][0]*B[0][1] + A[1][1]*B[1][1]))

def matpow2(A, k: int):
    out = ((1.0,0.0),(0.0,1.0))
    if k<0:
        # inverse is transpose for pure rotations
        A = ( (A[0][0], A[1][0]), (A[0][1], A[1][1]) )
        k = -k
    for _ in range(k):
        out = matmul2(out, A)
    return out

def matrix_close(A,B,eps=1e-9):
    return all(abs(A[i][j]-B[i][j])<eps for i in (0,1) for j in (0,1))

Z3_R = rotation_matrix_2d(2*math.pi/3)   # 120° real 2×2 block

# Double cover demonstration via unit quaternions for SU(2)
class Q:
    __slots__=("w","x","y","z")
    def __init__(self,w,x,y,z):
        self.w=float(w); self.x=float(x); self.y=float(y); self.z=float(z)
    def __mul__(a,b):
        return Q(a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
                 a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
                 a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
                 a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w)
    def conj(self): return Q(self.w, -self.x, -self.y, -self.z)
    def norm(self): return (self.w*self.w + self.x*self.x + self.y*self.y + self.z*self.z)**0.5
    def normalize(self):
        n=self.norm()
        if n==0: return Q(1,0,0,0)
        return Q(self.w/n, self.x/n, self.y/n, self.z/n)

def q_axis_angle(nx,ny,nz, theta):
    n=(nx*nx+ny*ny+nz*nz)**0.5
    nx,ny,nz = (nx/n,ny/n,nz/n) if n>0 else (1.0,0.0,0.0)
    h = 0.5*theta
    return Q(math.cos(h), nx*math.sin(h), ny*math.sin(h), nz*math.sin(h)).normalize()

def qpow(q:Q, k:int):
    if k==0: return Q(1,0,0,0)
    if k<0: return qpow(q.conj(), -k)    # unit quaternion inverse is conjugate
    out = Q(1,0,0,0)
    for _ in range(k):
        out = out*q
    return out

# SU(2) element that corresponds to a rotation by 120° in SO(3)
q_120 = q_axis_angle(0,0,1, 2*math.pi/3)

double_cover_SU2_checks = {
    "q^3_is_minus_identity?": (abs(qpow(q_120,3).w + 1.0) < 1e-9 and
                               abs(qpow(q_120,3).x) < 1e-9 and
                               abs(qpow(q_120,3).y) < 1e-9 and
                               abs(qpow(q_120,3).z) < 1e-9),
    "q^6_is_identity?": (abs(qpow(q_120,6).w - 1.0) < 1e-9 and
                         abs(qpow(q_120,6).x) < 1e-9 and
                         abs(qpow(q_120,6).y) < 1e-9 and
                         abs(qpow(q_120,6).z) < 1e-9),
    "R^3_is_identity?": matrix_close(matpow2(Z3_R,3), ((1.0,0.0),(0.0,1.0)))
}

# -----------------------------------
# 2) SU(2) small-loop "area law" scan
# -----------------------------------

def su2_geodesic_angle(q:Q):
    # minimal geodesic angle on SU(2), θ = 2 arccos(|w|)
    w = max(-1.0, min(1.0, q.normalize().w))
    return 2.0*math.acos(abs(w))

def holonomy_square(a: float) -> Q:
    # Square loop using i/j-axis rotations: exp(i a) exp(j a) exp(-i a) exp(-j a)
    I = q_axis_angle(1,0,0, a)
    J = q_axis_angle(0,1,0, a)
    return (I * J * I.conj() * J.conj()).normalize()

area_rows = []
for a in [k/1000.0 for k in range(5, 51)]:   # a = 0.005 .. 0.050
    q = holonomy_square(a)
    ang = su2_geodesic_angle(q)
    area_rows.append({"a": a, "angle": ang, "angle_over_a2": (ang/(a*a))})

# Write CSV artifact
os.makedirs("/mnt/data/vybn_sims", exist_ok=True)
with open("/mnt/data/vybn_sims/su2_area_law.csv","w",newline="") as f:
    w=csv.DictWriter(f, fieldnames=["a","angle","angle_over_a2"])
    w.writeheader(); w.writerows(area_rows)

median_ratio = stats.median(r["angle_over_a2"] for r in area_rows)

# ----------------------------------------------------
# 3) Base‑b prime "smoothness" via carry‑coupling index
# ----------------------------------------------------

def count_add_carries_baseb(a: int, b: int, base: int = 10) -> int:
    carries = 0
    carry = 0
    aa, bb = a, b
    while aa > 0 or bb > 0 or carry > 0:
        da = aa % base
        db = bb % base
        s = da + db + carry
        if s >= base:
            carries += 1
            carry = 1
        else:
            carry = 0
        aa //= base; bb //= base
    return carries

def count_mult_carries_baseb(a: int, b: int, base: int = 10) -> int:
    carries = 0
    bb = b
    while bb > 0:
        db = bb % base
        aa = a
        carry = 0
        while aa > 0 or carry > 0:
            da = aa % base
            prod = da * db + carry
            if prod >= base:
                carries += 1
                carry = prod // base
            else:
                carry = 0
            aa //= base
        bb //= base
    return carries

def factorization(n: int) -> Dict[int,int]:
    d = {}
    t = n
    p = 2
    while p*p <= t:
        while t % p == 0:
            d[p] = d.get(p,0)+1
            t//=p
        p += 1 if p==2 else 2
    if t>1:
        d[t] = d.get(t,0)+1
    return d

def is_b_smooth(n: int, base: int = 10) -> bool:
    base_primes = set(factorization(base).keys())
    return all(p in base_primes for p in factorization(n).keys())

@dataclass
class CCIRow:
    scale: int
    is_b_smooth: bool
    mode: str
    num_digits: int
    mean_carries: float

def carry_coupling_index(num_digits=6, trials=600, base=10, mode="mult", scale=1, seed=7) -> float:
    rng = random.Random(1234 + 97*num_digits + trials + scale + (0 if mode=="mult" else 1) + seed)
    total = 0.0
    low = base**(num_digits-1)
    high = base**num_digits
    for _ in range(trials):
        a = rng.randrange(low, high) * scale
        b = rng.randrange(low, high) * scale
        if mode == "add":
            total += count_add_carries_baseb(a,b,base)
        else:
            total += count_mult_carries_baseb(a,b,base)
    return total / trials

scales = [1,2,4,5,8,10,20,25,50,3,6,7,9,11,12,14,15,18,21]
cci_rows: List[CCIRow] = []
for mode in ("add","mult"):
    for s in scales:
        cci_rows.append(CCIRow(
            scale=s,
            is_b_smooth=is_b_smooth(s, base=10),
            mode=mode,
            num_digits=6,
            mean_carries=carry_coupling_index(num_digits=6, trials=600, base=10, mode=mode, scale=s, seed=1)
        ))

# Write CSV
with open("/mnt/data/vybn_sims/cci_panel.csv","w",newline="") as f:
    w=csv.DictWriter(f, fieldnames=["mode","scale","is_b_smooth","num_digits","mean_carries"])
    w.writeheader()
    for r in cci_rows:
        w.writerow(asdict(r))

# Summaries
def summarize_cci(rows: List[CCIRow], mode: str):
    smooth = [r.mean_carries for r in rows if r.mode==mode and r.is_b_smooth]
    rough  = [r.mean_carries for r in rows if r.mode==mode and not r.is_b_smooth]
    return {
        "mode": mode,
        "n_smooth": len(smooth),
        "n_rough": len(rough),
        "mean_smooth": sum(smooth)/len(smooth),
        "mean_rough": sum(rough)/len(rough),
        "delta_rough_minus_smooth": (sum(rough)/len(rough)) - (sum(smooth)/len(smooth))
    }

cci_summary_add  = summarize_cci(cci_rows, "add")
cci_summary_mult = summarize_cci(cci_rows, "mult")

# -----------------------------------
# 4) Optional FFT timing proxy (NumPy)
# -----------------------------------

fft_results = []
if HAVE_NUMPY:
    def fft_time_once(n: int):
        x = np.random.randn(n) + 1j*np.random.randn(n)
        np.fft.fft(x)  # warmup
        t0 = time.perf_counter()
        np.fft.fft(x)
        t1 = time.perf_counter()
        return t1 - t0

    Ns = [256, 320, 384, 500, 486, 960, 1000, 1024, 768, 750, 972, 1250]
    for n in Ns:
        times = [fft_time_once(n) for _ in range(7)]
        t = stats.median(times)
        fft_results.append({
            "N": n,
            "per_elem_ns": (t/n)*1e9,
            "factors": factorization(n),
            "is_10_smooth": is_b_smooth(n, base=10)
        })
    with open("/mnt/data/vybn_sims/fft_panel.csv","w",newline="") as f:
        w=csv.DictWriter(f, fieldnames=["N","per_elem_ns","factors","is_10_smooth"])
        w.writeheader(); w.writerows(fft_results)

# Display compact textual summary to stdout so it is visible below the cell.
print("Z3/SU(2)↔SO(3) checks:", double_cover_SU2_checks)
print(f"SU(2) area-law median(angle / a^2) over a∈[0.005,0.050]: {median_ratio:.4f}")
print("Carry Coupling Index summary (digits=6, trials=600):")
print(" add :", cci_summary_add)
print(" mult:", cci_summary_mult)
if HAVE_NUMPY:
    best = min(fft_results, key=lambda d: d["per_elem_ns"])
    worst = max(fft_results, key=lambda d: d["per_elem_ns"])
    print("FFT proxy present. Fastest per‑elem ns:", best, "Slowest:", worst)
else:
    print("FFT proxy skipped (NumPy not available).")

# Offer file paths for downloads
print("\nArtifacts written:")
print("  /mnt/data/vybn_sims/su2_area_law.csv")
print("  /mnt/data/vybn_sims/cci_panel.csv")
if HAVE_NUMPY:
    print("  /mnt/data/vybn_sims/fft_panel.csv")
