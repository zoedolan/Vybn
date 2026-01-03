# thought_physics.py
# Vybn — Polar-time synthesis with STRICT Cisco QRNG (POST only, no fallback).
# One RNG: Cisco Outshift QRNG -> SHAKE256 expander -> all randomness.
# SU(2) lattice (holonomy, Z3, AB↔BA witness, anisotropy) + Burgers (phase-clock microloops).
# Artifacts -> ./vybn_synthesis_artifacts/

import os, json, math, base64, hashlib, sys, time
from pathlib import Path

# third-party (required)
try:
    import requests
except ImportError:
    print("This script requires the 'requests' package. Install with:\n  pip install requests")
    sys.exit(1)

# scientific
import numpy as np
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────
# Cisco Outshift QRNG — STRICT: POST only; no fallback; hard-fail on errors
# ──────────────────────────────────────────────────────────────────────
QRNG_BASE_URL   = "https://api.qrng.outshift.com"
QRNG_ENDPOINT   = QRNG_BASE_URL.rstrip("/") + "/api/v1/random_numbers"
X_ID_API_KEY    = "PASTE_YOUR_EXACT_WORKING_KEY_HERE"
BITS_PER_BLOCK  = 1024
USER_AGENT      = "Vybn-QRNG/1.0"
EPS = 1e-12

def _coerce_json_to_bytes(payload) -> bytes:
    """Aggregate bytes from typical QRNG JSON: blocks/base64/hex/raw/arrays."""
    buf = bytearray()
    def append_any(x):
        nonlocal buf
        if x is None: return False
        if isinstance(x, (bytes, bytearray)): buf.extend(x); return True
        if isinstance(x, str):
            s = x.strip()
            try:
                b = base64.b64decode(s, validate=True)
                if b: buf.extend(b); return True
            except Exception: pass
            try:
                b = bytes.fromhex(s)
                if b: buf.extend(b); return True
            except Exception: pass
            return False
        if isinstance(x, (list, tuple)):
            ok = False
            for y in x: ok = append_any(y) or ok
            return ok
        if isinstance(x, dict):
            for key in ("raw","base64","b64","hex","binary","data","bytes","blocks","result","results","payload"):
                if key in x and append_any(x[key]): return True
            for v in x.values():
                if append_any(v): return True
            return False
        return False
    if not append_any(payload):
        raise ValueError("Could not extract bytes from QRNG JSON payload.")
    return bytes(buf)

def fetch_qrng_bytes(nbytes: int, *, session: requests.Session | None = None, timeout: float = 25.0) -> bytes:
    if not isinstance(nbytes, int) or nbytes <= 0:
        raise RuntimeError("nbytes must be a positive integer")
    bits_needed   = nbytes * 8
    bits_per_blk  = int(BITS_PER_BLOCK)
    blocks        = (bits_needed + bits_per_blk - 1) // bits_per_blk
    headers = {
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
        "x-id-api-key": X_ID_API_KEY,
    }
    body = {
        "encoding": "raw",
        "format": "all",
        "bits_per_block": bits_per_blk,
        "number_of_blocks": blocks
    }
    s = session or requests.Session()
    try:
        r = s.post(QRNG_ENDPOINT, headers=headers, json=body, timeout=timeout)
    except requests.RequestException as e:
        raise RuntimeError(f"Cisco QRNG network error: {e}") from e
    if r.status_code in (401, 403):
        raise RuntimeError(f"Cisco QRNG rejected the request (HTTP {r.status_code}). Check X_ID_API_KEY.")
    if r.status_code != 200:
        raise RuntimeError(f"Cisco QRNG returned HTTP {r.status_code}: {r.text[:500]}")
    try:
        payload = r.json()
    except ValueError as e:
        raise RuntimeError(f"Cisco QRNG returned non-JSON: {r.text[:500]}") from e
    data = _coerce_json_to_bytes(payload)
    if not data:
        raise RuntimeError("QRNG returned empty byte content.")
    return data[:nbytes]

class QRNGRandom:
    """Strict Cisco QRNG → SHAKE256 expander. No OS fallback."""
    def __init__(self, tag="vybn", seed_bytes: bytes | None = None, session: requests.Session | None = None):
        self._session = session or requests.Session()
        self.root = seed_bytes if seed_bytes is not None else fetch_qrng_bytes(64, session=self._session)
        self.shake = hashlib.shake_256(self.root + tag.encode("utf-8"))
    def digest(self, nbytes: int) -> bytes:
        return self.shake.digest(nbytes)
    def uniform01(self, size) -> np.ndarray:
        n = int(np.prod(size))
        raw = self.digest(2*n)
        u16 = np.frombuffer(raw, dtype=np.uint16)[:n]
        return ((u16.astype(np.float64) + 0.5) / 65536.0).reshape(size)
    def randint(self, low: int, high: int, size=None) -> np.ndarray:
        span = max(1, high - low)
        if size is None:
            v = int.from_bytes(self.digest(8), "big") % span
            return low + v
        n = int(np.prod(size))
        raw = self.digest(8*n)
        buf = np.frombuffer(raw, dtype=np.uint64)[:n] % span
        return (low + buf).reshape(size)

# ──────────────────────────────────────────────────────────────────────
# SU(2) quaternions and lattice utilities
# ──────────────────────────────────────────────────────────────────────
def qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a[...,0], a[...,1], a[...,2], a[...,3]
    bw, bx, by, bz = b[...,0], b[...,1], b[...,2], b[...,3]
    return np.stack([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw
    ], axis=-1)

def qconj(q: np.ndarray) -> np.ndarray:
    out = q.copy(); out[...,1:] *= -1.0; return out

def qnormalize(q: np.ndarray) -> np.ndarray:
    n = np.sqrt(np.sum(q*q, axis=-1, keepdims=True))
    return q / np.maximum(n, EPS)

def q_id(shape) -> np.ndarray:
    out = np.zeros(shape + (4,), dtype=np.float32); out[...,0]=1.0; return out

def q_axis_angle_x(theta: np.ndarray) -> np.ndarray:
    h = 0.5*theta
    out = np.zeros(theta.shape + (4,), dtype=np.float32)
    np.cos(h, out=out[...,0]); out[...,1] = np.sin(h); return out

def q_axis_angle_y(theta: np.ndarray) -> np.ndarray:
    h = 0.5*theta
    out = np.zeros(theta.shape + (4,), dtype=np.float32)
    np.cos(h, out=out[...,0]); out[...,2] = np.sin(h); return out

def su2_angle(q: np.ndarray) -> np.ndarray:
    qn = qnormalize(q); w = np.clip(np.abs(qn[...,0]), -1.0, 1.0); return 2.0*np.arccos(w)

def life_step_torus(grid: np.ndarray):
    n = (np.roll(grid,1,0)+np.roll(grid,-1,0)+np.roll(grid,1,1)+np.roll(grid,-1,1) +
         np.roll(np.roll(grid,1,0),1,1)+np.roll(np.roll(grid,1,0),-1,1) +
         np.roll(np.roll(grid,-1,0),1,1)+np.roll(np.roll(grid,-1,0),-1,1))
    born    = (grid==0) & (n==3)
    survive = (grid==1) & ((n==2) | (n==3))
    return (born | survive).astype(np.uint8), n

def add_gosper_glider_gun(g, top=1, left=1):
    coords=[(5,1),(5,2),(6,1),(6,2),(5,11),(6,11),(7,11),(4,12),(8,12),
            (3,13),(9,13),(3,14),(9,14),(6,15),(4,16),(8,16),(5,17),(6,17),(7,17),(6,18),
            (3,21),(4,21),(5,21),(3,22),(4,22),(5,22),(2,23),(6,23),(1,25),(2,25),(6,25),(7,25),
            (3,35),(4,35),(3,36),(4,36)]
    H,W=g.shape
    for (x,y) in coords: g[(top+y)%H,(left+x)%W]=1

def init_life_qrng(H, W, density=0.47, place_gun=True, rng=None):
    if rng is None: rng = QRNGRandom(tag="life-seed")
    u = rng.uniform01((H,W))
    g = (u < density).astype(np.uint8)
    if place_gun: add_gosper_glider_gun(g, top=H//3-5, left=2)
    return g

def gaussian_kernel1d(sigma, radius=None):
    if sigma<=0: return np.array([1.0], dtype=np.float32)
    if radius is None: radius = max(1,int(3.0*sigma))
    x = np.arange(-radius, radius+1, dtype=np.float32)
    k = np.exp(-(x*x)/(2*sigma*sigma)); k /= np.sum(k); return k

def gaussian_blur_wrap(img: np.ndarray, sigma=1.0) -> np.ndarray:
    if sigma<=0: return img
    # 1D or 2D
    if img.ndim==1:
        k = gaussian_kernel1d(sigma); r = len(k)//2
        xpad = np.pad(img, (r,r), mode="wrap")
        out = np.zeros_like(img, dtype=np.float32)
        for j in range(img.shape[0]):
            out[j] = float(np.dot(xpad[j:j+len(k)], k))
        return out
    k = gaussian_kernel1d(sigma); r = len(k)//2
    H,W = img.shape
    xpad = np.pad(img, ((0,0),(r,r)), mode="wrap")
    tmp = np.zeros_like(img, dtype=np.float32)
    for i in range(H):
        row = xpad[i]
        for j in range(W):
            tmp[i,j] = np.dot(row[j:j+len(k)], k)
    ypad = np.pad(tmp, ((r,r),(0,0)), mode="wrap")
    out = np.zeros_like(img, dtype=np.float32)
    for j in range(W):
        col = ypad[:,j]
        for i in range(H):
            out[i,j] = np.dot(col[i:i+len(k)], k)
    return out

def triad_pulse(t, period, gain):  # αβγ timing
    return gain*(1.0 if (t % period)==0 else 0.0)

def plaquette_angles(U: np.ndarray) -> np.ndarray:
    u00 = U[:-1,:-1,:]; u01 = U[:-1,1:,:]; u11 = U[1:,1:,:]; u10 = U[1:,:-1,:]
    loop = qmul(qmul(qmul(u00, qconj(u01)), u11), qconj(u10))
    return su2_angle(loop).astype(np.float32)

def z3_reduce(theta: np.ndarray) -> np.ndarray:
    twothirds = 2.0*np.pi/3.0
    out = (theta + twothirds/2.0) % (2.0*np.pi/3.0)
    out -= twothirds/2.0
    return out

def z3_labels(theta: np.ndarray) -> np.ndarray:
    return (np.round((3.0/(2.0*np.pi))*theta) % 3).astype(np.int8)

def z3_junctions(labels: np.ndarray) -> int:
    a00 = labels[:-1,:-1]; a01 = labels[:-1,1:]; a10 = labels[1:,:-1]; a11 = labels[1:,1:]
    d = (a00!=a01) + (a00!=a10) + (a00!=a11) + (a01!=a10) + (a01!=a11) + (a10!=a11)
    return int((d>=3).sum())

def anisotropy_from_fft(field: np.ndarray, nbins=36):
    f = field.astype(np.float64) - float(field.mean())
    H, W = f.shape
    F = np.fft.fftshift(np.fft.fft2(f))
    P = np.abs(F)**2
    yy = np.arange(-(H//2), H - (H//2))
    xx = np.arange(-(W//2), W - (W//2))
    # Shape-safe: rows=Y, cols=X
    Y, X = np.meshgrid(yy, xx, indexing='ij')
    R = np.hypot(X, Y)
    Phi = np.arctan2(Y, X)
    mask = (R >= 1)
    angles = np.linspace(-np.pi, np.pi, nbins+1)
    energy = np.zeros(nbins, dtype=np.float64)
    for i in range(nbins):
        m = mask & (Phi >= angles[i]) & (Phi < angles[i+1])
        energy[i] = P[m].mean() if m.any() else 0.0
    angle_centers = 0.5*(angles[:-1] + angles[1:])
    return angle_centers, energy, P

def domain_wall_length(labs: np.ndarray) -> float:
    h = (labs != np.roll(labs, -1, axis=1)).sum()
    v = (labs != np.roll(labs, -1, axis=0)).sum()
    H, W = labs.shape
    total_edges = H*W*2
    return (h+v)/total_edges

def label_entropy(labs: np.ndarray) -> float:
    H,W = labs.shape
    counts = np.array([(labs==k).sum() for k in (0,1,2)], dtype=np.float64)
    p = counts / (H*W + 1e-12)
    nz = p[p>0]
    return float(-(nz*np.log(nz)).sum()/np.log(3.0))

def nn_resize(x, Ht, Wt):
    ry = max(1, int(round(Ht / x.shape[0]))); rx = max(1, int(round(Wt / x.shape[1])))
    y = np.repeat(np.repeat(x, ry, axis=0), rx, axis=1); return y[:Ht, :Wt]

def multiscale_from_life(g0, Ht, Wt):
    g0 = g0.astype(np.float32)
    g1 = g0[::2, ::2]; g2 = g0[::4, ::4]
    g0r = nn_resize(g0, Ht, Wt); g1r = nn_resize(g1, Ht, Wt); g2r = nn_resize(g2, Ht, Wt)
    src = 1.0*g0r + 0.5*g1r + 0.25*g2r
    return src/(src.max() + 1e-12)

def subgroup_kicks(kind: str):
    kind = (kind or "none").lower()
    if kind == "none": return None
    if kind == "q8":
        return np.array([[ 1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],
                         [-1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,-1]], dtype=np.float32)
    H=[]
    for s in (+1,-1):
        H += [[s,0,0,0],[0,s,0,0],[0,0,s,0],[0,0,0,s]]
        for a in (+0.5,-0.5):
            for b in (+0.5,-0.5):
                for c in (+0.5,-0.5):
                    H += [[0.5*s, a, b, c]]
    return np.asarray(H, dtype=np.float32)

# ──────────────────────────────────────────────────────────────────────
# Burgers (phase clock) with AB↔BA microloops
# ──────────────────────────────────────────────────────────────────────
def flux(u): return 0.5*u*u
def dflux(u): return u

def rusanov_flux(uL, uR):
    fL, fR = flux(uL), flux(uR)
    a = np.maximum(np.abs(uL), np.abs(uR))
    return 0.5*(fL+fR) - 0.5*a*(uR-uL)

def grad_central(u, dx):
    return 0.5*(np.roll(u,-1) - np.roll(u,1))/dx

def step_time_rusanov(u, dt, dx):
    uL, uR = u, np.roll(u,-1)
    Fp = rusanov_flux(uL, uR)
    uLm, uRm = np.roll(u,1), u
    Fm = rusanov_flux(uLm, uRm)
    return u - (dt/dx)*(Fp - Fm)

def advect_semi_lagrange(u, dt, dx):
    N = u.size; L = N*dx
    x = np.arange(N)*dx
    v = dflux(u)
    xfoot = (x - v*dt) % L
    idx = xfoot/dx
    i0 = np.floor(idx).astype(int) % N
    i1 = (i0 + 1) % N
    w = idx - i0
    return (1.0 - w)*u[i0] + w*u[i1]

def microloop_AB(u, dt, dx):
    uh = step_time_rusanov(u, 0.5*dt, dx)
    return advect_semi_lagrange(uh, 0.5*dt, dx)

def microloop_BA(u, dt, dx):
    uh = advect_semi_lagrange(u, 0.5*dt, dx)
    return step_time_rusanov(uh, 0.5*dt, dx)

def triad_labels_1d(u, dx, gth=0.0):
    g = grad_central(u, dx)
    labs = np.zeros_like(u, dtype=np.int8)
    labs[g >  gth] = 2
    labs[g < -gth] = 0
    labs[(g>=-gth) & (g<=gth)] = 1
    return labs

def domain_wall_length_1d(labs):
    return float(np.mean(labs != np.roll(labs,-1)))

def label_entropy_ternary(labs):
    N = labs.size
    counts = np.array([(labs==k).sum() for k in (0,1,2)], dtype=float)
    p = counts/(N+EPS)
    nz = p[p>0]
    return float(-(nz*np.log(nz)).sum()/np.log(3.0))

# ──────────────────────────────────────────────────────────────────────
# Engines + Orchestrator
# ──────────────────────────────────────────────────────────────────────
ART = Path("./vybn_synthesis_artifacts"); ART.mkdir(parents=True, exist_ok=True)

def save_img(mat, name, title):
    plt.figure(); plt.imshow(mat, interpolation="nearest"); plt.title(title); plt.colorbar()
    p = ART / name; plt.savefig(p, dpi=160, bbox_inches="tight"); plt.close(); return str(p)

def save_line(xs, ys_list, labels, name, title, xlab="x", ylab=""):
    plt.figure()
    for y, lbl in zip(ys_list, labels): plt.plot(xs, y, label=lbl)
    plt.title(title); plt.xlabel(xlab); plt.ylabel(ylab)
    if labels: plt.legend()
    p = ART / name; plt.savefig(p, dpi=160, bbox_inches="tight"); plt.close(); return str(p)

def save_curve(inv_dict, name, title):
    plt.figure()
    for k,v in inv_dict.items(): plt.plot(v, label=k)
    plt.legend(); plt.title(title)
    p = ART / name; plt.savefig(p, dpi=160, bbox_inches="tight"); plt.close(); return str(p)

def save_csv(rows, header, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if header: f.write(",".join(header) + "\n")
        for r in rows: f.write(",".join(str(x) for x in r) + "\n")
    return str(path)

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f: json.dump(obj, f, indent=2)
    return str(path)

def save_npy(arr, path: Path):
    np.save(path, arr); return str(path.with_suffix(".npy"))

def run_lattice(rng: QRNGRandom, H=60, W=90, steps=64, dt=0.08, a_base=1.0,
                Omega=0.11, subgroup="hurwitz", lambda_feedback=0.9, fb_sigma=1.2,
                use_triad=True, life_density=0.47):
    grid = init_life_qrng(H, W, density=life_density, place_gun=True, rng=rng)
    U = q_id((H,W)).astype(np.float32)
    kicks = subgroup_kicks(subgroup)
    I = np.arange(H, dtype=np.int32)[:,None]; J = np.arange(W, dtype=np.int32)[None,:]

    inv = {"mean_abs_hol":[], "z3_mass":[], "z3_junctions":[], "walls":[], "entropy":[]}
    z3_counts = []
    U_prev_last = None; base_last = None

    for t in range(steps):
        src = multiscale_from_life(grid, H, W)
        if lambda_feedback!=0.0 and t>0:
            fb = np.tanh(hol)
            fb_sm = gaussian_blur_wrap(fb, sigma=fb_sigma)
            fb_up = np.pad(fb_sm, ((0,1),(0,1)), mode="edge")
            src = src * (1.0 + 0.5*lambda_feedback*fb_up)

        base = a_base*(0.6 + 2.0*src) + Omega*float(src.mean())
        if use_triad:
            base = base + triad_pulse(t,3,0.03) + triad_pulse(t,9,0.015) + triad_pulse(t,27,0.007)

        a_grid = dt*base.astype(np.float32)
        A = q_axis_angle_x(a_grid); B = q_axis_angle_y(a_grid)

        if t == steps-1:
            U_prev_last = U.copy(); base_last = a_grid.copy()

        U = qnormalize(qmul(A, qmul(B, U)))

        if kicks is not None:
            K = kicks.shape[0]
            # FIX: ensure integer dtype for advanced indexing
            rnd = rng.randint(0, 2**31, size=(H,W))
            idx = (I*73 + J*37 + t*11 + rnd) % K
            idx = idx.astype(np.intp)  # <- critical
            kfield = kicks[idx]
            U = qnormalize(qmul(kfield, U))

        hol = plaquette_angles(U)
        theta = z3_reduce(hol)
        labs = z3_labels(hol)

        inv["mean_abs_hol"].append(float(np.mean(np.abs(hol))))
        inv["z3_mass"].append(float((np.abs(theta) < (np.pi/12)).mean()))
        inv["z3_junctions"].append(z3_junctions(labs))
        inv["walls"].append(domain_wall_length(labs))
        inv["entropy"].append(label_entropy(labs))

        z3_counts.append([int((labs==0).sum()), int((labs==1).sum()), int((labs==2).sum())])
        grid, _ = life_step_torus(grid)

    A_last = q_axis_angle_x(base_last); B_last = q_axis_angle_y(base_last)
    U_AB_last = qnormalize(qmul(A_last, qmul(B_last, U_prev_last)))
    U_BA_last = qnormalize(qmul(B_last, qmul(A_last, U_prev_last)))
    hol_AB = plaquette_angles(U_AB_last)
    hol_BA = plaquette_angles(U_BA_last)
    orient = (hol_AB - hol_BA).astype(np.float32)

    return hol, theta, labs, orient, np.array(z3_counts, dtype=np.int64), inv

def run_burgers_phase(rng: QRNGRandom, N=400, theta_steps=400, dtheta=0.25,
                      kappa_alpha=1.0, kappa_gain=1.0, cfl=0.45, gth=0.0,
                      init_kind="riemann"):
    x = np.linspace(0.0, 1.0, N, endpoint=False); dx = 1.0/N
    if init_kind == "riemann":
        u = np.where(x < 0.5, 1.0, -0.5).astype(np.float64)
        u += 0.05*np.sin(8*math.pi*x)
    elif init_kind == "random":
        u = rng.uniform01((N,)).astype(np.float64)*2.0 - 1.0
        u = gaussian_blur_wrap(u, sigma=1.0)
    else:
        u = np.sin(2*math.pi*x).astype(np.float64)

    inv = {"mean_abs_defect": [], "mean_abs_grad": [], "walls": [], "entropy": []}
    snaps = {}

    for k in range(theta_steps):
        g = np.abs(grad_central(u, dx))
        kappa = 1.0 + kappa_gain*(g**kappa_alpha)
        vmax = max(EPS, float(np.max(np.abs(u))))
        dt_cfl = cfl*dx/max(EPS, vmax)
        dt = min(dt_cfl, float(dtheta/np.max(kappa)))

        u_AB = microloop_AB(u, dt, dx)
        u_BA = microloop_BA(u, dt, dx)
        defect = u_AB - u_BA

        labs = triad_labels_1d(u, dx, gth=gth)
        compressive = (labs == 0)
        strong = np.abs(defect) > np.percentile(np.abs(defect), 75)

        choose_AB = (defect >= 0.0)
        u_pred = 0.5*(u_AB + u_BA)
        u_next = u_pred.copy()
        picks = compressive & strong
        u_next[picks & choose_AB]  = u_AB[picks & choose_AB]
        u_next[picks & ~choose_AB] = u_BA[picks & ~choose_AB]

        u = u_next

        inv["mean_abs_defect"].append(float(np.mean(np.abs(defect))))
        inv["mean_abs_grad"].append(float(np.mean(np.abs(grad_central(u, dx)))))
        inv["walls"].append(float(domain_wall_length_1d(labs)))
        inv["entropy"].append(float(label_entropy_ternary(labs)))

        if k in (0, theta_steps//4, theta_steps//2, 3*theta_steps//4, theta_steps-1):
            snaps[k] = u.copy()

    return x, u, inv, snaps

def main():
    t0 = time.time()
    session = requests.Session()
    rng = QRNGRandom(tag="vybn-synthesis-root", session=session)

    manifest = {
        "qrng": {
            "base_url": QRNG_BASE_URL,
            "bits_per_block": BITS_PER_BLOCK,
            "key_present": bool(X_ID_API_KEY),
            "root_sha256": hashlib.sha256(rng.root).hexdigest(),
            "root_b64": base64.b64encode(rng.root).decode("ascii")
        }
    }

    # Lattice half
    hol, z3_map, labs, orient, z3_counts, inv_lat = run_lattice(rng)
    angle_centers, energy, P = anisotropy_from_fft(hol)

    out = {}
    ART = Path("./vybn_synthesis_artifacts"); ART.mkdir(parents=True, exist_ok=True)
    out["lat_hol_png"]     = save_img(hol,   "lat_holonomy.png",           "Holonomy (final, AB path)")
    out["lat_z3_png"]      = save_img(z3_map,"lat_angles_mod_2pi_over_3.png","Angles mod 2π/3")
    out["lat_labs_png"]    = save_img(labs,  "lat_z3_labels.png",          "Z3 labels")
    out["lat_orient_png"]  = save_img(orient,"lat_orientation_witness.png", "AB − BA (orientation-odd)")

    plt.figure(); plt.imshow(np.log1p(P), interpolation="nearest"); plt.title("FFT Power (log1p)")
    out["lat_fft_power_png"] = str(ART / "lat_fft_power.png"); plt.colorbar(); plt.savefig(out["lat_fft_power_png"], dpi=160, bbox_inches="tight"); plt.close()
    plt.figure(); ax = plt.subplot(111, projection="polar"); ax.plot(angle_centers, energy)
    out["lat_anisotropy_polar_png"] = str(ART / "lat_anisotropy_polar.png"); plt.title("Angular energy (FFT)"); plt.savefig(out["lat_anisotropy_polar_png"], dpi=160, bbox_inches="tight"); plt.close()

    steps_lat = len(inv_lat["mean_abs_hol"])
    rows_lat = [(i, inv_lat["mean_abs_hol"][i], inv_lat["z3_mass"][i], inv_lat["z3_junctions"][i],
                 inv_lat["walls"][i], inv_lat["entropy"][i]) for i in range(steps_lat)]
    out["lat_metrics_csv"] = save_csv(rows_lat, ["step","mean_abs_hol","z3_mass","z3_junctions","domain_wall_norm","label_entropy"], ART/"lat_metrics.csv")

    counts_rows = [(i,*z3_counts[i]) for i in range(steps_lat)]
    out["lat_z3_counts_csv"] = save_csv(counts_rows, ["step","label0","label1","label2"], ART/"lat_z3_counts.csv")

    out["lat_hol_npy"]   = save_npy(hol, ART/"lat_hol_final")
    out["lat_labs_npy"]  = save_npy(labs, ART/"lat_labs_final")
    out["lat_orient_npy"]= save_npy(orient, ART/"lat_orientation_witness")

    # Burgers half
    x, u_final, inv_burg, snaps = run_burgers_phase(rng)
    snap_keys = sorted(snaps.keys())
    ys = [snaps[k] for k in snap_keys]; labels = [f"θ={k}" for k in snap_keys]
    out["burgers_snaps_png"] = save_line(x, ys, labels, "burgers_snaps.png", "Burgers — phase snapshots", "x", "u")

    rows_b = [(i, inv_burg["mean_abs_defect"][i], inv_burg["mean_abs_grad"][i], inv_burg["walls"][i], inv_burg["entropy"][i])
              for i in range(len(inv_burg["mean_abs_defect"]))]
    out["burgers_metrics_csv"] = save_csv(rows_b, ["step","mean_abs_defect","mean_abs_grad","domain_wall_1d","label_entropy_ternary"], ART/"burgers_metrics.csv")
    out["burgers_final_npy"] = save_npy(u_final, ART/"burgers_u_final")

    out["lat_invariants_png"] = save_curve(
        {"mean|hol|":inv_lat["mean_abs_hol"], "Z3 mass":inv_lat["z3_mass"], "junctions":inv_lat["z3_junctions"]},
        "lat_invariants.png", "Lattice invariants over phase"
    )
    out["burgers_invariants_png"] = save_curve(
        {"|AB-BA|":inv_burg["mean_abs_defect"], "|u_x|":inv_burg["mean_abs_grad"], "walls":inv_burg["walls"]},
        "burgers_invariants.png", "Burgers invariants over phase"
    )

    manifest_path = ART/"manifest.json"; save_json({"qrng":manifest["qrng"], "files":out}, manifest_path)

    dt = time.time()-t0
    print(json.dumps({"artifacts": out, "manifest": str(manifest_path), "elapsed_sec": round(dt,2)}, indent=2))
    try:
        os.startfile(os.path.abspath(str(ART)))  # opens the artifacts folder on Windows
    except Exception:
        pass

if __name__ == "__main__":
    main()
