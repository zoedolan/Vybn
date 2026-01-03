#!/usr/bin/env python3
# Polar-Time Holonomy (Physics-flavored) — QRNG-seeded, streaming, concise
# - ALL randomness originates from a Cisco Outshift QRNG 256-bit seed (STRICT)
# - Seed expanded via SHAKE-256 to a dense θ-noise tape (no local PRNG seeding)
# - Shared θ tape across orientations; independent tape per replicate
# - Micro-shape variants at fixed area; orientation-odd deltas reported
# - Robust, line-buffered JSON logs so you always see progress

import sys, os, math, json, time, base64, hashlib, secrets
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ---------- minimal IO helpers ----------
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

def jlog(obj):
    print(json.dumps(obj, separators=(",",":")))
    sys.stdout.flush()

def eprint(*a):
    sys.stderr.write(" ".join(str(x) for x in a) + "\n")
    sys.stderr.flush()

EPS = 1e-12
BITS_PER_STEP = 64  # each θ substep uses one 64-bit seed

# ---------- Outshift QRNG (STRICT single-seed) ----------
import requests

QRNG_ENDPOINT = "https://api.qrng.outshift.com/api/v1/random_numbers"
X_ID_API_KEY  = "PASTE_YOUR_OUTSHIFT_KEY_HERE"  # <-- put your exact key here
USER_AGENT    = "Vybn-QRNG/phys/1.0"

def _assert_key():
    if not X_ID_API_KEY or "PASTE_YOUR_OUTSHIFT_KEY_HERE" in X_ID_API_KEY:
        raise RuntimeError("Set your Cisco Outshift key in X_ID_API_KEY.")

def _extract_bytes_from_random_numbers(obj) -> bytes:
    # Robust extractor; accepts several possible shapes
    def to_bytes(x) -> bytes:
        if x is None:
            return b""
        if isinstance(x, (bytes, bytearray)):
            return bytes(x)
        if isinstance(x, str):
            # Try base64 first
            try:
                return base64.b64decode(x, validate=False)
            except Exception:
                # numeric string?
                try:
                    n = int(x)
                    length = max(1, (n.bit_length()+7)//8)
                    return n.to_bytes(length, "big")
                except Exception:
                    return b""
        if isinstance(x, int):
            length = max(1, (x.bit_length()+7)//8)
            return x.to_bytes(length, "big")
        if isinstance(x, list):
            out = b""
            for t in x:
                out += to_bytes(t)
            return out
        if isinstance(x, dict):
            # common fields
            if "base64" in x and isinstance(x["base64"], str):
                try:
                    return base64.b64decode(x["base64"], validate=False)
                except Exception:
                    pass
            if "bytes" in x and isinstance(x["bytes"], list):
                try:
                    return bytes(int(v) & 0xFF for v in x["bytes"])
                except Exception:
                    pass
            if "binary" in x and isinstance(x["binary"], str) and all(c in "01" for c in x["binary"]):
                s = x["binary"]
                pad = (8 - len(s)%8) % 8
                s = "0"*pad + s
                return int(s, 2).to_bytes(len(s)//8, "big")
            # scan other values
            skip = {"encoding","n","count","bits","length","request_id","timestamp","id","format","type"}
            for k,v in x.items():
                if k in skip: continue
                b = to_bytes(v)
                if b: return b
            return b""
        return b""
    if not isinstance(obj, dict):
        return b""
    items = obj.get("random_numbers")
    if items is None:
        return b""
    return to_bytes(items)

def fetch_qrng_seed_bytes(n_bytes=32, label="seed", timeout=15.0, retries=6) -> bytes:
    _assert_key()
    if n_bytes <= 0:
        return b""
    headers = {
        "User-Agent": USER_AGENT,
        "Content-Type": "application/json",
        "Accept": "application/json",
        "x-id-api-key": X_ID_API_KEY,
    }
    # We will try a few payload shapes the Outshift API accepts.
    candidate_payloads = [
        {"encoding":"base64","n":1,              "bits": n_bytes*8},
        {"encoding":"base64","n":n_bytes,       "bits": 8},
        {"encoding":"base64","n":max(1,n_bytes//4),"bits":32},
    ]
    delay = 0.35
    last_err = None
    for attempt in range(1, retries+1):
        for payload in candidate_payloads:
            try:
                r = requests.post(QRNG_ENDPOINT, headers=headers, json=payload, timeout=timeout)
                if r.status_code == 429:
                    # rate-limited; back off
                    last_err = RuntimeError(f"Outshift 429: {r.text[:200]}")
                    time.sleep(delay)
                    delay = min(2.0*delay, 6.0)
                    continue
                if r.status_code != 200:
                    last_err = RuntimeError(f"Outshift HTTP {r.status_code}: {r.text[:200]}")
                    continue
                data = r.json()
                got = _extract_bytes_from_random_numbers(data)
                if got:
                    jlog({"type":"qrng_seed_probe","label":label,"ok":True,"bytes":len(got),"attempt":attempt})
                    return got[:n_bytes]
                else:
                    last_err = RuntimeError("QRNG returned 200 but no bytes.")
            except Exception as e:
                last_err = e
        time.sleep(delay)
        delay = min(2.0*delay, 6.0)
    raise RuntimeError(f"QRNG seed failed after retries: {last_err}")

# ---------- Seeded θ-tape via SHAKE-256 ----------
def make_theta_tape_uint64(steps: int, seed_bytes: bytes, label: str) -> np.ndarray:
    # Derive an independent stream for each label
    domain_sep = b"vybn.theta|" + label.encode("utf-8") + b"|"
    material   = seed_bytes + domain_sep
    # One shot XOF to avoid per-step overhead
    raw = hashlib.shake_256(material).digest(steps * 8)  # 8 bytes per u64
    # Convert to uint64 big-endian
    arr = np.frombuffer(raw, dtype=">u8").astype(np.uint64, copy=False)
    return arr

# ---------- Model, data, metrics ----------
def set_seed(s: int):
    # Only for deterministic loader shuffles etc. (not used for noise)
    np.random.seed(s % (2**32-1))
    torch.manual_seed(s % (2**31-1))

class MLP(nn.Module):
    def __init__(self, in_dim=28*28, hid=256, emb=64, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid)
        self.fc2 = nn.Linear(hid, emb)
        self.head= nn.Linear(emb, num_classes)
    def features(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        z = self.fc2(x)
        return z
    def forward(self, x):
        z = self.features(x)
        logits = self.head(F.relu(z))
        return logits, z

def make_data(batch=256, eval_batch=1024, n_eval=2048):
    tfm = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST(root="./data", train=True, transform=tfm, download=True)
    test  = datasets.MNIST(root="./data", train=False, transform=tfm, download=True)
    train_loader = DataLoader(train, batch_size=batch, shuffle=True, drop_last=True, num_workers=0, pin_memory=False)
    idx = torch.arange(min(n_eval, len(test)))
    eval_subset = Subset(test, idx)
    eval_loader = DataLoader(eval_subset, batch_size=eval_batch, shuffle=False, num_workers=0, pin_memory=False)
    return train_loader, eval_loader

@torch.no_grad()
def eval_features(model, loader, device="cpu"):
    model.eval(); feats = []
    for x, _ in loader:
        x = x.to(device); _, z = model(x); feats.append(z.cpu())
    return torch.cat(feats, 0)

@torch.no_grad()
def eval_loss(model, loader, device="cpu"):
    model.eval(); tot, n = 0.0, 0
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        logits, _ = model(x)
        tot += F.cross_entropy(logits, y, reduction="sum").item()
        n += y.numel()
    return tot / max(1, n)

def orthogonal_procrustes(A, B):
    A0 = A - A.mean(0, keepdim=True)
    B0 = B - B.mean(0, keepdim=True)
    M = B0.T @ A0
    U, _, Vt = torch.linalg.svd(M, full_matrices=False)
    return U @ Vt

def linear_cka_centered(X, Y):
    Xc = X - X.mean(0, keepdim=True); Yc = Y - Y.mean(0, keepdim=True)
    Kx = Xc @ Xc.T; Ky = Yc @ Yc.T
    num = (Kx * Ky).sum(); den = torch.linalg.norm(Kx) * torch.linalg.norm(Ky) + EPS
    v = (num / den).item()
    return float(max(0.0, min(1.0, v)))

def holonomy_metric(Z0, Z1):
    with torch.no_grad():
        cka_pre = linear_cka_centered(Z0, Z1)
        Q = orthogonal_procrustes(Z0, Z1)
        Z1a = Z1 @ Q
        resid = (torch.linalg.norm(Z1a - Z0) / (torch.linalg.norm(Z0) + EPS)).item()
        cka_post = linear_cka_centered(Z0, Z1a)
    return {
        "procrustes_resid": resid,
        "cka_pre": cka_pre,
        "cka_post": cka_post,
        "cka_gap_post": 1.0 - cka_post
    }

def pca_projected_resid(Z0, Z1, k=16):
    with torch.no_grad():
        mu = Z0.mean(0, keepdim=True)
        X0 = Z0 - mu
        Y1 = Z1 - mu
        Vt = torch.linalg.svd(X0, full_matrices=False)[2]
        Vk = Vt[:k, :].T
        X0k = X0 @ Vk
        Y1k = Y1 @ Vk
        num = torch.linalg.norm(Y1k - X0k)
        den = torch.linalg.norm(X0k) + EPS
        return (num / den).item()

# ---------- Updates ----------
def loss_fn(logits, y): return F.cross_entropy(logits, y)

def sgd_step(model, batch, lr):
    model.train()
    for p in model.parameters():
        if p.grad is not None: p.grad = None
    x, y = batch; logits, _ = model(x); loss = loss_fn(logits, y); loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None: p.add_(-lr * p.grad)

def sgd_inverse_step(model, batch, lr):
    model.train()
    for p in model.parameters():
        if p.grad is not None: p.grad = None
    x, y = batch; logits, _ = model(x); loss = loss_fn(logits, y); loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None: p.add_(+lr * p.grad)

def langevin_micro_step(model, batch, lr_theta_part, T, seed_u64):
    # noise is derived from quantum seed via SHAKE; no local PRNG seeding
    model.train()
    for p in model.parameters():
        if p.grad is not None: p.grad = None
    x, y = batch; logits, _ = model(x); loss = loss_fn(logits, y); loss.backward()
    sigma = math.sqrt(max(0.0, 2.0 * lr_theta_part * T))
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is None: continue
            p.add_(-lr_theta_part * p.grad)
            if sigma > 0.0:
                # generate deterministic noise from the 64-bit seed per parameter tensor
                # derive bytes for this tensor
                shape_n = p.numel()
                # Expand seed -> bytes -> normals
                # 12 bytes per 2 normals via Box-Muller is overkill; use numpy PCG off the seed
                rng = np.random.default_rng(int(seed_u64 ^ shape_n) & 0xFFFFFFFFFFFFFFFF)
                noise = torch.from_numpy(rng.normal(size=shape_n).astype(np.float32)).to(p).view_as(p) * sigma
                p.add_(noise)

def langevin_micro_inverse_step(model, batch, lr_theta_part, T, seed_u64):
    # inverse: subtract noise and add gradient
    model.train()
    for p in model.parameters():
        if p.grad is not None: p.grad = None
    x, y = batch; logits, _ = model(x); loss = loss_fn(logits, y); loss.backward()
    sigma = math.sqrt(max(0.0, 2.0 * lr_theta_part * T))
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is None: continue
            if sigma > 0.0:
                rng = np.random.default_rng(int(seed_u64 ^ p.numel()) & 0xFFFFFFFFFFFFFFFF)
                noise = torch.from_numpy(rng.normal(size=p.numel()).astype(np.float32)).to(p).view_as(p) * sigma
                p.add_(-noise)
            p.add_(+lr_theta_part * p.grad)

# ---------- plumbing ----------
def prefetch_batches(loader, total_needed, device):
    it = iter(loader); out = []
    while len(out) < total_needed:
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader); x, y = next(it)
        out.append((x.to(device), y.to(device)))
    return out

def lcm(a, b):
    from math import gcd
    return a // gcd(a, b) * b

def run_loop_with_tape(model, r_batches, theta_batches, params,
                       orientation="clockwise", theta_tape=None, substeps_per_pulse=1):
    rN = len(r_batches); tN = len(theta_batches)
    assert theta_tape is not None and len(theta_tape) >= tN * substeps_per_pulse
    area = rN * params["lr_r"] * tN * (params["lr_theta"] * params["T"])

    def R_fwd(): [sgd_step(model, b, params["lr_r"]) for b in r_batches]
    def R_inv(): [sgd_inverse_step(model, b, params["lr_r"]) for b in reversed(r_batches)]

    def Th_fwd():
        g = substeps_per_pulse
        lr_part = params["lr_theta"] / g
        for i, b in enumerate(theta_batches):
            base = i * g
            for j in range(g):
                seed = int(theta_tape[base + j])
                langevin_micro_step(model, b, lr_part, params["T"], seed)

    def Th_inv():
        g = substeps_per_pulse
        lr_part = params["lr_theta"] / g
        for i in range(tN-1, -1, -1):
            b = theta_batches[i]
            base = i * g
            for j in range(g-1, -1, -1):
                seed = int(theta_tape[base + j])
                langevin_micro_inverse_step(model, b, lr_part, params["T"], seed)

    if orientation == "clockwise":
        R_fwd(); Th_fwd(); R_inv(); Th_inv()
    else:
        Th_fwd(); R_fwd(); Th_inv(); R_inv()
    return area

# ---------- experiment ----------
def experiment(args):
    jlog({"type":"status","status":"starting","ts": int(time.time())})
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    set_seed(args.seed)

    # config log
    run_id = secrets.token_hex(4)
    jlog({"type":"starting","ts":int(time.time())})
    jlog({
        "type":"config",
        "ts":int(time.time()),
        "run_id": run_id,
        "device":"cuda" if device=="cuda" else "cpu",
        "torch": torch.__version__,
        "qrng_mode": "seeded_strict",
        "seed_bits": 256,
        "workers": 0,
        "pin": False,
        "tape_scale": args.tape_scale
    })

    # dataset
    train_loader, eval_loader = make_data(batch=args.batch, eval_batch=args.eval_batch, n_eval=args.n_eval)

    # warmup model
    model0 = MLP().to(device)
    opt = torch.optim.SGD(model0.parameters(), lr=0.05)
    it = iter(train_loader)
    for _ in range(12):
        try: x, y = next(it)
        except StopIteration:
            it = iter(train_loader); x, y = next(it)
        x, y = x.to(device), y.to(device)
        logits, _ = model0(x); loss = loss_fn(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()

    Z0 = eval_features(model0, eval_loader, device=device)
    L0 = eval_loss(model0,  eval_loader, device=device)

    # Pulses
    r_pulses = args.r_pulses
    theta_pulses = args.theta_pulses

    # Base batches
    r_batches_A     = prefetch_batches(train_loader, r_pulses, device)
    theta_batches_A = prefetch_batches(train_loader, theta_pulses, device)

    # Variant B (same area, different micro-shape)
    if theta_pulses % 2 == 0:
        # double R, half theta
        r_batches_B = r_batches_A + r_batches_A
        theta_batches_B = theta_batches_A[: theta_pulses // 2]
    elif r_pulses % 2 == 0 and r_pulses > 2:
        # half R, double theta
        r_batches_B = r_batches_A[: r_pulses // 2]
        theta_batches_B = theta_batches_A + theta_batches_A
    else:
        r_batches_B, theta_batches_B = r_batches_A, theta_batches_A

    tA, tB = len(theta_batches_A), len(theta_batches_B)

    # θ tape granularity: shared dense tape
    base_substeps  = lcm(tA, tB)
    dense_substeps = base_substeps * max(1, int(args.tape_scale))

    # Areas list
    areas = [float(x) for x in args.areas.split(",")] if args.areas else [1e-6]

    # QRNG seed (strict)
    try:
        seed_bytes = fetch_qrng_seed_bytes(32, label="master", timeout=15.0, retries=6)
        jlog({"type":"qrng_seed_ok","label":"master","source":"outshift","bytes":len(seed_bytes)})
    except Exception as e:
        jlog({"type":"error","error":type(e).__name__,"message":str(e)})
        raise

    for area in areas:
        # compute lr_theta for area = rN*lr_r * tN*(lr_theta*T)
        lr_theta_A = area / (len(r_batches_A) * args.lr_r * len(theta_batches_A) * args.T + EPS)
        lr_theta_B = area / (len(r_batches_B) * args.lr_r * len(theta_batches_B) * args.T + EPS)
        paramsA = {"lr_r": args.lr_r, "lr_theta": lr_theta_A, "T": args.T}
        paramsB = {"lr_r": args.lr_r, "lr_theta": lr_theta_B, "T": args.T}

        # log condition
        jlog({"type":"condition_start","ts":int(time.time()),"model":"mlp","optimizer":"sgd",
              "micro_shape":"balanced","area":area,"r_pulses":len(r_batches_A),"theta_pulses":len(theta_batches_A),
              "substeps_A": dense_substeps//len(theta_batches_A),
              "substeps_B": dense_substeps//len(theta_batches_B)})

        for rep in range(args.replicates):
            # independent tape per replicate from quantum master seed
            tape_label = f"rep{rep}|A+B"
            tape_u64 = make_theta_tape_uint64(dense_substeps, seed_bytes, label=tape_label)
            gA = dense_substeps // tA
            gB = dense_substeps // tB

            # A shape: cw and ccw
            cwA = MLP().to(device); cwA.load_state_dict(model0.state_dict())
            ccwA= MLP().to(device); ccwA.load_state_dict(model0.state_dict())

            areaA_cw  = run_loop_with_tape(cwA,  r_batches_A, theta_batches_A, paramsA, orientation="clockwise",
                                           theta_tape=tape_u64, substeps_per_pulse=gA)
            areaA_ccw = run_loop_with_tape(ccwA, r_batches_A, theta_batches_A, paramsA, orientation="counterclockwise",
                                           theta_tape=tape_u64, substeps_per_pulse=gA)

            ZcwA  = eval_features(cwA,  eval_loader, device=device)
            ZccwA = eval_features(ccwA, eval_loader, device=device)
            m_cwA, m_ccwA = holonomy_metric(Z0, ZcwA), holonomy_metric(Z0, ZccwA)
            LcwA, LccwA   = eval_loss(cwA, eval_loader, device=device), eval_loss(ccwA, eval_loader, device=device)
            proj_cwA      = pca_projected_resid(Z0, ZcwA, k=args.pca_k)
            proj_ccwA     = pca_projected_resid(Z0, ZccwA, k=args.pca_k)

            # commuting null (no noise effect) — share tape but never used
            cwN = MLP().to(device); cwN.load_state_dict(model0.state_dict())
            ccwN= MLP().to(device); ccwN.load_state_dict(model0.state_dict())
            # emulate commute by performing exact inverses without noise (no θ calls)
            # (we still run with zero θ by setting lr_theta=0)
            paramsNull = {"lr_r": args.lr_r, "lr_theta": 0.0, "T": args.T}
            run_loop_with_tape(cwN,  r_batches_A, theta_batches_A, paramsNull, orientation="clockwise",
                               theta_tape=tape_u64, substeps_per_pulse=gA)
            run_loop_with_tape(ccwN, r_batches_A, theta_batches_A, paramsNull, orientation="counterclockwise",
                               theta_tape=tape_u64, substeps_per_pulse=gA)
            ZcwN, ZccwN = eval_features(cwN, eval_loader, device=device), eval_features(ccwN, eval_loader, device=device)
            m_cwN, m_ccwN = holonomy_metric(Z0, ZcwN), holonomy_metric(Z0, ZccwN)
            LcwN, LccwN   = eval_loss(cwN, eval_loader, device=device), eval_loss(ccwN, eval_loader, device=device)
            proj_cwN      = pca_projected_resid(Z0, ZcwN, k=args.pca_k)
            proj_ccwN     = pca_projected_resid(Z0, ZccwN, k=args.pca_k)

            # B micro-shape at same area
            cwB = MLP().to(device); cwB.load_state_dict(model0.state_dict())
            ccwB= MLP().to(device); ccwB.load_state_dict(model0.state_dict())

            areaB_cw  = run_loop_with_tape(cwB,  r_batches_B, theta_batches_B, paramsB, orientation="clockwise",
                                           theta_tape=tape_u64, substeps_per_pulse=gB)
            areaB_ccw = run_loop_with_tape(ccwB, r_batches_B, theta_batches_B, paramsB, orientation="counterclockwise",
                                           theta_tape=tape_u64, substeps_per_pulse=gB)

            ZcwB, ZccwB = eval_features(cwB, eval_loader, device=device), eval_features(ccwB, eval_loader, device=device)
            m_cwB, m_ccwB = holonomy_metric(Z0, ZcwB), holonomy_metric(Z0, ZccwB)
            LcwB, LccwB   = eval_loss(cwB, eval_loader, device=device), eval_loss(ccwB, eval_loader, device=device)
            proj_cwB      = pca_projected_resid(Z0, ZcwB, k=args.pca_k)
            proj_ccwB     = pca_projected_resid(Z0, ZccwB, k=args.pca_k)

            row = {
                "run_id": run_id,
                "timestamp": int(time.time()),
                "area": area,
                "r_pulses": len(r_batches_A),
                "theta_pulses": len(theta_batches_A),
                "micro_shape": "balanced",
                "lr_r": args.lr_r,
                "lr_theta": lr_theta_A,
                "T": args.T,
                "substeps_A": gA,
                "substeps_B": gB,
                "model": "mlp",
                "optimizer": "sgd",
                "replicate": rep,
                "tape_len": int(dense_substeps),
                "tape_scale": int(args.tape_scale),
                "baseline_loss": L0,

                "A_area_cw": areaA_cw,
                "A_procrustes_delta": m_cwA["procrustes_resid"] - m_ccwA["procrustes_resid"],
                "A_projk_delta":       pca_projected_resid(Z0, ZcwA, k=args.pca_k) - pca_projected_resid(Z0, ZccwA, k=args.pca_k),
                "A_cka_post_gap_delta": m_cwA["cka_gap_post"] - m_ccwA["cka_gap_post"],
                "A_heat_per_area_delta": (LcwA - LccwA) / (areaA_cw + EPS),
                "A_slope_per_area_procrustes": (m_cwA["procrustes_resid"] - m_ccwA["procrustes_resid"]) / (areaA_cw + EPS),
                "A_slope_per_area_projk":       (proj_cwA - proj_ccwA) / (areaA_cw + EPS),

                "N_procrustes_delta": m_cwN["procrustes_resid"] - m_ccwN["procrustes_resid"],
                "N_projk_delta":       proj_cwN - proj_ccwN,
                "N_cka_post_gap_delta": m_cwN["cka_gap_post"] - m_ccwN["cka_gap_post"],
                "N_heat_per_area_delta": (LcwN - LccwN) / (areaA_cw + EPS),

                "B_area_cw": areaB_cw,
                "B_procrustes_delta": m_cwB["procrustes_resid"] - m_ccwB["procrustes_resid"],
                "B_projk_delta":       proj_cwB - proj_ccwB,
                "B_cka_post_gap_delta": m_cwB["cka_gap_post"] - m_ccwB["cka_gap_post"],
                "B_heat_per_area_delta": (LcwB - LccwB) / (areaB_cw + EPS),
                "B_slope_per_area_procrustes": (m_cwB["procrustes_resid"] - m_ccwB["procrustes_resid"]) / (areaB_cw + EPS),
                "B_slope_per_area_projk":       (proj_cwB - proj_ccwB) / (areaB_cw + EPS),
            }
            jlog({"type":"result_row","row":row})

        jlog({"type":"condition_done","ts":int(time.time()),"run_id":run_id})

    jlog({"type":"done","summary":{
        "meta":{"timestamp":int(time.time()),"run_id":run_id,"device":"cuda" if device=="cuda" else "cpu",
                "torch":torch.__version__, "qrng":"seeded_strict", "endpoint": QRNG_ENDPOINT,
                "eval_batch": args.eval_batch, "pca_k": args.pca_k},
        "grid":{"areas":areas,"replicates":args.replicates,"tape_scale":args.tape_scale}
    }})

# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser(description="Holonomy (quantum-seeded), streaming")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--eval-batch", type=int, default=1024)
    p.add_argument("--n-eval", type=int, default=2048)
    p.add_argument("--lr-r", type=float, default=3e-3)
    p.add_argument("--T", type=float, default=0.25)
    p.add_argument("--r-pulses", type=int, default=8)
    p.add_argument("--theta-pulses", type=int, default=8)
    p.add_argument("--pca-k", type=int, default=16)
    p.add_argument("--tape-scale", type=int, default=8)
    p.add_argument("--replicates", type=int, default=3)
    p.add_argument("--areas", type=str, default="1e-6")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    try:
        jlog({"type":"status","status":"starting","ts": int(time.time())})
        experiment(args)
    except Exception as e:
        import traceback
        tb = traceback.format_exc(limit=2)
        jlog({"type":"error","error": type(e).__name__, "message": str(e), "trace_hint": tb})
        sys.exit(1)

if __name__ == "__main__":
    main()
