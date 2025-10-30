# holonomy_v2.py
# Polar-Time Representation Holonomy with Cisco Outshift QRNG (STRICT, hard-coded key)
# Deterministic batching, true dθ=0 null, micro-shape reuse at fixed area,
# representation holonomy + "heat per area" companion. No env vars. No fallbacks.

import os, math, json, time, random, argparse, base64
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ── Outshift QRNG (STRICT) ─────────────────────────────────────────────

QRNG_BASE_URL   = "https://api.qrng.outshift.com"
QRNG_ENDPOINT   = QRNG_BASE_URL.rstrip("/") + "/api/v1/random_numbers"
X_ID_API_KEY    = "PASTE_YOUR_EXACT_WORKING_KEY_HERE"  # ← paste your exact key
USER_AGENT      = "Vybn-QRNG/1.0"
BITS_PER_BLOCK  = 1024
EPS             = 1e-12

try:
    import requests
except Exception:
    raise SystemExit("Install requests first:  pip install requests")

def _require_key():
    if not X_ID_API_KEY or X_ID_API_KEY.strip() == "" or "PASTE_YOUR_EXACT_WORKING_KEY_HERE" in X_ID_API_KEY:
        raise RuntimeError("Outshift key not set. Paste your key into X_ID_API_KEY near the top of this file.")

def _post_outshift(payload: dict, timeout_sec: float = 20.0) -> dict:
    _require_key()
    headers = {
        "User-Agent": USER_AGENT,
        "Content-Type": "application/json",
        "Accept": "application/json",
        "x-id-api-key": X_ID_API_KEY,
    }
    r = requests.post(QRNG_ENDPOINT, headers=headers, json=payload, timeout=timeout_sec)
    if r.status_code != 200:
        raise RuntimeError(f"Outshift QRNG HTTP {r.status_code}: {r.text[:400]}")
    try:
        return r.json()
    except Exception:
        raise RuntimeError("Outshift QRNG returned non‑JSON payload.")

def _looks_like_bits(s: str) -> bool:
    return isinstance(s, str) and len(s) > 0 and all(c in "01" for c in s)

def _hex_to_bits_str(s: str) -> str:
    t = s.strip()
    if t.startswith(("0x","0X")):
        t = t[2:]
    t = "".join(ch for ch in t if ch.strip() != "")
    if not t:
        return ""
    return bin(int(t, 16))[2:]  # raw entropy, no width forcing

def _b64_to_bits_str(s: str) -> str:
    raw = base64.b64decode(s, validate=False)
    return "".join(f"{b:08b}" for b in raw)

def _coerce_any_to_bits(it, declared_enc: str, depth: int = 0) -> str:
    """
    Convert any plausible 'random_numbers' element into a raw bitstring.
    Handles strings, ints, bytes, lists of bytes/ints/strings/dicts, and nested dicts.
    Skips obvious metadata keys when descending dicts.
    """
    if depth > 4:
        raise RuntimeError("Outshift: excessive nesting in random_numbers element.")

    enc = (declared_enc or "").strip().lower()

    if isinstance(it, str):
        if enc in ("binary", "bit", "bits"):
            if not _looks_like_bits(it):
                raise RuntimeError("Outshift: declared 'binary' but element string is not a bitstring.")
            return it
        if enc in ("hex", "hexadecimal"):
            bs = _hex_to_bits_str(it)
            if not bs:
                raise RuntimeError("Outshift: empty hex element.")
            return bs
        if enc in ("base64", "b64", "bytes"):
            bs = _b64_to_bits_str(it)
            if not bs:
                raise RuntimeError("Outshift: empty base64 element.")
            return bs
        if _looks_like_bits(it):
            return it
        if it.lower().startswith("0x"):
            return _hex_to_bits_str(it)
        if it.isdigit():
            return bin(int(it))[2:]
        # Best-effort: try base64 then hex; if both fail, error.
        try:
            return _b64_to_bits_str(it)
        except Exception:
            try:
                return _hex_to_bits_str(it)
            except Exception:
                raise RuntimeError("Outshift: unrecognized string element under random_numbers.")

    if isinstance(it, int):
        return bin(int(it))[2:]

    if isinstance(it, (bytes, bytearray)):
        return "".join(f"{b:08b}" for b in bytes(it))

    if isinstance(it, list):
        if all(isinstance(x, int) for x in it):
            return "".join(f"{int(x) & 0xFF:08b}" for x in it)
        # concatenate recursively for mixed lists (strings, dicts, nested lists)
        bits = []
        for x in it:
            bits.append(_coerce_any_to_bits(x, enc, depth+1))
        return "".join(bits)

    if isinstance(it, dict):
        # Prefer canonical randomness-bearing fields if present
        for key in ("binary","bits"):
            v = it.get(key)
            if isinstance(v, str) and _looks_like_bits(v):
                return v
        for key in ("hex","hexadecimal"):
            v = it.get(key)
            if isinstance(v, str):
                return _hex_to_bits_str(v)
        for key in ("base64","b64"):
            v = it.get(key)
            if isinstance(v, str):
                return _b64_to_bits_str(v)
        for key in ("value","number"):
            v = it.get(key)
            if isinstance(v, int):
                return bin(int(v))[2:]
            if isinstance(v, str) and v.isdigit():
                return bin(int(v))[2:]
        for key in ("bytes","data"):
            v = it.get(key)
            if isinstance(v, list) and all(isinstance(x, int) for x in v):
                return "".join(f"{int(x) & 0xFF:08b}" for x in v)
        # If none of the standard fields exist, descend into child values but skip obvious metadata.
        skip = {"encoding","bits","n","count","length","request_id","id","timestamp","time","format","type"}
        bits = []
        for k, v in it.items():
            if k in skip:
                continue
            try:
                b = _coerce_any_to_bits(v, enc, depth+1)
                if b:
                    bits.append(b)
            except Exception:
                continue
        if bits:
            return "".join(bits)
        raise RuntimeError("Outshift: dict element lacks recognizable randomness fields.")

    raise RuntimeError(f"Outshift: unhandled element type under random_numbers: {type(it).__name__}")

def _extract_random_bitstrings(obj: dict) -> list[str]:
    """
    Expected top-level shape: {"encoding": "...", "random_numbers": [...]}
    Coerces each element of random_numbers to bits, returns a list of per-element bitstrings.
    """
    if not isinstance(obj, dict):
        raise RuntimeError("Outshift: top-level payload is not a dict.")
    if "encoding" not in obj or "random_numbers" not in obj:
        raise RuntimeError(f"Outshift: expected top-level 'encoding' and 'random_numbers'. Got keys: {list(obj.keys())[:6]}")
    enc = str(obj["encoding"]).strip().lower()
    items = obj["random_numbers"]
    if not isinstance(items, list) or len(items) == 0:
        raise RuntimeError("Outshift: 'random_numbers' is not a non-empty list.")
    out = []
    for it in items:
        out.append(_coerce_any_to_bits(it, enc))
    return out

def outshift_qrng_uint64(k: int) -> np.ndarray:
    """Accumulate exactly 64*k quantum bits, posting again until enough payload arrives."""
    if k <= 0:
        return np.zeros((0,), dtype=np.uint64)
    need_bits = 64 * k
    bitstream = ""
    tries = 0
    while len(bitstream) < need_bits:
        tries += 1
        if tries > 16:
            raise RuntimeError(f"Outshift: insufficient bits after multiple requests ({len(bitstream)} < {need_bits}).")
        payload = {"n": 1, "bits": BITS_PER_BLOCK, "format": "binary"}
        data    = _post_outshift(payload)
        blocks  = _extract_random_bitstrings(data)
        if not blocks:
            raise RuntimeError("Outshift: response carried no randomness.")
        bitstream += "".join(blocks)
    seeds = []
    idx = 0
    for _ in range(k):
        chunk = bitstream[idx: idx + 64]
        if len(chunk) < 64:
            raise RuntimeError("Outshift: bitstream underflow while packing seeds.")
        seeds.append(int(chunk, 2))
        idx += 64
    return np.array(seeds, dtype=np.uint64)

# ── Geometry core ─────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

class MLP(nn.Module):
    def __init__(self, in_dim=28*28, hid=256, emb=64, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid)
        self.fc2 = nn.Linear(hid, emb)
        self.head = nn.Linear(emb, num_classes)
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
    train_loader = DataLoader(train, batch_size=batch, shuffle=True, drop_last=True)
    idx = torch.arange(min(n_eval, len(test)))
    eval_subset = Subset(test, idx)
    eval_loader = DataLoader(eval_subset, batch_size=eval_batch, shuffle=False)
    return train_loader, eval_loader

def eval_features(model, loader, device="cpu"):
    model.eval(); feats = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            _, z = model(x)
            feats.append(z.cpu())
    return torch.cat(feats, 0)

def eval_loss(model, loader, device="cpu"):
    model.eval(); tot, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            logits, _ = model(x)
            tot += F.cross_entropy(logits, y, reduction="sum").item()
            n += y.numel()
    return tot / max(1, n)

def orthogonal_procrustes(A, B):
    A0 = A - A.mean(0, keepdim=True); B0 = B - B.mean(0, keepdim=True)
    M = B0.T @ A0
    U, _, Vt = torch.linalg.svd(M, full_matrices=False)
    return U @ Vt

def linear_cka_centered(X, Y):
    Xc = X - X.mean(0, keepdim=True); Yc = Y - Y.mean(0, keepdim=True)
    Kx = Xc @ Xc.T; Ky = Yc @ Yc.T
    num = (Kx * Ky).sum(); den = torch.linalg.norm(Kx) * torch.linalg.norm(Ky) + EPS
    val = (num / den).item()
    return float(max(0.0, min(1.0, val)))

def holonomy_metric(Z0, Z1):
    with torch.no_grad():
        cka_pre = linear_cka_centered(Z0, Z1)
        Q = orthogonal_procrustes(Z0, Z1)
        Z1a = Z1 @ Q
        resid = (torch.linalg.norm(Z1a - Z0) / (torch.linalg.norm(Z0) + EPS)).item()
        cka_post = linear_cka_centered(Z0, Z1a)
    return {"procrustes_resid": resid, "cka_pre": cka_pre, "cka_post": cka_post, "cka_gap_post": 1.0 - cka_post}

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

def langevin_step(model, batch, lr_theta, T, step_seed: int, commute=False):
    model.train()
    if commute: return
    for p in model.parameters():
        if p.grad is not None: p.grad = None
    x, y = batch; logits, _ = model(x); loss = loss_fn(logits, y); loss.backward()
    sigma = math.sqrt(max(0.0, 2.0 * lr_theta * T))
    local = np.random.default_rng(step_seed)
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is None: continue
            p.add_(-lr_theta * p.grad)
            if sigma > 0.0:
                noise = torch.from_numpy(local.normal(size=p.shape)).to(p) * sigma
                p.add_(noise)

def langevin_inverse_step(model, batch, lr_theta, T, step_seed: int, commute=False):
    model.train()
    if commute: return
    for p in model.parameters():
        if p.grad is not None: p.grad = None
    x, y = batch; logits, _ = model(x); loss = loss_fn(logits, y); loss.backward()
    sigma = math.sqrt(max(0.0, 2.0 * lr_theta * T))
    local = np.random.default_rng(step_seed)
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is None: continue
            if sigma > 0.0:
                noise = torch.from_numpy(local.normal(size=p.shape)).to(p) * sigma
                p.add_(-noise)
            p.add_(+lr_theta * p.grad)

def prefetch_batches(loader, total_needed, device):
    it = iter(loader); out = []
    while len(out) < total_needed:
        try: x, y = next(it)
        except StopIteration:
            it = iter(loader); x, y = next(it)
        out.append((x.to(device), y.to(device)))
    return out

def run_loop(model, r_batches, theta_batches, params, orientation="clockwise",
             commute=False, theta_seeds=None):
    rN = len(r_batches); tN = len(theta_batches)
    area = rN * params["lr_r"] * tN * (params["lr_theta"] * params["T"])

    def R_fwd(): [sgd_step(model, b, params["lr_r"]) for b in r_batches]
    def R_inv(): [sgd_inverse_step(model, b, params["lr_r"]) for b in reversed(r_batches)]
    def Th_fwd():
        for i, b in enumerate(theta_batches):
            s = int(theta_seeds[i]) if theta_seeds is not None else 0
            langevin_step(model, b, params["lr_theta"], params["T"], s, commute=commute)
    def Th_inv():
        for i, b in enumerate(reversed(theta_batches)):
            s = int(theta_seeds[tN-1-i]) if theta_seeds is not None else 0
            langevin_inverse_step(model, b, params["lr_theta"], params["T"], s, commute=commute)

    if orientation == "clockwise":
        R_fwd(); Th_fwd(); R_inv(); Th_inv()
    else:
        Th_fwd(); R_fwd(); Th_inv(); R_inv()
    return area

def reuse_microshape(r_batches_A, theta_batches_A):
    rA, tA = len(r_batches_A), len(theta_batches_A)
    if tA % 2 == 0:
        r_batches_B     = r_batches_A + r_batches_A
        theta_batches_B = theta_batches_A[: tA // 2]
    elif rA % 2 == 0 and rA > 2:
        r_batches_B     = r_batches_A[: rA // 2]
        theta_batches_B = theta_batches_A + theta_batches_A
    else:
        r_batches_B, theta_batches_B = r_batches_A, theta_batches_A
    return r_batches_B, theta_batches_B

# ── Experiment ────────────────────────────────────────────────────────

def experiment(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    set_seed(args.seed)

    train_loader, eval_loader = make_data(batch=args.batch, eval_batch=args.eval_batch, n_eval=args.n_eval)
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

    params = dict(lr_r=args.lr_r, lr_theta=args.lr_theta, T=args.T)

    rA, tA = args.r_pulses, args.theta_pulses
    r_batches_A     = prefetch_batches(train_loader, rA, device)
    theta_batches_A = prefetch_batches(train_loader, tA, device)

    seedsA = outshift_qrng_uint64(tA).tolist()

    cw = MLP().to(device); cw.load_state_dict(model0.state_dict())
    ccw = MLP().to(device); ccw.load_state_dict(model0.state_dict())

    areaA_cw  = run_loop(cw,  r_batches_A, theta_batches_A, params, orientation="clockwise",        commute=False, theta_seeds=seedsA)
    areaA_ccw = run_loop(ccw, r_batches_A, theta_batches_A, params, orientation="counterclockwise", commute=False, theta_seeds=seedsA)

    ZcwA = eval_features(cw,  eval_loader, device=device)
    ZccwA= eval_features(ccw, eval_loader, device=device)
    m_cwA, m_ccwA = holonomy_metric(Z0, ZcwA), holonomy_metric(Z0, ZccwA)
    LcwA, LccwA   = eval_loss(cw, eval_loader, device=device), eval_loss(ccw, eval_loader, device=device)

    orient_odd_A = {
        "procrustes_resid_delta": m_cwA["procrustes_resid"] - m_ccwA["procrustes_resid"],
        "cka_post_gap_delta":     m_cwA["cka_gap_post"]     - m_ccwA["cka_gap_post"],
        "heat_per_area_delta":    (LcwA - LccwA) / (areaA_cw + EPS),
        "area":                   areaA_cw,
        "slope_per_area_procrustes": (m_cwA["procrustes_resid"] - m_ccwA["procrustes_resid"]) / (areaA_cw + EPS),
    }

    cwN = MLP().to(device); cwN.load_state_dict(model0.state_dict())
    ccwN= MLP().to(device); ccwN.load_state_dict(model0.state_dict())
    run_loop(cwN,  r_batches_A, theta_batches_A, params, orientation="clockwise",        commute=True,  theta_seeds=seedsA)
    run_loop(ccwN, r_batches_A, theta_batches_A, params, orientation="counterclockwise", commute=True,  theta_seeds=seedsA)
    ZcwN, ZccwN = eval_features(cwN, eval_loader, device=device), eval_features(ccwN, eval_loader, device=device)
    m_cwN, m_ccwN = holonomy_metric(Z0, ZcwN), holonomy_metric(Z0, ZccwN)
    LcwN, LccwN   = eval_loss(cwN, eval_loader, device=device), eval_loss(ccwN, eval_loader, device=device)

    orient_odd_null = {
        "procrustes_resid_delta": m_cwN["procrustes_resid"] - m_ccwN["procrustes_resid"],
        "cka_post_gap_delta":     m_cwN["cka_gap_post"]     - m_ccwN["cka_gap_post"],
        "heat_per_area_delta":    (LcwN - LccwN) / (areaA_cw + EPS)
    }

    r_batches_B, theta_batches_B = reuse_microshape(r_batches_A, theta_batches_A)
    tB   = len(theta_batches_B)
    seedsB = outshift_qrng_uint64(tB).tolist()

    cwB = MLP().to(device); cwB.load_state_dict(model0.state_dict())
    ccwB= MLP().to(device); ccwB.load_state_dict(model0.state_dict())
    areaB_cw  = run_loop(cwB,  r_batches_B, theta_batches_B, params, orientation="clockwise",        commute=False, theta_seeds=seedsB)
    areaB_ccw = run_loop(ccwB, r_batches_B, theta_batches_B, params, orientation="counterclockwise", commute=False, theta_seeds=seedsB)

    ZcwB, ZccwB = eval_features(cwB, eval_loader, device=device), eval_features(ccwB, eval_loader, device=device)
    m_cwB, m_ccwB = holonomy_metric(Z0, ZcwB), holonomy_metric(Z0, ZccwB)
    LcwB, LccwB   = eval_loss(cwB, eval_loader, device=device), eval_loss(ccwB, eval_loader, device=device)

    orient_odd_B = {
        "procrustes_resid_delta": m_cwB["procrustes_resid"] - m_ccwB["procrustes_resid"],
        "cka_post_gap_delta":     m_cwB["cka_gap_post"]     - m_ccwB["cka_gap_post"],
        "heat_per_area_delta":    (LcwB - LccwB) / (areaB_cw + EPS),
        "area":                   areaB_cw,
        "slope_per_area_procrustes": (m_cwB["procrustes_resid"] - m_ccwB["procrustes_resid"]) / (areaB_cw + EPS),
    }

    out = {
        "meta": {
            "timestamp": int(time.time()),
            "device": "cuda" if device == "cuda" else "cpu",
            "torch": torch.__version__,
            "qrng_path": "outshift"
        },
        "params": params | {"seed": args.seed},
        "loops": {
            "A": {"r_pulses": len(r_batches_A), "theta_pulses": len(theta_batches_A)},
            "B": {"r_pulses": len(r_batches_B), "theta_pulses": len(theta_batches_B)}
        },
        "qrng": {
            "endpoint": QRNG_ENDPOINT,
            "seeds_loop_A_len": len(seedsA),
            "seeds_loop_B_len": len(seedsB)
        },
        "baseline_loss": L0,
        "metrics": {
            "cwA": m_cwA, "ccwA": m_ccwA, "orient_odd_A": orient_odd_A,
            "null": orient_odd_null,
            "cwB": m_cwB, "ccwB": m_ccwB, "orient_odd_B": orient_odd_B
        }
    }
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Polar-Time Representation Holonomy with Cisco Outshift QRNG (STRICT, hard-coded key)")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--eval-batch", type=int, default=1024)
    p.add_argument("--n-eval", type=int, default=2048)
    p.add_argument("--lr-r", type=float, default=3e-3)
    p.add_argument("--lr-theta", type=float, default=1.2e-4)
    p.add_argument("--T", type=float, default=0.25)
    p.add_argument("--r-pulses", type=int, default=8)
    p.add_argument("--theta-pulses", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    experiment(args)
