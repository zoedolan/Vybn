# holonomy_qmin.py
# runtip: python holonomy_matter.py --area 1e-7 --replicates 8 --r-pulses 8 --theta-pulses 8 --timeout-sec 5 --overall-sec 20 --max-calls-per-src 200 --tape-scale 8
# Minimal holonomy experiment with quantum-only randomness, fail-fast IO, and 1s heartbeats.
# - Prints JSON lines continuously.
# - Uses Cisco Outshift (if X_ID_API_KEY present) then ANU QRNG, with strict caps and timeouts.
# - No local PRNG fallback: if quantum bytes are insufficient -> clear error and exit.
# - Minimal model + FakeData to avoid dataset downloads; small, fast, iterative.

import os, sys, time, json, math, base64, secrets, argparse
import requests
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets

# ---------- logging helpers ----------
def jlog(**kv):
    kv.setdefault("ts", int(time.time()))
    print(json.dumps(kv), flush=True)

class Heartbeat:
    def __init__(self, every_sec=1.0, tag="hb"):
        self.every = every_sec
        self.last = 0.0
        self.tag = tag
    def tick(self, **extra):
        now = time.monotonic()
        if now - self.last >= self.every:
            self.last = now
            jlog(type="heartbeat", tag=self.tag, **extra)

# ---------- QRNG Clients (quantum-only) ----------
OUTSHIFT_URL = "https://api.qrng.outshift.com/api/v1/random_numbers"
ANU_URL      = "https://qrng.anu.edu.au/API/jsonI.php"  # ?length=N&type=uint8

def outshift_fetch_bytes(need_bytes, timeout_sec=3.0, max_calls=15, label="outshift"):
    """Fetch bytes from Cisco Outshift. Requires X_ID_API_KEY in env. Returns bytes (may be < need_bytes)."""
    KEY = os.environ.get("X_ID_API_KEY", "").strip()
    if not KEY:
        return b"", {"used": False, "reason": "no_key"}
    hb = Heartbeat(1.0, tag="qrng-outshift")
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "x-id-api-key": KEY,
        "user-agent": "Vybn-QMinimal/1.0",
    }
    got = bytearray()
    zero_streak = 0
    per_call_bits = min(8 * need_bytes, 8192)  # request up to 8 Ki bits per call
    for call in range(1, max_calls + 1):
        payload = {"n": 1, "bits": int(per_call_bits), "encoding": "base64"}
        try:
            r = requests.post(OUTSHIFT_URL, headers=headers, json=payload, timeout=timeout_sec)
            if r.status_code == 429:
                jlog(type="qrng_tick", src="outshift", label=label, call=call, ok=False, http=429,
                     msg="rate_limited", total_bytes=len(got), need_bytes=need_bytes)
                break
            if r.status_code != 200:
                jlog(type="qrng_tick", src="outshift", label=label, call=call, ok=False, http=r.status_code,
                     total_bytes=len(got), need_bytes=need_bytes)
                break
            data = r.json()
            items = data.get("random_numbers", [])
            enc = str(data.get("encoding", "")).lower()
            chunk = b""
            if enc == "base64" and isinstance(items, list):
                for it in items:
                    if isinstance(it, str):
                        try:
                            chunk += base64.b64decode(it)
                        except Exception:
                            pass
                    elif isinstance(it, dict):
                        b64 = it.get("base64") or it.get("b64")
                        if isinstance(b64, str):
                            try:
                                chunk += base64.b64decode(b64)
                            except Exception:
                                pass
                        arr = it.get("bytes")
                        if isinstance(arr, list) and all(isinstance(x, int) for x in arr):
                            chunk += bytes([x & 0xFF for x in arr])
            # If we got nothing, count and potentially bail
            ok = len(chunk) > 0
            if not ok:
                zero_streak += 1
            else:
                zero_streak = 0
                got.extend(chunk)
            jlog(type="qrng_tick", src="outshift", label=label, call=call, ok=ok,
                 got_bytes=len(chunk), total_bytes=len(got), need_bytes=need_bytes)
            hb.tick(src="outshift", total=len(got), need=need_bytes)
            if len(got) >= need_bytes:
                return bytes(got[:need_bytes]), {"used": True, "calls": call, "http": 200}
            if zero_streak >= 3:
                jlog(type="qrng_info", src="outshift", label=label, msg="three_empty_responses")
                break
        except requests.RequestException as e:
            jlog(type="qrng_tick", src="outshift", label=label, call=call, ok=False, http="EXC",
                 exc=str(e), total_bytes=len(got), need_bytes=need_bytes)
            break
    return bytes(got), {"used": True, "calls": call if 'call' in locals() else 0, "partial": True}

def anu_fetch_bytes(need_bytes, timeout_sec=3.0, max_calls=30, label="anu"):
    """Fetch bytes from ANU QRNG JSON API. Returns bytes (may be < need_bytes)."""
    hb = Heartbeat(1.0, tag="qrng-anu")
    got = bytearray()
    for call in range(1, max_calls + 1):
        length = min(1024, need_bytes - len(got))
        if length <= 0:
            break
        params = {"length": int(length), "type": "uint8"}
        try:
            r = requests.get(ANU_URL, params=params, timeout=timeout_sec)
            if r.status_code != 200:
                jlog(type="qrng_tick", src="anu", label=label, call=call, ok=False, http=r.status_code,
                     got_bytes=0, total_bytes=len(got), need_bytes=need_bytes)
                break
            data = r.json()
            arr = data.get("data")
            if not isinstance(arr, list):
                jlog(type="qrng_tick", src="anu", label=label, call=call, ok=False, http=200,
                     msg="no_data", got_bytes=0, total_bytes=len(got), need_bytes=need_bytes)
                break
            chunk = bytes(int(x) & 0xFF for x in arr)
            ok = len(chunk) > 0
            got.extend(chunk)
            jlog(type="qrng_tick", src="anu", label=label, call=call, ok=ok,
                 got_bytes=len(chunk), total_bytes=len(got), need_bytes=need_bytes)
            hb.tick(src="anu", total=len(got), need=need_bytes)
            if len(got) >= need_bytes:
                return bytes(got[:need_bytes]), {"used": True, "calls": call, "http": 200}
        except requests.RequestException as e:
            jlog(type="qrng_tick", src="anu", label=label, call=call, ok=False, http="EXC",
                 exc=str(e), total_bytes=len(got), need_bytes=need_bytes)
            break
    return bytes(got), {"used": True, "calls": call if 'call' in locals() else 0, "partial": True}

def quantum_fetch_bytes(total_bytes, overall_sec=20.0, timeout_sec=3.0,
                        max_calls_per_src=25, label="qbytes"):
    """Try Outshift (if key present) then ANU; fail fast if insufficient within overall_sec."""
    start = time.monotonic()
    got = bytearray()
    hb = Heartbeat(1.0, tag="qrng-all")
    # 1) Outshift first (if key present)
    out_bytes, meta_o = outshift_fetch_bytes(
        need_bytes=total_bytes, timeout_sec=timeout_sec, max_calls=max_calls_per_src, label=label
    )
    got.extend(out_bytes)
    hb.tick(stage="outshift", total=len(got), need=total_bytes)
    if len(got) < total_bytes and (time.monotonic() - start) < overall_sec:
        # 2) ANU for the remainder
        remain = total_bytes - len(got)
        anu_bytes, meta_a = anu_fetch_bytes(
            need_bytes=remain, timeout_sec=timeout_sec, max_calls=max_calls_per_src, label=label
        )
        got.extend(anu_bytes)
        hb.tick(stage="anu", total=len(got), need=total_bytes)
    # Final check / fail fast
    if len(got) < total_bytes:
        jlog(type="error", error="QRNGInsufficientData",
             message=f"Quantum sources did not deliver enough bytes. got={len(got)} need={total_bytes}")
        raise SystemExit(1)
    return bytes(got[:total_bytes])

# ---------- Minimal model + utilities ----------
def set_seed_all(seed: int):
    np.random.seed(seed & 0xFFFFFFFF)
    torch.manual_seed(seed & 0xFFFFFFFF)
    torch.cuda.manual_seed_all(seed & 0xFFFFFFFF)

class MLP(nn.Module):
    def __init__(self, in_dim=28*28, hid=128, emb=32, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid)
        self.fc2 = nn.Linear(hid, emb)
        self.head = nn.Linear(emb, num_classes)
    def features(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    def forward(self, x):
        z = self.features(x)
        return self.head(F.relu(z)), z

def make_data(batch=256, eval_batch=512, n_eval=1024):
    # FakeData => no downloads, fast iteration, MNIST-like shape
    tfm = transforms.Compose([transforms.ToTensor()])
    train = datasets.FakeData(size=4096, image_size=(1,28,28), num_classes=10, transform=tfm)
    evald = datasets.FakeData(size=max(n_eval, 1024), image_size=(1,28,28), num_classes=10, transform=tfm)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch, shuffle=True, drop_last=True)
    eval_loader  = torch.utils.data.DataLoader(evald, batch_size=eval_batch, shuffle=False)
    return train_loader, eval_loader

@torch.no_grad()
def eval_features(model, loader, device="cpu"):
    model.eval(); feats=[]
    for x, _ in loader:
        x = x.to(device); _, z = model(x); feats.append(z.cpu())
    return torch.cat(feats, 0)

@torch.no_grad()
def eval_loss(model, loader, device="cpu"):
    model.eval(); tot=0.0; n=0
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
    EPS = 1e-12
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
        resid = (torch.linalg.norm(Z1a - Z0) / (torch.linalg.norm(Z0) + 1e-12)).item()
        cka_post = linear_cka_centered(Z0, Z1a)
    return {
        "procrustes_resid": float(resid),
        "cka_pre": float(cka_pre),
        "cka_post": float(cka_post),
        "cka_gap_post": float(1.0 - cka_post),
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
        den = torch.linalg.norm(X0k) + 1e-12
    return float((num / den).item())

# ---------- update rules ----------
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

def langevin_micro_step(model, batch, lr_theta_part, T, rng: np.random.Generator, commute=False):
    if commute: return
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
                noise = torch.from_numpy(rng.normal(size=p.shape)).to(p) * sigma
                p.add_(noise)

def langevin_micro_inverse_step(model, batch, lr_theta_part, T, rng: np.random.Generator, commute=False):
    if commute: return
    model.train()
    for p in model.parameters():
        if p.grad is not None: p.grad = None
    x, y = batch; logits, _ = model(x); loss = loss_fn(logits, y); loss.backward()
    sigma = math.sqrt(max(0.0, 2.0 * lr_theta_part * T))
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is None: continue
            if sigma > 0.0:
                noise = torch.from_numpy(rng.normal(size=p.shape)).to(p) * sigma
                p.add_(-noise)
            p.add_(+lr_theta_part * p.grad)

def prefetch_batches(loader, total_needed, device):
    it = iter(loader); out=[]
    while len(out) < total_needed:
        try: x, y = next(it)
        except StopIteration:
            it = iter(loader); x, y = next(it)
        out.append((x.to(device), y.to(device)))
    return out

# ---------- loop with quantum tape ----------
def run_loop_with_quantum_tape(model, r_batches, theta_batches, params,
                               orientation="clockwise", seeds=None, substeps_per_pulse=1, label="loop"):
    if seeds is None: raise RuntimeError("seeds required")
    rN = len(r_batches); tN = len(theta_batches)
    area = rN * params["lr_r"] * tN * (params["lr_theta"] * params["T"])
    g = substeps_per_pulse
    def Th_fwd():
        lr_part = params["lr_theta"] / g
        for i, b in enumerate(theta_batches):
            base = i * g
            for j in range(g):
                s = int(seeds[base + j])
                rng = np.random.default_rng(s)
                langevin_micro_step(model, b, lr_part, params["T"], rng, commute=False)
    def Th_inv():
        lr_part = params["lr_theta"] / g
        for i in range(tN-1, -1, -1):
            b = theta_batches[i]; base = i * g
            for j in range(g-1, -1, -1):
                s = int(seeds[base + j])
                rng = np.random.default_rng(s)
                langevin_micro_inverse_step(model, b, lr_part, params["T"], rng, commute=False)
    def R_fwd(): [sgd_step(model, b, params["lr_r"]) for b in r_batches]
    def R_inv(): [sgd_inverse_step(model, b, params["lr_r"]) for b in reversed(r_batches)]
    if orientation == "clockwise":
        R_fwd(); Th_fwd(); R_inv(); Th_inv()
    else:
        Th_fwd(); R_fwd(); Th_inv(); R_inv()
    return area

# ---------- experiment ----------
def experiment(args):
    jlog(type="status", status="starting")
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    run_id = secrets.token_hex(4)
    jlog(type="starting")
    jlog(type="config", run_id=run_id, device=device, torch=torch.__version__,
         qrng_mode="quantum-only", bits_per_step=64, workers=0, pin=False, tape_scale=args.tape_scale)

    # data & model
    train_loader, eval_loader = make_data(batch=args.batch, eval_batch=args.eval_batch, n_eval=args.n_eval)
    model0 = MLP().to(device)

    # minimal warmup
    opt = torch.optim.SGD(model0.parameters(), lr=0.05)
    it = iter(train_loader)
    for _ in range(5):
        try: x, y = next(it)
        except StopIteration:
            it = iter(train_loader); x, y = next(it)
        x, y = x.to(device), y.to(device)
        logits, _ = model0(x); loss = loss_fn(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()

    Z0 = eval_features(model0, eval_loader, device=device)
    L0 = eval_loss(model0,  eval_loader, device=device)

    # condition: single micro-shape "balanced"
    rA, tA = args.r_pulses, args.theta_pulses
    r_batches_A     = prefetch_batches(train_loader, rA, device)
    theta_batches_A = prefetch_batches(train_loader, tA, device)

    dense_substeps = tA * max(1, int(args.tape_scale))
    steps_needed = dense_substeps  # seeds count
    need_bytes = steps_needed * 8  # 64-bit per step

    jlog(type="qrng_reserve", label="balanced-A", steps=steps_needed, bytes_per_step=8, total_bytes=need_bytes)
    t0 = time.monotonic()
    qbytes = quantum_fetch_bytes(
        total_bytes=need_bytes,
        overall_sec=args.overall_sec,
        timeout_sec=args.timeout_sec,
        max_calls_per_src=args.max_calls_per_src,
        label="balanced-A"
    )
    elapsed = time.monotonic() - t0
    jlog(type="qrng_ready", label="balanced-A", bytes=len(qbytes), elapsed_sec=round(elapsed,3))

    # pack into uint64 seeds little-endian
    seeds = np.frombuffer(qbytes[:steps_needed*8], dtype="<u8")
    if len(seeds) != steps_needed:
        jlog(type="error", error="SeedPackingError", message="packed seeds mismatch")
        raise SystemExit(1)

    # params for holonomy
    params = dict(lr_r=args.lr_r, lr_theta=(args.area/(rA * args.T * tA)), T=args.T) if args.area is not None \
             else dict(lr_r=args.lr_r, lr_theta=args.lr_theta, T=args.T)

    # two orientations
    for rep in range(args.replicates):
        jlog(type="condition_start", model="mlp", optimizer="sgd", micro_shape="balanced",
             area=(rA*params["lr_r"]*tA*(params["lr_theta"]*params["T"])),
             r_pulses=rA, theta_pulses=tA, substeps_A=dense_substeps//tA, replicate=rep)

        cw = MLP().to(device); cw.load_state_dict(model0.state_dict())
        ccw = MLP().to(device); ccw.load_state_dict(model0.state_dict())

        # progress heartbeat during loops
        hb = Heartbeat(1.0, tag="train")
        areaA_cw  = run_loop_with_quantum_tape(cw,  r_batches_A, theta_batches_A, params,
                                               orientation="clockwise", seeds=seeds,
                                               substeps_per_pulse=dense_substeps//tA, label="cw")
        hb.tick(stage="cw_done")
        areaA_ccw = run_loop_with_quantum_tape(ccw, r_batches_A, theta_batches_A, params,
                                               orientation="counterclockwise", seeds=seeds,
                                               substeps_per_pulse=dense_substeps//tA, label="ccw")
        hb.tick(stage="ccw_done")

        ZcwA = eval_features(cw,  eval_loader, device=device)
        ZccwA= eval_features(ccw, eval_loader, device=device)
        m_cwA, m_ccwA = holonomy_metric(Z0, ZcwA), holonomy_metric(Z0, ZccwA)
        proj_cwA      = pca_projected_resid(Z0, ZcwA, k=args.pca_k)
        proj_ccwA     = pca_projected_resid(Z0, ZccwA, k=args.pca_k)
        LcwA, LccwA   = eval_loss(cw, eval_loader, device=device), eval_loss(ccw, eval_loader, device=device)

        row = {
            "run_id": run_id,
            "timestamp": int(time.time()),
            "area": areaA_cw,
            "r_pulses": rA, "theta_pulses": tA, "micro_shape": "balanced",
            "lr_r": params["lr_r"], "lr_theta": params["lr_theta"], "T": params["T"],
            "substeps_A": dense_substeps//tA,
            "model": "mlp", "optimizer": "sgd", "replicate": rep,
            "tape_len": len(seeds), "tape_scale": args.tape_scale,
            "baseline_loss": float(L0),
            "A_area_cw": float(areaA_cw),
            "A_procrustes_delta": float(m_cwA["procrustes_resid"] - m_ccwA["procrustes_resid"]),
            "A_projk_delta": float(proj_cwA - proj_ccwA),
            "A_cka_post_gap_delta": float(m_cwA["cka_gap_post"] - m_ccwA["cka_gap_post"]),
            "A_heat_per_area_delta": float((LcwA - LccwA) / (areaA_cw + 1e-12)),
            "A_slope_per_area_procrustes": float((m_cwA["procrustes_resid"] - m_ccwA["procrustes_resid"]) / (areaA_cw + 1e-12)),
            "A_slope_per_area_projk": float((proj_cwA - proj_ccwA) / (areaA_cw + 1e-12)),
        }
        jlog(type="result_row", row=row)

    jlog(type="done", summary={"run_id": run_id})

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Minimal holonomy with quantum-only randomness, fail-fast IO.")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--eval-batch", type=int, default=512)
    p.add_argument("--n-eval", type=int, default=1024)
    p.add_argument("--lr-r", type=float, default=3e-3)
    p.add_argument("--lr-theta", type=float, default=None, help="If --area not set, use this directly.")
    p.add_argument("--T", type=float, default=0.25)
    p.add_argument("--area", type=float, default=1e-6, help="If set, compute lr_theta from area=r*lr_r * t*(lr_theta*T).")
    p.add_argument("--r-pulses", type=int, default=8)
    p.add_argument("--theta-pulses", type=int, default=8)
    p.add_argument("--pca-k", type=int, default=16)
    p.add_argument("--tape-scale", type=int, default=8)
    p.add_argument("--replicates", type=int, default=1)
    p.add_argument("--timeout-sec", type=float, default=3.0, help="Per HTTP request timeout.")
    p.add_argument("--overall-sec", type=float, default=20.0, help="Overall QRNG budget seconds.")
    p.add_argument("--max-calls-per-src", type=int, default=25)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    # sanity: if user provided lr_theta explicitly, keep it; otherwise compute from area
    if args.lr_theta is None and args.area is None:
        args.lr_theta = 2.0833333333333333e-05  # fallback to earlier default
    try:
        jlog(type="status", status="starting", ts=int(time.time()))
        experiment(args)
    except SystemExit as e:
        # Propagate cleanly but make sure a final line is printed
        jlog(type="exit", code=int(e.code) if isinstance(e.code, int) else 1)
        raise
    except Exception as e:
        import traceback
        tb = traceback.format_exc(limit=3)
        jlog(type="error", error=type(e).__name__, message=str(e), trace_hint=tb)
        sys.exit(1)
