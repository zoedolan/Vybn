# holonomy_full.py
# Polar-Time Representation Holonomy — streaming, QRNG-seeded tapes, area sweeps, micro-shape variants.
# This program prints NDJSON "events" to stdout as it runs and writes CSV/JSONL artifacts to disk.
#
# Design notes summarized in prose for clarity.
# One small, once-per-run QRNG seed is fetched from Cisco Outshift using encoding="base64".
# If the provider is rate-limited or returns empty payloads, you can either allow a local fallback
# or inject your own master seed via --seed-hex or --seed-file. A keyed BLAKE2b DRBG expands the
# master seed deterministically into long uint64 tapes with per-condition personalization so each
# replicate is independent yet perfectly reproducible. The holonomy loops are executed with the same
# tape for CW vs CCW (and for the null “commute” control) to isolate orientation effects. The code
# streams progress and results immediately; no silent pooling or waiting.

import sys, os, math, json, time, random, argparse, csv, uuid, pathlib, hashlib, base64
from dataclasses import dataclass, asdict
from typing import List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ── stdout as a live event stream ─────────────────────────────────────────────────────────────────

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

def sprint(obj: Any):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

def eprint(msg: str):
    sys.stderr.write(str(msg) + "\n")
    sys.stderr.flush()

# ── QRNG seed acquisition (Outshift; tolerant decoder; graceful fallback) ─────────────────────────

QRNG_BASE_URL   = "https://api.qrng.outshift.com"
QRNG_ENDPOINT   = QRNG_BASE_URL.rstrip("/") + "/api/v1/random_numbers"
X_ID_API_KEY    = os.environ.get("OUTSHIFT_API_KEY", "PASTE_YOUR_EXACT_WORKING_KEY_HERE")
USER_AGENT      = "Vybn-QRNG/seed-2.0"

try:
    import requests
except Exception:
    requests = None  # guarded; you can still run with --seed-hex/--seed-file or --strict off

def _require_key():
    k = X_ID_API_KEY
    if not k or "PASTE_YOUR_EXACT_WORKING_KEY_HERE" in k:
        raise RuntimeError("Outshift key not set. Set OUTSHIFT_API_KEY or edit X_ID_API_KEY near the top.")

def _new_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": USER_AGENT,
        "Content-Type": "application/json",
        "Accept": "application/json",
        "x-id-api-key": X_ID_API_KEY,
    })
    return s

def _post_outshift(session, payload: dict, timeout_sec: float = 12.0) -> dict:
    _require_key()
    r = session.post(QRNG_ENDPOINT, json=payload, timeout=timeout_sec)
    if r.status_code == 200:
        try:
            return r.json()
        except Exception:
            raise RuntimeError("Outshift: 200 with non‑JSON payload")
    raise RuntimeError(f"Outshift QRNG HTTP {r.status_code}: {r.text[:400]}")

def _hex_to_bytes(s: str) -> bytes:
    t = s.strip()
    if t.startswith(("0x","0X")): t = t[2:]
    t = "".join(ch for ch in t if ch.strip() != "")
    return b"" if not t else int(t, 16).to_bytes((len(t)+1)//2, "big")

def _looks_like_bits(s: str) -> bool:
    return isinstance(s, str) and len(s) > 0 and all(c in "01" for c in s)

def _bits_to_bytes(bitstr: str) -> bytes:
    if not bitstr: return b""
    pad = (-len(bitstr)) % 8
    if pad: bitstr = ("0"*pad) + bitstr
    val = int(bitstr, 2)
    return val.to_bytes(len(bitstr)//8, "big")

def _coerce_any_to_bytes(it: Any, enc_hint: str = "") -> bytes:
    enc = (enc_hint or "").lower()
    if isinstance(it, (bytes, bytearray)):
        return bytes(it)
    if isinstance(it, str):
        if enc in ("base64","b64","raw"):
            try: return base64.b64decode(it, validate=False)
            except Exception: pass
        if _looks_like_bits(it): return _bits_to_bytes(it)
        if it.lower().startswith("0x"): return _hex_to_bytes(it)
        try: return base64.b64decode(it, validate=False)
        except Exception:
            b = _hex_to_bytes(it)
            return b if b else (_bits_to_bytes(it) if _looks_like_bits(it) else b"")
    if isinstance(it, int):
        return it.to_bytes((it.bit_length()+7)//8 or 1, "big")
    if isinstance(it, list):
        if all(isinstance(x, int) for x in it):
            return bytes([int(x) & 0xFF for x in it])
        out = bytearray()
        for x in it: out += _coerce_any_to_bytes(x, enc)
        return bytes(out)
    if isinstance(it, dict):
        for k in ("base64","b64","bytes"):
            v = it.get(k)
            if isinstance(v, str):
                try: return base64.b64decode(v, validate=False)
                except Exception: pass
        for k in ("binary","bits"):
            v = it.get(k)
            if isinstance(v, str) and _looks_like_bits(v): return _bits_to_bytes(v)
        for k in ("hex","hexadecimal"):
            v = it.get(k)
            if isinstance(v, str):
                b = _hex_to_bytes(v)
                if b: return b
        v = it.get("value") or it.get("number")
        if isinstance(v, (int, str)):
            try:
                return _coerce_any_to_bytes(v, enc)
            except Exception:
                pass
        seq = it.get("data") or it.get("bytes")
        if isinstance(seq, list) and all(isinstance(x,int) for x in seq):
            return bytes([int(x) & 0xFF for x in seq])
        out = bytearray()
        for k, val in it.items():
            if k in {"encoding","n","count","length","request_id","id","timestamp","time","format","type"}: 
                continue
            out += _coerce_any_to_bytes(val, enc)
        return bytes(out)
    return b""

def _extract_bytes_seed(obj: dict) -> bytes:
    if not isinstance(obj, dict): return b""
    enc = str(obj.get("encoding","")).lower()
    rn  = obj.get("random_numbers", None)
    if isinstance(rn, list) and rn:
        return _coerce_any_to_bytes(rn[0], enc)
    if "data" in obj:
        return _coerce_any_to_bytes(obj["data"], enc)
    if "numbers" in obj:
        return _coerce_any_to_bytes(obj["numbers"], enc)
    if "result" in obj:
        return _coerce_any_to_bytes(obj["result"], enc)
    return b""

def get_qrng_seed_bytes(seed_bits: int, strict: bool, label: str) -> bytes:
    want_bytes = max(16, (seed_bits + 7) // 8)
    if requests is None:
        if strict:
            raise RuntimeError("requests not installed; cannot contact Outshift in strict mode.")
        seed = os.urandom(want_bytes)
        sprint({"type":"qrng_seed_fallback","label":label,"mode":"no_requests_module","bytes":len(seed)})
        return seed
    sess = _new_session()
    payload = {"n": 1, "bits": int(seed_bits), "encoding": "base64"}
    try:
        t0 = time.time()
        data = _post_outshift(sess, payload, timeout_sec=12.0)
        preview = {"keys": list(data.keys())[:8]}
        if isinstance(data.get("random_numbers"), list):
            preview["rn_len"] = len(data["random_numbers"])
            if data["random_numbers"]:
                preview["rn0_type"] = type(data["random_numbers"][0]).__name__
        raw = _extract_bytes_seed(data)
        sprint({"type":"qrng_seed_probe","label":label,"ok":bool(raw), "elapsed_sec":round(time.time()-t0,3), "preview":preview})
        if not raw:
            if strict:
                raise RuntimeError("QRNG seed probe returned 200 but no bytes")
            seed = os.urandom(want_bytes)
            sprint({"type":"qrng_seed_fallback","label":label,"mode":"no_bytes_200","bytes":len(seed)})
            return seed
        seed = raw[:want_bytes] if len(raw) >= want_bytes else raw + os.urandom(want_bytes - len(raw))
        sprint({"type":"qrng_seed_ok","label":label,"source":"outshift","bytes":len(seed)})
        return seed
    except Exception as ex:
        msg = str(ex)
        if strict:
            raise RuntimeError(f"QRNG seed failed in strict mode: {msg}")
        mode = "quota_429" if "HTTP 429" in msg else "http_error_or_other"
        seed = os.urandom(want_bytes)
        sprint({"type":"qrng_seed_fallback","label":label,"mode":mode,"bytes":len(seed)})
        return seed

# ── DRBG: keyed BLAKE2b in counter mode; deterministic and fast ──────────────────────────────────

class Blake2bDRBG:
    def __init__(self, master_key: bytes, personalization: bytes = b""):
        if not isinstance(master_key, (bytes, bytearray)) or len(master_key) == 0:
            raise ValueError("master_key required")
        self.key = bytes(master_key)
        self.info = bytes(personalization or b"")
        self.counter = 0

    def _block(self) -> bytes:
        h = hashlib.blake2b(digest_size=64, key=self.key)
        h.update(self.info)
        h.update(self.counter.to_bytes(16, "big"))
        self.counter += 1
        return h.digest()

    def gen_bytes(self, n: int) -> bytes:
        chunks = []
        need = int(max(0, n))
        while need > 0:
            b = self._block()
            take = min(need, len(b))
            chunks.append(b[:take]); need -= take
        return b"".join(chunks)

    def gen_uint64(self, k: int) -> np.ndarray:
        if k <= 0:
            return np.zeros((0,), dtype=np.uint64)
        raw = self.gen_bytes(k * 8)
        return np.frombuffer(raw, dtype=np.uint64, count=k)

def derive_personalization(run_id: str, cond: dict, replicate: int) -> bytes:
    j = json.dumps({"run_id": run_id, "cond": cond, "replicate": replicate},
                   sort_keys=True, separators=(",",":")).encode("utf-8")
    return j

# ── data / model / optimizer ─────────────────────────────────────────────────────────────────────

def set_seed(seed: int, threads: int | None):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    if threads and threads > 0:
        try: torch.set_num_threads(int(threads))
        except Exception: pass

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

class ConvSmall(nn.Module):
    def __init__(self, emb=64, num_classes=10):
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32*7*7, emb)
        self.head = nn.Linear(emb, num_classes)
    def features(self, x):
        x = F.relu(self.c1(x))
        x = F.max_pool2d(F.relu(self.c2(x)), 2)  # 14x14 -> 7x7
        x = x.view(x.size(0), -1)
        z = self.fc(x)
        return z
    def forward(self, x):
        z = self.features(x)
        logits = self.head(F.relu(z))
        return logits, z

def make_model(name: str):
    n = name.lower()
    if n == "mlp": return MLP()
    if n == "conv": return ConvSmall()
    raise ValueError(f"unknown model {name}")

def make_opt(name: str, params, lr: float):
    n = name.lower()
    if n == "sgd":    return torch.optim.SGD(params, lr=lr)
    if n == "adam":   return torch.optim.Adam(params, lr=lr)
    if n == "adamw":  return torch.optim.AdamW(params, lr=lr)
    if n == "rmsprop":return torch.optim.RMSprop(params, lr=lr)
    raise ValueError(f"unknown optimizer {name}")

def make_data(batch=256, eval_batch=1024, n_eval=2048, workers=0, pin=False):
    tfm = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST(root="./data", train=True, transform=tfm, download=True)
    test  = datasets.MNIST(root="./data", train=False, transform=tfm, download=True)
    train_loader = DataLoader(train, batch_size=batch, shuffle=True, drop_last=True,
                              num_workers=workers, pin_memory=pin, persistent_workers=bool(workers))
    idx = torch.arange(min(n_eval, len(test)))
    eval_subset = Subset(test, idx)
    eval_loader = DataLoader(eval_subset, batch_size=eval_batch, shuffle=False,
                             num_workers=workers, pin_memory=pin, persistent_workers=bool(workers))
    return train_loader, eval_loader

# ── metrics ──────────────────────────────────────────────────────────────────────────────────────

EPS = 1e-12

def eval_features(model, loader, device="cpu"):
    model.eval(); feats = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True); _, z = model(x); feats.append(z.cpu())
    return torch.cat(feats, 0)

def eval_loss(model, loader, device="cpu"):
    model.eval(); tot, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            logits, _ = model(x)
            tot += F.cross_entropy(logits, y, reduction="sum").item()
            n += y.numel()
    return tot / max(1, n)

def orthogonal_procrustes(A, B):
    A0 = A - A.mean(0, keepdim=True); B0 = B - B.mean(0, keepdim=True)
    M = B0.T @ A0
    U, _, Vh = torch.linalg.svd(M, full_matrices=False)
    return U @ Vh

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
    return {"procrustes_resid": resid, "cka_pre": cka_pre, "cka_post": cka_post, "cka_gap_post": 1.0 - cka_post}

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

def loss_fn(logits, y): return F.cross_entropy(logits, y)

# ── update rules and tape execution ───────────────────────────────────────────────────────────────

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
    model.train()
    if commute: return
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
    model.train()
    if commute: return
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
    it = iter(loader); out = []
    while len(out) < total_needed:
        try: x, y = next(it)
        except StopIteration:
            it = iter(loader); x, y = next(it)
        out.append((x.to(device, non_blocking=True), y.to(device, non_blocking=True)))
    return out

def lcm(a, b):
    from math import gcd
    return a // gcd(a, b) * b

def run_loop_with_tape(model, r_batches, theta_batches, params,
                       orientation="clockwise", commute=False,
                       theta_tape=None, substeps_per_pulse=1):
    rN = len(r_batches); tN = len(theta_batches)
    if theta_tape is None or len(theta_tape) < tN * substeps_per_pulse:
        raise RuntimeError("theta_tape too short for requested pulses.")
    area = rN * params["lr_r"] * tN * (params["lr_theta"] * params["T"])

    rBatches = r_batches

    def R_fwd():
        for b in rBatches:
            sgd_step(model, b, params["lr_r"])

    def R_inv():
        for b in reversed(rBatches):
            sgd_inverse_step(model, b, params["lr_r"])

    def Th_fwd():
        g = substeps_per_pulse
        lr_part = params["lr_theta"] / g
        for i, b in enumerate(theta_batches):
            base = i * g
            for j in range(g):
                rng = np.random.default_rng(int(theta_tape[base + j]))
                langevin_micro_step(model, b, lr_part, params["T"], rng, commute=commute)

    def Th_inv():
        g = substeps_per_pulse
        lr_part = params["lr_theta"] / g
        for i in range(tN-1, -1, -1):
            b = theta_batches[i]
            base = i * g
            for j in range(g-1, -1, -1):
                rng = np.random.default_rng(int(theta_tape[base + j]))
                langevin_micro_inverse_step(model, b, lr_part, params["T"], rng, commute=commute)

    if orientation == "clockwise":
        R_fwd(); Th_fwd(); R_inv(); Th_inv()
    else:
        Th_fwd(); R_fwd(); Th_inv(); R_inv()
    return area

# ── orchestration ─────────────────────────────────────────────────────────────────────────────────

@dataclass
class Condition:
    area: float
    r_pulses: int
    theta_pulses: int
    micro_shape: str
    lr_r: float
    lr_theta: float
    T: float
    substeps_A: int
    substeps_B: int
    model: str
    optimizer: str
    replicate: int
    tape_len: int
    tape_scale: int
    commute_null: bool

def parse_csvish(s: str) -> List[str]:
    return [t.strip() for t in s.split(",") if t.strip()]

def build_micro_shapes(kind: str, r_base: int, t_base: int) -> Tuple[int,int,int,int]:
    k = kind.lower()
    if k == "balanced":
        return r_base, t_base, r_base, t_base
    if k == "long_r":
        return r_base, t_base, r_base*2, max(1, t_base//2)
    if k == "long_theta":
        return r_base, t_base, max(1, r_base//2), t_base*2
    return r_base, t_base, r_base, t_base

def warm_start(model, train_loader, device, steps=12, lr=0.05, optimizer="sgd"):
    opt = make_opt(optimizer, model.parameters(), lr=lr)
    it = iter(train_loader)
    for _ in range(steps):
        try: x, y = next(it)
        except StopIteration:
            it = iter(train_loader); x, y = next(it)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits, _ = model(x); loss = loss_fn(logits, y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

def experiment(args):
    run_id = str(uuid.uuid4())[:8]
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    set_seed(args.seed, args.threads)

    default_workers = 0 if sys.platform.startswith("win") else max(0, (os.cpu_count() or 2)//2)
    workers = default_workers if args.workers is None else args.workers
    pin = (device == "cuda") and (not args.no_pin)
    train_loader, eval_loader = make_data(batch=args.batch, eval_batch=args.eval_batch,
                                          n_eval=args.n_eval, workers=workers, pin=pin)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"holonomy_runs_{run_id}.csv")
    jsonl_path = os.path.join(out_dir, f"holonomy_runs_{run_id}.jsonl")

    sprint({"type":"starting","ts":int(time.time())})
    sprint({"type":"config","ts":int(time.time()),"run_id":run_id,
            "device":device,"torch":torch.__version__,
            "qrng_mode":args.qrng,"seed_bits":args.seed_bits,"strict":bool(args.strict),
            "workers":workers,"pin":pin,"tape_scale":args.tape_scale})

    # master seed selection: command-line overrides take precedence, then Outshift, then local OS
    master_seed = None
    if args.seed_hex:
        master_seed = _hex_to_bytes(args.seed_hex)
        if not master_seed:
            raise RuntimeError("seed-hex provided but did not parse to bytes")
        sprint({"type":"qrng_seed_ok","label":"master","source":"seed_hex","bytes":len(master_seed)})
    elif args.seed_file:
        with open(args.seed_file, "rb") as fh:
            master_seed = fh.read()
        if not master_seed:
            raise RuntimeError("seed-file read 0 bytes")
        sprint({"type":"qrng_seed_ok","label":"master","source":"seed_file","bytes":len(master_seed)})
    elif args.qrng == "seeded":
        master_seed = get_qrng_seed_bytes(args.seed_bits, strict=bool(args.strict), label="master")
    elif args.qrng == "local":
        master_seed = os.urandom(max(16, (args.seed_bits + 7)//8))
        sprint({"type":"qrng_seed_ok","label":"master","source":"local","bytes":len(master_seed)})
    elif args.qrng == "none":
        # fully deterministic default for hermetic tests; do not use for experiments requiring real entropy
        master_seed = hashlib.sha256(b"vybn_holonomy_default_seed").digest()
        sprint({"type":"qrng_seed_ok","label":"master","source":"none_fixed","bytes":len(master_seed)})
    else:
        raise ValueError("unknown qrng mode")

    csv_header = None
    csv_file = open(csv_path, "w", newline="")
    csv_writer = None
    all_rows = []

    areas = [float(x) for x in parse_csvish(args.area_grid)] if args.area_grid else [args.area]
    micro_shapes = parse_csvish(args.micro_shapes) if args.micro_shapes else ["balanced"]
    model_names = parse_csvish(args.models) if args.models else ["mlp"]
    optimizer_names = parse_csvish(args.optimizers) if args.optimizers else ["sgd"]

    meta0 = None
    cond_index = 0
    total_conds = len(model_names) * len(optimizer_names) * len(micro_shapes) * len(areas)

    for model_name in model_names:
        for optimizer_name in optimizer_names:
            base_model = make_model(model_name).to(device)
            warm_start(base_model, train_loader, device, steps=args.warm_steps, lr=args.warm_lr, optimizer=optimizer_name)
            Z0 = eval_features(base_model, eval_loader, device=device)
            L0 = eval_loss(base_model,  eval_loader, device=device)

            for mshape in micro_shapes:
                rA, tA, rB, tB = build_micro_shapes(mshape, args.r_pulses, args.theta_pulses)
                max_r = max(rA, rB); max_t = max(tA, tB)
                r_batches_all = prefetch_batches(train_loader, max_r, device)
                t_batches_all = prefetch_batches(train_loader, max_t, device)
                r_batches_A = r_batches_all[:rA]; t_batches_A = t_batches_all[:tA]
                r_batches_B = r_batches_all[:rB]; t_batches_B = t_batches_all[:tB]

                base_substeps  = lcm(tA, tB)
                dense_substeps = base_substeps * max(1, int(args.tape_scale))
                gA = dense_substeps // tA
                gB = dense_substeps // tB

                for area_target in areas:
                    cond_index += 1
                    lr_r = args.lr_r
                    denom = (rA * tA * args.T * max(EPS, lr_r))
                    lr_theta = float(area_target) / max(EPS, denom)
                    if lr_theta <= 0:
                        continue

                    cond_desc = {
                        "model": model_name, "optimizer": optimizer_name,
                        "micro_shape": mshape, "area": area_target,
                        "rA": rA, "tA": tA, "rB": rB, "tB": tB,
                        "gA": gA, "gB": gB, "T": args.T, "lr_r": lr_r
                    }

                    drbgs = []
                    for rep in range(args.replicas):
                        info = derive_personalization(run_id, cond_desc, rep)
                        per_key = hashlib.blake2b(info, key=master_seed, digest_size=32).digest()
                        drbgs.append(Blake2bDRBG(per_key, personalization=info))

                    sprint({"type":"condition_start","ts":int(time.time()),"run_id":run_id,
                            "index":cond_index,"total":total_conds,
                            "model":model_name,"optimizer":optimizer_name,
                            "micro_shape":mshape,"area":area_target,
                            "r_pulses":rA,"theta_pulses":tA,
                            "substeps_A":gA,"substeps_B":gB})

                    for rep in range(args.replicas):
                        theta_tape = drbgs[rep].gen_uint64(dense_substeps).tolist()

                        def clone_base():
                            m = make_model(model_name).to(device)
                            m.load_state_dict(base_model.state_dict())
                            return m

                        params = dict(lr_r=lr_r, lr_theta=lr_theta, T=args.T)

                        # Loop A (balanced R/Theta)
                        cwA, ccwA = clone_base(), clone_base()
                        areaA_cw  = run_loop_with_tape(cwA, r_batches_A, t_batches_A, params, "clockwise",
                                                       commute=False, theta_tape=theta_tape, substeps_per_pulse=gA)
                        areaA_ccw = run_loop_with_tape(ccwA, r_batches_A, t_batches_A, params, "counterclockwise",
                                                       commute=False, theta_tape=theta_tape, substeps_per_pulse=gA)
                        ZcwA, ZccwA = eval_features(cwA, eval_loader, device=device), eval_features(ccwA, eval_loader, device=device)
                        m_cwA, m_ccwA = holonomy_metric(Z0, ZcwA), holonomy_metric(Z0, ZccwA)
                        LcwA, LccwA = eval_loss(cwA, eval_loader, device=device), eval_loss(ccwA, eval_loader, device=device)
                        proj_cwA = pca_projected_resid(Z0, ZcwA, k=args.pca_k)
                        proj_ccwA= pca_projected_resid(Z0, ZccwA, k=args.pca_k)

                        # Null commute control
                        cwN, ccwN = clone_base(), clone_base()
                        run_loop_with_tape(cwN,  r_batches_A, t_batches_A, params, "clockwise",
                                           commute=True, theta_tape=theta_tape, substeps_per_pulse=gA)
                        run_loop_with_tape(ccwN, r_batches_A, t_batches_A, params, "counterclockwise",
                                           commute=True, theta_tape=theta_tape, substeps_per_pulse=gA)
                        ZcwN, ZccwN = eval_features(cwN, eval_loader, device=device), eval_features(ccwN, eval_loader, device=device)
                        m_cwN, m_ccwN = holonomy_metric(Z0, ZcwN), holonomy_metric(Z0, ZccwN)
                        LcwN, LccwN = eval_loss(cwN, eval_loader, device=device), eval_loss(ccwN, eval_loader, device=device)
                        proj_cwN = pca_projected_resid(Z0, ZcwN, k=args.pca_k)
                        proj_ccwN= pca_projected_resid(Z0, ZccwN, k=args.pca_k)

                        # Loop B (alternate micro-shape at same nominal area)
                        cwB, ccwB = clone_base(), clone_base()
                        areaB_cw  = run_loop_with_tape(cwB, r_batches_B, t_batches_B, params, "clockwise",
                                                       commute=False, theta_tape=theta_tape, substeps_per_pulse=gB)
                        areaB_ccw = run_loop_with_tape(ccwB, r_batches_B, t_batches_B, params, "counterclockwise",
                                                       commute=False, theta_tape=theta_tape, substeps_per_pulse=gB)
                        ZcwB, ZccwB = eval_features(cwB, eval_loader, device=device), eval_features(ccwB, eval_loader, device=device)
                        m_cwB, m_ccwB = holonomy_metric(Z0, ZcwB), holonomy_metric(Z0, ZccwB)
                        LcwB, LccwB = eval_loss(cwB, eval_loader, device=device), eval_loss(ccwB, eval_loader, device=device)
                        proj_cwB = pca_projected_resid(Z0, ZcwB, k=args.pca_k)
                        proj_ccwB= pca_projected_resid(Z0, ZccwB, k=args.pca_k)

                        cond = Condition(
                            area=area_target, r_pulses=rA, theta_pulses=tA, micro_shape=mshape,
                            lr_r=lr_r, lr_theta=lr_theta, T=args.T,
                            substeps_A=gA, substeps_B=gB, model=model_name, optimizer=optimizer_name,
                            replicate=rep, tape_len=len(theta_tape), tape_scale=args.tape_scale,
                            commute_null=True
                        )

                        row = {
                            "run_id": run_id,
                            "timestamp": int(time.time()),
                            **asdict(cond),
                            "baseline_loss": L0,
                            # A
                            "A_area_cw": areaA_cw,
                            "A_procrustes_delta": m_cwA["procrustes_resid"] - m_ccwA["procrustes_resid"],
                            "A_projk_delta":       proj_cwA - proj_ccwA,
                            "A_cka_post_gap_delta":m_cwA["cka_gap_post"] - m_ccwA["cka_gap_post"],
                            "A_heat_per_area_delta": (LcwA - LccwA) / (areaA_cw + EPS),
                            "A_slope_per_area_procrustes": (m_cwA["procrustes_resid"] - m_ccwA["procrustes_resid"]) / (areaA_cw + EPS),
                            "A_slope_per_area_projk": (proj_cwA - proj_ccwA) / (areaA_cw + EPS),
                            # Null
                            "N_procrustes_delta": m_cwN["procrustes_resid"] - m_ccwN["procrustes_resid"],
                            "N_projk_delta":       proj_cwN - proj_ccwN,
                            "N_cka_post_gap_delta":m_cwN["cka_gap_post"] - m_ccwN["cka_gap_post"],
                            "N_heat_per_area_delta": (LcwN - LccwN) / (areaA_cw + EPS),
                            # B
                            "B_area_cw": areaB_cw,
                            "B_procrustes_delta": m_cwB["procrustes_resid"] - m_ccwB["procrustes_resid"],
                            "B_projk_delta":       proj_cwB - proj_ccwB,
                            "B_cka_post_gap_delta":m_cwB["cka_gap_post"] - m_ccwB["cka_gap_post"],
                            "B_heat_per_area_delta": (LcwB - LccwB) / (areaB_cw + EPS),
                            "B_slope_per_area_procrustes": (m_cwB["procrustes_resid"] - m_ccwB["procrustes_resid"]) / (areaB_cw + EPS),
                            "B_slope_per_area_projk": (proj_cwB - proj_ccwB) / (areaB_cw + EPS),
                        }

                        sprint({"type":"result_row","row":row})

                        with open(jsonl_path, "a") as jf:
                            jf.write(json.dumps(row) + "\n")

                        if csv_header is None:
                            csv_header = list(row.keys())
                            csv_writer = csv.DictWriter(csv_file, fieldnames=csv_header)
                            csv_writer.writeheader()
                        if csv_writer is None:
                            csv_writer = csv.DictWriter(csv_file, fieldnames=csv_header)
                        csv_writer.writerow(row)
                        csv_file.flush()

                        if meta0 is None:
                            meta0 = {
                                "device": device,
                                "torch": torch.__version__,
                                "qrng": args.qrng,
                                "endpoint": QRNG_ENDPOINT if args.qrng not in ("local","none") else args.qrng,
                                "eval_batch": args.eval_batch,
                                "pca_k": args.pca_k
                            }

                        all_rows.append(row)

                    sprint({"type":"condition_done","ts":int(time.time()),"run_id":run_id,
                            "index":cond_index,"total":total_conds})

    csv_file.close()

    # small-area linear slope digest (A loop)
    bins = {}
    for r in all_rows:
        bins.setdefault(r["area"], []).append(r)
    slope_est = {}
    for a, rows in bins.items():
        vals = [r["A_slope_per_area_procrustes"] for r in rows if np.isfinite(r["A_slope_per_area_procrustes"])]
        if len(vals) > 0:
            slope_est[str(a)] = float(np.median(vals))

    out = {
        "meta": {
            "timestamp": int(time.time()),
            "run_id": run_id,
            "log_csv": csv_path,
            "log_jsonl": jsonl_path,
            **(meta0 or {})
        },
        "grid_summary": {
            "num_rows": len(all_rows),
            "unique_areas": sorted({r["area"] for r in all_rows}),
            "unique_models": sorted({r["model"] for r in all_rows}),
            "unique_opts": sorted({r["optimizer"] for r in all_rows}),
            "micro_shapes": sorted({r["micro_shape"] for r in all_rows}),
            "replicas": args.replicas
        },
        "small_area_linear_slope_median_A_procrustes": slope_est
    }
    sprint({"type":"done","summary":out})

# ── CLI ───────────────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Holonomy — streaming with QRNG seed expansion and robust I/O")
    # data / eval
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--eval-batch", type=int, default=1024)
    p.add_argument("--n-eval", type=int, default=2048)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--no-pin", action="store_true")
    # base hyper
    p.add_argument("--lr-r", type=float, default=3e-3)
    p.add_argument("--T", type=float, default=0.25)
    p.add_argument("--pca-k", type=int, default=16)
    # pulses
    p.add_argument("--r-pulses", type=int, default=8)
    p.add_argument("--theta-pulses", type=int, default=8)
    # area control
    p.add_argument("--area", type=float, default=1e-6)
    p.add_argument("--area-grid", type=str, default="")
    # tapes / entropy
    p.add_argument("--qrng", type=str, default="seeded", choices=["seeded","local","none"])
    p.add_argument("--seed-bits", type=int, default=256)
    p.add_argument("--seed-hex", type=str, default="")
    p.add_argument("--seed-file", type=str, default="")
    p.add_argument("--strict", action="store_true", help="require Outshift seed; otherwise fall back")
    # models/opts
    p.add_argument("--models", type=str, default="mlp", help="comma: mlp,conv")
    p.add_argument("--optimizers", type=str, default="sgd", help="comma: sgd,adam,adamw,rmsprop")
    # micro-shapes
    p.add_argument("--micro-shapes", type=str, default="balanced,long_r,long_theta")
    # warm start
    p.add_argument("--warm-steps", type=int, default=12)
    p.add_argument("--warm-lr", type=float, default=0.05)
    # system
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--replicas", type=int, default=3)
    p.add_argument("--tape-scale", type=int, default=8, help="LCM multiplier for theta substeps")
    p.add_argument("--out-dir", type=str, default="./vybn_artifacts")
    args = p.parse_args()

    try:
        experiment(args)
    except Exception as e:
        import traceback
        tb = traceback.format_exc(limit=2)
        sprint({"type":"error","error": type(e).__name__, "message": str(e), "trace_hint": tb})
        sys.exit(1)
