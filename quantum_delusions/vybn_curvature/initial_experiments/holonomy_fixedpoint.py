# holonomy_trans_vybn.py
# A careful, fixed-point holonomy experiment with reproducibility, summaries, and optional non-quantum RNG.
# 
# Key design choices:
#   • Fixed point per (micro_shape, area): one model is initialized + preburned ONCE, then reused identically for all replicates.
#   • Fixed tape and fixed mini-batches options give true replicate consistency.
#   • "Commute" mode collapses Θ so CW ≈ CCW as a sanity check.
#   • Results are written to JSONL and CSV; per-(area,shape) summaries include sign-test p-values.
#
# Example (Windows CMD; one line):
#   python holonomy_trans_vybn.py --model vit --areas 3e-10,1e-9,3e-9,1e-8,3e-8 --replicates 8 --fixed-tape --fixed-batches --preburn 50 --out results
#
# If you want a debugging run without quantum RNG:
#   python holonomy_trans_vybn.py --areas 3e-9 --replicates 4 --fixed-tape --fixed-batches --rng numpy --seed 42 --out debug_np
#
# If you want the commute sanity check:
#   python holonomy_trans_vybn.py --areas 3e-9 --replicates 8 --fixed-tape --fixed-batches --preburn 50 --commute --out commute_test
#
# Notes:
#   • On Windows CMD, do not end the line with a backslash. Use a single line, or use PowerShell with the backtick ` for line continuation.
#   • MNIST is downloaded automatically to ./data on first run.
#
# MIT License (c) 2025 Vybn co‑lab

import os, sys, time, math, json, csv, argparse, base64, secrets, threading, requests, numpy as np
from collections import defaultdict
from statistics import mean
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ---------- util ----------
def jprint(o):
    print(json.dumps(o, ensure_ascii=False))
    sys.stdout.flush()

try:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
except Exception:
    pass

# ---------- RNG backends ----------
ANU_URL = "https://qrng.anu.edu.au/API/jsonI.php"
OUTSHIFT_URL = "https://api.qrng.outshift.com/api/v1/random_numbers"
USER_AGENT = "Vybn-QRNG/3.2"
OUTSHIFT_TIMEOUT = 8.0
ANU_TIMEOUT = 8.0
ANU_MIN_INTERVAL = float(os.environ.get("ANU_MIN_INTERVAL", "60.0"))
OUTSHIFT_API_KEY = (os.environ.get("OUTSHIFT_API_KEY") or "ENTER_API_KEY").strip()

_ANU_LOCK = threading.Lock()
_ANU_NEXT_ALLOWED = 0.0

class QRNGInsufficientData(RuntimeError): pass

class ANUClient:
    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": USER_AGENT})
    def can_call(self):
        with _ANU_LOCK:
            return time.time() >= _ANU_NEXT_ALLOWED
    def _gate(self):
        global _ANU_NEXT_ALLOWED
        with _ANU_LOCK:
            _ANU_NEXT_ALLOWED = time.time() + ANU_MIN_INTERVAL
    def fetch_bytes(self, n, label=""):
        self._gate()
        try:
            r = self.s.get(ANU_URL, params={"length": min(1024, max(1, n)), "type": "uint8"}, timeout=ANU_TIMEOUT)
            if r.status_code == 200 and (js := r.json()).get("success") and js.get("data"):
                raw = bytes(int(x) & 0xFF for x in js["data"])[:n]
                jprint({"type":"qrng_tick","src":"anu","label":label,"call":1,"ok":True,
                        "got_bytes":len(raw),"total_bytes":len(raw),"need_bytes":n,"ts":int(time.time())})
                return raw
            jprint({"type":"qrng_tick","src":"anu","label":label,"call":1,"ok":False,
                    "http":r.status_code,"got_bytes":0,"total_bytes":0,"need_bytes":n,"ts":int(time.time())})
            return b""
        except Exception:
            jprint({"type":"qrng_tick","src":"anu","label":label,"call":1,"ok":False,
                    "http":"exc","got_bytes":0,"total_bytes":0,"need_bytes":n,"ts":int(time.time())})
            return b""

class OutshiftClient:
    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": USER_AGENT, "Content-Type":"application/json",
                               "Accept":"application/json","x-id-api-key": OUTSHIFT_API_KEY})
    @staticmethod
    def _pack(nums, bits):
        acc = 0; accb = 0; out = bytearray(); mask = (1<<bits)-1
        for v in nums:
            v = int(v) & mask; acc |= (v << accb); accb += bits
            while accb >= 8:
                out.append(acc & 0xFF); acc >>= 8; accb -= 8
        return bytes(out)
    def fetch_bytes(self, n, label=""):
        if not OUTSHIFT_API_KEY:
            return b""
        out = bytearray(); call = 0
        while len(out) < n and call < 64:
            need = n - len(out); bits = 10; bpb = (bits+7)//8; blocks = max(1, min(1000, (need+bpb-1)//bpb))
            try:
                r = self.s.post(OUTSHIFT_URL, json={"encoding":"raw","format":"all",
                                                     "bits_per_block":bits,"number_of_blocks":int(blocks)}, timeout=OUTSHIFT_TIMEOUT)
                call += 1
                if r.status_code == 200:
                    js = r.json(); got = 0
                    nums = js.get("numbers") or js.get("data")
                    if isinstance(nums, list) and nums:
                        raw = self._pack([int(x) for x in nums], bits); out.extend(raw); got += len(raw)
                    rn = js.get("random_numbers")
                    if isinstance(rn, dict):
                        if isinstance(rn.get("numbers"), list):
                            raw = self._pack([int(x) for x in rn["numbers"]], bits); out.extend(raw); got += len(raw)
                        blocks = rn.get("blocks") or rn.get("values")
                        if isinstance(blocks, list):
                            for it in blocks:
                                b64 = (it.get("base64") if isinstance(it, dict) else (it if isinstance(it, str) else None))
                                if isinstance(b64, str):
                                    try:
                                        dec = base64.b64decode(b64, validate=False)
                                        out.extend(dec); got += len(dec)
                                    except Exception:
                                        pass
                    if got == 0:
                        blocks = js.get("blocks") or js.get("values") or js.get("data")
                        if isinstance(blocks, list):
                            for it in blocks:
                                if isinstance(it, dict) and "base64" in it:
                                    try:
                                        raw = base64.b64decode(it["base64"], validate=False)
                                        out.extend(raw); got += len(raw)
                                    except Exception:
                                        pass
                    jprint({"type":"qrng_tick","src":"outshift","label":label,"call":call,"ok":True,
                            "got_bytes":got,"total_bytes":len(out),"need_bytes":n,"ts":int(time.time())})
                    if got == 0:
                        time.sleep(min(0.25*call, 2.0))
                else:
                    jprint({"type":"qrng_tick","src":"outshift","label":label,"call":call,"ok":False,
                            "http":r.status_code,"got_bytes":0,"total_bytes":len(out),"need_bytes":n,"ts":int(time.time())})
            except Exception:
                jprint({"type":"qrng_tick","src":"outshift","label":label,"call":call+1,"ok":False,
                        "http":"exc","got_bytes":0,"total_bytes":len(out),"need_bytes":n,"ts":int(time.time())})
            time.sleep(min(0.2*(call+1), 2.0))
        return bytes(out[:n])

CACHE_FILE = "./vybn_quantum_cache.bin"

class QuantumPool:
    def __init__(self, cache_file=CACHE_FILE):
        self.cache = cache_file
        self.lock = threading.Lock()
    def _read(self, n):
        if not os.path.exists(self.cache):
            return b""
        try:
            with self.lock, open(self.cache, "rb") as f:
                data = f.read()
            if not data:
                return b""
            take = min(len(data), n); rem = data[take:]
            with self.lock, open(self.cache, "wb") as f:
                f.write(rem)
            return data[:take]
        except Exception:
            return b""
    def fetch(self, n, label=""):
        got = bytearray(self._read(n))
        if got:
            jprint({"type":"qrng_cache","label":label,"bytes":len(got),"ts":int(time.time())})
        need = n - len(got)
        if need > 0:
            got.extend(OutshiftClient().fetch_bytes(need, label=label))
        remain = n - len(got)
        if remain > 0:
            anu = ANUClient()
            if anu.can_call():
                got.extend(anu.fetch_bytes(remain, label=label))
            else:
                jprint({"type":"qrng_skip","src":"anu","label":label,"reason":"cooldown","ts":int(time.time())})
        if len(got) < n:
            raise QRNGInsufficientData(f"got={len(got)} need={n}")
        return bytes(got[:n])

def bytes_to_u64_list(b):
    return [int.from_bytes(b[i:i+8], "little") for i in range(0, len(b)//8*8, 8)]

class PseudoPool:
    """Optional deterministic / non-quantum RNG, useful if quantum endpoints rate-limit."""
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(int(seed) & ((1<<64)-1))
    def fetch(self, n, label=""):
        data = self.rng.integers(0, 256, size=n, dtype=np.uint8).tobytes()
        jprint({"type":"rng_tick","src":"numpy","label":label,"bytes":len(data),"ts":int(time.time())})
        return data

# ---------- Torch bits ----------
def set_seed(s):
    import random
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
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
        return self.fc2(x)
    def forward(self, x):
        z = self.features(x)
        return self.head(F.relu(z)), z

class VisionTransformer(nn.Module):
    def __init__(self, img_size=28, patch=7, in_chans=1, emb=128, depth=4, heads=4, mlp_ratio=2.0, num_classes=10, dropout=0.0):
        super().__init__()
        assert img_size % patch == 0
        self.patch = nn.Conv2d(in_chans, emb, kernel_size=patch, stride=patch, bias=True)
        n = (img_size//patch)**2
        self.cls = nn.Parameter(torch.zeros(1,1,emb))
        self.pos = nn.Parameter(torch.randn(1,1+n,emb)*0.02)
        self.drop = nn.Dropout(dropout)
        ff = int(emb*mlp_ratio*2)
        enc = nn.TransformerEncoderLayer(d_model=emb, nhead=heads, dim_feedforward=ff, dropout=dropout,
                                         activation="gelu", batch_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=depth)
        self.norm = nn.LayerNorm(emb)
        self.head = nn.Linear(emb, num_classes)
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
    def tokenize(self, x):
        x = self.patch(x)
        return x.flatten(2).transpose(1,2)
    def features(self, x):
        B = x.size(0); t = self.tokenize(x)
        z = torch.cat([self.cls.expand(B,-1,-1), t], 1) + self.pos[:,:t.size(1)+1,:]
        z = self.drop(z)
        z = self.enc(z)
        return self.norm(z[:,0])
    def forward(self, x):
        z = self.features(x)
        return self.head(z), z

def make_model(kind="vit"):
    k = (kind or "vit").lower()
    if k in ("vit","transformer","vision_transformer"):
        return VisionTransformer()
    if k == "mlp":
        return MLP()
    return VisionTransformer()

def loss_fn(logits, y):
    return F.cross_entropy(logits, y)

def make_data(batch=256, eval_batch=1024, n_eval=2048, fixed_batches=False, seed=42, workers=0, pin=False):
    tfm = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST(root="./data", train=True, transform=tfm, download=True)
    test  = datasets.MNIST(root="./data", train=False, transform=tfm, download=True)
    if fixed_batches:
        def make_fixed(n_batches, device):
            rng = np.random.default_rng(int(seed))
            idx = rng.permutation(len(train))[:n_batches*batch].reshape(n_batches, batch)
            out = []
            for row in idx:
                xs, ys = [], []
                for i in row:
                    x, y = train[int(i)]
                    xs.append(x); ys.append(y)
                out.append((torch.stack(xs,0).to(device), torch.tensor(ys, dtype=torch.long, device=device)))
            return out
    else:
        loader = DataLoader(train, batch_size=batch, shuffle=True, drop_last=True, num_workers=workers, pin_memory=pin)
        def make_fixed(n_batches, device):
            it = iter(loader); out = []
            while len(out) < n_batches:
                try:
                    x, y = next(it)
                except StopIteration:
                    it = iter(loader); x, y = next(it)
                out.append((x.to(device), y.to(device)))
            return out
    idx = torch.arange(min(n_eval, len(test)))
    eval_loader = DataLoader(Subset(test, idx), batch_size=eval_batch, shuffle=False, num_workers=workers, pin_memory=pin)
    return eval_loader, make_fixed

# ---------- Metrics ----------
def orthogonal_procrustes(A,B):
    A0 = A - A.mean(0, keepdim=True); B0 = B - B.mean(0, keepdim=True); M = B0.T @ A0
    U,_,Vt = torch.linalg.svd(M, full_matrices=False)
    return U @ Vt

def linear_cka_centered(X,Y):
    EPS = 1e-12
    Xc = X - X.mean(0, keepdim=True); Yc = Y - Y.mean(0, keepdim=True)
    Kx = Xc @ Xc.T; Ky = Yc @ Yc.T
    num = (Kx*Ky).sum()
    den = torch.linalg.norm(Kx)*torch.linalg.norm(Ky) + EPS
    v = (num/den).item()
    return float(max(0.0, min(1.0, v)))

def holonomy_metric(Z0, Z1):
    EPS = 1e-12
    with torch.no_grad():
        cka_pre = linear_cka_centered(Z0, Z1)
        Q = orthogonal_procrustes(Z0, Z1)
        Z1a = Z1 @ Q
        resid = (torch.linalg.norm(Z1a - Z0) / (torch.linalg.norm(Z0) + EPS)).item()
        cka_post = linear_cka_centered(Z0, Z1a)
    return {"procrustes_resid":resid, "cka_pre":cka_pre, "cka_post":cka_post, "cka_gap_post": 1.0-cka_post}

def pca_projected_resid(Z0, Z1, k=16):
    EPS = 1e-12
    with torch.no_grad():
        mu = Z0.mean(0, keepdim=True); X0 = Z0 - mu; Y1 = Z1 - mu
        Vt = torch.linalg.svd(X0, full_matrices=False)[2]
        Vk = Vt[:k, :].T
        X0k = X0 @ Vk; Y1k = Y1 @ Vk
        num = torch.linalg.norm(Y1k - X0k)
        den = torch.linalg.norm(X0k) + EPS
    return (num/den).item()

# ---------- Updates and loop ----------
def sgd_step(model, b, lr):
    model.train()
    for p in model.parameters():
        if p.grad is not None: p.grad = None
    x, y = b; logits,_ = model(x); loss = loss_fn(logits, y); loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None: p.add_(-lr * p.grad)

def sgd_inverse_step(model, b, lr):
    model.train()
    for p in model.parameters():
        if p.grad is not None: p.grad = None
    x, y = b; logits,_ = model(x); loss = loss_fn(logits, y); loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None: p.add_(+lr * p.grad)

def langevin_micro_step(model, b, lr_theta_part, T, seed_u64, commute=False):
    if commute: return
    model.train()
    for p in model.parameters():
        if p.grad is not None: p.grad = None
    x, y = b; logits,_ = model(x); loss = loss_fn(logits, y); loss.backward()
    sigma = math.sqrt(max(0.0, 2.0 * lr_theta_part * T))
    rng = np.random.default_rng(int(seed_u64 & ((1<<64)-1)))
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is None: continue
            p.add_(-lr_theta_part * p.grad)
            if sigma > 0.0:
                noise = torch.from_numpy(rng.normal(0.0, sigma, size=p.shape)).to(p)
                p.add_(noise)

def langevin_micro_inverse_step(model, b, lr_theta_part, T, seed_u64, commute=False):
    if commute: return
    model.train()
    for p in model.parameters():
        if p.grad is not None: p.grad = None
    x, y = b; logits,_ = model(x); loss = loss_fn(logits, y); loss.backward()
    sigma = math.sqrt(max(0.0, 2.0 * lr_theta_part * T))
    rng = np.random.default_rng(int(seed_u64 & ((1<<64)-1)))
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is None: continue
            if sigma > 0.0:
                noise = torch.from_numpy(rng.normal(0.0, sigma, size=p.shape)).to(p)
                p.add_(-noise)
            p.add_(+lr_theta_part * p.grad)

def run_loop_with_tape(model, r_batches, t_batches, lr_r, lr_theta, T, orientation="clockwise",
                       commute=False, theta_tape=None, substeps_per_pulse=1):
    rN = len(r_batches); tN = len(t_batches)
    if theta_tape is None or len(theta_tape) < tN * substeps_per_pulse:
        raise RuntimeError("theta_tape too short")
    def R_fwd(): [sgd_step(model, b, lr_r) for b in r_batches]
    def R_inv(): [sgd_inverse_step(model, b, lr_r) for b in reversed(r_batches)]
    def Th_fwd():
        g = substeps_per_pulse; lr_part = lr_theta / g
        for i, b in enumerate(t_batches):
            base = i*g
            for j in range(g): langevin_micro_step(model, b, lr_part, T, theta_tape[base+j], commute=commute)
    def Th_inv():
        g = substeps_per_pulse; lr_part = lr_theta / g
        for i in range(tN-1, -1, -1):
            b = t_batches[i]; base = i*g
            for j in range(g-1, -1, -1): langevin_micro_inverse_step(model, b, lr_part, T, theta_tape[base+j], commute=commute)
    if orientation == "clockwise":
        R_fwd(); Th_fwd(); R_inv(); Th_inv()
    else:
        Th_fwd(); R_fwd(); Th_inv(); R_inv()

# ---------- One condition at a FIXED POINT ----------
def compute_substeps_A(micro_shape, tape_scale):
    if micro_shape in ("balanced", "long_r"):
        return tape_scale
    if micro_shape in ("long_theta",):
        return max(1, tape_scale // 2)
    return tape_scale

def eval_features(model, loader, device="cpu"):
    model.eval(); feats = []
    with torch.no_grad():
        for x,_ in loader:
            x = x.to(device); _, z = model(x); feats.append(z.cpu())
    return torch.cat(feats, 0)

def eval_loss(model, loader, device="cpu"):
    model.eval(); tot = 0.0; n = 0
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device); y = y.to(device); logits,_ = model(x)
            tot += F.cross_entropy(logits, y, reduction="sum").item(); n += y.numel()
    return tot / max(1, n)

def one_condition_fixedpoint(device, eval_loader, make_batches_fn,
                             base_state_dict, area, r_pulses, theta_pulses, tape_scale,
                             micro_shape, lr_r, T, replicate,
                             pool, pca_k, model_kind="vit", optimizer_kind="sgd",
                             commute=False, tape_override_bytes=None):
    # Recreate the fixed-point model and eval baseline
    model0 = make_model(model_kind).to(device); model0.load_state_dict(base_state_dict)
    Z0 = eval_features(model0, eval_loader, device=device); L0 = eval_loss(model0, eval_loader, device=device)

    substeps_A = compute_substeps_A(micro_shape, tape_scale)
    r_batches = make_batches_fn(r_pulses, device); t_batches = make_batches_fn(theta_pulses, device)

    dense_substeps = theta_pulses * substeps_A; need_bytes = dense_substeps * 8
    if tape_override_bytes is None:
        label = f"{micro_shape}-rep{replicate}"
        jprint({"type":"qrng_reserve","label":label,"steps":dense_substeps,"bytes_per_step":8,"total_bytes":need_bytes})
        raw_tape = pool.fetch(need_bytes, label=label); jprint({"type":"qrng_ready","label":label,"bytes":len(raw_tape),"ts":int(time.time())})
    else:
        raw_tape = tape_override_bytes; jprint({"type":"qrng_reuse","label":f"{micro_shape}-FIXED","bytes":len(raw_tape),"ts":int(time.time())})

    theta_tape_cw = bytes_to_u64_list(raw_tape); theta_tape_ccw = list(reversed(theta_tape_cw))
    cw = make_model(model_kind).to(device); cw.load_state_dict(model0.state_dict())
    ccw = make_model(model_kind).to(device); ccw.load_state_dict(model0.state_dict())

    rN = len(r_batches); tN = len(t_batches)
    lr_theta = max(1e-12, area / (max(1,rN) * lr_r * max(1,tN) * max(1e-12, T)))
    run_loop_with_tape(cw, r_batches, t_batches, lr_r, lr_theta, T, "clockwise", commute, theta_tape_cw, substeps_A)
    run_loop_with_tape(ccw, r_batches, t_batches, lr_r, lr_theta, T, "counterclockwise", commute, theta_tape_ccw, substeps_A)

    Zcw = eval_features(cw, eval_loader, device=device); Zccw = eval_features(ccw, eval_loader, device=device)
    Lcw = eval_loss(cw, eval_loader, device=device); Lccw = eval_loss(ccw, eval_loader, device=device)
    m_cw = holonomy_metric(Z0, Zcw); m_ccw = holonomy_metric(Z0, Zccw)

    row = {
        "area":area,"r_pulses":r_pulses,"theta_pulses":theta_pulses,"micro_shape":micro_shape,
        "lr_r":lr_r,"lr_theta":lr_theta,"T":T,"substeps_A":substeps_A,"model":model_kind,"optimizer":optimizer_kind,
        "replicate":replicate,"tape_len":len(theta_tape_cw),"tape_scale":tape_scale,"baseline_loss":L0,
        "A_area_cw":area,
        "A_procrustes_delta": m_cw["procrustes_resid"] - m_ccw["procrustes_resid"],
        "A_projk_delta":       pca_projected_resid(Z0, Zcw, pca_k) - pca_projected_resid(Z0, Zccw, pca_k),
        "A_cka_post_gap_delta":m_cw["cka_gap_post"] - m_ccw["cka_gap_post"],
        "A_heat_per_area_delta": (Lcw - Lccw) / (area + 1e-12),
        "A_slope_per_area_procrustes": (m_cw["procrustes_resid"] - m_ccw["procrustes_resid"]) / (area + 1e-12),
        "A_slope_per_area_projk":       (pca_projected_resid(Z0, Zcw, pca_k) - pca_projected_resid(Z0, Zccw, pca_k)) / (area + 1e-12),
    }
    jprint({"type":"result_row","row":row,"ts":int(time.time())})
    return row

# ---------- Stats ----------
def binom_two_sided_p(k, n):
    # exact two-sided p for sign test under p=0.5
    if n == 0: return 1.0
    k = int(k); n = int(n)
    from math import comb
    tail = min(k, n-k)
    cdf = 0.0
    for i in range(0, tail+1):
        cdf += comb(n, i) / (2.0**n)
    p = 2.0 * cdf
    return min(1.0, p)

def summarize(rows, key_fields=("area","micro_shape"), value_fields=("A_slope_per_area_procrustes","A_slope_per_area_projk","A_heat_per_area_delta")):
    groups = defaultdict(list)
    for r in rows:
        key = tuple(r[k] for k in key_fields)
        groups[key].append(r)
    out = []
    for key, rs in groups.items():
        rec = {k:v for k,v in zip(key_fields, key)}
        rec["n"] = len(rs)
        for vf in value_fields:
            vals = [float(x.get(vf, float("nan"))) for x in rs]
            vals_clean = [v for v in vals if math.isfinite(v)]
            if not vals_clean:
                rec[f"{vf}_mean"] = float("nan"); rec[f"{vf}_std"] = float("nan"); rec[f"{vf}_p_sign"] = 1.0
                continue
            m = float(np.mean(vals_clean)); sd = float(np.std(vals_clean, ddof=1)) if len(vals_clean) > 1 else 0.0
            rec[f"{vf}_mean"] = m; rec[f"{vf}_std"] = sd
            kpos = sum(1 for v in vals_clean if v > 0.0)
            rec[f"{vf}_p_sign"] = binom_two_sided_p(kpos, len(vals_clean))
        out.append(rec)
    return out

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_csv(path, rows, fieldnames=None):
    if fieldnames is None:
        # collect union of keys
        keys = set()
        for r in rows: keys.update(r.keys())
        fieldnames = sorted(keys)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows: w.writerow(r)

# ---------- Argparse & experiment ----------
def parse_list_arg(s):
    s = (s or "").strip()
    return [t.strip() for t in s.split(",") if t.strip()] if s else []

def build_argparser():
    p = argparse.ArgumentParser(description="Holonomy (Transformer) fixed-point with summaries and reproducibility controls")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--eval-batch", type=int, default=1024)
    p.add_argument("--n-eval", type=int, default=2048)
    p.add_argument("--lr-r", type=float, default=3e-3)
    p.add_argument("--T", type=float, default=0.25)
    p.add_argument("--area", type=float, default=None)
    p.add_argument("--areas", type=str, default="")
    p.add_argument("--r-pulses", type=int, default=8)
    p.add_argument("--theta-pulses", type=int, default=8)
    p.add_argument("--pca-k", type=int, default=16)
    p.add_argument("--tape-scale", type=int, default=8)
    p.add_argument("--replicates", type=int, default=8)
    p.add_argument("--model", type=str, default="vit")
    p.add_argument("--optimizer", type=str, default="sgd")
    p.add_argument("--micro-shapes", type=str, default="balanced")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--pin", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--commute", action="store_true")
    p.add_argument("--fixed-tape", action="store_true")
    p.add_argument("--fixed-batches", action="store_true")
    p.add_argument("--preburn", type=int, default=50)
    p.add_argument("--rng", type=str, default="quantum", choices=["quantum","numpy","os","python"],
                   help="quantum uses Outshift+ANU; numpy/os/python are non-quantum fallbacks")
    p.add_argument("--out", type=str, default="", help="output prefix; will write <prefix>.jsonl, <prefix>.csv, <prefix>_summary.csv")
    return p

def get_pool(rng_mode, seed):
    if rng_mode == "quantum":
        return QuantumPool(CACHE_FILE)
    if rng_mode == "numpy":
        return PseudoPool(seed=seed)
    if rng_mode == "os":
        class _OSPool:
            def fetch(self, n, label=""): 
                b = os.urandom(n); jprint({"type":"rng_tick","src":"os.urandom","label":label,"bytes":len(b),"ts":int(time.time())}); return b
        return _OSPool()
    if rng_mode == "python":
        class _PyPool:
            def __init__(self, seed=seed):
                import random; self.rng = random.Random(seed)
            def fetch(self, n, label=""):
                b = bytes(self.rng.randrange(0,256) for _ in range(n))
                jprint({"type":"rng_tick","src":"python.Random","label":label,"bytes":len(b),"ts":int(time.time())})
                return b
        return _PyPool()
    return QuantumPool(CACHE_FILE)

def experiment(args):
    jprint({"type":"status","status":"starting","ts":int(time.time())})
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    set_seed(args.seed)

    areas = parse_list_arg(args.areas) if args.areas else ([float(args.area)] if args.area is not None else [3e-9])
    # normalize to floats
    areas = [float(a) for a in areas]
    shapes = parse_list_arg(args.micro_shapes) or ["balanced"]

    jprint({"type":"starting","ts":int(time.time())})
    jprint({"type":"config","run_id":secrets.token_hex(4),"device":device,"torch":torch.__version__,
            "qrng_mode":args.rng,"bits_per_step":64,"workers":args.workers,"pin":bool(args.pin),
            "tape_scale":args.tape_scale,"ts":int(time.time()),"model":args.model})

    # data + batch maker
    eval_loader, make_batches_fn = make_data(batch=args.batch, eval_batch=args.eval_batch, n_eval=args.n_eval,
                                             fixed_batches=args.fixed_batches, seed=args.seed,
                                             workers=args.workers, pin=args.pin)
    pool = get_pool(args.rng, args.seed)

    all_rows = []

    # optional fixed tapes per (shape, pulses, substeps)
    tape_bank = {}
    for shape in shapes:
        subA = compute_substeps_A(shape, args.tape_scale)
        dense = args.theta_pulses * subA
        key = (shape, args.theta_pulses, subA)
        if args.fixed_tape:
            label = f"{shape}-FIXED"; need = dense * 8
            jprint({"type":"qrng_reserve","label":label,"steps":dense,"bytes_per_step":8,"total_bytes":need})
            tape_bank[key] = pool.fetch(need, label=label)
            jprint({"type":"qrng_ready","label":label,"bytes":len(tape_bank[key]),"ts":int(time.time())})

        for area in areas:
            # ---------- FIX THE POINT: build one model and preburn ONCE for this (shape, area) ----------
            base = make_model(args.model).to(device)
            if args.preburn > 0:
                opt = torch.optim.SGD(base.parameters(), lr=0.05)
                for b in make_batches_fn(args.preburn, device):
                    logits,_ = base(b[0])
                    loss = loss_fn(logits, b[1])
                    opt.zero_grad(); loss.backward(); opt.step()
            base_state = base.state_dict()  # snapshot for all replicates at this point

            jprint({"type":"condition_start","ts":int(time.time()),"model":args.model,"optimizer":args.optimizer,
                    "micro_shape":shape,"area":area,"r_pulses":args.r_pulses,"theta_pulses":args.theta_pulses,
                    "substeps_A":"tbd","substeps_B":"tbd"})

            for rep in range(args.replicates):
                try:
                    tape = tape_bank.get(key) if args.fixed_tape else None
                    row = one_condition_fixedpoint(
                        device=device, eval_loader=eval_loader, make_batches_fn=make_batches_fn,
                        base_state_dict=base_state, area=float(area),
                        r_pulses=args.r_pulses, theta_pulses=args.theta_pulses, tape_scale=args.tape_scale,
                        micro_shape=shape, lr_r=args.lr_r, T=args.T, replicate=rep, pool=pool, pca_k=args.pca_k,
                        model_kind=args.model, optimizer_kind=args.optimizer, commute=args.commute,
                        tape_override_bytes=tape
                    )
                    all_rows.append(row)
                except QRNGInsufficientData as e:
                    jprint({"type":"error","error":"QRNGInsufficientData","message":str(e),"ts":int(time.time())})
                    break  # move on to next condition

    # summaries
    sums = summarize(all_rows)

    # outputs
    if args.out:
        base = args.out
        write_jsonl(base + ".jsonl", all_rows)
        write_csv(base + ".csv", all_rows)
        write_csv(base + "_summary.csv", sums)
        jprint({"type":"saved","rows":len(all_rows),"summary_rows":len(sums),
                "files":[base + ".jsonl", base + ".csv", base + "_summary.csv"]})

    jprint({"type":"done","summary":{"ok":True,"rows":len(all_rows),"summary_rows":len(sums)},"ts":int(time.time())})

if __name__ == "__main__":
    args = build_argparser().parse_args()
    try:
        experiment(args)
    except KeyboardInterrupt:
        jprint({"type":"exit","code":130,"ts":int(time.time())}); sys.exit(130)
    except Exception as e:
        import traceback
        tb = traceback.format_exc(limit=2)
        jprint({"type":"error","error":type(e).__name__,"message":str(e),"trace_hint":tb}); sys.exit(1)
