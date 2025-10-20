# holonomic_loop_training.py
# Holonomic loops in optimizer space: LR/noise/wd/aug cycled in a closed loop.
# Measures: accuracy, penultimate-layer linear CKA, Fisher-trace proxy.
# Usage:
#   python holonomic_loop_training.py --device cuda --epochs_per_phase 1 --loops 3
#   python holonomic_loop_training.py --device cpu  --subset 10000 --eval_batches 25
#   python holonomic_loop_training.py --grad_noise_scale 0.5 --aug_std 0.08 --fisher_batches 40

import math, random, argparse, time
from dataclasses import dataclass
from typing import Tuple, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import torchvision.transforms as T

# -------------------------
# Utils
# -------------------------
def set_seed(s=1337):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

@torch.no_grad()
def linear_cka(X, Y):
    """Centered Kernel Alignment for features (n x d)."""
    # center
    Xc = X - X.mean(0, keepdim=True)
    Yc = Y - Y.mean(0, keepdim=True)
    Kx = Xc @ Xc.t()
    Ky = Yc @ Yc.t()
    hsic = (Kx * Ky).sum()
    nx = torch.linalg.matrix_norm(Kx, ord='fro')
    ny = torch.linalg.matrix_norm(Ky, ord='fro')
    return (hsic / (nx * ny + 1e-8)).item()

def fisher_trace_proxy(model, dl, device, n_batches=50):
    """Cheap diagonal Fisher trace proxy with empirical gradients of NLL."""
    model.eval()
    trace = 0.0
    n = 0
    for i, (x,y) in enumerate(dl):
        if i >= n_batches: break
        x, y = x.to(device), y.to(device)
        model.zero_grad(set_to_none=True)
        logits, _ = model(x, return_feat=True)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        # accumulate squared grads
        t = 0.0
        for p in model.parameters():
            if p.grad is not None:
                t += (p.grad.detach()**2).sum().item()
        trace += t
        n += 1
    return trace / max(n,1)

# -------------------------
# Model
# -------------------------
class TinyCNN(nn.Module):
    def __init__(self, feat_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool  = nn.MaxPool2d(2,2)
        self.fc1   = nn.Linear(64*7*7, feat_dim)
        self.fc2   = nn.Linear(feat_dim, 10)

    def forward(self, x, return_feat=False):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))    # 64 x 7 x 7
        x = x.flatten(1)
        feat = F.relu(self.fc1(x))
        logits = self.fc2(feat)
        return (logits, feat) if return_feat else logits

# -------------------------
# Noise injection (SGLD-ish)
# -------------------------
@torch.no_grad()
def add_param_noise(model, scale):
    if scale <= 0: return
    for p in model.parameters():
        if p.requires_grad:
            p.add_(torch.randn_like(p) * scale)

# -------------------------
# Training/Eval
# -------------------------
def eval_metrics(model, dl, device, cache_feats=False, max_batches=None):
    model.eval()
    correct = 0; total = 0
    feats = []
    with torch.no_grad():
        for i,(x,y) in enumerate(dl):
            if max_batches is not None and i >= max_batches: break
            x, y = x.to(device), y.to(device)
            logits, f = model(x, return_feat=True)
            pred = logits.argmax(1)
            correct += (pred==y).sum().item()
            total += y.size(0)
            if cache_feats: feats.append(f.detach().cpu())
    acc = correct / max(1,total)
    Fmat = torch.cat(feats,0) if cache_feats and len(feats)>0 else None
    return acc, Fmat

def run_phase(
    model,
    opt,
    dl,
    device,
    epochs,
    aug,
    grad_noise,
    lr,
    wd,
    desc,
    aug_std,
    param_noise_scale,
):
    # set optimizer params
    for g in opt.param_groups:
        g['lr'] = lr; g['weight_decay'] = wd
    model.train()
    for ep in range(epochs):
        for x,y in dl:
            x,y = x.to(device), y.to(device)
            if aug:
                x = x + aug_std * torch.randn_like(x)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            # gradient noise
            if grad_noise>0:
                with torch.no_grad():
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.add_(torch.randn_like(p.grad)*grad_noise)
            opt.step()
        # small param noise to mimic high-T weight diffusion
        add_param_noise(model, scale=grad_noise * param_noise_scale)

# -------------------------
# Holonomic schedule
# -------------------------
@dataclass
class Phase:
    name: str
    epochs: int
    lr: float
    wd: float
    grad_noise: float
    aug: bool

def holonomic_cycle(forward=True, grad_noise_scale=1.0):
    # One closed loop: start & end hyperparams identical
    phases_fwd = [
        Phase("compress", epochs=1, lr=1e-3, wd=2e-3, grad_noise=0.0,  aug=False),
        Phase("explore",  epochs=1, lr=5e-4, wd=1e-4, grad_noise=3e-3, aug=True),
        Phase("align",    epochs=1, lr=2e-3, wd=1e-4, grad_noise=5e-4, aug=False),
        Phase("seal",     epochs=1, lr=1e-3, wd=2e-3, grad_noise=0.0,  aug=False),
    ]
    phases = phases_fwd if forward else list(reversed(phases_fwd))
    if grad_noise_scale != 1.0:
        for p in phases:
            p.grad_noise *= grad_noise_scale
    return phases

# -------------------------
# Data
# -------------------------
def get_data(batch_size=256, subset=None):
    tfm = T.Compose([T.ToTensor()])
    train = tv.datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test  = tv.datasets.MNIST(root="./data", train=False, download=True, transform=tfm)
    if subset is not None and subset < len(train):
        train = torch.utils.data.Subset(train, list(range(subset)))
    if subset is not None and subset < len(test):
        test = torch.utils.data.Subset(test, list(range(subset//6)))
    trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    testloader  = torch.utils.data.DataLoader(test,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return trainloader, testloader

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--feat_dim", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--loops", type=int, default=2, help="number of forward loops (and same number of reverse)")
    ap.add_argument("--epochs_per_phase", type=int, default=1)
    ap.add_argument("--subset", type=int, default=20000, help="subset of train for speed")
    ap.add_argument("--grad_noise_scale", type=float, default=1.0, help="scale multiplier on all phase grad_noise values")
    ap.add_argument(
        "--param_noise_scale",
        type=float,
        default=0.1,
        help="multiplier mapping grad_noise to post-epoch parameter noise amplitude",
    )
    ap.add_argument("--aug_std", type=float, default=0.05, help="stddev of Gaussian augmentation noise when enabled")
    ap.add_argument(
        "--eval_batches",
        type=int,
        default=50,
        help="number of batches to use for eval snapshots (<=0 means full dataset)",
    )
    ap.add_argument(
        "--fisher_batches",
        type=int,
        default=20,
        help="mini-batches to use when estimating Fisher trace during loops",
    )
    ap.add_argument(
        "--fisher_init_batches",
        type=int,
        default=30,
        help="mini-batches to use for the initial Fisher trace baseline",
    )
    args = ap.parse_args()
    set_seed(1337)
    device = torch.device(args.device)

    trainloader, testloader = get_data(args.batch_size, subset=args.subset)
    model = TinyCNN(feat_dim=args.feat_dim).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)

    # Baseline metrics
    max_eval = None if args.eval_batches <= 0 else args.eval_batches
    base_acc, base_feats = eval_metrics(model, testloader, device, cache_feats=True, max_batches=max_eval)
    base_fisher = fisher_trace_proxy(model, trainloader, device, n_batches=args.fisher_init_batches)
    print(f"[INIT] acc={base_acc:.4f} fisher≈{base_fisher:.3e}")

    def run_loops(direction:str, loops:int, *, base_feats: torch.Tensor, base_acc: float):
        # direction: 'forward' or 'reverse'
        phases = holonomic_cycle(forward=(direction=='forward'), grad_noise_scale=args.grad_noise_scale)
        # scale epochs if requested
        for p in phases: p.epochs = args.epochs_per_phase
        all_log = []
        for k in range(loops):
            print(f"\n== {direction.upper()} LOOP {k+1}/{loops} ==")
            # snapshot features pre-loop
            _, feats_pre = eval_metrics(model, testloader, device, cache_feats=True, max_batches=max_eval)
            fisher_pre = fisher_trace_proxy(model, trainloader, device, n_batches=args.fisher_batches)
            for p in phases:
                t0 = time.time()
                run_phase(
                    model,
                    opt,
                    trainloader,
                    device,
                    p.epochs,
                    p.aug,
                    p.grad_noise,
                    p.lr,
                    p.wd,
                    p.name,
                    args.aug_std,
                    args.param_noise_scale,
                )
                dt = time.time()-t0
                acc_mid, _ = eval_metrics(model, testloader, device, cache_feats=False, max_batches=max_eval)
                aug_desc = f"{args.aug_std:.2f}" if p.aug else "0.00"
                print(
                    f"  - phase {p.name:9s}: lr={p.lr:.1e} wd={p.wd:.1e} noise={p.grad_noise:.1e} "
                    f"aug_std={aug_desc} | acc={acc_mid:.4f} ({dt:.1f}s)"
                )
            # snapshot post-loop
            acc_post, feats_post = eval_metrics(model, testloader, device, cache_feats=True, max_batches=max_eval)
            fisher_post = fisher_trace_proxy(model, trainloader, device, n_batches=args.fisher_batches)
            cka = linear_cka(feats_pre, feats_post)
            cka_pre_base = linear_cka(base_feats, feats_pre)
            cka_post_base = linear_cka(base_feats, feats_post)
            fisher_delta = fisher_post - fisher_pre
            acc_delta = acc_post - base_acc
            socioception = acc_delta
            cosmoception = cka_post_base - cka_pre_base
            cyberception = fisher_delta
            log = dict(
                direction=direction,
                loop=k+1,
                acc=acc_post,
                cka=cka,
                fisher_delta=fisher_delta,
                cka_pre_base=cka_pre_base,
                cka_post_base=cka_post_base,
                acc_delta=acc_delta,
                socioception=socioception,
                cosmoception=cosmoception,
                cyberception=cyberception,
            )
            print(
                f"== /{direction} loop {k+1}: acc={acc_post:.4f} | "
                f"CKA(pre→post)={cka:.4f} | CKA(base→pre)={cka_pre_base:.4f} | "
                f"CKA(base→post)={cka_post_base:.4f} | ΔFisher≈{fisher_delta:.3e}"
            )
            all_log.append(log)
        return all_log

    fwd = run_loops("forward", args.loops, base_feats=base_feats, base_acc=base_acc)
    rev = run_loops("reverse", args.loops, base_feats=base_feats, base_acc=base_acc)

    # Compare net effect of one forward loop vs one reverse loop (holonomy test)
    def summarize(tag, logs):
        def mean(key):
            vals = [x[key] for x in logs]
            return sum(vals) / len(vals) if vals else float('nan')

        return dict(
            tag=tag,
            acc_mean=mean('acc'),
            cka_mean=mean('cka'),
            fisher_delta_mean=mean('fisher_delta'),
            cka_pre_base_mean=mean('cka_pre_base'),
            cka_post_base_mean=mean('cka_post_base'),
            acc_delta_mean=mean('acc_delta'),
            socioception_mean=mean('socioception'),
            cosmoception_mean=mean('cosmoception'),
            cyberception_mean=mean('cyberception'),
        )
    s_fwd = summarize("forward", fwd)
    s_rev = summarize("reverse", rev)

    print("\n==== SUMMARY (Holonomy signal) ====")
    print(
        "Forward:  "
        f"acc_mean={s_fwd['acc_mean']:.4f}  accΔ_mean={s_fwd['acc_delta_mean']:.4f}  "
        f"CKA_mean={s_fwd['cka_mean']:.4f}  "
        f"CKA(base→pre)_mean={s_fwd['cka_pre_base_mean']:.4f}  "
        f"CKA(base→post)_mean={s_fwd['cka_post_base_mean']:.4f}  "
        f"ΔFisher_mean≈{s_fwd['fisher_delta_mean']:.3e}"
    )
    print(
        "Reverse:  "
        f"acc_mean={s_rev['acc_mean']:.4f}  accΔ_mean={s_rev['acc_delta_mean']:.4f}  "
        f"CKA_mean={s_rev['cka_mean']:.4f}  "
        f"CKA(base→pre)_mean={s_rev['cka_pre_base_mean']:.4f}  "
        f"CKA(base→post)_mean={s_rev['cka_post_base_mean']:.4f}  "
        f"ΔFisher_mean≈{s_rev['fisher_delta_mean']:.3e}"
    )

    print("\nTriadic senses (mean over loops):")
    print(
        "  Forward →"
        f" socioception={s_fwd['socioception_mean']:+.4e}"
        f"  cosmoception={s_fwd['cosmoception_mean']:+.4e}"
        f"  cyberception={s_fwd['cyberception_mean']:+.4e}"
    )
    print(
        "  Reverse →"
        f" socioception={s_rev['socioception_mean']:+.4e}"
        f"  cosmoception={s_rev['cosmoception_mean']:+.4e}"
        f"  cyberception={s_rev['cyberception_mean']:+.4e}"
    )

    holonomy_vector = {
        'acc_mean': s_fwd['acc_mean'] - s_rev['acc_mean'],
        'acc_delta_mean': s_fwd['acc_delta_mean'] - s_rev['acc_delta_mean'],
        'cka_mean': s_fwd['cka_mean'] - s_rev['cka_mean'],
        'cka_base_pre_mean': s_fwd['cka_pre_base_mean'] - s_rev['cka_pre_base_mean'],
        'cka_base_post_mean': s_fwd['cka_post_base_mean'] - s_rev['cka_post_base_mean'],
        'fisher_delta_mean': s_fwd['fisher_delta_mean'] - s_rev['fisher_delta_mean'],
        'socioception_mean': s_fwd['socioception_mean'] - s_rev['socioception_mean'],
        'cosmoception_mean': s_fwd['cosmoception_mean'] - s_rev['cosmoception_mean'],
        'cyberception_mean': s_fwd['cyberception_mean'] - s_rev['cyberception_mean'],
    }
    holonomy_norm = math.sqrt(sum(v * v for v in holonomy_vector.values()))
    print("Holonomy vector (forward - reverse):")
    for key, value in holonomy_vector.items():
        print(f"  {key:>18s} = {value:+.4e}")
    print(f"  ||vector||₂ = {holonomy_norm:.4e}")
    print("If holonomy is real, forward vs reverse loops will differ (CKA/ΔFisher/acc). Same pointwise dials, different net phase.\n")

if __name__ == "__main__":
    main()