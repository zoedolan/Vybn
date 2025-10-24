#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vybn® — "show, don't tell" empirical script
===========================================

This file is a single-button demonstration of three claims that recur across the Vybn corpus:

1) SU(2)×mod‑24 is a PATH instrument on S^3: the residue-indexed trivializations step along
   the 24‑cell adjacency rather than jumping to antipodes (D4‑glue lives in the edges).
2) Small non‑abelian loops obey an area law on SU(2): a square made of i/j rotations yields
   a k‑axis rotation of angle ≈ 2 a^2 (BCH / non‑abelian Stokes in quaternion clothing).
3) Gödel curvature appears whenever you alternate "update" with "projection": a tiny 2‑atom
   model returns to its exact ensemble but not to its compressed state, generating a measurable
   loop residue and strictly positive KL "heat".

Minimal deps (pure Python). If 'state_sum.py' is present in the same folder we import it
to mirror v0.1 seeding; otherwise we use a local fallback with the same definitions.

Usage examples:
    python vybn_show_dont_tell.py --all
    python vybn_show_dont_tell.py --d4 --area --godel --out ./_artifacts

Outputs:
    out/d4/*.json, *.csv   — residue→vertex map and 24‑cell scores
    out/area/*.csv         — angle vs 2a^2 calibration
    out/godel/*.json       — loop residue and heat, plus grid scans
"""

import os, json, math, csv, argparse
import statistics
from typing import Dict, Tuple, List

# ---------- Quaternion / SU(2) utilities ----------

EPS = 1e-12

class Q:
    __slots__ = ("w","x","y","z")
    def __init__(self,w,x,y,z):
        self.w=float(w); self.x=float(x); self.y=float(y); self.z=float(z)
    def __mul__(a,b):
        return Q(
            a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
            a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
            a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
            a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w,
        )
    def conj(self): return Q(self.w,-self.x,-self.y,-self.z)
    def norm(self): return (self.w*self.w + self.x*self.x + self.y*self.y + self.z*self.z)**0.5
    def normalize(self):
        n = self.norm()
        if n < EPS: return Q(1,0,0,0)
        return Q(self.w/n, self.x/n, self.y/n, self.z/n)
    def rounded_key(self, places=6):
        return (round(self.w,places), round(self.x,places), round(self.y,places), round(self.z,places))

def q_id(): return Q(1,0,0,0)
def q_inv(q:Q): return q.conj()
def q_axis_angle(nx,ny,nz,theta):
    n = (nx*nx+ny*ny+nz*nz)**0.5
    if n < EPS: return q_id()
    nx/=n; ny/=n; nz/=n
    h = 0.5*theta
    return Q(math.cos(h), nx*math.sin(h), ny*math.sin(h), nz*math.sin(h))

def su2_geodesic_angle(q:Q):
    # Minimal geodesic angle on SU(2): θ = 2 arccos(|w|)
    w = max(-1.0, min(1.0, q.normalize().w))
    return 2.0*math.acos(abs(w))

def pow_q(q:Q,k:int):
    if k==0: return q_id()
    if k<0: return pow_q(q_inv(q), -k)
    out = q_id()
    for _ in range(k):
        out = out*q
    return out

# ---------- Hurwitz 24 and 24‑cell adjacency ----------

def hurwitz_vertices()->List[Q]:
    V = []
    # 8 coordinate units
    V += [Q(1,0,0,0),Q(-1,0,0,0),Q(0,1,0,0),Q(0,-1,0,0),Q(0,0,1,0),Q(0,0,-1,0),Q(0,0,0,1),Q(0,0,0,-1)]
    # 16 half‑units
    for sw in (-1,1):
        for sx in (-1,1):
            for sy in (-1,1):
                for sz in (-1,1):
                    V.append(Q(0.5*sw,0.5*sx,0.5*sy,0.5*sz))
    # unique, normalized
    uniq = {}
    for q in V:
        u = q.normalize()
        uniq[u.rounded_key()] = u
    return list(uniq.values())

def is_coordinate(q:Q, tol=1e-9):
    comps = [abs(q.w),abs(q.x),abs(q.y),abs(q.z)]
    return sum(abs(c-1.0)<tol for c in comps)==1 and sum(c<tol for c in comps if abs(c-1.0)>=tol)==3

def is_half(q:Q, tol=1e-9):
    comps = [abs(q.w),abs(q.x),abs(q.y),abs(q.z)]
    return all(abs(c-0.5)<tol for c in comps)

def dot4(a:Q,b:Q): return a.w*b.w + a.x*b.x + a.y*b.y + a.z*b.z

def sign_tuple_half(q:Q):
    def sgn(v): return 1 if v>0 else (-1 if v<0 else 0)
    return (sgn(q.w),sgn(q.x),sgn(q.y),sgn(q.z))

def build_24cell_adjacency():
    V = hurwitz_vertices()
    idx = {v.rounded_key():i for i,v in enumerate(V)}
    edges=set()
    # Coordinate↔Half: connect when dot = +1/2
    for i,v in enumerate(V):
        if is_coordinate(v):
            for j,u in enumerate(V):
                if is_half(u) and abs(dot4(v,u)-0.5)<1e-9:
                    a,b=min(i,j),max(i,j); edges.add((a,b))
    # Half↔Half: Hamming distance 1 in sign tuples
    half = [(i,sign_tuple_half(v)) for i,v in enumerate(V) if is_half(v)]
    for a,(i,si) in enumerate(half):
        for j,(k,sk) in enumerate(half[a+1:], start=a+1):
            hdist = sum(int(si[t]!=sk[t]) for t in range(4))
            if hdist==1:
                a1,b1=min(i,k),max(i,k); edges.add((a1,b1))
    adj = {i:set() for i in range(len(V))}
    for a,b in edges:
        adj[a].add(b); adj[b].add(a)
    degs=[len(adj[i]) for i in range(len(V))]
    return V, adj, degs, len(edges)

# ---------- Optional import of v0.1 seeding ----------
def try_import_state_sum():
    try:
        import importlib.util, sys, os
        here = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(here, "state_sum.py")
        if os.path.exists(candidate):
            spec = importlib.util.spec_from_file_location("state_sum", candidate)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["state_sum"] = mod
            spec.loader.exec_module(mod)
            return mod
    except Exception:
        return None
    return None

# Local fallback mirrors v0.1 choices
def fallback_generate_2T(maxlen=8):
    s = Q(0.5,0.5,0.5,0.5)     # 120° around (1,1,1)
    t = Q(0.5,0.5,0.5,-0.5)    # 120° around (1,1,-1)
    gens=[s,t,q_inv(s),q_inv(t)]
    seen={q_id().rounded_key(): q_id()}
    frontier=[q_id()]
    for _ in range(maxlen):
        new=[]
        for g in frontier:
            for h in gens:
                u=(g*h).normalize()
                k=u.rounded_key()
                if k not in seen:
                    seen[k]=u; new.append(u)
        frontier=new
        if not new: break
    G=list(seen.values())
    if len(G)<24 and maxlen<12: return fallback_generate_2T(maxlen=maxlen+2)
    return G

def fallback_seed_loop_for_residue(r:int):
    s = Q(0.5,0.5,0.5,0.5)
    t = Q(0.5,0.5,0.5,-0.5)
    a = 1 + (r % 3)
    b = 1 + ((r // 3) % 3)
    c = 1 + ((r // 9) % 3)
    U = (pow_q(s,a) * pow_q(t,b) * pow_q(s,c)).normalize()
    m = r % 3
    axis = (1,0,0) if m==0 else ((0,1,0) if m==1 else (0,0,1))
    return U, axis

def state_sum_map(beta=25.0, maxlen=8, epsilon_deg=5.0):
    mod = try_import_state_sum()
    if mod is None:
        G = fallback_generate_2T(maxlen=maxlen)
        residues = list(range(1,24,2))
        eps = math.radians(epsilon_deg)
        mapping={}
        for r in residues:
            U, axis = fallback_seed_loop_for_residue(r)
            twist = q_axis_angle(axis[0],axis[1],axis[2], eps*(1+(r%2)))
            Ur=(twist*U).normalize()
            best_g=None; best_cost=1e9; best_ang=None
            for g in G:
                q=(Ur*g).normalize()
                ang=su2_geodesic_angle(q)
                c=ang*ang
                if c<best_cost-1e-15:
                    best_cost=c; best_ang=ang; best_g=g
            mapping[r]={"best_g":best_g, "angle_deg":best_ang*180.0/math.pi}
        return mapping
    else:
        # Use exact v0.1 logic
        results = mod.state_sum(beta=beta, maxlen=maxlen, epsilon_deg=epsilon_deg)
        # Convert labels to quaternions for adjacency scoring; use nearest in Hurwitz set
        H = hurwitz_vertices()
        def nearest(label_q:Q):
            # nearest among H
            best=None; bd=1e9
            for v in H:
                d=((label_q.w-v.w)**2 + (label_q.x-v.x)**2 + (label_q.y-v.y)**2 + (label_q.z-v.z)**2)**0.5
                if d<bd: bd=d; best=v
            return best
        mapping={}
        # state_sum returns labels, but we can recompute best_g by label lookup using mod.hurwitz_units
        # Fall back to using mod.generate_2T search to reconstruct the quaternion corresponding to each label.
        G = mod.generate_2T(maxlen=maxlen)
        residues = list(range(1,24,2))
        eps = math.radians(epsilon_deg)
        for r in residues:
            U, axis = mod.seed_loop_for_residue(r)
            twist = mod.q_axis_angle(axis[0],axis[1],axis[2], eps*(1+(r%2)))
            Ur=(twist*U).normalize()
            best_g=None; best_cost=1e9; best_ang=None
            for g in G:
                q=(Ur*g).normalize()
                ang=mod.su2_geodesic_angle(q)
                c=ang*ang
                if c<best_cost-1e-15:
                    best_cost=c; best_ang=ang; best_g=g
            mapping[r]={"best_g":Q(best_g.w,best_g.x,best_g.y,best_g.z), "angle_deg":best_ang*180.0/math.pi}
        return mapping

# ---------- (A) D4‑glue: adjacency vs antipodes ----------

def analyze_d4(mapping:Dict[int,Dict], outdir:str):
    V, adj, degs, nedges = build_24cell_adjacency()
    def nearest_index(q:Q):
        best=None; bd=1e9; bi=None
        for i,v in enumerate(V):
            d=((q.w-v.w)**2 + (q.x-v.x)**2 + (q.y-v.y)**2 + (q.z-v.z)**2)**0.5
            if d<bd: bd=d; bi=i; best=v
        return bi, bd
    # residue→vertex index
    r2i = {}
    for r,d in mapping.items():
        i,_ = nearest_index(d["best_g"])
        r2i[r]=i
    # scores
    consec = [(r, r+2) for r in range(1,23,2)]
    adj_hits = sum(1 for a,b in consec if r2i[b] in adj[r2i[a]])
    adj_score = adj_hits/len(consec)
    anti_pairs = [(r, 24-r) for r in range(1,12,2)]
    def q_neg(q:Q): return Q(-q.w,-q.x,-q.y,-q.z)
    anti_hits=0
    for a,b in anti_pairs:
        qa=mapping[a]["best_g"]; qb=mapping[b]["best_g"]
        if abs(qb.w + qa.w)<1e-6 and abs(qb.x + qa.x)<1e-6 and abs(qb.y + qa.y)<1e-6 and abs(qb.z + qa.z)<1e-6:
            anti_hits+=1
    anti_score = anti_hits/len(anti_pairs)
    # coverage
    used = sorted(set(r2i.values()))
    cov = len(used)/24.0
    # write artifacts
    os.makedirs(outdir, exist_ok=True)
    summary = {
        "vertex_degrees_first8": [degs[i] for i in range(8)],
        "num_edges": nedges,
        "coverage_ratio": cov,
        "adjacency_score_consecutive_residues": adj_score,
        "antipodal_pairing_score_r_vs_24_minus_r": anti_score,
        "mapping": {str(r): {"vertex_index": r2i[r], "angle_deg": mapping[r]["angle_deg"]} for r in sorted(mapping)},
    }
    with open(os.path.join(outdir,"d4_summary.json"),"w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(outdir,"d4_mapping.csv"),"w",newline="") as f:
        w=csv.writer(f); w.writerow(["residue","vertex_index","angle_deg"])
        for r in sorted(mapping):
            w.writerow([r, r2i[r], f"{mapping[r]['angle_deg']:.6f}"])
    return adj_score, anti_score, cov

# ---------- (B) Area law on SU(2) ----------

def holonomy_square(a:float)->Q:
    # Q = exp(i a) exp(j a) exp(-i a) exp(-j a)
    I = q_axis_angle(1,0,0, a)
    J = q_axis_angle(0,1,0, a)
    return (I * J * q_inv(I) * q_inv(J)).normalize()

def area_law_scan(outdir:str):
    os.makedirs(outdir, exist_ok=True)
    rows=[]
    for a in [k/1000.0 for k in range(5,51,1)]:  # 0.005..0.050
        q=holonomy_square(a)
        ang=su2_geodesic_angle(q)
        ratio = ang / (a*a) if a>0 else float("nan")
        rows.append((a, ang, ratio))
    with open(os.path.join(outdir,"area_law.csv"),"w",newline="") as f:
        w=csv.writer(f); w.writerow(["a","angle","angle_over_a2"])
        for a,ang,ratio in rows:
            w.writerow([f"{a:.6f}", f"{ang:.9f}", f"{ratio:.6f}"])
    # Simple summary: median ratio near 1
    median_ratio = statistics.median(r for _,_,r in rows)
    return median_ratio

# ---------- (C) Gödel curvature: update⊚project loop ----------

def softmax(weights):
    m = max(weights)
    exps = [math.exp(w-m) for w in weights]
    Z = sum(exps)
    return [e/Z for e in exps]

def godel_loop(eps=0.1, delta=0.1, outdir:str="godel"):
    # Universe Ω = {(a,b)∈{0,1}^2}; exact reference q = uniform; compressed family f tracks product of marginals
    states=[(0,0),(0,1),(1,0),(1,1)]
    def energy(a,b, lam_parity, lam_a):
        phi_xor = 1.0 if (a ^ b)==1 else 0.0
        return lam_parity*phi_xor + lam_a*(1.0 if a==1 else 0.0)
    def tilt(r, lam_parity, lam_a):
        weights=[math.log(r[i])+energy(a,b,lam_parity,lam_a) for i,(a,b) in enumerate(states)]
        return softmax(weights)
    def project_to_product(r):
        # m‑projection onto independent Bernoulli(a)*Bernoulli(b)
        p_a = r[2] + r[3]
        p_b = r[1] + r[3]
        return [(1-p_a)*(1-p_b), (1-p_a)*p_b, p_a*(1-p_b), p_a*p_b]
    def KL(p,q): return sum(0 if p[i]==0 else p[i]*(math.log(p[i]) - math.log(max(q[i],1e-300))) for i in range(4))
    # Start at uniform compressed
    p = [0.25]*4
    Qheat=0.0
    # 1) +ε parity
    r = tilt(p, eps, 0.0); p = project_to_product(r); Qheat += KL(r,p)
    # 2) +δ a
    r = tilt(p, 0.0, delta); p = project_to_product(r); Qheat += KL(r,p)
    # 3) -ε parity
    r = tilt(p, -eps, 0.0); p = project_to_product(r); Qheat += KL(r,p)
    # 4) -δ a
    r = tilt(p, 0.0, -delta); p = project_to_product(r); Qheat += KL(r,p)
    # Return b‑marginal
    p_b = p[1] + p[3]
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "godel_summary.json"),"w") as f:
        json.dump({"eps":eps,"delta":delta,"p_b":p_b,"Q_heat_nats":Qheat}, f, indent=2)
    return p_b, Qheat

def godel_grid(outdir:str):
    os.makedirs(outdir, exist_ok=True)
    rows=[]
    for eps in [k/100.0 for k in range(5,31,5)]:      # 0.05..0.30
        for delta in [k/100.0 for k in range(5,31,5)]:# 0.05..0.30
            pb, Qh = godel_loop(eps, delta, outdir)
            rows.append((eps,delta,pb,Qh))
    with open(os.path.join(outdir,"godel_grid.csv"),"w",newline="") as f:
        w=csv.writer(f); w.writerow(["eps","delta","p_b","Q_heat_nats"])
        for row in rows: w.writerow([f"{row[0]:.3f}",f"{row[1]:.3f}",f"{row[2]:.9f}",f"{row[3]:.9f}"])
    # estimate slope κ in p_b ≈ 1/2 + κ εδ
    # Use least squares through origin on y = (p_b-1/2) vs (eps*delta)
    xs=[row[0]*row[1] for row in rows]
    ys=[row[2]-0.5 for row in rows]
    num=sum(x*y for x,y in zip(xs,ys))
    den=sum(x*x for x in xs)
    kappa = num/den if den>0 else float('nan')
    return kappa

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="./out", help="output directory")
    ap.add_argument("--d4", action="store_true", help="run SU(2)×mod‑24 D4‑glue adjacency probe")
    ap.add_argument("--area", action="store_true", help="run SU(2) area‑law square‑loop scan")
    ap.add_argument("--godel", action="store_true", help="run Gödel‑curvature update⊚project loop")
    ap.add_argument("--all", action="store_true", help="run everything")
    args = ap.parse_args()
    if args.all: args.d4=args.area=args.godel=True
    os.makedirs(args.out, exist_ok=True)

    if args.d4:
        print("[D4] Running SU(2)×mod‑24 state‑sum mapping and 24‑cell scoring…")
        mapping = state_sum_map(beta=25.0, maxlen=8, epsilon_deg=5.0)
        d4_dir = os.path.join(args.out,"d4"); os.makedirs(d4_dir, exist_ok=True)
        adj_score, anti_score, cov = analyze_d4(mapping, d4_dir)
        print(f"[D4] adjacency score (r→r+2): {adj_score:.3f}; antipodal score (r↔24−r): {anti_score:.3f}; coverage: {cov*100:.1f}%")

    if args.area:
        print("[AREA] Scanning square‑loop holonomy angle vs 2 a^2…")
        area_dir = os.path.join(args.out,"area"); os.makedirs(area_dir, exist_ok=True)
        med = area_law_scan(area_dir)
        print(f"[AREA] median(angle / a^2) ≈ {med:.4f} (→ 1.0 = ideal)")

    if args.godel:
        print("[GÖDEL] Running loop with ε=δ=0.1; then grid scan for κ in p_b ≈ 1/2 + κ εδ …")
        godel_dir = os.path.join(args.out,"godel"); os.makedirs(godel_dir, exist_ok=True)
        pb, Qh = godel_loop(0.1,0.1,godel_dir)
        kappa = godel_grid(godel_dir)
        print(f"[GÖDEL] p_b(ε=δ=0.1) = {pb:.9f}; heat = {Qh:.6f} nats; estimated κ ≈ {kappa:.3f} (theory: 1/8 = 0.125)")

if __name__ == "__main__":
    main()
