# Vybn — simulation harness to probe three load-bearing claims
# 1) SU(2) small-loop area law on the square commutator
# 2) SU(2)×mod-24 "path" behavior (24‑cell adjacency vs antipodes)
# 3) Gödel curvature in update∘project loops (2‑atom example)
# 4) Z3 double‑cover check: 120° rotations close in SO(3) after 3 steps, in SU(2) after 6

import os, math, json, csv, statistics
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

OUTROOT = "/mnt/data/vybn_sim"
os.makedirs(OUTROOT, exist_ok=True)

# ---------- Quaternion / SU(2) utilities ----------

EPS = 1e-12

@dataclass
class Q:
    w: float
    x: float
    y: float
    z: float

    def __mul__(a, b):
        return Q(
            a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
            a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
            a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
            a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w,
        )

    def conj(self):
        return Q(self.w, -self.x, -self.y, -self.z)

    def norm(self):
        return (self.w*self.w + self.x*self.x + self.y*self.y + self.z*self.z)**0.5

    def normalize(self):
        n = self.norm()
        if n < EPS:
            return Q(1,0,0,0)
        return Q(self.w/n, self.x/n, self.y/n, self.z/n)

def q_id():
    return Q(1,0,0,0)

def q_axis_angle(nx, ny, nz, theta):
    n = (nx*nx+ny*ny+nz*nz)**0.5
    if n < EPS: 
        return q_id()
    nx/=n; ny/=n; nz/=n
    h = 0.5*theta
    return Q(math.cos(h), nx*math.sin(h), ny*math.sin(h), nz*math.sin(h))

def su2_angle(q: Q):
    # Minimal geodesic angle on SU(2): θ = 2 arccos(|w|) for unit quaternion
    w = max(-1.0, min(1.0, q.normalize().w))
    return 2.0*math.acos(abs(w))

# ---------- (1) Area law on SU(2) square loop ----------

def holonomy_square(a: float) -> Q:
    # Q = exp(i a) exp(j a) exp(-i a) exp(-j a)
    I = q_axis_angle(1,0,0,a)
    J = q_axis_angle(0,1,0,a)
    return (I * J * I.conj() * J.conj()).normalize()

def run_area_law_scan():
    outdir = os.path.join(OUTROOT, "area")
    os.makedirs(outdir, exist_ok=True)
    rows = []
    a_vals = [k/1000.0 for k in range(5, 51, 1)]  # 0.005..0.050
    for a in a_vals:
        q = holonomy_square(a)
        ang = su2_angle(q)
        ratio = ang / (a*a)
        rows.append((a, ang, ratio))
    # write CSV
    csv_path = os.path.join(outdir, "area_law.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["a", "angle", "angle_over_a2"])
        for a,ang,ratio in rows:
            w.writerow([f"{a:.6f}", f"{ang:.9f}", f"{ratio:.6f}"])
    # plot ratio vs a
    plt.figure()
    plt.plot([r[0] for r in rows], [r[2] for r in rows])
    plt.xlabel("a (radians)")
    plt.ylabel("angle / a^2")
    plt.title("SU(2) square-loop holonomy: angle ≈ κ a^2  (κ≈const)")
    fig_path = os.path.join(outdir, "area_law_ratio.png")
    plt.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close()
    median_ratio = statistics.median(r[2] for r in rows)
    return {"csv": csv_path, "plot": fig_path, "median_ratio": median_ratio}

# ---------- (2) 24‑cell vertices and adjacency ----------

def hurwitz_vertices() -> List[Q]:
    V = []
    # 8 coordinate units
    V += [Q(1,0,0,0),Q(-1,0,0,0),Q(0,1,0,0),Q(0,-1,0,0),Q(0,0,1,0),Q(0,0,-1,0),Q(0,0,0,1),Q(0,0,0,-1)]
    # 16 half‑units
    for sw in (-1,1):
        for sx in (-1,1):
            for sy in (-1,1):
                for sz in (-1,1):
                    V.append(Q(0.5*sw,0.5*sx,0.5*sy,0.5*sz))
    # unique normalized
    uniq = {}
    for q in V:
        u = q.normalize()
        uniq[(round(u.w,6), round(u.x,6), round(u.y,6), round(u.z,6))] = u
    return list(uniq.values())

def is_coordinate(q:Q, tol=1e-9):
    comps = [abs(q.w),abs(q.x),abs(q.y),abs(q.z)]
    return sum(abs(c-1.0)<tol for c in comps)==1 and sum(c<tol for c in comps if abs(c-1.0)>=tol)==3

def is_half(q:Q, tol=1e-9):
    comps = [abs(q.w),abs(q.x),abs(q.y),abs(q.z)]
    return all(abs(c-0.5)<tol for c in comps)

def dot4(a:Q,b:Q): 
    return a.w*b.w + a.x*b.x + a.y*b.y + a.z*b.z

def sign_tuple_half(q:Q):
    def sgn(v): return 1 if v>0 else (-1 if v<0 else 0)
    return (sgn(q.w),sgn(q.x),sgn(q.y),sgn(q.z))

def build_24cell_adjacency():
    V = hurwitz_vertices()
    edges=set()
    # Coordinate↔Half: connect when dot = +1/2
    for i,v in enumerate(V):
        if is_coordinate(v):
            for j,u in enumerate(V):
                if is_half(u) and abs(dot4(v,u)-0.5)<1e-9:
                    a,b = (i,j) if i<j else (j,i)
                    edges.add((a,b))
    # Half↔Half: Hamming distance 1 in sign tuples
    half = [(i,sign_tuple_half(v)) for i,v in enumerate(V) if is_half(v)]
    for a,(i,si) in enumerate(half):
        for j,(k,sk) in enumerate(half[a+1:], start=a+1):
            hdist = sum(int(si[t]!=sk[t]) for t in range(4))
            if hdist==1:
                a1,b1 = (i,k) if i<k else (k,i)
                edges.add((a1,b1))
    adj = {i:set() for i in range(len(V))}
    for a,b in edges:
        adj[a].add(b); adj[b].add(a)
    return V, adj

# fallback 2T word generator and residue seeding
def fallback_generate_2T(maxlen=8):
    s = q_axis_angle(1,1,1, 2*math.pi/3)     # 120° around (1,1,1)
    t = q_axis_angle(1,1,-1, 2*math.pi/3)    # 120° around (1,1,-1)
    gens=[s,t,s.conj(),t.conj()]
    seen={(1.0,0.0,0.0,0.0): q_id()}
    frontier=[q_id()]
    for _ in range(maxlen):
        new=[]
        for g in frontier:
            for h in gens:
                u=(g*h).normalize()
                k=(round(u.w,6),round(u.x,6),round(u.y,6),round(u.z,6))
                if k not in seen:
                    seen[k]=u; new.append(u)
        frontier=new
        if not new: break
    return list(seen.values())

def fallback_seed_loop_for_residue(r:int):
    s = q_axis_angle(1,1,1, 2*math.pi/3)
    t = q_axis_angle(1,1,-1, 2*math.pi/3)
    def pow_q(q:Q,k:int):
        if k==0: return q_id()
        if k<0: 
            q=q.conj(); k=-k
        out=q_id()
        for _ in range(k): out = (out*q).normalize()
        return out
    a = 1 + (r % 3)
    b = 1 + ((r // 3) % 3)
    c = 1 + ((r // 9) % 3)
    U = (pow_q(s,a) * pow_q(t,b) * pow_q(s,c)).normalize()
    m = r % 3
    axis = (1,0,0) if m==0 else ((0,1,0) if m==1 else (0,0,1))
    return U, axis

def state_sum_map(maxlen=8, epsilon_deg=5.0):
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
            ang=su2_angle(q)
            c=ang*ang
            if c<best_cost-1e-15:
                best_cost=c; best_ang=ang; best_g=g
        mapping[r]={"best_g":best_g, "angle_deg":best_ang*180.0/math.pi}
    return mapping

def analyze_d4_path(mapping: Dict[int,Dict]):
    outdir = os.path.join(OUTROOT, "d4")
    os.makedirs(outdir, exist_ok=True)
    V, adj = build_24cell_adjacency()
    def nearest_index(q:Q):
        best_i=None; best_d=1e9
        for i,v in enumerate(V):
            d=((q.w-v.w)**2 + (q.x-v.x)**2 + (q.y-v.y)**2 + (q.z-v.z)**2)**0.5
            if d<best_d: best_d=d; best_i=i
        return best_i
    r2i = { r: nearest_index(d["best_g"]) for r,d in mapping.items() }
    consec = [(r, r+2) for r in range(1,23,2)]
    adj_hits = sum(1 for a,b in consec if r2i[b] in adj[r2i[a]])
    adj_score = adj_hits/len(consec)
    anti_pairs = [(r, 24-r) for r in range(1,12,2)]
    anti_hits=0
    for a,b in anti_pairs:
        qa=mapping[a]["best_g"]; qb=mapping[b]["best_g"]
        if abs(qb.w + qa.w)<1e-6 and abs(qb.x + qa.x)<1e-6 and abs(qb.y + qa.y)<1e-6 and abs(qb.z + qa.z)<1e-6:
            anti_hits+=1
    anti_score = anti_hits/len(anti_pairs)
    # artifacts
    with open(os.path.join(outdir,"d4_mapping.csv"),"w",newline="") as f:
        w=csv.writer(f); w.writerow(["residue","vertex_index","angle_deg"])
        for r in sorted(mapping):
            w.writerow([r, r2i[r], f"{mapping[r]['angle_deg']:.6f}"])
    with open(os.path.join(outdir,"d4_summary.json"),"w") as f:
        json.dump({"adjacency_score_r_to_r_plus_2": adj_score, 
                   "antipodal_pairing_score_r_to_24_minus_r": anti_score}, f, indent=2)
    return {"adjacency_score": adj_score, "antipodal_score": anti_score,
            "csv": os.path.join(outdir,"d4_mapping.csv"),
            "json": os.path.join(outdir,"d4_summary.json")}

# ---------- (3) Gödel curvature toy engine ----------

def godel_loop(eps=0.1, delta=0.1):
    # Universe Ω = {(a,b)∈{0,1}^2}; exact reference q = uniform; compressed family f tracks independent marginals
    states=[(0,0),(0,1),(1,0),(1,1)]
    def softmax(weights):
        m = max(weights); exps = [math.exp(w-m) for w in weights]; Z = sum(exps)
        return [e/Z for e in exps]
    def tilt(r, lam_parity, lam_a):
        weights=[math.log(r[i]) + (1.0 if (a ^ b)==1 else 0.0)*lam_parity + (1.0 if a==1 else 0.0)*lam_a 
                 for i,(a,b) in enumerate(states)]
        return softmax(weights)
    def project_to_product(r):
        p_a = r[2] + r[3]
        p_b = r[1] + r[3]
        return [(1-p_a)*(1-p_b), (1-p_a)*p_b, p_a*(1-p_b), p_a*p_b]
    def KL(p,q): 
        return sum(0 if p[i]==0 else p[i]*(math.log(p[i]) - math.log(max(q[i],1e-300))) for i in range(4))
    p = [0.25]*4
    Qheat=0.0
    r = tilt(p, eps, 0.0); p = project_to_product(r); Qheat += KL(r,p)
    r = tilt(p, 0.0, delta); p = project_to_product(r); Qheat += KL(r,p)
    r = tilt(p, -eps, 0.0); p = project_to_product(r); Qheat += KL(r,p)
    r = tilt(p, 0.0, -delta); p = project_to_product(r); Qheat += KL(r,p)
    p_b = p[1] + p[3]
    return p_b, Qheat

def godel_grid():
    rows=[]
    for eps in [k/100.0 for k in range(5,31,5)]:      # 0.05..0.30
        for delta in [k/100.0 for k in range(5,31,5)]:# 0.05..0.30
            pb, Qh = godel_loop(eps, delta)
            rows.append((eps,delta,pb,Qh))
    # estimate κ in p_b ≈ 1/2 + κ εδ (least squares through origin)
    xs=[row[0]*row[1] for row in rows]
    ys=[row[2]-0.5 for row in rows]
    num=sum(x*y for x,y in zip(xs,ys))
    den=sum(x*x for x in xs)
    kappa = num/den if den>0 else float('nan')
    # dump
    outdir = os.path.join(OUTROOT,"godel")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir,"godel_grid.csv"),"w",newline="") as f:
        w=csv.writer(f); w.writerow(["eps","delta","p_b","Q_heat_nats"])
        for row in rows: w.writerow([f"{row[0]:.3f}",f"{row[1]:.3f}",f"{row[2]:.9f}",f"{row[3]:.9f}"])
    return {"kappa_estimate": kappa, 
            "grid_csv": os.path.join(outdir,"godel_grid.csv")}

# ---------- (4) Z3 double‑cover check ----------

def z3_double_cover_demo(theta=2*math.pi/3):
    # Pick a generic axis
    R = q_axis_angle(1,2,3, theta).normalize()
    R3 = (R*R*R).normalize()
    R6 = (R3*R3).normalize()
    # SU(2) "identity" is q=±1; +1 is exact identity in SO(3), -1 is center element
    def approx(q:Q, target:Q, tol=1e-9):
        return abs(q.w-target.w)<tol and abs(q.x-target.x)<tol and abs(q.y-target.y)<tol and abs(q.z-target.z)<tol
    return {"R3_is_minus_identity": approx(R3, Q(-1,0,0,0)), 
            "R6_is_plus_identity": approx(R6, Q(1,0,0,0)),
            "R_geodesic_angle_rad": su2_angle(R)}

# ---------- Run all ----------

area_result = run_area_law_scan()
d4_mapping = state_sum_map(maxlen=8, epsilon_deg=5.0)
d4_result = analyze_d4_path(d4_mapping)
pb, Qh = godel_loop(0.1,0.1)
ggrid = godel_grid()
z3 = z3_double_cover_demo()

print("AREA median(angle/a^2) ~", round(area_result["median_ratio"], 4))
print("D4 adjacency_score=", round(d4_result["adjacency_score"],3), "antipodal_score=", round(d4_result["antipodal_score"],3))
print("Gödel loop p_b(eps=delta=0.1) =", round(pb,9), "heat=", round(Qh,6), "kappa≈", round(ggrid["kappa_estimate"],3))
print("Z3 cover: R^3 = -I ?", z3["R3_is_minus_identity"], "; R^6 = +I ?", z3["R6_is_plus_identity"])

area_result, d4_result, {"pb_at_0.1": pb, "heat": Qh, "kappa": ggrid["kappa_estimate"]}, z3
