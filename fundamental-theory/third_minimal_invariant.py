
import json, math, os, csv, statistics, random, time
from dataclasses import dataclass, asdict
from typing import Dict, List

try:
    import numpy as np
except Exception:
    np = None

OUTDIR = os.path.dirname(__file__)

# ---------- Linear algebra helpers ----------
def mat_mul(A, B):
    return [[A[0][0]*B[0][0] + A[0][1]*B[1][0],
             A[0][0]*B[0][1] + A[0][1]*B[1][1]],
            [A[1][0]*B[0][0] + A[1][1]*B[1][0],
             A[1][0]*B[0][1] + A[1][1]*B[1][1]]]

def mat_pow(A, k):
    I = [[1.0, 0.0],[0.0, 1.0]]
    if k == 0:
        return I
    if k < 0:
        At = [[A[0][0], A[1][0]],[A[0][1], A[1][1]]]
        return mat_pow(At, -k)
    out = I
    for _ in range(k):
        out = mat_mul(out, A)
    return out

def mat_close(A, B, tol=1e-9):
    return all(abs(A[i][j]-B[i][j]) < tol for i in range(2) for j in range(2))

# ---------- SU(2) / quaternion utilities ----------
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
def q_id(): return Q(1,0,0,0)
def q_inv(q:Q): return q.conj()
def q_axis_angle(nx,ny,nz,theta):
    n = (nx*nx+ny*ny+nz*nz)**0.5
    if n < EPS: return q_id()
    nx/=n; ny/=n; nz/=n
    h = 0.5*theta
    return Q(math.cos(h), nx*math.sin(h), ny*math.sin(h), nz*math.sin(h))
def su2_geodesic_angle(q:Q):
    w = max(-1.0, min(1.0, q.normalize().w))
    return 2.0*math.acos(abs(w))
def pow_q(q:Q,k:int):
    if k==0: return q_id()
    if k<0: return pow_q(q_inv(q), -k)
    out = q_id()
    for _ in range(k):
        out = out*q
    return out

# ---------- 1) Z3 check ----------
def z3_check():
    theta = 2*math.pi/3
    R = [[math.cos(theta), -math.sin(theta)],[math.sin(theta), math.cos(theta)]]
    I2 = [[1.0,0.0],[0.0,1.0]]
    R3 = mat_pow(R, 3)
    so3_order3 = mat_close(R3, I2, 1e-9)
    U = q_axis_angle(1,1,1, theta).normalize()
    U3 = pow_q(U, 3)
    U6 = pow_q(U, 6)
    minusI = Q(-1,0,0,0)
    is_U3_minusI = (abs(U3.w - minusI.w) < 1e-9 and abs(U3.x) < 1e-9 and abs(U3.y) < 1e-9 and abs(U3.z) < 1e-9)
    is_U6_I = (abs(U6.w - 1.0) < 1e-9 and abs(U6.x) < 1e-9 and abs(U6.y) < 1e-9 and abs(U6.z) < 1e-9)
    out = {"SO3_R_120deg_order3": so3_order3,
           "SU2_U_120deg_halfangle": {"U^3 == -I": is_U3_minusI, "U^6 == I": is_U6_I}}
    with open(os.path.join(OUTDIR,"z3_check.json"),"w") as f: json.dump(out, f, indent=2)
    return out

# ---------- 2) Area law ----------
def holonomy_square(a:float)->Q:
    I = q_axis_angle(1,0,0, a)
    J = q_axis_angle(0,1,0, a)
    return (I * J * q_inv(I) * q_inv(J)).normalize()
def su2_area_law_scan():
    rows = []
    for k in range(5, 51):
        a = k/1000.0
        q = holonomy_square(a)
        ang = su2_geodesic_angle(q)
        rows.append((a, ang, ang/(a*a)))
    path = os.path.join(OUTDIR, "area_law.csv")
    with open(path,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["a","angle","angle_over_a2"])
        for a,ang,ratio in rows: w.writerow([f"{a:.6f}", f"{ang:.9f}", f"{ratio:.6f}"])
    summary = {"median_angle_over_a2": statistics.median(r for _,_,r in rows),
               "mean_angle_over_a2": statistics.mean(r for _,_,r in rows),
               "min_angle_over_a2": min(r for _,_,r in rows),
               "max_angle_over_a2": max(r for _,_,r in rows),
               "csv": path}
    with open(os.path.join(OUTDIR,"area_law_summary.json"),"w") as f: json.dump(summary, f, indent=2)
    return summary

# ---------- 3) Gödel loop ----------
def softmax(weights):
    m = max(weights); exps = [math.exp(w-m) for w in weights]; Z = sum(exps); return [e/Z for e in exps]
def godel_loop(eps=0.1, delta=0.1):
    states=[(0,0),(0,1),(1,0),(1,1)]
    def energy(a,b, lam_parity, lam_a):
        phi_xor = 1.0 if (a ^ b)==1 else 0.0
        return lam_parity*phi_xor + lam_a*(1.0 if a==1 else 0.0)
    def tilt(r, lam_parity, lam_a):
        weights=[math.log(r[i])+energy(a,b,lam_parity,lam_a) for i,(a,b) in enumerate(states)]
        return softmax(weights)
    def project_to_product(r):
        p_a = r[2] + r[3]; p_b = r[1] + r[3]
        return [(1-p_a)*(1-p_b), (1-p_a)*p_b, p_a*(1-p_b), p_a*p_b]
    def KL(p,q): return sum(0 if p[i]==0 else p[i]*(math.log(p[i]) - math.log(max(q[i],1e-300))) for i in range(4))
    p = [0.25]*4; Qheat=0.0
    r = tilt(p, eps, 0.0); p = project_to_product(r); Qheat += KL(r,p)
    r = tilt(p, 0.0, delta); p = project_to_product(r); Qheat += KL(r,p)
    r = tilt(p, -eps, 0.0); p = project_to_product(r); Qheat += KL(r,p)
    r = tilt(p, 0.0, -delta); p = project_to_product(r); Qheat += KL(r,p)
    p_b = p[1] + p[3]
    return p_b, Qheat
def godel_grid_scan():
    rows=[]
    for eps in [k/100.0 for k in range(5,31,5)]:
        for delta in [k/100.0 for k in range(5,31,5)]:
            pb, Qh = godel_loop(eps, delta)
            rows.append((eps,delta,pb,Qh))
    path = os.path.join(OUTDIR, "godel_grid.csv")
    with open(path,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["eps","delta","p_b","Q_heat_nats"])
        for row in rows: w.writerow([f"{row[0]:.3f}",f"{row[1]:.3f}",f"{row[2]:.9f}",f"{row[3]:.9f}"])
    xs=[row[0]*row[1] for row in rows]; ys=[row[2]-0.5 for row in rows]
    num=sum(x*y for x,y in zip(xs,ys)); den=sum(x*x for x in xs)
    kappa = num/den if den>0 else float('nan')
    out = {"p_b_eps_eq_delta_0.1": godel_loop(0.1,0.1)[0],
           "Q_heat_eps_eq_delta_0.1": godel_loop(0.1,0.1)[1],
           "kappa_estimate": kappa,
           "theory_kappa": 1.0/8.0,
           "csv": path}
    with open(os.path.join(OUTDIR,"godel_summary.json"),"w") as f: json.dump(out, f, indent=2)
    return out

# ---------- 4) D4 adjacency (fallback) ----------
def hurwitz_vertices()->List[Q]:
    V = []
    V += [Q(1,0,0,0),Q(-1,0,0,0),Q(0,1,0,0),Q(0,-1,0,0),Q(0,0,1,0),Q(0,0,-1,0),Q(0,0,0,1),Q(0,0,0,-1)]
    for sw in (-1,1):
        for sx in (-1,1):
            for sy in (-1,1):
                for sz in (-1,1):
                    V.append(Q(0.5*sw,0.5*sx,0.5*sy,0.5*sz))
    uniq = {}
    for q in V:
        u = q.normalize()
        uniq[u.w, u.x, u.y, u.z] = u
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
    V = hurwitz_vertices(); edges=set()
    for i,v in enumerate(V):
        if is_coordinate(v):
            for j,u in enumerate(V):
                if is_half(u) and abs(dot4(v,u)-0.5)<1e-9:
                    a,b=min(i,j),max(i,j); edges.add((a,b))
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
def fallback_generate_2T(maxlen=8):
    s = Q(0.5,0.5,0.5,0.5); t = Q(0.5,0.5,0.5,-0.5)
    gens=[s,t,q_inv(s),q_inv(t)]
    seen={(1.0,0.0,0.0,0.0): Q(1,0,0,0)}
    frontier=[Q(1,0,0,0)]
    for _ in range(maxlen):
        new=[]
        for g in frontier:
            for h in gens:
                u=(g*h).normalize()
                k=(round(u.w,6), round(u.x,6), round(u.y,6), round(u.z,6))
                if k not in seen:
                    seen[k]=u; new.append(u)
        frontier=new
        if not new: break
    G=list(seen.values())
    if len(G)<24 and maxlen<12: return fallback_generate_2T(maxlen=maxlen+2)
    return G
def fallback_seed_loop_for_residue(r:int):
    s = Q(0.5,0.5,0.5,0.5); t = Q(0.5,0.5,0.5,-0.5)
    a = 1 + (r % 3); b = 1 + ((r // 3) % 3); c = 1 + ((r // 9) % 3)
    U = (pow_q(s,a) * pow_q(t,b) * pow_q(s,c)).normalize()
    m = r % 3; axis = (1,0,0) if m==0 else ((0,1,0) if m==1 else (0,0,1))
    return U, axis
def state_sum_map(beta=25.0, maxlen=8, epsilon_deg=5.0):
    G = fallback_generate_2T(maxlen=maxlen)
    residues = list(range(1,24,2)); eps = math.radians(epsilon_deg)
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
def analyze_d4(mapping:Dict[int,Dict], outdir:str):
    V, adj, degs, nedges = build_24cell_adjacency()
    def nearest_index(q:Q):
        bd = 1e9; bi=None
        for i,v in enumerate(V):
            d=((q.w-v.w)**2 + (q.x-v.x)**2 + (q.y-v.y)**2 + (q.z-v.z)**2)**0.5
            if d<bd: bd=d; bi=i
        return bi, bd
    r2i = {r: nearest_index(d["best_g"])[0] for r,d in mapping.items()}
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
    used = sorted(set(r2i.values()))
    cov = len(used)/24.0
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir,"d4_mapping.csv"),"w",newline="") as f:
        w=csv.writer(f); w.writerow(["residue","vertex_index","angle_deg"])
        for r in sorted(mapping):
            w.writerow([r, r2i[r], f"{mapping[r]['angle_deg']:.6f}"])
    summary = {"vertex_degrees_first8": [degs[i] for i in range(8)],
               "num_edges": nedges,
               "coverage_ratio": cov,
               "adjacency_score_r_to_r+2": adj_score,
               "antipodal_pairing_score_r_vs_24-r": anti_score,
               "csv": os.path.join(outdir,"d4_mapping.csv")}
    with open(os.path.join(outdir,"d4_summary.json"),"w") as f: json.dump(summary, f, indent=2)
    return summary

# ---------- 5) Prime-smooth CCI and FFT ----------
def factorization(n: int) -> Dict[int, int]:
    if n < 1: raise ValueError("n must be >= 1")
    d: Dict[int, int] = {}; t = n; p = 2
    while p * p <= t:
        while t % p == 0:
            d[p] = d.get(p,0) + 1; t //= p
        p += 1 if p == 2 else 2
    if t > 1: d[t] = d.get(t,0) + 1
    return d
def base_prime_support(b: int) -> List[int]:
    return list(factorization(b).keys())
def is_b_smooth(n: int, b: int = 10) -> bool:
    bp = set(base_prime_support(b))
    return all(p in bp for p in factorization(n))
def count_add_carries_baseb(a: int, b: int, base: int = 10) -> int:
    carries = 0; carry = 0; aa, bb = a, b
    while aa > 0 or bb > 0 or carry > 0:
        da = aa % base; db = bb % base; s = da + db + carry
        if s >= base: carries += 1; carry = 1
        else: carry = 0
        aa //= base; bb //= base
    return carries
def count_mult_carries_baseb(a: int, b: int, base: int = 10) -> int:
    carries = 0; bb = b
    while bb > 0:
        db = bb % base; aa = a; carry = 0
        while aa > 0 or carry > 0:
            da = aa % base; prod = da * db + carry
            if prod >= base: carries += 1; carry = prod // base
            else: carry = 0
            aa //= base
        bb //= base
    return carries
def carry_coupling_index(num_digits: int = 8, trials: int = 2000, scale: int = 1,
                         base: int = 10, mode: str = "mult", seed: int = 0) -> float:
    rng = random.Random(42 + seed + num_digits * 911 + trials + scale + (0 if mode == "mult" else 1))
    total = 0; low = base ** (num_digits - 1); high = base ** num_digits
    for _ in range(trials):
        a = rng.randrange(low, high) * scale; b = rng.randrange(low, high) * scale
        if mode == "add": total += count_add_carries_baseb(a, b, base)
        else: total += count_mult_carries_baseb(a, b, base)
    return total / trials
@dataclass
class CCIRow:
    mode: str; digits: int; scale: int; is_b_smooth: bool; CCI: float
def run_cci_panel(scales: List[int], digits_list: List[int], base:int=10, trials:int=600, seed:int=1) -> List[CCIRow]:
    rows: List[CCIRow] = []
    for mode in ["add", "mult"]:
        for D in digits_list:
            for s in scales:
                cci = carry_coupling_index(num_digits=D, trials=trials, scale=s, base=base, mode=mode, seed=seed)
                rows.append(CCIRow(mode=mode, digits=D, scale=s, is_b_smooth=is_b_smooth(s, base), CCI=cci))
    return rows
@dataclass
class FFTResult:
    N: int; time_s: float; per_elem_ns: float; factors: Dict[int,int]; b_smooth: bool
def fft_time(n: int, reps: int = 5) -> float:
    if np is None: raise RuntimeError("NumPy not available; FFT proxy disabled")
    x = np.random.randn(n) + 1j * np.random.randn(n); np.fft.fft(x)
    times: List[float] = []
    for _ in range(reps):
        x2 = x.copy(); t0 = time.perf_counter(); np.fft.fft(x2); t1 = time.perf_counter(); times.append(t1 - t0)
    return statistics.median(times)
def run_fft_panel(Ns: List[int], base:int=10, reps:int=7) -> List[FFTResult]:
    out: List[FFTResult] = []
    for n in Ns:
        t = fft_time(n, reps=reps)
        out.append(FFTResult(N=n, time_s=t, per_elem_ns=(t/n)*1e9, factors=factorization(n), b_smooth=is_b_smooth(n, base)))
    return out

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    z3_out = z3_check()
    area_out = su2_area_law_scan()
    godel_out = godel_grid_scan()
    mapping = state_sum_map(beta=25.0, maxlen=8, epsilon_deg=5.0)
    d4_out = analyze_d4(mapping, os.path.join(OUTDIR,"d4"))
    scales = [1,2,3,4,5,6,7,8,9,10]
    rows = run_cci_panel(scales=scales, digits_list=[6,7], base=10, trials=500, seed=2)
    with open(os.path.join(OUTDIR,"cci_panel.json"),"w") as f: json.dump([asdict(r) for r in rows], f, indent=2)
    smooth_vals = [r.CCI for r in rows if r.is_b_smooth]
    nonsmooth_vals = [r.CCI for r in rows if not r.is_b_smooth]
    cci_summary = {"avg_CCI_smooth": statistics.mean(smooth_vals) if smooth_vals else None,
                   "avg_CCI_nonsmooth": statistics.mean(nonsmooth_vals) if nonsmooth_vals else None,
                   "n_rows": len(rows),
                   "json": os.path.join(OUTDIR,"cci_panel.json")}
    with open(os.path.join(OUTDIR,"cci_summary.json"),"w") as f: json.dump(cci_summary, f, indent=2)
    if np is not None:
        Ns = [256, 320, 384, 486, 500, 512, 640, 768, 960, 972, 1000, 1024]
        results = run_fft_panel(Ns, base=10, reps=5)
        with open(os.path.join(OUTDIR,"fft_panel.json"),"w") as f: json.dump([asdict(r) for r in results], f, indent=2)
        per_elem_ns_smooth = [r.per_elem_ns for r in results if r.b_smooth]
        per_elem_ns_nonsmooth = [r.per_elem_ns for r in results if not r.b_smooth]
        fft_summary = {"median_per_elem_ns_smooth": statistics.median(per_elem_ns_smooth) if per_elem_ns_smooth else None,
                       "median_per_elem_ns_nonsmooth": statistics.median(per_elem_ns_nonsmooth) if per_elem_ns_nonsmooth else None,
                       "json": os.path.join(OUTDIR,"fft_panel.json")}
        with open(os.path.join(OUTDIR,"fft_summary.json"),"w") as f: json.dump(fft_summary, f, indent=2)
    print("Done. Artifacts written to", OUTDIR)

if __name__ == "__main__":
    main()
