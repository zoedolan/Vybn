# smoking_gun_invariants.py
# Minimal, dependency-light harness producing four "smoking-gun" invariants:
# 1) SU(2) area-law calibration with orientation/degen checks
# 2) SU(2)→SO(3) double-cover closure at 2π/3 (ℤ3 downstairs, ℤ6 upstairs)
# 3) 24-cell (Hurwitz) adjacency ablation for SU(2)×mod-24 instrument
# 4) Gödel curvature unit test with κ ≈ 1/8 and strictly positive heat
#
# See notebook runner for identical logic; this file mirrors that code for repository use.
# Usage: python smoking_gun_invariants.py --out out_dir

import argparse, json, math, os, csv, statistics
from typing import List

EPS=1e-12
class Q:
    __slots__=("w","x","y","z")
    def __init__(self,w,x,y,z): self.w=float(w); self.x=float(x); self.y=float(y); self.z=float(z)
    def __mul__(a,b):
        return Q(a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
                 a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
                 a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
                 a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w)
    def conj(self): return Q(self.w,-self.x,-self.y,-self.z)
    def norm(self): return (self.w*self.w + self.x*self.x + self.y*self.y + self.z*self.z)**0.5
    def normalize(self):
        n=self.norm()
        return Q(1,0,0,0) if n<EPS else Q(self.w/n,self.x/n,self.y/n,self.z/n)

def q_id(): return Q(1,0,0,0)
def q_inv(q:Q): return q.conj()
def q_axis_angle(nx,ny,nz,theta):
    n=(nx*nx+ny*ny+nz*nz)**0.5
    if n<EPS: return q_id()
    nx/=n; ny/=n; nz/=n
    h=0.5*theta
    return Q(math.cos(h), nx*math.sin(h), ny*math.sin(h), nz*math.sin(h))
def su2_geodesic_angle(q:Q):
    w=max(-1.0, min(1.0, q.normalize().w)); return 2.0*math.acos(abs(w))

def holonomy_square(a:float)->Q:
    I=q_axis_angle(1,0,0,a); J=q_axis_angle(0,1,0,a)
    return (I*J*q_inv(I)*q_inv(J)).normalize()

def area_law_scan():
    rows=[]
    for a in [k/1000.0 for k in range(5,51,1)]:
        q=holonomy_square(a); ang=su2_geodesic_angle(q)
        rows.append((a, ang, ang/(a*a), ang/(2*a*a)))
    med_a2=statistics.median(r[2] for r in rows)
    med_2a2=statistics.median(r[3] for r in rows)
    a0=0.03; I=q_axis_angle(1,0,0,a0); J=q_axis_angle(0,1,0,a0)
    q_fwd=(I*J*q_inv(I)*q_inv(J)).normalize()
    q_rev=(J*I*q_inv(J)*q_inv(I)).normalize()
    inv=q_inv(q_fwd).normalize()
    sign_ok=all(abs(getattr(q_rev,k)-getattr(inv,k))<1e-9 for k in ("w","x","y","z"))
    q_line=(I*q_inv(I)).normalize()
    deg_ok=su2_geodesic_angle(q_line)<1e-12
    return rows, med_a2, med_2a2, sign_ok, deg_ok

def z3_plateau_checks():
    theta=2*math.pi/3; q=q_axis_angle(1,1,1,theta)
    def mul_pow(q,k):
        out=q_id()
        for _ in range(k): out=(out*q).normalize()
        return out
    q3=mul_pow(q,3); q6=mul_pow(q,6)
    su2_3 = abs(q3.w + 1.0)<1e-12 and abs(q3.x)<1e-12 and abs(q3.y)<1e-12 and abs(q3.z)<1e-12
    su2_6 = abs(q6.w - 1.0)<1e-12 and abs(q6.x)<1e-12 and abs(q6.y)<1e-12 and abs(q6.z)<1e-12
    def rotate_vec(q, v):
        vv=Q(0.0, *v); res=q*vv*q_inv(q); return (res.x,res.y,res.z)
    v=(1.0,0.0,0.0); v3=v
    for _ in range(3): v3=rotate_vec(q, v3)
    so3_3 = (sum((v3[i]-v[i])**2 for i in range(3))**0.5) < 1e-12
    return su2_3, su2_6, so3_3

def hurwitz_vertices()->List[Q]:
    V=[Q(1,0,0,0),Q(-1,0,0,0),Q(0,1,0,0),Q(0,-1,0,0),Q(0,0,1,0),Q(0,0,-1,0),Q(0,0,0,1),Q(0,0,0,-1)]
    for sw in (-1,1):
        for sx in (-1,1):
            for sy in (-1,1):
                for sz in (-1,1):
                    V.append(Q(0.5*sw,0.5*sx,0.5*sy,0.5*sz))
    uniq={}
    for q in V:
        u=q.normalize()
        uniq[(round(u.w,6),round(u.x,6),round(u.y,6),round(u.z,6))]=u
    return list(uniq.values())
def is_coordinate(q:Q, tol=1e-9):
    comps=[abs(q.w),abs(q.x),abs(q.y),abs(q.z)]
    return sum(abs(c-1.0)<tol for c in comps)==1 and sum(c<tol for c in comps if abs(c-1.0)>=tol)==3
def is_half(q:Q, tol=1e-9):
    comps=[abs(q.w),abs(q.x),abs(q.y),abs(q.z)]
    return all(abs(c-0.5)<tol for c in comps)
def dot4(a:Q,b:Q): return a.w*b.w + a.x*b.x + a.y*b.y + a.z*b.z
def sign_tuple_half(q:Q):
    def sgn(v): return 1 if v>0 else (-1 if v<0 else 0)
    return (sgn(q.w),sgn(q.x),sgn(q.y),sgn(q.z))
def build_24cell_adjacency():
    V=hurwitz_vertices(); edges=set()
    for i,v in enumerate(V):
        if is_coordinate(v):
            for j,u in enumerate(V):
                if is_half(u) and abs(dot4(v,u)-0.5)<1e-9:
                    a,b=min(i,j),max(i,j); edges.add((a,b))
    half=[(i,sign_tuple_half(v)) for i,v in enumerate(V) if is_half(v)]
    for a,(i,si) in enumerate(half):
        for j,(k,sk) in enumerate(half[a+1:], start=a+1):
            hdist=sum(int(si[t]!=sk[t]) for t in range(4))
            if hdist==1:
                a1,b1=min(i,k),max(i,k); edges.add((a1,b1))
    adj={i:set() for i in range(len(V))}
    for a,b in edges:
        adj[a].add(b); adj[b].add(a)
    return V, adj
def fallback_generate_2T(maxlen=8):
    s=q_axis_angle(1,1,1, 2*math.pi/3); t=q_axis_angle(1,1,-1, 2*math.pi/3)
    gens=[s,t,q_inv(s),q_inv(t)]; seen={(1.0,0.0,0.0,0.0): q_id()}; frontier=[q_id()]
    for _ in range(maxlen):
        new=[]
        for g in frontier:
            for h in gens:
                u=(g*h).normalize(); k=(round(u.w,6),round(u.x,6),round(u.y,6),round(u.z,6))
                if k not in seen: seen[k]=u; new.append(u)
        frontier=new
        if not new: break
    return list(seen.values())
def fallback_seed_loop_for_residue(r:int):
    s=q_axis_angle(1,1,1, 2*math.pi/3); t=q_axis_angle(1,1,-1, 2*math.pi/3)
    def pow_q(q,k):
        out=q_id()
        for _ in range(k): out=(out*q).normalize()
        return out
    a=1+(r%3); b=1+((r//3)%3); c=1+((r//9)%3)
    U=(pow_q(s,a)*pow_q(t,b)*pow_q(s,c)).normalize()
    m=r%3; axis=(1,0,0) if m==0 else ((0,1,0) if m==1 else (0,0,1))
    twist=q_axis_angle(*axis, math.radians(5.0)*(1+(r%2)))
    return (twist*U).normalize()
def d4_adjacency_probe():
    V, adj = build_24cell_adjacency()
    def nearest_index(q:Q):
        best=None; bd=1e9; bi=None
        for i,v in enumerate(V):
            d=((q.w-v.w)**2 + (q.x-v.x)**2 + (q.y-v.y)**2 + (q.z-v.z)**2)**0.5
            if d<bd: bd=d; bi=i; best=v
        return bi
    G=fallback_generate_2T(); residues=list(range(1,24,2)); r2i={}
    for r in residues:
        Ur=fallback_seed_loop_for_residue(r)
        best_g=None; best_cost=1e9
        for g in G:
            q=(Ur*g).normalize(); ang=su2_geodesic_angle(q)
            if ang*ang<best_cost: best_cost=ang*ang; best_g=g
        r2i[r]=nearest_index(best_g)
    consec=[(r, r+2) for r in range(1,23,2)]
    adj_hits=sum(1 for a,b in consec if r2i[b] in adj[r2i[a]])
    adj_score=adj_hits/len(consec)
    anti_pairs=[(r,24-r) for r in range(1,12,2)]
    anti_hits=0
    for a,b in anti_pairs:
        qa=fallback_seed_loop_for_residue(a); qb=fallback_seed_loop_for_residue(b)
        if abs(qb.w + qa.w)<1e-6 and abs(qb.x + qa.x)<1e-6 and abs(qb.y + qa.y)<1e-6 and abs(qb.z + qa.z)<1e-6:
            anti_hits+=1
    anti_score=anti_hits/len(anti_pairs)
    coverage=len(set(r2i.values()))/24.0
    return adj_score, anti_score, coverage

def godel_unit(eps=0.1, delta=0.1):
    states=[(0,0),(0,1),(1,0),(1,1)]
    def softmax(weights):
        m=max(weights); exps=[math.exp(w-m) for w in weights]; Z=sum(exps); return [e/Z for e in exps]
    def energy(a,b, lam_parity, lam_a):
        return (1.0 if (a ^ b)==1 else 0.0)*lam_parity + (1.0 if a==1 else 0.0)*lam_a
    def tilt(r, lam_parity, lam_a):
        weights=[math.log(r[i])+energy(a,b,lam_parity,lam_a) for i,(a,b) in enumerate(states)]
        return softmax(weights)
    def project_to_product(r):
        p_a=r[2]+r[3]; p_b=r[1]+r[3]
        return [(1-p_a)*(1-p_b), (1-p_a)*p_b, p_a*(1-p_b), p_a*p_b]
    def KL(p,q): return sum(0 if p[i]==0 else p[i]*(math.log(p[i]) - math.log(max(q[i],1e-300))) for i in range(4))
    p=[0.25]*4; Qh=0.0
    r=tilt(p, eps, 0.0); p=project_to_product(r); Qh+=KL(r,p)
    r=tilt(p, 0.0, delta); p=project_to_product(r); Qh+=KL(r,p)
    r=tilt(p, -eps, 0.0); p=project_to_product(r); Qh+=KL(r,p)
    r=tilt(p, 0.0, -delta); p=project_to_product(r); Qh+=KL(r,p)
    return p[1]+p[3], Qh

def godel_grid():
    rows=[]; xs=[]; ys=[]
    for eps in [k/100.0 for k in range(5,31,5)]:
        for delta in [k/100.0 for k in range(5,31,5)]:
            pb, Qh = godel_unit(eps, delta)
            rows.append((eps,delta,pb,Qh)); xs.append(eps*delta); ys.append(pb-0.5)
    num=sum(x*y for x,y in zip(xs,ys)); den=sum(x*x for x in xs)
    return (num/den if den>0 else float('nan')), rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out", type=str, default="./out"); args=ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows, med_a2, med_2a2, sign_ok, deg_ok = area_law_scan()
    with open(os.path.join(args.out,"area_law.csv"),"w",newline="") as f:
        w=csv.writer(f); w.writerow(["a","angle","angle_over_a2","angle_over_2a2"])
        for a,ang,r1,r2 in rows: w.writerow([f"{a:.6f}", f"{ang:.9f}", f"{r1:.6f}", f"{r2:.6f}"])
    su2_3, su2_6, so3_3 = z3_plateau_checks()
    adj_score, anti_score, cov = d4_adjacency_probe()
    kappa, grows = godel_grid()
    with open(os.path.join(args.out,"godel_grid.csv"),"w",newline="") as f:
        w=csv.writer(f); w.writerow(["eps","delta","p_b","Q_heat_nats"])
        for eps,delta,pb,Qh in grows: w.writerow([f"{eps:.3f}", f"{delta:.3f}", f"{pb:.9f}", f"{Qh:.9f}"])
    summary={
        "area_law":{"median_angle_over_a2":med_a2,"median_angle_over_2a2":med_2a2,"orientation_sign_flip_inversion_ok":sign_ok,"degenerate_line_is_identity_ok":deg_ok},
        "z3_plateau":{"su2_three_powers_to_minusI":su2_3,"su2_six_powers_to_plusI":su2_6,"so3_three_is_identity":so3_3},
        "adjacency_ablation":{"consecutive_residue_adjacency_score":adj_score,"antipodal_pairing_score":anti_score,"coverage_ratio":cov},
        "godel_curvature":{"kappa_estimate":kappa,"theory_kappa":0.125,"example_loop":{"eps":0.1,"delta":0.1,"p_b":godel_unit(0.1,0.1)[0],"Q_heat":godel_unit(0.1,0.1)[1]}}
    }
    with open(os.path.join(args.out,"smoking_gun_results.json"),"w") as f: json.dump(summary,f,indent=2)
    print(json.dumps(summary, indent=2))

if __name__=='__main__':
    main()
