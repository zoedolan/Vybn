
import numpy as np, math, cmath, itertools, csv, os, sys, json
import matplotlib.pyplot as plt
import pandas as pd

ARTDIR = "/mnt/data"
os.makedirs(ARTDIR, exist_ok=True)

# -------------------------
# Pauli & utility operators
# -------------------------

I2 = np.eye(2, dtype=complex)
X = np.array([[0,1],[1,0]], dtype=complex)
Z = np.array([[1,0],[0,-1]], dtype=complex)
Y = np.array([[0,-1j],[1j,0]], dtype=complex)

def kron_n(ops):
    out = np.array([[1]], dtype=complex)
    for op in ops:
        out = np.kron(out, op)
    return out

def op_on_qubit(op, i, n):
    ops = [I2]*n
    ops[i] = op
    return kron_n(ops)

def ZZ(i,j,n):
    return op_on_qubit(Z,i,n) @ op_on_qubit(Z,j,n)

def Ysum(n):
    H = np.zeros((2**n,2**n), dtype=complex)
    for i in range(n):
        H += op_on_qubit(Y,i,n)
    return H

def Xsum(n):
    H = np.zeros((2**n,2**n), dtype=complex)
    for i in range(n):
        H += op_on_qubit(X,i,n)
    return H

def H_cost(edges, n):
    H = np.zeros((2**n,2**n), dtype=complex)
    for (i,j) in edges:
        H += 0.5*(np.eye(2**n, dtype=complex) - ZZ(i,j,n))
    return H

def ground_state(H):
    vals, vecs = np.linalg.eigh(H)
    return vals[0].real, vecs[:,0]

def commutator(A,B):
    return A@B - B@A

def pancharatnam_phase(states):
    prod = 1+0j
    for k in range(len(states)-1):
        prod *= np.vdot(states[k], states[k+1])
    prod *= np.vdot(states[-1], states[0])
    return cmath.phase(prod)

# -------------------------
# Graph / Max-Cut helpers
# -------------------------

class Graph:
    def __init__(self, n, edges, weight=1.0):
        self.n = n
        self.edges = [(min(i,j), max(i,j)) for (i,j) in edges]
        self.weights = {e: float(weight) for e in self.edges}

    def cost(self, bits):
        c = 0.0
        for (i,j) in self.edges:
            if bits[i] ^ bits[j]:
                c += self.weights[(i,j)]
        return c

    def max_cost_bruteforce(self):
        best = -1.0
        best_bits = None
        for x in range(1<<self.n):
            bits = np.array([(x>>k)&1 for k in range(self.n)], dtype=np.uint8)
            c = self.cost(bits)
            if c > best:
                best = c
                best_bits = bits
        return best, best_bits

def apply_rx_mixer(amplitude, n, beta):
    c = np.cos(beta); s = np.sin(beta)
    for q in range(n):
        step = 1<<q
        block = step<<1
        for base in range(0, amplitude.size, block):
            for i in range(step):
                i0 = base + i; i1 = i0 + step
                a0 = amplitude[i0]; a1 = amplitude[i1]
                amplitude[i0] = c*a0 - 1j*s*a1
                amplitude[i1] = -1j*s*a0 + c*a1

def holonomy_qaoa_distribution(g: Graph, kappa: float, beta: float):
    n = g.n; N = 1<<n
    costs = np.zeros(N, dtype=float)
    for x in range(N):
        bits = np.array([(x>>k)&1 for k in range(n)], dtype=np.uint8)
        costs[x] = g.cost(bits)
    cmax, _ = g.max_cost_bruteforce()
    amp = np.ones(N, dtype=np.complex128)/np.sqrt(N)
    phases = np.exp(1j * (-np.pi * kappa * (costs / cmax)))
    amp *= phases
    apply_rx_mixer(amp, n, beta)
    probs = np.abs(amp)**2
    return probs, costs, cmax

# -------------------------
# Paths in (alpha,beta) space
# -------------------------

def rectangle_path(a0,b0, da, db, K_per_edge=24, orientation=+1, schedule="uniform"):
    pts = []
    def edge_samples(K, nonuniform=False, edge_id=0):
        if schedule=="uniform":
            return np.linspace(0,1,K,endpoint=False)
        if schedule=="ease":
            t = np.linspace(0,1,K,endpoint=False)
            return t**2*(3-2*t)  # smoothstep
        if schedule=="zigzag":
            t = np.linspace(0,1,K,endpoint=False)
            if edge_id%2==0:
                return t
            else:
                return 1 - t
        return np.linspace(0,1,K,endpoint=False)
    if orientation>0:
        edges = [(a0, b0, da, 0),
                 (a0+da, b0, 0, db),
                 (a0+da, b0+db, -da, 0),
                 (a0, b0+db, 0, -db)]
    else:
        edges = [(a0, b0, 0, db),
                 (a0, b0+db, da, 0),
                 (a0+da, b0+db, 0, -db),
                 (a0+da, b0, -da, 0)]
    for idx,(ax,by, dax, dby) in enumerate(edges):
        for t in edge_samples(K_per_edge, schedule!="uniform", idx):
            pts.append((ax + t*dax, by + t*dby))
    return pts

def diamond_path(a0,b0, da, db, K_per_edge=24, orientation=+1):
    cx = a0 + da/2; cy = b0 + db/2
    wx = da/2; wy = db/2
    corners = [(cx, cy+wy), (cx+wx, cy), (cx, cy-wy), (cx-wx, cy)]
    if orientation<0:
        corners = list(reversed(corners))
    pts = []
    def seg(p,q,K,edge_id):
        xs = np.linspace(p[0], q[0], K, endpoint=False)
        ys = np.linspace(p[1], q[1], K, endpoint=False)
        return list(zip(xs,ys))
    for i in range(4):
        pts += seg(corners[i], corners[(i+1)%4], K_per_edge, i)
    return pts

def lissajous_path(a0,b0, da, db, K=256, orientation=+1):
    t = np.linspace(0, 2*np.pi, K, endpoint=False)
    x = a0 + da/2 + (da/2)*np.cos(2*t)
    y = b0 + db/2 + (db/2)*np.sin(3*t)
    if orientation<0:
        x = x[::-1]; y = y[::-1]
    return list(zip(x,y))

def states_along_path(HC, HM, HY, eps, pts):
    states = []
    gaps = []
    for (a,b) in pts:
        H = a*HC + b*HM + eps*HY
        vals, vecs = np.linalg.eigh(H)
        gaps.append((vals[1]-vals[0]).real)
        psi = vecs[:,0]
        states.append(psi/np.linalg.norm(psi))
    return states, np.array(gaps)

def loop_phase_for_path(HC, HM, HY, eps, pts):
    states, gaps = states_along_path(HC, HM, HY, eps, pts)
    return pancharatnam_phase(states), gaps


# -------------------------
# Bloch-sphere probe (single qubit) for clean Berry curvature
# H_b(α,β) = α Z + β X + ε Y, with ε > 0 fixing a north tilt; ground-state Berry phase equals half the solid angle.
# -------------------------

def H_bloch(alpha, beta, eps):
    Z = np.array([[1,0],[0,-1]], dtype=complex)
    X = np.array([[0,1],[1,0]], dtype=complex)
    Y = np.array([[0,-1j],[1j,0]], dtype=complex)
    return alpha*Z + beta*X + eps*Y

def ground_state_bloch(alpha, beta, eps):
    H = H_bloch(alpha, beta, eps)
    vals, vecs = np.linalg.eigh(H)
    return vals[0].real, vecs[:,0]

def loop_phase_bloch(a0,b0, da, db, K_per_edge=64, orientation=+1, schedule="uniform", eps=0.2):
    pts = rectangle_path(a0,b0, da, db, K_per_edge=K_per_edge, orientation=orientation, schedule=schedule)
    states = []
    for (a,b) in pts:
        _, psi = ground_state_bloch(a,b,eps)
        states.append(psi/np.linalg.norm(psi))
    return pancharatnam_phase(states)

def run_bloch_probe():
    a0,b0 = 0.1,0.1
    eps = 0.2
    rows = []
    areas = []
    phases = []
    for da in np.linspace(0.02, 0.6, 12):
        for db in np.linspace(0.02, 0.6, 12):
            A = da*db
            phi = loop_phase_bloch(a0,b0, da, db, K_per_edge=96, orientation=+1, schedule="uniform", eps=eps)
            rows.append([A, da, db, phi])
            areas.append(A); phases.append(phi)
    df = pd.DataFrame(rows, columns=["area","da","db","phase"])
    df.to_csv(os.path.join(ARTDIR, "bloch_area_scan.csv"), index=False)

    # Fit phase ~ s*area locally
    idx = np.argsort(np.array(areas))
    areas_sorted = np.array(areas)[idx]
    phases_sorted = np.unwrap(np.array(phases)[idx])
    A_mat = np.vstack([areas_sorted, np.ones_like(areas_sorted)]).T
    slope, intercept = np.linalg.lstsq(A_mat, phases_sorted, rcond=None)[0]

    # Orientation flip and schedule/shape checks at a representative area
    da, db = 0.35, 0.22
    phi_plus  = loop_phase_bloch(a0,b0, da, db, K_per_edge=96, orientation=+1, schedule="uniform", eps=eps)
    phi_minus = loop_phase_bloch(a0,b0, da, db, K_per_edge=96, orientation=-1, schedule="uniform", eps=eps)
    phi_ease  = loop_phase_bloch(a0,b0, da, db, K_per_edge=96, orientation=+1, schedule="ease", eps=eps)
    phi_zig   = loop_phase_bloch(a0,b0, da, db, K_per_edge=96, orientation=+1, schedule="zigzag", eps=eps)

    # Triadic area prediction from slope
    A_star = (2*np.pi/3)/abs(slope) if slope!=0 else np.nan

    # Plot phase vs area
    plt.figure()
    plt.plot(areas_sorted, phases_sorted, marker=".", linestyle="none")
    xfit = np.linspace(areas_sorted.min(), areas_sorted.max(), 200)
    yfit = slope*xfit + intercept
    plt.plot(xfit, yfit)
    plt.title(f"Bloch probe: phase vs area; slope≈{slope:.3f}, eps={eps}")
    plt.xlabel("Area Δα·Δβ")
    plt.ylabel("Phase (rad)")
    plt.savefig(os.path.join(ARTDIR, "bloch_phase_vs_area.png"), dpi=160); plt.close()

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "A_star": float(A_star),
        "orientation": {"phi_plus": float(phi_plus), "phi_minus": float(phi_minus), "sum": float(phi_plus+phi_minus)},
        "schedule": {"uniform": float(phi_plus), "ease": float(phi_ease), "zigzag": float(phi_zig)},
        "eps": float(eps)
    }

# -------------------------
# Unified experiment
# -------------------------

def run_experiment(seed=0):
    np.random.seed(seed)
    # Build graph: 6-node ring + chord
    n=6
    edges = [(i,(i+1)%n) if i<(i+1)%n else ((i+1)%n,i) for i in range(n)]
    edges += [(0,3)]
    edges = sorted({(min(i,j),max(i,j)) for (i,j) in edges})
    G = Graph(n, edges, weight=1.0)

    HC = H_cost(edges, n)
    HM = Xsum(n)
    HY = Ysum(n)
    eps = 0.03  # small symmetry-breaking Y-field

    C = commutator(HC, HM)
    comm_norm2 = float(np.linalg.norm(C, ord=2))

    alpha0, beta0 = 0.2, 0.2

    # Sweep rectangles for area law
    rect_rows = []
    areas = []
    ph_rect = []
    for da in np.linspace(0.05, 0.6, 7):
        for db in np.linspace(0.05, 0.6, 7):
            A = da*db
            pts = rectangle_path(alpha0, beta0, da, db, K_per_edge=24, orientation=+1, schedule="uniform")
            phi, gaps = loop_phase_for_path(HC, HM, HY, eps, pts)
            rect_rows.append([A, da, db, phi, gaps.min()])
            areas.append(A); ph_rect.append(phi)

    rect_df = pd.DataFrame(rect_rows, columns=["area","da","db","phase","min_gap"])
    rect_csv = os.path.join(ARTDIR, "unified_rect_area_scan.csv")
    rect_df.to_csv(rect_csv, index=False)

    # Orientation flip test
    da, db = 0.4, 0.25
    pts_plus  = rectangle_path(alpha0, beta0, da, db, K_per_edge=30, orientation=+1)
    pts_minus = rectangle_path(alpha0, beta0, da, db, K_per_edge=30, orientation=-1)
    phi_plus, gaps_plus = loop_phase_for_path(HC, HM, HY, eps, pts_plus)
    phi_minus, gaps_minus = loop_phase_for_path(HC, HM, HY, eps, pts_minus)

    # Shape invariance test (same area, different shapes/aspects)
    A_target = da*db
    pts_rect_tall = rectangle_path(alpha0, beta0, da*0.5, db*2.0, K_per_edge=24, orientation=+1)
    pts_diamond   = diamond_path(alpha0, beta0, da, db, K_per_edge=24, orientation=+1)
    phi_rect_tall, gaps_tall = loop_phase_for_path(HC, HM, HY, eps, pts_rect_tall)
    phi_diamond, gaps_diamond = loop_phase_for_path(HC, HM, HY, eps, pts_diamond)

    # Schedule invariance (same rectangle, different timing)
    pts_uniform = rectangle_path(alpha0, beta0, da, db, K_per_edge=24, schedule="uniform")
    pts_ease    = rectangle_path(alpha0, beta0, da, db, K_per_edge=24, schedule="ease")
    pts_zigzag  = rectangle_path(alpha0, beta0, da, db, K_per_edge=24, schedule="zigzag")
    phi_uniform, gaps_u = loop_phase_for_path(HC, HM, HY, eps, pts_uniform)
    phi_ease,    gaps_e = loop_phase_for_path(HC, HM, HY, eps, pts_ease)
    phi_zigzag,  gaps_z = loop_phase_for_path(HC, HM, HY, eps, pts_zigzag)

    # Triadic scan: search for phase near ±2π/3 modulo 2π
    triadic_rows = []
    for da_scan in np.linspace(0.05, 0.8, 40):
        db_scan = A_target/da_scan
        pts = rectangle_path(alpha0, beta0, da_scan, db_scan, K_per_edge=36, orientation=+1)
        phi, gaps = loop_phase_for_path(HC, HM, HY, eps, pts)
        phi_mod = ((phi+np.pi)%(2*np.pi))-np.pi
        triadic_rows.append([da_scan*db_scan, da_scan, db_scan, phi, phi_mod])
    triadic_df = pd.DataFrame(triadic_rows, columns=["area","da","db","phase","phase_mod_pi"])
    triadic_csv = os.path.join(ARTDIR, "unified_triadic_scan.csv")
    triadic_df.to_csv(triadic_csv, index=False)

    # Fit phase ~ s * area (rectangles)
    areas_arr = np.array(areas)
    idx = np.argsort(areas_arr)
    areas_sorted = areas_arr[idx]
    ph_rect_arr = np.unwrap(np.array(ph_rect)[idx])
    A_mat = np.vstack([areas_sorted, np.ones_like(areas_sorted)]).T
    slope, intercept = np.linalg.lstsq(A_mat, ph_rect_arr, rcond=None)[0]

    # Interference scaffold calibration
    def run_holonomy_qaoa_scan(G: Graph, kappa_grid, beta_grid):
        n = G.n; N = 1<<n
        best = None
        best_params = None
        best_expected = -1e18
        for k in kappa_grid:
            for b in beta_grid:
                probs, costs, cmax = holonomy_qaoa_distribution(G, k, b)
                expected = float(np.sum(probs * costs))
                if expected > best_expected:
                    best_expected = expected
                    best = (probs, costs, cmax)
                    best_params = (k, b)
        probs, costs, cmax = best
        x_star = int(np.argmax(probs))
        bits = np.array([(x_star>>i)&1 for i in range(G.n)], dtype=np.uint8)
        best_cost = G.cost(bits)
        return dict(kappa=best_params[0], beta=best_params[1], expected=best_expected,
                    best_cost=best_cost, probs=probs, costs=costs, cmax=cmax)

    kappa_grid = np.linspace(0.0, 2.0, 81)
    beta_grid  = np.linspace(0.0, np.pi/2, 81)
    qaoa_res = run_holonomy_qaoa_scan(G, kappa_grid, beta_grid)

    # Bloch probe
    bloch = run_bloch_probe()

    # Summaries & artifacts
    # Estimate A_star for ±2π/3 via slope
    A_star = (2*np.pi/3)/abs(slope) if slope!=0 else np.nan

    # Identify closest triadic hit
    target_angles = np.array([2*np.pi/3, -2*np.pi/3])
    tri_phases = np.array(triadic_df["phase"].values)
    tri_areas  = np.array(triadic_df["area"].values)
    # distance to nearest triadic target modulo 2π
    def circ_dist(phi, target):
        return min(abs(((phi-target)+np.pi)%(2*np.pi)-np.pi),
                   abs(((phi+2*np.pi-target)+np.pi)%(2*np.pi)-np.pi),
                   abs(((phi-2*np.pi-target)+np.pi)%(2*np.pi)-np.pi))
    dists = np.array([min(circ_dist(phi, target_angles[0]), circ_dist(phi, target_angles[1])) for phi in tri_phases])
    best_idx = int(np.argmin(dists))
    tri_best = dict(area=float(tri_areas[best_idx]), phase=float(tri_phases[best_idx]), delta=float(dists[best_idx]))

    summary = {
        "graph_n": n,
        "edges": edges,
        "commutator_norm_2": float(commutator(H_cost(edges,n), Xsum(n)).astype(np.complex128).view(np.float64).max() if False else float(np.linalg.norm(commutator(HC,HM), ord=2))),
        "phase_area_slope": float(slope),
        "phase_area_intercept": float(intercept),
        "A_star_from_slope_for_2pi_over_3": float(A_star),
        "orientation_flip": {"phi_plus": float(phi_plus), "phi_minus": float(phi_minus), "sum": float(phi_plus+phi_minus)},
        "shape_invariance": {"phi_rect_tall": float(phi_rect_tall), "phi_diamond": float(phi_diamond)},
        "schedule_invariance": {"uniform": float(phi_uniform), "ease": float(phi_ease), "zigzag": float(phi_zigzag)},
        "qaoa": {"kappa": float(qaoa_res["kappa"]), "beta": float(qaoa_res["beta"]), "expected": float(qaoa_res["expected"]), "best_cost": float(qaoa_res["best_cost"])},
        "bloch": bloch,
        "triadic_best": tri_best
    }

    summary_json = os.path.join(ARTDIR, "unified_summary.json")
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    # Plots
    # 1) Phase vs area (rectangles)
    plt.figure()
    plt.plot(areas_sorted, ph_rect_arr, marker="o", linestyle="none")
    xfit = np.linspace(areas_sorted.min(), areas_sorted.max(), 200)
    yfit = slope*xfit + intercept
    plt.plot(xfit, yfit)
    plt.title(f"Phase vs area (rectangles); slope≈{slope:.3f}")
    plt.xlabel("Area Δα·Δβ")
    plt.ylabel("Phase (radians)")
    plot1 = os.path.join(ARTDIR, "unified_phase_vs_area.png")
    plt.savefig(plot1, dpi=160)
    plt.close()

    # 2) Orientation flip check
    plt.figure()
    plt.bar([0,1], [phi_plus, -phi_minus])
    plt.title("Orientation flip: φ(+) vs -φ(−)")
    plt.xlabel("Loop orientation")
    plt.ylabel("Phase (radians)")
    plot2 = os.path.join(ARTDIR, "unified_orientation_flip.png")
    plt.savefig(plot2, dpi=160)
    plt.close()

    # 3) Shape & schedule invariance
    plt.figure()
    vals = [phi_uniform, phi_ease, phi_zigzag, phi_rect_tall, phi_diamond]
    labels = ["uniform","ease","zigzag","rect_tall","diamond"]
    plt.bar(range(len(vals)), vals)
    plt.xticks(range(len(vals)), labels, rotation=15)
    plt.title("Same area: shape & schedule invariance test")
    plt.ylabel("Phase (radians)")
    plot3 = os.path.join(ARTDIR, "unified_shape_schedule.png")
    plt.savefig(plot3, dpi=160)
    plt.close()

    # 4) Triadic scan
    plt.figure()
    plt.plot(triadic_df["area"].values, ((triadic_df["phase"]+np.pi)%(2*np.pi))-np.pi, marker=".", linestyle="none")
    plt.axhline(2*np.pi/3, linestyle="--")
    plt.axhline(-2*np.pi/3, linestyle="--")
    plt.title("Phase modulo 2π vs area (triadic markers at ±2π/3)")
    plt.xlabel("Area")
    plt.ylabel("Phase mod 2π (rad)")
    plot4 = os.path.join(ARTDIR, "unified_triadic_plot.png")
    plt.savefig(plot4, dpi=160)
    plt.close()

    # 5) Interference aggregation: prob mass vs cut value
    probs = qaoa_res["probs"]; costs = qaoa_res["costs"]
    df_out = pd.DataFrame({"cost": costs, "prob": probs})
    agg = df_out.groupby("cost", as_index=False)["prob"].sum().sort_values("cost")
    plt.figure()
    plt.plot(agg["cost"].values, agg["prob"].values, marker="o")
    plt.title(f"Interference prob mass vs cut; κ={qaoa_res['kappa']:.3f}, β={qaoa_res['beta']:.3f}")
    plt.xlabel("Cut value")
    plt.ylabel("Probability mass")
    plot5 = os.path.join(ARTDIR, "unified_interference_mass.png")
    plt.savefig(plot5, dpi=160)
    plt.close()

    return {
        "rect_csv": rect_csv,
        "triadic_csv": triadic_csv,
        "summary_json": summary_json,
        "plots": [plot1, plot2, plot3, plot4, plot5],
        "rect_df_sample": rect_df.sort_values("area").head(12),
        "summary": summary
    }

if __name__ == "__main__":
    res = run_experiment()
    print("Polar-time unified test — concise report")
    print("----------------------------------------")
    s = res["summary"]
    print(f"Graph n=6, edges={len(s['edges'])}; ||[H_C,H_M]||_2 ≈ {s['commutator_norm_2']:.3f}")
    print(f"Phase–area fit (rectangles): slope ≈ {s['phase_area_slope']:.4f}, intercept ≈ {s['phase_area_intercept']:.4f}")
    print(f"A* for ±2π/3 from slope: {s['A_star_from_slope_for_2pi_over_3']:.4f}")
    print(f"Orientation flip: φ(+) ≈ {s['orientation_flip']['phi_plus']:.4f}, φ(−) ≈ {s['orientation_flip']['phi_minus']:.4f}, sum ≈ {s['orientation_flip']['sum']:.4e}")
    print(f"Shape invariance (same area): rect_tall ≈ {s['shape_invariance']['phi_rect_tall']:.4f}, diamond ≈ {s['shape_invariance']['phi_diamond']:.4f}")
    print(f"Schedule invariance (same area): uniform ≈ {s['schedule_invariance']['uniform']:.4f}, ease ≈ {s['schedule_invariance']['ease']:.4f}, zigzag ≈ {s['schedule_invariance']['zigzag']:.4f}")
    print(f"Interference (p=1): κ ≈ {s['qaoa']['kappa']:.3f}, β ≈ {s['qaoa']['beta']:.3f}, E[cost] ≈ {s['qaoa']['expected']:.3f}, best_cost = {s['qaoa']['best_cost']:.1f}")
    print(f"Best triadic hit: area ≈ {s['triadic_best']['area']:.4f}, phase ≈ {s['triadic_best']['phase']:.4f}, |Δ| ≈ {s['triadic_best']['delta']:.3e}")
    print(f"Bloch probe slope ≈ {s['bloch']['slope']:.4f}, A* ≈ {s['bloch']['A_star']:.4f}, eps={s['bloch']['eps']:.2f}")
    print("Artifacts:")
    print(f"- CSV: {res['rect_csv']}")
    print(f"- CSV: {res['triadic_csv']}")
    print(f"- JSON: {res['summary_json']}")
    for p in res["plots"]:
        print(f"- PNG: {p}")
