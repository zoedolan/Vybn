# Vybn — Polar-Time Manifold · Unification + Recursion + Moiré + Residue (full script)
# Matplotlib only. One figure per save. Colors only for phase encodings.
# Artifacts -> ./vybn_show/

import numpy as np, math, json
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path

# -------------------- Output directory --------------------
OUT = Path.cwd() / "vybn_show"
OUT.mkdir(parents=True, exist_ok=True)

# -------------------- Life on a torus ---------------------
def life_step_torus(grid: np.ndarray):
    n = (np.roll(grid,1,0)+np.roll(grid,-1,0)+np.roll(grid,1,1)+np.roll(grid,-1,1) +
         np.roll(np.roll(grid,1,0),1,1)+np.roll(np.roll(grid,1,0),-1,1) +
         np.roll(np.roll(grid,-1,0),1,1)+np.roll(np.roll(grid,-1,0),-1,1))
    born = (grid==0)&(n==3)
    survive = (grid==1)&((n==2)|(n==3))
    return (born|survive).astype(np.uint8), n

def add_gosper_glider_gun(g, top=1, left=1):
    coords=[(5,1),(5,2),(6,1),(6,2),(5,11),(6,11),(7,11),(4,12),(8,12),
            (3,13),(9,13),(3,14),(9,14),(6,15),(4,16),(8,16),(5,17),(6,17),(7,17),(6,18),
            (3,21),(4,21),(5,21),(3,22),(4,22),(5,22),(2,23),(6,23),(1,25),(2,25),(6,25),(7,25),
            (3,35),(4,35),(3,36),(4,36)]
    H,W=g.shape
    for (x,y) in coords: g[(top+y)%H,(left+x)%W]=1

def init_seed(H,W,guns=1,noise=0.015):
    g=np.zeros((H,W),np.uint8)
    if guns>=1: add_gosper_glider_gun(g, top=H//3-5, left=2)
    if guns>=2: add_gosper_glider_gun(g, top=2*H//3-5, left=W//2-20)
    if noise>0: g|=(np.random.rand(H,W)<noise).astype(np.uint8)
    return g

# ------------- Polar-time memory & curvature --------------
def polar_field(H,W,kappa=1.0,center=None):
    cy,cx=(H//2,W//2) if center is None else center
    y=np.arange(H)[:,None]; x=np.arange(W)[None,:]
    return kappa*np.arctan2(y-cy,x-cx)

def update_memory(M,grid,angle_xy,alpha):
    return alpha*M + grid*np.exp(1j*angle_xy)

def lattice_curvature(M,eps=1e-9):
    mag=np.abs(M)
    u=np.ones_like(M,dtype=np.complex128)
    nz=mag>eps
    u[nz]=M[nz]/mag[nz]
    Ux=np.roll(u,-1,1)*np.conj(u)
    Uy=np.roll(u,-1,0)*np.conj(u)
    Up=Ux*np.roll(Uy,-1,1)*np.conj(np.roll(Ux,-1,0))*np.conj(Uy)
    return np.angle(Up)

# --------------- Order parameters (Life) -------------------
def F_k(grid_like,k,xs,ys):
    N=grid_like.shape[0]   # assume square normalization; fine for motif readout
    phase=2*np.pi*(k[0]*xs + k[1]*ys)/N
    return (grid_like*np.exp(1j*phase)).sum()

def embed_cos_sin(thetas):
    T,M=thetas.shape
    X=np.zeros((T,2*M))
    for i in range(M):
        X[:,2*i], X[:,2*i+1] = np.cos(thetas[:,i]), np.sin(thetas[:,i])
    return X

def pca3(X):
    Xc=X - X.mean(0,keepdims=True)
    U,S,Vt=np.linalg.svd(Xc,full_matrices=False)
    comps=Vt[:3].T
    Y=Xc@comps
    return Y,S[:3]

# ----------------- Weave (moiré) helpers ------------------
def make_canvas(N=480):
    y,x=np.ogrid[:N,:N]
    cy,cx=N//2,N//2
    r=np.sqrt((x-cx)**2+(y-cy)**2)
    th=np.arctan2(y-cy,x-cx)
    radial=np.exp(-r**2/(2*(0.35*N)**2))
    return x,y,r,th,radial

def timemap_from_x(T,x,N):
    t=(x+0.5)/N
    return np.clip(np.round(t*(T-1)).astype(int),0,T-1)

def timemap_spiral(T,r,th,N,turns=2.0):
    s=(r/(0.48*N)) + (turns/(2*np.pi))*(th+np.pi)
    s=(s-s.min())/(s.max()-s.min()+1e-12)
    return np.clip(np.round(s*(T-1)).astype(int),0,T-1)

def combine_phases(weights,thetas):
    W=np.array(weights,float)
    W=W/(np.linalg.norm(W)+1e-12)
    z=np.zeros(thetas.shape[:2],dtype=np.complex128)
    for k,w in enumerate(W): z += w*np.exp(1j*thetas[...,k])
    return np.angle(z)

def make_hsv(h,s,v):
    hsv=np.zeros(h.shape+(3,),float)
    hsv[...,0]=(h+np.pi)/(2*np.pi)
    hsv[...,1]=np.clip(s,0,1)
    hsv[...,2]=np.clip(v,0,1)
    return hsv_to_rgb(hsv)

def phase_vortices(hue):
    z=np.exp(1j*hue)
    z1=z[:-1,:-1]; z2=z[:-1,1:]; z3=z[1:,1:]; z4=z[1:,:-1]
    loop=(z2*np.conj(z1))*(z3*np.conj(z2))*(z4*np.conj(z3))*(z1*np.conj(z4))
    return np.angle(loop)

# --------------- Robust NN resize (Windows-safe) ----------
def nn_resize(x, Ht, Wt):
    """Nearest-neighbor resize via repetition; exact (Ht, Wt)."""
    ry = max(1, int(round(Ht / x.shape[0])))
    rx = max(1, int(round(Wt / x.shape[1])))
    y = np.repeat(np.repeat(x, ry, axis=0), rx, axis=1)
    return y[:Ht, :Wt]

def multiscale_from_life(g0, Ht, Wt):
    """Multi-scale source from Life (recursion in scale), resized to (Ht, Wt)."""
    g0 = g0.astype(float)
    g1 = g0[::2, ::2]
    g2 = g0[::4, ::4]
    g0r = nn_resize(g0, Ht, Wt)
    g1r = nn_resize(g1, Ht, Wt)
    g2r = nn_resize(g2, Ht, Wt)
    src = 1.0*g0r + 0.5*g1r + 0.25*g2r
    return src / (src.max() + 1e-12)

# ---------------- Triadic pulses (2π/3) -------------------
def triad_pulse(t, period, gain):
    return gain * (1.0 if (t % period)==0 else 0.0)

# ------------------ SU(2) quaternion core -----------------
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
def q_inv(q): return q.conj()
def q_axis_angle(nx,ny,nz,theta):
    n=(nx*nx+ny*ny+nz*nz)**0.5
    if n<EPS: return q_id()
    nx/=n; ny/=n; nz/=n; h=0.5*theta
    return Q(math.cos(h),nx*math.sin(h),ny*math.sin(h),nz*math.sin(h))
def su2_geodesic_angle(q):
    w=max(-1.0,min(1.0,q.normalize().w))
    return 2.0*math.acos(abs(w))

def q_to_mat(q: Q):
    """Map unit quaternion q=(w,x,y,z) to SU(2) matrix: U = w I + i (x σx + y σy + z σz)."""
    w,x,y,z = q.w, q.x, q.y, q.z
    return np.array([[w + 1j*z,    x + 1j*y],
                     [ -x + 1j*y,  w - 1j*z]], dtype=np.complex128)

# --- Reversible SU(2) with recursive Life sourcing & pulses; returns final quaternions for U(2) readout
def reversible_cutglue_recoupled(H=36,W=48,steps=90,E_over_hbar=1.0,step_dt=0.08,
                                 kappa=1.0,omega=2*np.pi/3*0.12, seam_j=None,
                                 life_sequence=None, baseline=0.6, coupling=2.0):
    if life_sequence is None:
        life_sequence=[np.zeros((H,W),np.uint8)]
    U=np.empty((H,W),dtype=object)
    for i in range(H):
        for j in range(W):
            U[i,j]=q_id()
    th0=polar_field(H,W,kappa=kappa)
    local_A=lambda a: q_axis_angle(1,0,0,a)
    local_B=lambda a: q_axis_angle(0,1,0,a)
    a_base=E_over_hbar*step_dt
    a2=a_base*a_base
    hol_accum=np.zeros((H-1,W-1))
    counts=0
    seam_j=W//2 if seam_j is None else seam_j

    for t in range(steps):
        life_grid_t = life_sequence[t % len(life_sequence)]
        src = multiscale_from_life(life_grid_t, H, W)
        mod = baseline + coupling*src
        # Triadic pulse stack (order-3 observable)
        twist = triad_pulse(t,3,0.03) + triad_pulse(t,9,0.015) + triad_pulse(t,27,0.007)
        a_grid = a_base*mod + twist

        Agrid=np.empty((H,W),dtype=object)
        Bgrid=np.empty((H,W),dtype=object)
        for i in range(H):
            for j in range(W):
                a_ij=float(a_grid[i,j])
                Agrid[i,j]=local_A(a_ij)
                Bgrid[i,j]=local_B(a_ij)

        U_AB=np.empty_like(U); U_BA=np.empty_like(U)
        for i in range(H):
            for j in range(W):
                U_AB[i,j]=(Agrid[i,j]*(Bgrid[i,j]*U[i,j])).normalize()
                U_BA[i,j]=(Bgrid[i,j]*(Agrid[i,j]*U[i,j])).normalize()

        def plaquette_angles(Uf):
            Hh,Ww=Uf.shape
            ang=np.zeros((Hh-1,Ww-1))
            for ii in range(Hh-1):
                for jj in range(Ww-1):
                    u00=Uf[ii,jj]; u01=Uf[ii,jj+1]; u11=Uf[ii+1,jj+1]; u10=Uf[ii+1,jj]
                    loop=(u00*q_inv(u01)*u11*q_inv(u10)).normalize()
                    ang[ii,jj]=su2_geodesic_angle(loop)
            return ang

        ang_AB = plaquette_angles(U_AB)
        ang_BA = plaquette_angles(U_BA)
        hol_accum += 0.5*(np.abs(ang_AB)+np.abs(ang_BA))
        counts += 1
        U = U_AB  # forward

    hol = hol_accum/max(counts,1)
    ratio = hol/(2.0*a2+1e-15)
    return hol, ratio, a_base, U

# ===================== MAIN RUN ===========================
if __name__ == "__main__":
    # --- Life + polar-time memory (matter layer) ---
    H,W,STEPS = 72,108,120
    grid = init_seed(H,W,guns=1,noise=0.015)
    polar = polar_field(H,W,1.0)
    alpha = 0.993
    omega = 2*np.pi/3*0.11
    xs,ys = np.meshgrid(np.arange(H),np.arange(W),indexing="ij")
    M = np.zeros((H,W),np.complex128)  # U(1) memory

    grids=[]; mem_mags=[]; curvs=[]; theta_series=[]
    modes=[(1,0),(0,1),(1,1),(2,1)]

    for t in range(STEPS):
        angle_xy=(omega*t + polar)%(2*np.pi)
        M = update_memory(M, grid, angle_xy, alpha)
        if t%2==0:
            grids.append(grid.copy())
            mem_mags.append(np.abs(M).copy())
            curvs.append(lattice_curvature(M).copy())
        theta_series.append([np.angle(F_k(grid,k,xs,ys)+1e-16) for k in modes])
        grid,_=life_step_torus(grid)

    theta_series=np.array(theta_series)
    np.savetxt(OUT/"theta_series.csv", theta_series, delimiter=",")

    # --- 3D manifold: static + GIF ---
    X=embed_cos_sin(theta_series)
    Y,S3=pca3(X)
    fig=plt.figure(figsize=(6,6)); ax=fig.add_subplot(111,projection="3d")
    ax.plot(Y[:,0],Y[:,1],Y[:,2],linewidth=1.2); ax.scatter(Y[0,0],Y[0,1],Y[0,2],s=25)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    fig.savefig(OUT/"manifold_pca3.png",dpi=170,bbox_inches="tight"); plt.close(fig)

    fig=plt.figure(figsize=(6,6)); ax=fig.add_subplot(111,projection="3d")
    line,=ax.plot(Y[:,0],Y[:,1],Y[:,2],linewidth=1.2); ax.scatter(Y[0,0],Y[0,1],Y[0,2],s=25)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    def init_view(): ax.view_init(elev=25,azim=30); return (line,)
    def update_view(frame): ax.view_init(elev=25,azim=30+frame); return (line,)
    FuncAnimation(fig,update_view,frames=100,init_func=init_view,blit=False).save(OUT/"manifold_pca3.gif",writer=PillowWriter(fps=20)); plt.close(fig)

    # --- Life + memory overlay GIF ---
    mem_stack=np.array(mem_mags); mem_norm=mem_stack/(mem_stack.max()+1e-12) if mem_stack.size else mem_stack
    fig=plt.figure(figsize=(6,4)); ax=fig.add_subplot(111)
    im=ax.imshow(grids[0]+0.35*mem_norm[0],interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
    def upd_gm(f): im.set_data(grids[f]+0.35*mem_norm[f]); return (im,)
    FuncAnimation(fig,upd_gm,frames=len(grids),blit=False).save(OUT/"life_memory.gif",writer=PillowWriter(fps=14)); plt.close(fig)

    # --- Curvature evolution GIF ---
    curv_stack=np.array(curvs)
    vmin,vmax=np.percentile(curv_stack,1),np.percentile(curv_stack,99)
    fig=plt.figure(figsize=(6,4)); ax=fig.add_subplot(111)
    imc=ax.imshow(curv_stack[0],interpolation="nearest",vmin=vmin,vmax=vmax); ax.set_xticks([]); ax.set_yticks([])
    def upd_c(f): imc.set_data(curv_stack[f]); return (imc,)
    FuncAnimation(fig,upd_c,frames=curv_stack.shape[0],blit=False).save(OUT/"curvature_evolution.gif",writer=PillowWriter(fps=14)); plt.close(fig)

    # --- Phase-weaves + vortex map ---
    T,K=theta_series.shape; N=480
    x,y,r,th,radial=make_canvas(N); idx_x=timemap_from_x(T,x,N); idx_sp=timemap_spiral(T,r,th,N,turns=2.0)
    phase_fields_x=[]; phase_fields_sp=[]
    for d in range(min(4,K)):
        theta_t=theta_series[:,d]
        carrier1=(2*x+1*y)*(2*np.pi/N)
        carrier2=(3*x+2*y)*(2*np.pi/N)
        phase_fields_x.append((theta_t[idx_x]+0.45*carrier1)%(2*np.pi))
        phase_fields_sp.append((theta_t[idx_sp]+0.45*carrier2)%(2*np.pi))
    phase_fields_x=np.stack(phase_fields_x,axis=-1); phase_fields_sp=np.stack(phase_fields_sp,axis=-1)
    weights=np.array([1.0,math.sqrt(2.0),math.sqrt(3.0),math.pi])[:phase_fields_x.shape[-1]]
    sat=radial; val=0.4+0.6*radial

    h_master_x=combine_phases(weights,phase_fields_x); rgb_master_x=make_hsv(h_master_x,sat,val)
    fig=plt.figure(figsize=(6,6)); ax=fig.add_subplot(111); ax.imshow(rgb_master_x,interpolation="bilinear"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Phase-weave · linear"); fig.savefig(OUT/"weave_master_linear.png",dpi=170,bbox_inches="tight"); plt.close(fig)

    h_master_sp=combine_phases(weights,phase_fields_sp); rgb_master_sp=make_hsv(h_master_sp,sat,val)
    fig=plt.figure(figsize=(6,6)); ax=fig.add_subplot(111); ax.imshow(rgb_master_sp,interpolation="bilinear"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Phase-weave · spiral"); fig.savefig(OUT/"weave_master_spiral.png",dpi=170,bbox_inches="tight"); plt.close(fig)

    for k in range(phase_fields_x.shape[-1]):
        rgb_k=make_hsv(phase_fields_x[...,k],sat,val)
        fig=plt.figure(figsize=(6,6)); ax=fig.add_subplot(111); ax.imshow(rgb_k,interpolation="bilinear"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"Phase-weave · dim {k}"); fig.savefig(OUT/f"weave_dim{k}_linear.png",dpi=160,bbox_inches="tight"); plt.close(fig)

    winding=phase_vortices(h_master_x); disp=np.clip(winding/(2*np.pi),-1,1)
    fig=plt.figure(figsize=(6,6)); ax=fig.add_subplot(111); ax.imshow(disp,interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Weave defect map"); fig.savefig(OUT/"weave_vortex_map.png",dpi=170,bbox_inches="tight"); plt.close(fig)

    # --- SU(2) with recursion-on coupling: Life -> field (multi-scale) + triadic pulses ---
    life_seq_for_su2 = [g for i,g in enumerate(grids) if i%2==0]  # modest subsample
    hol, ratio, a_base, U_quat = reversible_cutglue_recoupled(
        H=36, W=48, steps=90, life_sequence=life_seq_for_su2,
        baseline=0.6, coupling=2.0
    )

    fig=plt.figure(figsize=(7,5)); ax=fig.add_subplot(111)
    im=ax.imshow(hol,interpolation="nearest"); ax.set_title("SU(2) residue (recursion-on)"); fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
    fig.savefig(OUT/"su2_holonomy_heatmap.png",dpi=170,bbox_inches="tight"); plt.close(fig)

    vals=ratio.flatten(); vals=vals[np.isfinite(vals)&(vals>0)]
    fig=plt.figure(figsize=(7,4)); ax=fig.add_subplot(111)
    ax.hist(vals,bins=40); ax.set_xlabel("angle / (2 a^2)"); ax.set_ylabel("count"); ax.set_title("Area-law calibration")
    fig.savefig(OUT/"su2_area_law_hist.png",dpi=170,bbox_inches="tight"); plt.close(fig)

    with open(OUT/"su2_certificate.json","w") as f:
        json.dump({"a_base":float(a_base),
                   "area_ratio_median":float(np.median(vals)) if vals.size else None,
                   "area_ratio_mean":float(np.mean(vals)) if vals.size else None,
                   "n_plaquettes":int(hol.size)}, f, indent=2)

    # --- U(2) readout: unify U(1) phase with SU(2) (trace + traceless)
    H2,W2 = hol.shape[0]+1, hol.shape[1]+1     # plaquette grid implies (H2,W2) vertex grid
    # final Life phase field resampled to SU(2) lattice size
    phi_final = np.angle(M)                     # use final U(1) memory phase
    phi_resized = nn_resize(phi_final, H2, W2)  # U(1) at SU(2) resolution

    # build SU(2) matrices from final quaternions
    Qmat = np.empty((H2,W2,2,2),dtype=np.complex128)
    for i in range(H2):
        for j in range(W2):
            q = U_quat[i,j]
            Qmat[i,j] = q_to_mat(q)

    # compute U(2) plaquette loop; save U(1) flux heatmap and compare with SU(2) residue
    Phi_flux = np.zeros((H2-1,W2-1))
    Su2_angle = np.zeros_like(Phi_flux)
    for i in range(H2-1):
        for j in range(W2-1):
            U00 = np.exp(0.5j*phi_resized[i,j])   * Qmat[i,j]
            U01 = np.exp(0.5j*phi_resized[i,j+1]) * Qmat[i,j+1]
            U11 = np.exp(0.5j*phi_resized[i+1,j+1])*Qmat[i+1,j+1]
            U10 = np.exp(0.5j*phi_resized[i+1,j]) * Qmat[i+1,j]
            Wloop = U00 @ U01 @ U11.conj().T @ U10.conj().T

            phi_sq = np.angle(np.linalg.det(Wloop))         # trace piece (U(1))
            W_su2  = Wloop / np.exp(0.5j*phi_sq)            # strip trace to get SU(2)
            su2ang = np.arccos(np.clip(0.5*(np.trace(W_su2)).real, -1, 1))
            Phi_flux[i,j] = phi_sq
            Su2_angle[i,j] = su2ang

    # save U(1) flux (trace) map
    fig=plt.figure(figsize=(7,5)); ax=fig.add_subplot(111)
    im=ax.imshow(Phi_flux,interpolation="nearest"); ax.set_title("U(1) trace flux (from U(2) loop)")
    fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
    fig.savefig(OUT/"u1_trace_flux.png",dpi=170,bbox_inches="tight"); plt.close(fig)

    # sanity: SU(2) angle from U(2) readout vs direct residue (close patterns)
    fig=plt.figure(figsize=(7,5)); ax=fig.add_subplot(111)
    im=ax.imshow(Su2_angle,interpolation="nearest"); ax.set_title("SU(2) geodesic angle (from U(2) loop)")
    fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
    fig.savefig(OUT/"su2_angle_from_u2.png",dpi=170,bbox_inches="tight"); plt.close(fig)

    # Done.
    print(f"Gallery written to: {OUT}")
