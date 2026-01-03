# %%
# Supplement: coarse search for the maximal CHSH S for each correlator family.
import math

def max_chsh(Efun, ngrid=121):
    # coarse grid over angles in [0, pi)
    xs = [i * (math.pi/(ngrid-1)) for i in range(ngrid)]
    Smax = 0.0
    argmax = None
    for a in xs:
        for ap in xs:
            for b in xs:
                for bp in xs:
                    S = abs(Efun(a-b) + Efun(a-bp) + Efun(ap-b) - Efun(ap-bp))
                    if S > Smax:
                        Smax = S
                        argmax = (a, ap, b, bp)
    return Smax, argmax

Smax_qm, _ = max_chsh(E_qm, 49)  # keep it modest for runtime
Smax_step, _ = max_chsh(E_triadic_step, 49)
Smax_smooth, _ = max_chsh(lambda x: E_triadic_smooth(x, 10.0), 49)

print("Coarse-grid CHSH maxima (ngrid=49):")
print(f"  QM cosine:        {Smax_qm:.6f} (target ~ 2.828)")
print(f"  Triadic step:     {Smax_step:.6f} (should not exceed 4 algebraic max)")
print(f"  Triadic smooth:   {Smax_smooth:.6f}")
