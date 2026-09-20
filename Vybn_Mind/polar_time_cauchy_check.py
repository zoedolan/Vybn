"""Check the free-evolution obstruction in THEORY.md, not another navigator.

Local symbolic calculus and exact rational witnesses; no network/model/GPU calls.
Python 3 + existing SymPy/mpmath. Units c=hbar=1. Angular L2 norms use dtheta/2pi.
Standard ultrahyperbolic instability, not a new theorem or universal two-time no-go.
The spatially homogeneous massless sector can be normalized in a periodic box.
The local-interaction calculation is a SUPPORT counterexample and forced linear
response, not a numerical solution of the full nonlinear field theory.
"""
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import mpmath as mp
import sympy as sp


def main():
    checks = []
    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)
    r, R = sp.symbols('r R', positive=True)
    theta, tau, t = sp.symbols('theta tau t', real=True)
    n = sp.symbols('n', integer=True, positive=True)
    phase = sp.exp(sp.I*n*theta)
    u = (r/R)**n * phase
    radial_operator = sp.diff(u,r,2) + sp.diff(u,r)/r + sp.diff(u,theta,2)/r**2
    check('exact_polar_field_residual_zero',sp.simplify(radial_operator)==0)
    check('angular_periodicity',sp.simplify(u.subs(theta,theta+2*sp.pi)-u)==0)
    check('unit_final_mode_amplitude',sp.simplify(u.subs(r,R)/phase)==1)
    check('initial_value_2_to_minus_n',sp.simplify(u.subs({r:1,R:2})/phase)==2**(-n))
    check('initial_radial_derivative',sp.simplify(sp.diff(u,r).subs({r:1,R:2})/phase)==n*2**(-n))

    # r/r0 = exp(tau): the two-timelike sector is Laplace, not wave, in tau,theta.
    bad = sp.exp(n*(tau-sp.log(2)))*phase
    good = 2**(-n)*(sp.cos(n*tau)+sp.sin(n*tau))*phase
    check('log_radial_laplace_equation',sp.simplify(sp.diff(bad,tau,2)+sp.diff(bad,theta,2))==0)
    check('one_time_sign_control_wave_equation',sp.simplify(sp.diff(good,tau,2)-sp.diff(good,theta,2))==0)
    check('same_initial_value',sp.simplify((good-bad).subs(tau,0))==0)
    check('same_initial_velocity',sp.simplify(sp.diff(good-bad,tau).subs(tau,0))==0)
    # Smooth at the origin: r^n exp(in theta)=(T+i S)^n is a polynomial.
    T,S = sp.symbols('T S', real=True)
    for j in (1,2,3,7,16):
        polynomial = (T+sp.I*S)**j
        check('cartesian_regular_harmonic_'+str(j),sp.simplify(sp.diff(polynomial,T,2)+sp.diff(polynomial,S,2))==0)

    # Every conserved real symmetric quadratic form on the FULL n-mode phase
    # space is indefinite (or zero); no positive norm is secretly conserved.
    A=sp.Matrix([[0,1],[n**2,0]])
    a,b,c=sp.symbols('a b c',real=True); G=sp.Matrix([[a,b],[b,c]])
    solved=sp.solve(list(A.T*G+G*A),(a,b),dict=True)
    check('all_conserved_quadratic_forms_solved',solved==[{a:-c*n**2,b:0}])
    check('conserved_form_not_positive_definite',sp.factor(G.subs(solved[0]).det())==-c**2*n**2)
    A1=sp.Matrix([[0,1],[-n**2,0]]); G1=sp.diag(n**2,1)
    check('one_time_positive_energy_conserved',A1.T*G1+G1*A1==sp.zeros(2))

    # Cartesian check for the actual metric, allowing mass and spatial momentum.
    x,s,k,kappa,mu,lam=sp.symbols('x s k kappa mu lam',real=True)
    psi=sp.exp(lam*t+sp.I*kappa*s+sp.I*k*x)
    kg=(sp.diff(psi,t,2)+sp.diff(psi,s,2)-sp.diff(psi,x,2)+mu**2*psi)/psi
    check('cartesian_massive_dispersion',sp.simplify(kg.subs(lam**2,kappa**2-k**2-mu**2))==0)

    mp.mp.dps=80
    rows=[]
    for j in (4,8,16,32,64,128):
        amp=Fraction(1,2**j)
        data_norm_sq=(1+2*j*j)*amp*amp
        # Exact coefficient residual is also checked independently of SymPy.
        check('exact_integer_mode_residual_'+str(j),j*(j-1)+j-j*j==0)
        q=mp.mpf(amp.numerator)/amp.denominator
        control=q*(mp.cos(j*mp.log(2))+mp.sin(j*mp.log(2)))
        check('one_time_bound_'+str(j),abs(control)<=mp.sqrt(2)*q)
        rows.append(dict(n=j,initial_amplitude=float(amp),
            initial_H1_times_L2_norm=float(mp.sqrt(mp.mpf(data_norm_sq.numerator)/data_norm_sq.denominator)),
            radial_final_amplitude=1,one_time_final_amplitude=float(control),
            one_time_final_bound=float(mp.sqrt(2)*q)))

    # A nonzero spatial-frequency check for the METRIC-DERIVED sign convention:
    # f=J_n(r)/J_n(2), f''+f'/r+(1-n^2/r^2)f=0. Not the printed +H_spatial^2 sign.
    bessel=[]
    for j in (8,16,32,64):
        den=mp.besselj(j,2)
        f=lambda z: mp.besselj(j,z)/den
        z=mp.mpf('1.3'); fp=mp.diff(f,z); fpp=mp.diff(f,z,2)
        residual=fpp+fp/z+(1-j*j/z**2)*f(z)
        scale=abs(fpp)+abs(fp/z)+abs((1-j*j/z**2)*f(z))
        check('nonzero_spatial_frequency_residual_'+str(j),abs(residual)<=mp.mpf('1e-60')*scale)
        bessel.append(dict(n=j,initial_amplitude=float(f(1)),initial_derivative=float(mp.diff(f,1)),
            final_amplitude=float(f(2)),relative_equation_residual=float(abs(residual)/scale)))

    # Even a simple Fourier filter rejecting negative squared frequencies is
    # NOT invariant under unrestricted local polynomial interactions.
    # Each tuple is (spatial k, second-time kappa); all omitted components zero.
    inputs=((5,3),(5,3),(-10,6))
    omega_sq=[kk*kk-ss*ss for kk,ss in inputs]
    summed=tuple(sum(v[i] for v in inputs) for i in (0,1))
    out_sq=summed[0]**2-summed[1]**2
    check('three_input_modes_strictly_stable',all(v>0 for v in omega_sq))
    check('cubic_product_outside_stable_support',out_sq==-144)
    # Real fields include these Fourier components and their conjugates. The
    # selected cubic product has temporal frequency 4+4+8=16 and growth rate 12.
    response=(sp.cosh(12*t)-sp.cos(16*t))/400
    check('forced_response_equation',sp.simplify(sp.diff(response,t,2)-144*response-sp.cos(16*t))==0)
    check('forced_response_zero_initial_data',response.subs(t,0)==0 and sp.diff(response,t).subs(t,0)==0)
    check('one_time_generated_mode_stable',summed[0]**2+summed[1]**2==144)

    report=dict(schema='polar-time-cauchy-check-1',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        examined_theory_sha256='acab3dc33424adb9ba6082fbf2f43f33230c9fd9243173fd83c7bf18de176bf4',
        libraries=dict(sympy=sp.__version__,mpmath=mp.__version__),
        units='c=hbar=1; r0=1,r1=2; angular norm uses dtheta/(2*pi)',
        exact_family='u_n(r,theta)=(r/2)^n exp(i*n*theta); integer n>0',
        conclusion='unrestricted radial Cauchy evolution is not continuous in fixed finite-order Sobolev norms',
        initial_norm_definition='squared H1(field) x L2(log-radial derivative) norm = (1+2*n^2)*4^(-n)',
        mode_rows=rows,nonzero_spatial_frequency_rows=bessel,
        mode_generator=dict(two_time='[[0,1],[n^2,0]]',one_time='[[0,1],[-n^2,0]]',
            all_two_time_conserved_forms='diag(-n^2*c,c); determinant=-n^2*c^2; never positive definite for n>0'),
        filter_failure=dict(input_wavevectors=inputs,input_squared_frequencies=omega_sq,
            generated_wavevector=summed,generated_squared_frequency=out_sq,
            local_interaction='a cubic field term, as in a quartic potential',
            unit_forcing_response='(cosh(12*t)-cos(16*t))/400; zero initial data',
            scope='Fourier-support nonclosure and first-order forced response, not a full nonlinear evolution'),
        checks_passed=len(checks),checks=checks,
        limits=['standard ultrahyperbolic obstruction, not a new theorem',
            'metric alone does not specify admissible data, constraints, or interactions',
            'a constrained/gauge or boundary-value two-time theory is not ruled out',
            'does not test consciousness or prove which time geometry is physical',
            'no empirical observation or quantum/provider/model call'])
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__': main()
