# -*- coding: utf-8 -*-
"""
nailbiter.py — Vybn | QPU commutator holonomy (Runtime V2, budget-aware, disciplined)

Key fixes vs v7:
  • tau_loop uses the scheduled circuit duration as-is (circuits already include m loops).
    We no longer multiply by m a second time. This corrects kappa_eff scaling.
  • Watchdog exits on error/cancel states; only enforces wall-clock while RUNNING.
  • Statistical error uses actual shot count n = sum(counts.values()) per PUB.
  • Transpiler seed is honored; comma-arg parsing strips whitespace.
  • Deterministic pair-preserving shuffle via sha256 of run_tag+plane.

Physics and flow unchanged:
  one transpile per plane (with loop multiplicity m in the template),
  interleaved cw/ccw per point, adaptive shots concentrated near the origin,
  nulls (θ=0, φ=0), plateau CV + same-sign gate, optional pair shuffling,
  kappa_eff reported when dt and durations are available.
"""

import os, sys, time, math, json, csv, argparse, statistics, hashlib, random
from typing import List, Tuple, Dict, Optional

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit import Parameter
from qiskit_ibm_runtime import QiskitRuntimeService, Session, SamplerV2 as Sampler
from qiskit_ibm_runtime.options import SamplerOptions


# ---------- tiny io ----------
def jprint(obj):
    print(json.dumps(obj, ensure_ascii=False))
    sys.stdout.flush()


def write_csv(path, rows, fieldnames=None):
    if not rows:
        return
    if fieldnames is None:
        keys = set()
        for r in rows: keys |= set(r.keys())
        fieldnames = sorted(keys)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows: w.writerow(r)


# ---------- helpers ----------
def p1_from_counts(counts: Dict[str, int]) -> Tuple[float, int]:
    n = int(sum(counts.values()))
    p1 = 0.0 if n == 0 else counts.get("1", 0) / n
    return p1, n


def sigma_p1(p: float, n: int) -> float:
    if n <= 0: return float("nan")
    # binomial stdev; we’re near p≈0.5 in this experiment
    return math.sqrt(max(0.0, p*(1.0 - p)) / n)


def plateau_score(slopes: List[float], k_small: int = 6) -> Tuple[float, bool]:
    if not slopes: return (float("nan"), False)
    xs = slopes[:max(1, min(k_small, len(slopes)))]
    if not xs: return (float("nan"), False)
    m = sum(xs)/len(xs)
    sd = statistics.pstdev(xs) if len(xs) > 1 else 0.0
    cv = float("inf") if abs(m) < 1e-12 else abs(sd/m)
    same = all(x >= 0 for x in xs) or all(x <= 0 for x in xs)
    return cv, same


def shape_params(area: float, shape: str, aspect: float) -> Tuple[float, float]:
    s = shape.lower()
    if s == "balanced":
        t = math.sqrt(area); return (t, t)
    if s == "long_theta":
        t = max(1e-9, aspect) * math.sqrt(area); return (t, area/max(t,1e-18))
    if s == "long_r":
        p = max(1e-9, aspect) * math.sqrt(area); return (area/max(p,1e-18), p)
    raise ValueError(f"unknown micro-shape: {shape}")


def make_schedule(n_points: int, max_angle: float) -> List[float]:
    amax = float(max_angle)**2
    return [amax * (i+1)/float(n_points) for i in range(n_points)]


def compute_adaptive_shots(area: float, amin: float, m: int,
                           base: int, mn: int, mx: int) -> int:
    # Concentrate budget near the origin. Scale ~ (amin/area)^1.5 and gently boost with sqrt(m).
    if area <= 0 or amin <= 0:
        return int(max(mn, min(mx, base)))
    exponent = 1.5
    scale = (amin / area) ** exponent
    scale *= max(1.0, math.sqrt(max(1, m)))
    shots = int(round(base * scale))
    return int(max(mn, min(mx, shots)))


def extract_counts(pub_result):
    # Prefer the V2 join_data().get_counts() path; fall back to data.meas or a scan.
    try:
        jd = getattr(pub_result, "join_data", None)
        if callable(jd):
            data = jd()
            gc = getattr(data, "get_counts", None)
            if callable(gc):
                return data.get_counts()
    except Exception:
        pass
    try:
        data = getattr(pub_result, "data", None)
        if data is not None:
            meas = getattr(data, "meas", None)
            if meas is not None and hasattr(meas, "get_counts"):
                return meas.get_counts()
            cr = getattr(data, "cr", None)
            if cr is not None and hasattr(cr, "get_counts"):
                return cr.get_counts()
            for name in dir(data):
                obj = getattr(data, name)
                if hasattr(obj, "get_counts"):
                    return obj.get_counts()
    except Exception:
        pass
    return {}


# ---------- circuit building ----------
def _apply_gate(qc: QuantumCircuit, name: str, angle, qubit=0):
    if name == "rx": qc.rx(angle, qubit)
    elif name == "ry": qc.ry(angle, qubit)
    elif name == "rz": qc.rz(angle, qubit)
    else: raise ValueError(f"unknown gate {name}")


def build_commutator_templates(plane: str = "xz", m_loops: int = 1
                               ) -> Tuple[QuantumCircuit, QuantumCircuit, Parameter, Parameter]:
    plane = (plane or "xz").lower()
    if plane == "xz": g1, g2 = "rz", "rx"
    elif plane == "yz": g1, g2 = "rz", "ry"
    elif plane == "xy": g1, g2 = "rx", "ry"
    else: raise ValueError("plane must be one of: xz, yz, xy")

    THETA = Parameter("theta")
    PHI   = Parameter("phi")

    def mk(order):
        qr = QuantumRegister(1, "q")
        cr = ClassicalRegister(1, "cr")
        qc = QuantumCircuit(qr, cr, name=f"comm_{plane}")
        qc.h(0)  # |+> prep so Z readout is sign-sensitive to tiny Ry-like residue
        for _ in range(max(1, int(m_loops))):
            for gate, s in order:
                if gate == "g1": _apply_gate(qc, g1, s*PHI, 0)
                elif gate == "g2": _apply_gate(qc, g2, s*THETA, 0)
                else: raise ValueError("bad order symbol")
        qc.measure(0, 0)
        return qc

    cw  = mk([("g1", +1), ("g2", +1), ("g1", -1), ("g2", -1)])
    ccw = mk([("g2", +1), ("g1", +1), ("g2", -1), ("g1", -1)])
    return cw, ccw, THETA, PHI


def sampler_for(mode_obj):
    opts = SamplerOptions()
    # Make cost behavior explicit and cheap. (Defaults are already lean; we pin them.)
    try:
        # Some SDKs gate these options; guard with hasattr to remain compatible.
        if hasattr(opts, "twirling"):
            opts.twirling.enable_gates = False
            opts.twirling.enable_measure = False
        if hasattr(opts, "dynamical_decoupling"):
            opts.dynamical_decoupling.enable = False
    except Exception:
        pass
    return Sampler(mode=mode_obj, options=opts)


def run_with_watchdog(job_callable, max_seconds: int):
    t0 = time.time()
    job = job_callable()
    jid_attr = getattr(job, "job_id", None)
    jid = jid_attr() if callable(jid_attr) else jid_attr
    jprint({"type": "job_submitted", "job_id": jid})
    # Enforce wall clock only while RUNNING; allow queue time.
    terminal_ok = {"done", "success", "succeeded", "completed"}
    terminal_bad = {"error", "errored", "failed", "cancelled", "canceled", "aborted"}
    while True:
        st = job.status()
        name = (getattr(st, "name", "") or "").lower()
        now = time.time()
        if name in ("running", "executing"):
            if now - t0 > max_seconds:
                try: job.cancel()
                except Exception: pass
                raise TimeoutError(f"run time exceeded {max_seconds}s; job cancelled")
        if name in terminal_ok:
            break
        if name in terminal_bad:
            # Bubble with context; let caller decide how to persist partials.
            raise RuntimeError(f"job ended in bad state: {name}")
        time.sleep(1.5)
    return job.result()


# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Vybn QPU commutator holonomy (Runtime V2)")
    p.add_argument("--backend", type=str, default="ibm_fez")
    p.add_argument("--instance", type=str, default=None)
    p.add_argument("--use-session", action="store_true")

    p.add_argument("--planes", type=str, default="xz", help="comma list: xz,yz,xy")
    p.add_argument("--micro-shapes", type=str, default="balanced,long_theta,long_r")
    p.add_argument("--aspect", type=float, default=4.0)

    p.add_argument("--n-points", type=int, default=16)
    p.add_argument("--max-angle", type=float, default=0.20)
    p.add_argument("--m", type=int, default=3, help="loop multiplicity per template")

    p.add_argument("--base-shots", type=int, default=256)
    p.add_argument("--min-shots", type=int, default=128)
    p.add_argument("--max-shots", type=int, default=2048)
    p.add_argument("--include-nulls", action="store_true")
    p.add_argument("--z-null", type=float, default=2.0, help="z-threshold for null pass")

    p.add_argument("--max-total-shots", type=int, default=250_000)
    p.add_argument("--max-seconds", type=int, default=120)

    p.add_argument("--seed", type=int, default=None, help="PRNG + transpiler seed")
    p.add_argument("--shuffle-pairs", action="store_true")
    p.add_argument("--run-tag", type=str, default="baseline")
    p.add_argument("--out", type=str, default="")
    return p.parse_args()


# ---------- main ----------
def main():
    args = parse_args()
    jprint({"type": "start", "note": "vybn qpu commutator v8", "ts": int(time.time())})

    service = QiskitRuntimeService() if args.instance is None else QiskitRuntimeService(instance=args.instance)
    backend = service.backend(args.backend)
    jprint({"type": "backend", "backend": args.backend, "ts": int(time.time())})

    # Parse comma args robustly
    planes = [t.strip() for t in (args.planes or "").split(",") if t.strip()]
    shapes = [s.strip() for s in (args.micro_shapes or "").split(",") if s.strip()]
    areas = make_schedule(args.n_points, args.max_angle)
    amin = min(areas) if areas else 0.0

    # Budget preflight (approximate; per-PUB shots dominate)
    def planned_sampler_shots():
        total = 0
        for plane in planes:
            for shape in shapes:
                for a in areas:
                    shots = compute_adaptive_shots(a, amin, args.m, args.base_shots, args.min_shots, args.max_shots)
                    total += 2 * shots  # cw + ccw
        if args.include_nulls and areas:
            shots_null = max(args.base_shots, args.min_shots)
            total += 4 * shots_null  # two nulls, cw+ccw each
        return total

    planned = planned_sampler_shots()
    jprint({"type": "budget_plan", "planned_sampler_shots": planned})
    if args.max_total_shots and planned > args.max_total_shots:
        raise RuntimeError(f"planned sampler shots {planned} exceed cap {args.max_total_shots}")

    # SNR proxy at smallest point (matches earlier prints: se ~ 1/sqrt(shots))
    if areas:
        shots0 = compute_adaptive_shots(areas[0], amin, args.m, args.base_shots, args.min_shots, args.max_shots)
        se_delta = 1.0 / math.sqrt(max(1, shots0))
        snr0 = (args.m * areas[0]) / se_delta
        jprint({"type": "preflight", "a_min": areas[0], "m": args.m, "snr_smallest_point_est": snr0})

    # dt context
    dt_seconds = getattr(backend, "dt", None)
    if dt_seconds is None:
        cfg = getattr(backend, "configuration", None)
        if callable(cfg):
            cfg = cfg()
        dt_seconds = getattr(cfg, "dt", None)
    if dt_seconds is not None:
        jprint({"type": "context", "dt_s": float(dt_seconds)})
    else:
        jprint({"type": "warn", "what": "no_dt", "note": "tau_loop/kappa_eff will be NaN"})

    all_rows = []
    verdicts = []

    # Random seed for reproducibility (pair shuffling only; transpiler seed is set below)
    if args.seed is not None:
        random.seed(args.seed)

    def run_plane_block(mode_obj, plane: str):
        # Build templates with m loops baked in; one compile per plane
        cw_tpl, ccw_tpl, THETA, PHI = build_commutator_templates(plane=plane, m_loops=args.m)
        cw_isa  = transpile(cw_tpl,  backend=backend, optimization_level=1,
                            scheduling_method="alap", seed_transpiler=args.seed)
        ccw_isa = transpile(ccw_tpl, backend=backend, optimization_level=1,
                            scheduling_method="alap", seed_transpiler=args.seed)

        # Scheduled durations (dt units) -> tau_loop seconds
        try:
            dur_dt_cw  = int(getattr(cw_isa,  "duration", 0) or 0)
            dur_dt_ccw = int(getattr(ccw_isa, "duration", 0) or 0)
        except Exception:
            dur_dt_cw = dur_dt_ccw = 0
        dur_dt = max(dur_dt_cw, dur_dt_ccw)
        tau_loop = (float(dur_dt) * float(dt_seconds)) if (dt_seconds is not None and dur_dt > 0) else float("nan")
        # NOTE: do NOT multiply by m here; m loops are already inside the scheduled circuit.

        # Build pair-preserving PUBs
        pair_list = []
        for shape in shapes:
            for a in areas:
                theta, phi = shape_params(a, shape, args.aspect)
                shots = compute_adaptive_shots(a, amin, args.m, args.base_shots, args.min_shots, args.max_shots)
                cw_pub  = (cw_isa,  {THETA: float(theta), PHI: float(phi)}, int(shots))
                ccw_pub = (ccw_isa, {THETA: float(theta), PHI: float(phi)}, int(shots))
                meta = (plane, shape, a, theta, phi, int(shots), float(tau_loop))
                pair_list.append((cw_pub, ccw_pub, meta))

        # Built-in nulls at smallest area (balanced map) if requested
        if args.include_nulls and areas:
            a0 = areas[0]
            theta0, phi0 = shape_params(a0, "balanced", args.aspect)
            shots_null = int(max(args.base_shots, args.min_shots))
            cw_t0  = (cw_isa,  {THETA: 0.0,        PHI: float(phi0)},  shots_null)
            ccw_t0 = (ccw_isa, {THETA: 0.0,        PHI: float(phi0)},  shots_null)
            cw_p0  = (cw_isa,  {THETA: float(theta0), PHI: 0.0},       shots_null)
            ccw_p0 = (ccw_isa, {THETA: float(theta0), PHI: 0.0},       shots_null)
            pair_list.append((cw_t0, ccw_t0, (plane, "null_theta", 0.0, 0.0, phi0, shots_null, float(tau_loop))))
            pair_list.append((cw_p0, ccw_p0, (plane, "null_phi",   0.0, theta0, 0.0, shots_null, float(tau_loop))))

        # Deterministic shuffle of pair order across shapes/areas if asked
        if args.shuffle_pairs and pair_list:
            seed_str = f"{args.run_tag}|{plane}|{len(pair_list)}"
            seed = int(hashlib.sha256(seed_str.encode()).hexdigest()[:8], 16)
            rng = random.Random(seed)
            rng.shuffle(pair_list)

        pubs = []
        meta_seq = []
        for cw_pub, ccw_pub, meta in pair_list:
            pubs.append(cw_pub)
            meta_seq.append(("cw",) + meta)
            pubs.append(ccw_pub)
            meta_seq.append(("ccw",) + meta)

        # Submit with watchdog
        sampler = sampler_for(mode_obj)
        try:
            result = run_with_watchdog(lambda: sampler.run(pubs), max_seconds=int(args.max_seconds))
        except TimeoutError as e:
            jprint({"type": "timeout", "message": str(e)})
            if args.out and all_rows:
                write_csv(args.out + ".csv", all_rows)
            return
        except RuntimeError as e:
            jprint({"type": "runtime_error", "message": str(e)})
            if args.out and all_rows:
                write_csv(args.out + ".csv", all_rows)
            return

        # Parse results
        per_pub = []
        # SamplerV2 returns an iterable of per-PUB results; we index in order
        for i, pubres in enumerate(result):
            which, plane_i, shape, area, theta, phi, shots_req, tau_s = meta_seq[i]
            counts = extract_counts(pubres) or {}
            p1, n_actual = p1_from_counts(counts)
            # use actual shots for sigma
            p1_sig = sigma_p1(p1, n_actual)
            per_pub.append({
                "plane": plane_i, "micro_shape": shape, "area": area, "theta": theta, "phi": phi,
                "which": which, "p1": p1, "p1_sigma": p1_sig, "shots_planned": shots_req,
                "shots_actual": n_actual, "m": args.m, "tau_loop_s": tau_s
            })

        # Reduce cw/ccw pairs
        by_key = {}
        for r in per_pub:
            key = (r["plane"], r["micro_shape"], r["area"], r["theta"], r["phi"], r["m"], r["tau_loop_s"])
            by_key.setdefault(key, {})[r["which"]] = r

        reduced = []
        for key, pair in by_key.items():
            if "cw" in pair and "ccw" in pair:
                plane_k, shape_k, area_k, theta_k, phi_k, m_k, tau_s = key
                cw, ccw = pair["cw"], pair["ccw"]
                delta = cw["p1"] - ccw["p1"]
                # combine using actual sigmas
                s_delta = math.hypot(cw.get("p1_sigma", 0.0) or 0.0, ccw.get("p1_sigma", 0.0) or 0.0)
                if area_k > 1e-12:
                    slope = delta / area_k
                    slope_sigma = s_delta / area_k
                else:
                    slope, slope_sigma = float("nan"), float("nan")

                if tau_s == tau_s and tau_s > 0.0:  # sane tau
                    kappa_eff = slope / tau_s
                    kappa_eff_sigma = slope_sigma / tau_s
                else:
                    kappa_eff = float("nan")
                    kappa_eff_sigma = float("nan")

                row = {
                    "plane": plane_k, "micro_shape": shape_k, "area": area_k,
                    "theta": theta_k, "phi": phi_k, "m": m_k,
                    "p1_cw": cw["p1"], "p1_ccw": ccw["p1"],
                    "shots_cw_planned": cw["shots_planned"], "shots_ccw_planned": ccw["shots_planned"],
                    "shots_cw_actual": cw["shots_actual"],   "shots_ccw_actual":   ccw["shots_actual"],
                    "delta": delta, "delta_sigma": s_delta,
                    "slope_per_area": slope, "slope_sigma": slope_sigma,
                    "tau_loop_s": tau_s,
                    "kappa_eff_per_area_Hz": kappa_eff, "kappa_eff_sigma_Hz": kappa_eff_sigma
                }
                all_rows.append(row)
                reduced.append(row)

        # Plane-level verdicts
        plane_verdicts = []
        for shape in shapes:
            Rs = sorted([r for r in reduced if r["plane"] == plane and r["micro_shape"] == shape and r["area"] > 0.0],
                        key=lambda r: r["area"])
            slopes = [r["slope_per_area"] for r in Rs]
            kappas = [r["kappa_eff_per_area_Hz"] for r in Rs if r.get("kappa_eff_per_area_Hz") == r.get("kappa_eff_per_area_Hz")]
            cv, same = plateau_score(slopes, k_small=min(6, max(2, len(slopes)//2)))
            mean_slope = (sum(slopes)/len(slopes)) if slopes else float("nan")
            mean_kappa = (sum(kappas)/len(kappas)) if kappas else float("nan")
            plane_verdicts.append({
                "plane": plane, "micro_shape": shape, "n_points": len(Rs),
                "cv_small": cv, "same_sign": bool(same),
                "mean_slope_per_area": mean_slope,
                "mean_kappa_eff_per_area_Hz": mean_kappa
            })

        # Null checks
        def null_pass(which_null: str) -> Optional[Dict]:
            Ns = [r for r in reduced if r["plane"] == plane and r["micro_shape"] == which_null and r["area"] == 0.0]
            if not Ns: return None
            mu = sum(r["delta"] for r in Ns)/len(Ns)
            # standard error of the mean using individual delta_sigma
            se_mean = math.sqrt(sum((r.get("delta_sigma", 0.0) or 0.0)**2 for r in Ns)) / max(1, len(Ns))
            z = abs(mu)/max(1e-12, se_mean)
            return {"plane": plane, "null": which_null, "z": z, "mean_delta": mu, "pass": bool(z <= args.z_null)}

        verdicts.append({
            "plane": plane, "plateau": plane_verdicts,
            "null_theta": null_pass("null_theta") if args.include_nulls else None,
            "null_phi": null_pass("null_phi") if args.include_nulls else None
        })

    # Execute plane blocks
    try:
        if args.use_session:
            with Session(backend=backend) as sess:
                for plane in planes:
                    run_plane_block(sess, plane)
        else:
            for plane in planes:
                run_plane_block(backend, plane)
    except KeyboardInterrupt:
        jprint({"type": "exit", "code": 130})
        if args.out and all_rows:
            write_csv(args.out + ".csv", all_rows)
        return

    # Emit summary + files
    jprint({"type": "summary", "verdicts": verdicts})

    if args.out:
        write_csv(args.out + ".csv", all_rows)
        flat_sum = []
        for v in verdicts:
            plane = v["plane"]
            for pv in v["plateau"]:
                flat_sum.append({
                    "plane": plane, "micro_shape": pv["micro_shape"], "n_points": pv["n_points"],
                    "cv_small": pv["cv_small"], "same_sign": pv["same_sign"],
                    "mean_slope_per_area": pv["mean_slope_per_area"],
                    "mean_kappa_eff_per_area_Hz": pv.get("mean_kappa_eff_per_area_Hz", float("nan"))
                })
            if v.get("null_theta"):
                nt = v["null_theta"]
                flat_sum.append({"plane": plane, "micro_shape": "null_theta", "null_z": nt["z"], "null_pass": nt["pass"], "mean_delta": nt["mean_delta"]})
            if v.get("null_phi"):
                npf = v["null_phi"]
                flat_sum.append({"plane": plane, "micro_shape": "null_phi", "null_z": npf["z"], "null_pass": npf["pass"], "mean_delta": npf["mean_delta"]})
        write_csv(args.out + "_summary.csv", flat_sum)
        jprint({"type": "saved", "files": [args.out + ".csv", args.out + "_summary.csv"]})


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        jprint({"type": "exit", "code": 130})
        sys.exit(130)
    except Exception as e:
        import traceback
        jprint({"type": "error", "error": type(e).__name__, "message": str(e),
                "trace_hint": traceback.format_exc(limit=2)})
        sys.exit(1)
