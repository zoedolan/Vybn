#!/usr/bin/env python3
"""The Boolean manifold rerun of 2026-11-14 (THEORY.md, section III.3). The original: ibm_fez Q0, 2025-12-26, 128 shots, transpiler level 1.
Every arm is the identity, so the score is how often the qubit reads 0. One qubit per circuit, transpiler level 0, gate counts asserted before sending.
Arms: readout (just measure), nand (the original: twelve rz-sx-rz), xor (the original: H, ten X, H), idle (H, wait as long as ten X, H),
barex (ten X, no H), sxeig (H, twenty sx, H), nand30 (the nand path with 32 sx).
Run as: python3 boolean_manifold_rerun.py sim | ibm [backend] | fetch | noise [backend]. sim needs qiskit and qiskit-aer and spends nothing.
The other three need an IBM Quantum account, which IBM's own library reads from its saved settings. This script never prints or stores it.
The code below the imports is the code that ran. Only this description and the last line were changed for publication.
Counts from the run: boolean_manifold_rerun_result.json."""
import json, math, time, pathlib
D = pathlib.Path(__file__).resolve().parent
SHOTS = 2048
QUBITS = [0, 13, 57, 96, 141]
NEED = dict(readout=(0, 0), nand=(12, 0), xor=(2, 10), idle=(2, 0), barex=(0, 10), sxeig=(22, 0), nand30=(32, 0))   # (sx, x) pulses that must survive
def circuit(arm, wait=16):
    from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
    q = QuantumRegister(1); m = ClassicalRegister(1, 'm'); c = QuantumCircuit(q, m)
    def H(): c.rz(math.pi / 2, 0); c.sx(0); c.rz(math.pi / 2, 0)
    if arm in ('nand', 'nand30'):
        for _ in range(12 if arm == 'nand' else 32): H()
    elif arm == 'xor':
        H(); [c.x(0) for _ in range(10)]; H()
    elif arm == 'idle':
        H(); c.delay(wait, 0, unit='dt'); H()
    elif arm == 'barex': [c.x(0) for _ in range(10)]
    elif arm == 'sxeig':
        H(); [c.sx(0) for _ in range(20)]; H()
    c.measure(0, m[0]); return c
def report(plan, counts):
    rep = {}
    for (q, arm), cn in zip(plan, counts): rep.setdefault(str(q), {})[arm] = round(cn.get('0', 0) / max(1, sum(cn.values())), 4)
    for q, r in rep.items(): r['xor-nand'] = round(r['xor'] - r['nand'], 4)
    return rep
def sim():
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    be = AerSimulator(); plan = [(0, a) for a in NEED]
    cs = transpile([circuit(a) for a in NEED], be, optimization_level=0)
    print('JSON ' + json.dumps(report(plan, be.run(cs, shots=SHOTS).result().get_counts())))
def build(be):
    from qiskit import transpile
    t = be.target; plan, tc, props = [], [], {}
    for q in QUBITS:
        xd = t['x'][(q,)].duration; wait = 16 * max(1, round(10 * xd / t.dt / 16)); p = dict(x_ns=round(xd * 1e9, 1), idle_dt=wait)
        for g in ('x', 'sx', 'measure'):
            try: p[g + '_err'] = round(float(t[g][(q,)].error), 5)
            except Exception: p[g + '_err'] = None
        try: qp = be.qubit_properties(q); p['T1_us'] = round(qp.t1 * 1e6, 1); p['T2_us'] = round(qp.t2 * 1e6, 1)
        except Exception: pass
        props[str(q)] = p
        for arm in NEED:
            c = transpile(circuit(arm, wait), backend=be, initial_layout=[q], optimization_level=0); ops = c.count_ops()
            assert (ops.get('sx', 0), ops.get('x', 0)) == NEED[arm], (q, arm, dict(ops))
            plan.append((q, arm)); tc.append(c)
    return plan, tc, props
def ibm(name='ibm_fez'):
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    be = QiskitRuntimeService().backend(name); plan, tc, props = build(be)
    job = SamplerV2(mode=be).run(tc, shots=SHOTS)
    (D / 'manifold-job.json').write_text(json.dumps(dict(job=job.job_id(), backend=be.name, at=time.time(), shots=SHOTS, plan=plan, props=props)))
    print('job', job.job_id(), 'on', be.name, 'circuits', len(tc), 'props', json.dumps(props)); fetch()
def fetch():
    from qiskit_ibm_runtime import QiskitRuntimeService
    j = json.loads((D / 'manifold-job.json').read_text()); job = QiskitRuntimeService().job(j['job'])
    try: res = job.result(timeout=75)
    except Exception as e: return print('no result yet:', job.status(), '|', type(e).__name__, '| ask again with: manifold fetch')
    counts = [res[i].data.m.get_counts() for i in range(len(j['plan']))]; rep = report(j['plan'], counts)
    (D / f"manifold-result-{j['job']}.json").write_text(json.dumps(dict(job=j, counts=counts, read=rep), indent=1)); print('JSON ' + json.dumps(rep))
def noise(name='ibm_fez'):
    from qiskit_ibm_runtime import QiskitRuntimeService
    from qiskit_aer import AerSimulator
    be = QiskitRuntimeService().backend(name); plan, tc, props = build(be); s = AerSimulator.from_backend(be)
    print('NOISE ' + json.dumps(report(plan, s.run(tc, shots=SHOTS).result().get_counts())))
def main(mode='sim', *rest): dict(sim=sim, ibm=ibm, fetch=fetch, noise=noise)[mode](*rest)
if __name__ == '__main__':
    import sys; main(*sys.argv[1:])
