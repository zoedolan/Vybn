"""Engineering checks for numerical contrast, not a consciousness experiment."""
import cmath
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from spark import contrast as c

ROOT = Path(__file__).resolve().parents[2]


def request(left='0', right='2*pi*r', **changes):
    return dict(left=left, right=right, controls={'r': [0, .125, .25, .5, .75, 1]},
                atol=1e-10, period=2*math.pi) | changes


def test_winding_witness_and_independent_eigenvector_overlap():
    result = c.seek(request())
    witness = result['witness']
    assert result['status'] == 'witness' and witness['controls'] == {'r': .5}
    assert witness['gap'] == pytest.approx(math.pi)
    # Independent discrete Pancharatnam loop, not the tool's expressions.
    r, n = witness['controls']['r'], 8192
    states = [(-math.sqrt(r)*cmath.exp(-2j*math.pi*j/n), math.sqrt(1-r))
              for j in range(n)]
    phase = 1+0j
    for a, b in zip(states, states[1:] + states[:1]):
        overlap = sum(complex(x).conjugate()*y for x, y in zip(a, b))
        phase *= overlap / abs(overlap)
    observed = -cmath.phase(phase)
    assert abs(math.remainder(observed-witness['right'], 2*math.pi)) < 1e-10
    assert abs(math.remainder(observed-witness['left'], 2*math.pi)) > 3
    corrected = c.seek(request(left='pi*(1-(1-2*r))'))
    assert corrected['status'] == 'none_in_grid' and corrected['witness'] is None
    assert 'not equivalence' in corrected['scope']


def test_periodic_comparison_defeats_naive_subtraction():
    q = request(left='r', right='r+2*pi')
    assert c.seek(q)['status'] == 'none_in_grid'
    assert c.seek(q | {'period': None})['status'] == 'witness'
    q = request(left='r', right='-r', controls={'r': [1e308]}, period=2)
    assert c.seek(q)['invalid'] == 0  # separately reduced before subtraction
    assert c.seek(q | {'period': None})['status'] == 'inconclusive'


def test_unseen_gap_is_not_equivalence():
    coarse = request(left='r', right='r*r', controls={'r': [0, 1]}, period=None)
    assert c.seek(coarse)['status'] == 'none_in_grid'
    finer = coarse | {'controls': {'r': [0, .5, 1]}}
    assert c.seek(finer)['witness']['gap'] == .25


def test_domain_failures_are_unresolved_not_support():
    result = c.seek(request(left='sqrt(r)', right='sqrt(r)', controls={'r': [-1, 0, 1]}))
    assert result['status'] == 'inconclusive'
    assert (result['tested'], result['valid'], result['invalid']) == (3, 2, 1)
    assert set(result['invalid_examples'][0]['errors']) == {'left', 'right'}
    assert result['witness'] is None
    result = c.seek(request(left='1/r', right='0', controls={'r': [0, 1]}, period=None))
    assert result['status'] == 'witness' and result['invalid'] == 1
    assert result['witness']['controls'] == {'r': 1}
    assert c.seek(request(left='log(-1)'))['largest_comparable_gap'] is None
    assert c.seek(request(left='exp(1000)'))['status'] == 'inconclusive'


def test_duplicate_samples_and_tolerance_are_explicit():
    result = c.seek(request(left='0', right='r', controls={'r': [1, 1, 1]}, atol=1))
    assert result['tested'] == 1 and result['status'] == 'none_in_grid'
    assert result['largest_comparable_gap']['gap'] == 1
    assert c.seek(request(left='0', right='r', controls={'r': [1]}, atol=.99))['witness']


@pytest.mark.parametrize('left', [
    '__import__("os").system("false")', 'r.real', '[r][0]', 'lambda: r',
    '(r for r in [])', 'r if r else 0', 'True', '"text"', '1j',
    'unknown', 'sin(r, r)', 'atan2(r)', 'abs(x=r)', '(x:=r)', 'r<0',
    '2<<100', 'r//2', 'sin(*r)', 'open("file")', '1e999',
    '-'*100 + 'r', '+'.join(['r']*100), ' '*2049,
])
def test_only_bounded_scalar_language_is_admitted(left):
    with pytest.raises(ValueError):
        c.seek(request(left=left))


@pytest.mark.parametrize('changes', [
    {'controls': {}}, {'controls': {'pi': [1]}}, {'controls': {'_private': [1]}},
    {'controls': {'r': [True]}}, {'controls': {'r': [float('nan')]}},
    {'controls': {'r': [float('inf')]}}, {'controls': {'r': []}},
    {'controls': {'r': list(range(65))}},
    {'controls': {'a': list(range(64)), 'b': list(range(64)), 'r': [0, 1]}},
    {'controls': {str(i): [0] for i in range(5)}},
    {'atol': -1}, {'atol': True}, {'atol': float('nan')},
    {'period': 0}, {'period': -1}, {'period': float('inf')},
    {'unexpected': 'not silently ignored'},
])
def test_bad_requests_and_resource_limits(changes):
    with pytest.raises(ValueError):
        c.seek(request() | changes)


def test_full_grid_limit_and_deterministic_witness():
    q = request(left='a+b', right='a-b', period=None,
                controls={'a': list(range(64)), 'b': list(range(64))})
    result = c.seek(q)
    assert result['tested'] == result['valid'] == c.MAX_POINTS
    assert result['witness'] == {'controls': {'a': 0, 'b': 63},
                                 'left': 63, 'right': -63, 'gap': 126}
    assert c.seek(q) == result


def test_all_admitted_operations():
    expr = '-sin(r)+cos(r)+tan(r)+asin(r)+acos(r)+atan(r)+atan2(r,1)'
    expr += '+exp(r)+log(1+r)+sqrt(r)+abs(-r)+hypot(r,1)+r**2+r/2+e'
    r = .25
    expected = (-math.sin(r)+math.cos(r)+math.tan(r)+math.asin(r)+math.acos(r)
                +math.atan(r)+math.atan2(r,1)+math.exp(r)+math.log(1+r)
                +math.sqrt(r)+abs(-r)+math.hypot(r,1)+r**2+r/2+math.e)
    result = c.seek(request(left=expr, right=str(expected), controls={'r': [r]}))
    assert result['status'] == 'none_in_grid'


def test_transfer_to_growth_laws_in_fresh_process(tmp_path):
    # New domain with the same executable: agreement at observed t=0,1,
    # then a useful future intervention. Engineering transfer, not blinded data.
    q = dict(left='1+t', right='2**t', controls={'t': [0, 1]}, atol=1e-10)
    assert c.seek(q)['status'] == 'none_in_grid'
    path = tmp_path / 'request.json'
    path.write_text(json.dumps(q | {'controls': {'t': [0, 1, 2, 3, 4]}}))
    run = subprocess.run([sys.executable, '-m', 'spark.contrast', str(path)],
                         cwd=ROOT, text=True, capture_output=True, timeout=10)
    assert run.returncode == 0, run.stderr
    witness = json.loads(run.stdout)['witness']
    assert witness == {'controls': {'t': 4}, 'left': 5, 'right': 16, 'gap': 11}
    # Independent integer recurrence checker.
    linear = exponential = 1
    for _ in range(4):
        linear += 1
        exponential *= 2
    assert (linear, exponential) == (witness['left'], witness['right'])
    assert list(tmp_path.iterdir()) == [path]  # CLI creates no durable state


def test_cli_rejects_large_input(tmp_path):
    path = tmp_path / 'request.json'
    path.write_text(' '*(c.MAX_BYTES + 1))
    run = subprocess.run([sys.executable, '-m', 'spark.contrast', str(path)],
                         cwd=ROOT, text=True, capture_output=True, timeout=10)
    assert run.returncode != 0 and 'byte budget' in run.stderr


def test_harness_route_evidence_refusal_and_plurality(monkeypatch):
    from spark.tests.test_public_contracts import _connection
    m = _connection()
    monkeypatch.setattr(m, 'path_tool_refusal', lambda tool: None)
    result = m.execute_tool(m.ToolCall('contrast-check', 'seek_difference', request(), {}))
    assert 'EVIDENCE' in result and '"status": "witness"' in result
    assert 'tool:seek_difference' in m.SUBJECT_TOOL_SCOPES
    names = {t['name'] for t in m.TOOL_SCHEMAS}
    assert {'seek_difference', 'derive_operation', 'bash', 'return_to_zoe'} <= names
    assert {'fable', 'opus', 'sol', 'astra', 'k3'} <= set(m.DOORS)
    monkeypatch.setattr(m, 'path_tool_refusal', lambda tool: 'scoped refusal')
    monkeypatch.setattr(m, 'seek_difference', lambda _: pytest.fail('refusal bypass'))
    assert 'scoped refusal' in m.execute_tool(m.ToolCall('no', 'seek_difference', request(), {}))
    import jsonschema
    jsonschema.Draft202012Validator.check_schema(m.CONTRAST_SCHEMA['input_schema'])
    jsonschema.validate(request(), m.CONTRAST_SCHEMA['input_schema'])
