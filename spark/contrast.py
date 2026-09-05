"""Find a numerical witness between explicit predictions on a finite control grid.

This is experimental-design assistance, not theory generation or equivalence
proving. The caller supplies the formulas, units, admissible controls, absolute
tolerance, and (optionally) observable period. Formula provenance and physical
meaning still need checking outside this instrument. A discrepancy can be mere
floating-point error; a null can miss everything between samples.

The language is scalar arithmetic, named controls, pi/e, and the functions below.
ASTs are interpreted, never executed as Python. No imports, attributes, source
execution, I/O, model calls, adoption or persistence occur in seek(). Optional CLI:
    python3 -m spark.contrast request.json
It reads one bounded local JSON file and writes the result to stdout only.
"""
from __future__ import annotations

import argparse
import ast
import itertools
import json
import math
import operator
import re

MAX_POINTS = 4096
MAX_NODES = 128
MAX_BYTES = 16384
FUNCTIONS = {name: (getattr(math, name), 1) for name in
             ('sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'exp', 'log', 'sqrt')}
FUNCTIONS.update(abs=(abs, 1), atan2=(math.atan2, 2), hypot=(math.hypot, 2))
CONSTANTS = {'pi': math.pi, 'e': math.e}
BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
          ast.Div: operator.truediv, ast.Pow: math.pow}
UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def finite(value):
    if type(value) not in (int, float):
        raise ValueError('expected a finite real number, not bool or another type')
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError('number exceeds floating-point range') from exc
    if not math.isfinite(result):
        raise ValueError('nonfinite number')
    return result


def expression(text, names):
    if type(text) is not str or not 1 <= len(text) <= 2048:
        raise ValueError('expression must contain 1..2048 characters')
    try:
        tree = ast.parse(text, mode='eval').body
    except (SyntaxError, RecursionError) as exc:
        raise ValueError('invalid expression') from exc
    if sum(1 for _ in ast.walk(tree)) > MAX_NODES:
        raise ValueError('expression node budget exceeded')

    def check(node, depth=0):
        if depth > 16:
            raise ValueError('expression depth budget exceeded')
        if isinstance(node, ast.Constant):
            finite(node.value)
        elif isinstance(node, ast.Name) and node.id in names | CONSTANTS.keys():
            pass
        elif isinstance(node, ast.BinOp) and type(node.op) in BINARY:
            check(node.left, depth + 1)
            check(node.right, depth + 1)
        elif isinstance(node, ast.UnaryOp) and type(node.op) in UNARY:
            check(node.operand, depth + 1)
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
              and node.func.id in FUNCTIONS and not node.keywords
              and len(node.args) == FUNCTIONS[node.func.id][1]):
            for arg in node.args:
                check(arg, depth + 1)
        else:
            raise ValueError('unsupported expression or unknown control')
    check(tree)
    return tree


def evaluate(tree, controls):
    def visit(node):
        if isinstance(node, ast.Constant):
            value = node.value
        elif isinstance(node, ast.Name):
            value = controls[node.id] if node.id in controls else CONSTANTS[node.id]
        elif isinstance(node, ast.UnaryOp):
            value = UNARY[type(node.op)](visit(node.operand))
        elif isinstance(node, ast.BinOp):
            value = BINARY[type(node.op)](visit(node.left), visit(node.right))
        else:  # only validated calls can reach this branch
            value = FUNCTIONS[node.func.id][0](*(visit(arg) for arg in node.args))
        return finite(value)
    return visit(tree)


def seek(request):
    if (not isinstance(request, dict)
            or set(request) - {'left', 'right', 'controls', 'atol', 'period'}
            or not {'left', 'right', 'controls', 'atol'} <= set(request)):
        raise ValueError('supply left, right, controls, atol, and optional period only')
    controls = request['controls']
    if not isinstance(controls, dict) or not 1 <= len(controls) <= 4:
        raise ValueError('supply 1..4 named controls')
    grids = {}
    for name, values in controls.items():
        if (type(name) is not str or not re.fullmatch('[a-zA-Z][a-zA-Z0-9_]{0,23}', name)
                or name in FUNCTIONS or name in CONSTANTS):
            raise ValueError('invalid or reserved control name')
        if not isinstance(values, list) or not 1 <= len(values) <= 64:
            raise ValueError('each control needs 1..64 explicit samples')
        grids[name] = list(dict.fromkeys(finite(v) for v in values))
    total = math.prod(map(len, grids.values()))
    if total > MAX_POINTS:
        raise ValueError('Cartesian grid exceeds 4096 points')
    atol = finite(request['atol'])
    period = None if request.get('period') is None else finite(request['period'])
    if atol < 0 or (period is not None and period <= 0):
        raise ValueError('atol must be nonnegative; period must be positive')
    left = expression(request['left'], set(grids))
    right = expression(request['right'], set(grids))
    best, valid, invalid, errors = None, 0, 0, []
    for values in itertools.product(*grids.values()):
        point = dict(zip(grids, values))
        outputs, faults = {}, {}
        for name, tree in (('left', left), ('right', right)):
            try:
                outputs[name] = evaluate(tree, point)
            except (ValueError, OverflowError, ZeroDivisionError) as exc:
                faults[name] = type(exc).__name__
        if not faults:
            try:
                a, b = outputs['left'], outputs['right']
                if period is not None:
                    # Reduce separately before subtraction to avoid overflow.
                    a, b = math.remainder(a, period), math.remainder(b, period)
                delta = finite(a - b)
                gap = abs(math.remainder(delta, period)) if period else abs(delta)
            except (ValueError, OverflowError) as exc:
                faults['comparison'] = type(exc).__name__
        if faults:
            invalid += 1
            if len(errors) < 3:
                errors.append({'controls': point, 'errors': faults, **outputs})
            continue
        valid += 1
        if best is None or gap > best['gap']:
            best = {'controls': point, **outputs, 'gap': gap}
    found = best is not None and best['gap'] > atol
    return {
        'status': ('witness' if found else 'inconclusive' if invalid else 'none_in_grid'),
        'left': request['left'], 'right': request['right'], 'controls': grids,
        'atol': atol, 'period': period, 'tested': total, 'valid': valid,
        'invalid': invalid, 'invalid_examples': errors,
        'witness': best if found else None, 'largest_comparable_gap': best,
        'scope': 'Finite-grid floating-point comparison only; not equivalence, '
                 'causation, physical validation, statistical significance, or subject scoring. '
                 'Invalid points are unresolved; unsampled controls are untested. '
                 'Atol is caller-supplied, not an error bound. No automatic action.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request', help='bounded local JSON; never executable source')
    args = parser.parse_args()
    with open(args.request, 'rb') as handle:
        raw = handle.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError('request byte budget exceeded')
    print(json.dumps(seek(json.loads(raw)), ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == '__main__':
    main()
