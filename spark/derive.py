"""Bounded example-to-program synthesis over JSON; no eval, I/O, or auto-adoption.

The language is deliberately small: field (also over records in a list), exact
record filtering, one-level flattening, and string joining. A learned program
is data, not Python. Fitting examples is not evidence of generalization. Search
returns every *encountered* shortest fit, never asserts uniqueness: equivalent
training behavior can hide different behavior elsewhere.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import deque

MAX_BYTES = 65536
MAX_NODES = 8192
MAX_DEPTH = 20
MAX_STEPS = 8
MAX_STATES = 1500
MAX_ATTEMPTS = 50000
MAX_SECONDS = 3.0
MAX_EXAMPLES = 12
MAX_OPERATIONS = 80


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(',', ':'))


def bounded(value):
    """Validate JSON size/shape, including intermediate results, before use."""
    todo, count = [(value, 0)], 0
    while todo:
        node, depth = todo.pop()
        count += 1
        if count > MAX_NODES or depth > MAX_DEPTH:
            raise ValueError('JSON shape budget exceeded')
        if isinstance(node, dict):
            if any(type(k) is not str for k in node):
                raise ValueError('JSON object keys must be strings')
            todo.extend((v, depth + 1) for v in node.values())
        elif isinstance(node, list):
            todo.extend((v, depth + 1) for v in node)
        elif type(node) not in (str, int, float, bool, type(None)):
            raise ValueError('not a JSON value')
    if len(canonical(value).encode()) > MAX_BYTES:
        raise ValueError('JSON byte budget exceeded')
    return value


def step(value, instruction):
    if not isinstance(instruction, list) or not instruction:
        raise ValueError('invalid instruction')
    op, *args = instruction
    if op == 'field' and len(args) == 1 and type(args[0]) is str:
        key = args[0]
        if isinstance(value, dict) and key in value:
            return value[key]
        if isinstance(value, list) and all(isinstance(v, dict) and key in v for v in value):
            return [v[key] for v in value]
    elif op == 'where' and len(args) == 2 and type(args[0]) is str:
        key, expected = args
        if type(expected) not in (str, int, float, bool, type(None)):
            raise ValueError('filter value must be scalar')
        if isinstance(value, list) and all(isinstance(v, dict) for v in value):
            # JSON types matter: true is not the integer 1.
            return [v for v in value if key in v and type(v[key]) is type(expected)
                    and v[key] == expected]
    elif op == 'flatten' and not args:
        if isinstance(value, list) and all(isinstance(v, list) for v in value):
            return [x for v in value for x in v]
    elif op == 'join' and len(args) == 1 and args[0] in ('', '\n', ' '):
        if isinstance(value, list) and all(type(v) is str for v in value):
            return args[0].join(value)
    raise ValueError('instruction is unsupported or inapplicable')


def apply(program, value):
    bounded(program)
    bounded(value)
    if not isinstance(program, list) or len(program) > MAX_STEPS:
        raise ValueError('program must have at most eight steps')
    for instruction in program:
        value = bounded(step(value, instruction))
    return value


def vocabulary(examples):
    keys, filters = set(), set()
    todo = [e['input'] for e in examples]
    while todo:
        value = todo.pop()
        if isinstance(value, dict):
            for key, item in value.items():
                keys.add(key)
                if type(item) in (str, int, float, bool, type(None)):
                    filters.add(canonical(['where', key, item]))
                todo.append(item)
        elif isinstance(value, list):
            todo.extend(value)
    ops = ([['field', k] for k in sorted(keys)] +
           [json.loads(f) for f in sorted(filters)] +
           [['flatten'], ['join', ''], ['join', '\n'], ['join', ' ']])
    if len(ops) > MAX_OPERATIONS:
        raise ValueError('example vocabulary exceeds operation budget')
    return ops


def learn(examples, max_steps=5):
    bounded(examples)
    if (not isinstance(examples, list) or not 1 <= len(examples) <= MAX_EXAMPLES
            or any(not isinstance(e, dict) or set(e) != {'input', 'output'} for e in examples)):
        raise ValueError('supply 1..12 exact input/output examples')
    if type(max_steps) is not int or not 0 <= max_steps <= MAX_STEPS:
        raise ValueError('max_steps must be an integer in 0..8')
    ops = vocabulary(examples)
    start = [e['input'] for e in examples]
    target = canonical([e['output'] for e in examples])
    queue = deque([([], start)])
    seen = {canonical(start)}
    fits, attempts, exhausted = [], 0, False
    deadline = time.monotonic() + MAX_SECONDS
    fit_depth = None
    while queue:
        program, values = queue.popleft()
        if canonical(values) == target:
            fits.append(program)
            fit_depth = len(program)
            if len(fits) >= 16:
                exhausted = True
                break
            continue
        if len(program) >= max_steps or (fit_depth is not None and len(program) >= fit_depth):
            continue
        for op in ops:
            attempts += 1
            if attempts > MAX_ATTEMPTS or time.monotonic() >= deadline:
                exhausted = True
                break
            try:
                result = [bounded(step(v, op)) for v in values]
                signature = canonical(result)
            except (ValueError, TypeError, OverflowError):
                continue
            candidate = program + [op]
            # Keep distinct final fits even though they agree on training data.
            # Intermediate equivalence pruning is not a uniqueness proof.
            if signature == target:
                queue.append((candidate, result))
            elif signature not in seen:
                if len(seen) >= MAX_STATES:
                    exhausted = True
                    break
                seen.add(signature)
                queue.append((candidate, result))
        if exhausted:
            break
    # A limit may interrupt a breadth layer. Never imply completed enumeration.
    if fits:
        shortest = min(map(len, fits))
        fits = [p for p in fits if len(p) == shortest]
    return {'status': ('ambiguous' if len(fits) > 1 else 'fit' if fits else
                       'budget_exhausted' if exhausted else 'no_fit'),
            'programs': fits, 'search_complete': not exhausted,
            'attempts': attempts, 'states': len(seen),
            'scope': 'training fit only; finite language; intermediate equivalence pruning; '
                     'no uniqueness or generalization claim; no automatic adoption'}


def operate(arguments):
    bounded(arguments)
    action = arguments.get('action')
    if action == 'learn':
        return learn(arguments.get('examples'), arguments.get('max_steps', 5))
    if action == 'apply':
        if 'value' not in arguments or 'program' not in arguments:
            raise ValueError('apply needs program and value')
        return {'value': apply(arguments['program'], arguments['value']),
                'scope': 'pure data result, not source authority or a verified interpretation'}
    raise ValueError('action must be learn or apply')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request', help='local JSON request file; no source executes')
    args = parser.parse_args()
    with open(args.request, 'rb') as f:
        raw = f.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError('request byte budget exceeded')
    print(json.dumps(operate(json.loads(raw)), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
