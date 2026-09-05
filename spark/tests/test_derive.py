"""Transfer and failure cases for the finite transformation learner, not a mind test."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from spark import derive as d

ROOT = Path(__file__).resolve().parents[2]


def text_examples():
    return [
        {'input': {'content': [{'type': 'text', 'text': 'a'},
                               {'type': 'tool', 'text': 'not text'},
                               {'type': 'text', 'text': 'b'}]}, 'output': 'a\nb'},
        {'input': {'content': [{'type': 'text', 'text': 'c'}]}, 'output': 'c'},
    ]


def test_composition_is_derived_and_transfers_in_fresh_process(tmp_path):
    result = d.learn(text_examples())
    assert result['status'] == 'fit' and result['search_complete']
    program, = result['programs']
    assert program == [['field', 'content'], ['where', 'type', 'text'],
                       ['field', 'text'], ['join', '\n']]
    # These cases enter only after the program has been frozen. The author knew
    # the task; this is an engineering transfer check, not blinded evaluation.
    for value, expected in [
        ({'content': []}, ''),
        ({'content': [{'type': 'tool', 'text': 'discard'},
                      {'type': 'text', 'text': 'λ'},
                      {'type': 'text', 'text': 'new\nline'},
                      {'type': 'tool', 'text': 'last'}]}, 'λ\nnew\nline'),
        ({'content': [{'type': 'text', 'text': ''}]}, ''),
    ]:
        request = tmp_path / 'apply.json'
        request.write_text(json.dumps({'action': 'apply', 'program': program, 'value': value}))
        run = subprocess.run([sys.executable, '-m', 'spark.derive', str(request)],
                             cwd=ROOT, capture_output=True, text=True, timeout=10)
        assert run.returncode == 0, run.stderr
        assert json.loads(run.stdout)['value'] == expected
        assert d.apply([], value) != expected  # named identity rival


def test_counterexample_changes_ambiguity_without_silent_selection():
    first = {'input': {'left': 'same', 'right': 'same'}, 'output': 'same'}
    ambiguous = d.learn([first])
    assert ambiguous['status'] == 'ambiguous'
    assert ambiguous['programs'] == [[['field', 'left']], [['field', 'right']]]
    correction = {'input': {'left': 'wrong', 'right': 'correct'}, 'output': 'correct'}
    corrected = d.learn([first, correction])
    assert corrected['programs'] == [[['field', 'right']]]
    assert d.apply(corrected['programs'][0], {'left': 99, 'right': ['a', 'b']}) == ['a', 'b']


def test_counterexample_can_kill_every_program():
    assert d.learn([{'input': 'ab', 'output': 'ba'}])['status'] == 'no_fit'
    contradictory = [{'input': {'a': 'x'}, 'output': 'x'},
                     {'input': {'a': 'x'}, 'output': 'y'}]
    assert not d.learn(contradictory)['programs']


def test_identity_is_explicit_not_fabricated_learning():
    assert d.learn([{'input': {'a': 2}, 'output': {'a': 2}}])['programs'] == [[]]


def test_typed_exact_filter_and_missing_field():
    value = [{'flag': True}, {'flag': 1}, {'flag': 'true'}, {}]
    assert d.apply([['where', 'flag', True]], value) == [{'flag': True}]
    with pytest.raises(ValueError):
        d.apply([['field', 'text']], [{'text': 'x'}, {}])
    assert d.apply([['flatten']], [[1, 2], [], [3]]) == [1, 2, 3]


@pytest.mark.parametrize('program', [
    [['eval', '__import__("os").system("echo forbidden")']], [['field']],
    [['join', 'arbitrary']], [['where', 'x', []]], [['flatten', 1]], [42],
    [['field', 'x']] * 9,
])
def test_language_has_no_arbitrary_code_or_unknown_instruction(program):
    with pytest.raises(ValueError):
        d.apply(program, {'x': 1})


@pytest.mark.parametrize('value', [float('nan'), float('inf'), {1: 'not JSON'},
                                  object(), 'x' * (d.MAX_BYTES + 1)])
def test_non_json_and_oversized_values_fail(value):
    with pytest.raises((ValueError, TypeError)):
        d.apply([], value)


def test_depth_and_circular_values_fail():
    value = []
    value.append(value)
    with pytest.raises(ValueError):
        d.apply([], value)
    with pytest.raises(ValueError):
        d.apply([], [[]] * (d.MAX_NODES + 1))


@pytest.mark.parametrize('limit', ['MAX_STATES', 'MAX_ATTEMPTS', 'MAX_SECONDS'])
def test_exhaustion_is_not_no_fit_or_success(monkeypatch, limit):
    monkeypatch.setattr(d, limit, 0)
    result = d.learn(text_examples())
    assert result['status'] == 'budget_exhausted'
    assert not result['search_complete'] and not result['programs']


def test_vocabulary_budget_and_bad_requests():
    with pytest.raises(ValueError):
        d.learn([{'input': {str(i): i for i in range(81)}, 'output': 1}])
    for examples in [[], [{}], [{'input': 1}], [{'input': 1, 'output': 1, 'extra': True}]]:
        with pytest.raises(ValueError):
            d.learn(examples)
    with pytest.raises(ValueError):
        d.learn(text_examples(), True)
    with pytest.raises(ValueError):
        d.operate({'action': 'apply', 'program': []})


def test_harness_tool_is_explicit_evidence_and_respects_path_refusal(monkeypatch):
    from spark.tests.test_public_contracts import _connection
    m = _connection()
    monkeypatch.setattr(m, 'path_tool_refusal', lambda tool: None)
    result = m.execute_tool(m.ToolCall('derive-check', 'derive_operation',
        {'action': 'learn', 'examples': text_examples()}, {}))
    assert 'EVIDENCE' in result and 'training fit only' in result
    assert 'derive_operation' in {t['name'] for t in m.TOOL_SCHEMAS}
    monkeypatch.setattr(m, 'path_tool_refusal', lambda tool: 'scoped refusal')
    assert 'scoped refusal' in m.execute_tool(m.ToolCall('no', 'derive_operation', {}, {}))
    assert 'bash' in {t['name'] for t in m.TOOL_SCHEMAS}  # no route replaced


def test_harness_schemas_are_valid_json_schema():
    from spark.tests.test_public_contracts import _connection
    import jsonschema
    for tool in _connection().TOOL_SCHEMAS:
        jsonschema.Draft202012Validator.check_schema(tool['input_schema'])
