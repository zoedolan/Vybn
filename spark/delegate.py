"""Bounded, private multi-model contributions; no retries, fallback, tools or publication.

CLI is preflight-only unless --run is supplied under separate spending authority.
Delivery is not editorial acceptance. See harness/README.md.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time


def validate(plan):
    jobs = plan['jobs']
    if not jobs or len(jobs) > plan['max_calls']:
        raise ValueError('empty plan or call ceiling exceeded')
    if sum(j['max_output_tokens'] for j in jobs) > plan['max_total_output_tokens']:
        raise ValueError('output ceiling exceeded (includes thinking)')
    if len({j['id'] for j in jobs}) != len(jobs):
        raise ValueError('duplicate job IDs')
    for j in jobs:
        if j['provider'] not in ('anthropic', 'openai'):
            raise ValueError('unsupported provider')
        if not all(isinstance(j[k], str) and j[k].strip()
                   for k in ('id', 'model', 'system', 'prompt')):
            raise ValueError('missing assignment or model')
        if not isinstance(j['max_output_tokens'], int) or j['max_output_tokens'] <= 0:
            raise ValueError('invalid output limit')
        if not 0 < j['timeout_seconds'] <= 600:
            raise ValueError('timeout must be bounded at 600 seconds or less')
        if len(j['system']) + len(j['prompt']) > plan['max_input_chars_per_call']:
            raise ValueError('input character ceiling exceeded')
        if j.get('thinking_budget') is not None:
            if (j['provider'] != 'anthropic' or j.get('effort') is not None or
                    not 1024 <= j['thinking_budget'] < j['max_output_tokens']):
                raise ValueError('invalid explicit thinking budget; do not mix with effort')
    return {'calls_at_most': len(jobs),
            'output_tokens_at_most': sum(j['max_output_tokens'] for j in jobs),
            'input_characters': sum(len(j['system']) + len(j['prompt']) for j in jobs),
            'warning': 'Not a dollar bound. Adaptive effort reserves no answer tokens. '
                       'Short deliverables and a live pilot still matter.'}


def save(path, value):
    """Exclusive, owner-only files. Never overwrite or adopt a prior run's work."""
    with path.open('x', encoding='utf-8') as f:
        os.chmod(path, 0o600)
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())


def send(job):
    """One SDK request only. Unsupported settings fail visibly, not via downgrade."""
    if job['provider'] == 'anthropic':
        import anthropic
        knobs = {}
        if job.get('thinking_budget') is not None:
            knobs['thinking'] = {'type': 'enabled', 'budget_tokens': job['thinking_budget']}
        elif job.get('effort') is not None:
            knobs = {'thinking': {'type': 'adaptive'},
                     'output_config': {'effort': job['effort']}}
        with anthropic.Anthropic(max_retries=0, timeout=job['timeout_seconds']) as client:
            response = client.messages.create(
                model=job['model'], max_tokens=job['max_output_tokens'],
                system=job['system'], messages=[{'role': 'user', 'content': job['prompt']}],
                **knobs)
    else:
        from openai import OpenAI
        knobs = {'reasoning': {'effort': job['effort']}} if job.get('effort') else {}
        with OpenAI(max_retries=0, timeout=job['timeout_seconds']) as client:
            response = client.responses.create(
                model=job['model'], max_output_tokens=job['max_output_tokens'], store=False,
                input=[{'role': 'developer', 'content': job['system']},
                       {'role': 'user', 'content': job['prompt']}], **knobs)
    return response.model_dump(mode='json')


def classify(job, response):
    """Transport/delivery checks only; retain partial text without accepting it."""
    if job['provider'] == 'anthropic':
        blocks = response.get('content', [])
        text = '\n'.join(b['text'] for b in blocks if b.get('type') == 'text')
        complete = response.get('stop_reason') == 'end_turn'
        refusal = response.get('stop_reason') == 'refusal' or any(b.get('type') == 'refusal' for b in blocks)
    else:
        blocks = [b for item in response.get('output', [])
                  if item.get('type') == 'message' for b in item.get('content', [])]
        text = '\n'.join(b['text'] for b in blocks if b.get('type') == 'output_text')
        complete = response.get('status') == 'completed'
        refusal = any(b.get('type') == 'refusal' for b in blocks)
    # Exact allowlist: a similar-looking model name is not permission to substitute.
    models = [job['model']] + job.get('accepted_model_ids', [])
    status = ('model_mismatch' if response.get('model') not in models else
              'refused' if refusal else 'incomplete' if not complete else
              'empty' if not text.strip() else 'delivered')
    return {'status': status, 'text': text, 'returned_model': response.get('model'),
            'usage': response.get('usage'), 'requested_model': job['model']}


def account(job, response):
    """Use connection's existing budget ledger, in addition to the raw receipt."""
    u = response.get('usage') or {}
    cached = u.get('cache_read_input_tokens', 0) or (u.get('input_tokens_details') or {}).get('cached_tokens', 0)
    row = {'ts': time.strftime('%Y-%m-%dT%H:%M:%S'), 'model': response.get('model'),
           'in': u.get('input_tokens', 0), 'out': u.get('output_tokens', 0),
           'cache_r': cached, 'cache_w': u.get('cache_creation_input_tokens', 0),
           'settings': {'task': 'delegation', 'requested_model': job['model'],
                        'effort': job.get('effort'), 'thinking_budget': job.get('thinking_budget')}}
    path = Path.home() / '.cache/vybn-phase/api_usage.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(row) + '\n')


def run(plan, root, sender=send, accountant=account):
    """Sequential pilot-first dispatch. Stop on failure; no automatic continuation.

    Successful delivery does not validate a setting for longer assignments, nor
    mean that the coordinator has reviewed evidence or creative value.
    """
    validate(plan)  # reject the whole plan before any request
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Runtime material may contain private source text and opaque provider blocks.
    if root.stat().st_mode & 0o077:
        raise ValueError('run root must be owner-only')
    directory = Path(tempfile.mkdtemp(prefix='delegation-', dir=root))
    save(directory / 'plan.json', plan)
    rows = []
    for i, job in enumerate(plan['jobs']):
        save(directory / f'{i}-request.json', job)
        try:
            response = sender(job)
        except Exception as exc:
            # Request may have been billed. Do not retry or persist secret-bearing error text.
            rows.append({'id': job['id'], 'status': 'provider_error',
                         'error_type': type(exc).__name__, 'billing': 'unknown'})
            break
        # Preserve ALL returned blocks before interpretation/accounting can fail.
        save(directory / f'{i}-response.json', response)
        try:
            accountant(job, response)
            result = classify(job, response)
        except Exception as exc:
            rows.append({'id': job['id'], 'status': 'receipt_processing_error',
                         'returned_model': response.get('model'),
                         'error_type': type(exc).__name__})
            break
        save(directory / f'{i}-result.json', result)
        rows.append({'id': job['id'], **{k: v for k, v in result.items() if k != 'text'}})
        if result['status'] != 'delivered':
            break
    delivered = len(rows) == len(plan['jobs']) and all(r['status'] == 'delivered' for r in rows)
    summary = {'status': 'needs_editorial_review' if delivered else 'stopped',
               'jobs': rows, 'unstarted': [j['id'] for j in plan['jobs'][len(rows):]]}
    save(directory / 'summary.json', summary)
    return directory, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('--run', action='store_true', help='paid dispatch; requires grounded authority')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    print(json.dumps(validate(plan), indent=2))
    if args.run:
        root = Path.home() / '.local/state/vybn/delegation'
        directory, summary = run(plan, root)
        print(json.dumps({'private_run': str(directory), **summary}, indent=2))
        raise SystemExit(0 if summary['status'] == 'needs_editorial_review' else 1)


if __name__ == '__main__':
    main()
