import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
from types import SimpleNamespace
from spark import delegate as d


def plan():
    job = dict(id='pilot', provider='anthropic', model='model-a', system='Explore freely.',
               prompt='One surprising possibility; at most 500 words.',
               max_output_tokens=8192, thinking_budget=4096, timeout_seconds=60)
    return dict(max_calls=2, max_total_output_tokens=16384,
                max_input_chars_per_call=2000,
                jobs=[job, dict(job, id='rival', model='model-b')])


def response(model='model-a', stop='end_turn', text='A distinct contribution.'):
    return dict(model=model, stop_reason=stop,
                content=[dict(type='opaque_test_block', payload='retained'),
                         dict(type='text', text=text)],
                usage=dict(input_tokens=200, output_tokens=18000,
                           output_tokens_details=dict(thinking_tokens=17999)))


class DelegationTests(unittest.TestCase):
    def execute(self, answers, p=None, accountant=lambda *_: None):
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        calls = []
        def sender(job):
            calls.append(job['id'])
            r = answers[len(calls)-1]
            if isinstance(r, Exception):
                raise r
            return r
        directory, summary = d.run(p or plan(), root.name, sender, accountant)
        return calls, directory, summary

    def test_observed_exhaustion_stops_fanout_and_keeps_all_blocks(self):
        raw = response(stop='max_tokens', text='')
        calls, directory, s = self.execute([raw])
        self.assertEqual(calls, ['pilot'])
        self.assertEqual(s['status'], 'stopped')
        self.assertEqual(s['jobs'][0]['status'], 'incomplete')
        self.assertEqual(s['unstarted'], ['rival'])
        self.assertEqual(json.loads((directory/'0-response.json').read_text()), raw)
        self.assertEqual((directory/'0-response.json').stat().st_mode & 0o777, 0o600)

    def test_success_keeps_distinct_outputs_not_completion_claim(self):
        calls, directory, s = self.execute([response(), response('model-b', text='Disagree.')])
        self.assertEqual(calls, ['pilot', 'rival'])
        self.assertEqual(s['status'], 'needs_editorial_review')
        self.assertEqual(json.loads((directory/'1-result.json').read_text())['text'], 'Disagree.')

    def test_partial_prose_is_salvaged_not_adopted(self):
        _, directory, s = self.execute([response(stop='max_tokens', text='Useful fragment')])
        self.assertEqual(s['status'], 'stopped')
        self.assertEqual(json.loads((directory/'0-result.json').read_text())['text'], 'Useful fragment')

    def test_empty_complete_is_not_success(self):
        self.assertEqual(self.execute([response(text=' ')])[2]['jobs'][0]['status'], 'empty')

    def test_unapproved_model_is_not_silently_used(self):
        s = self.execute([response('model-a-other')])[2]
        self.assertEqual(s['jobs'][0]['status'], 'model_mismatch')
        self.assertEqual(s['jobs'][0]['returned_model'], 'model-a-other')

    def test_explicit_alias(self):
        p = plan(); p['jobs'] = p['jobs'][:1]
        p['jobs'][0]['accepted_model_ids'] = ['model-a-dated']
        self.assertEqual(self.execute([response('model-a-dated')], p)[2]['status'], 'needs_editorial_review')

    def test_timeout_no_retry_and_unknown_billing(self):
        calls, _, s = self.execute([TimeoutError('private detail')])
        self.assertEqual(calls, ['pilot'])
        self.assertEqual(s['jobs'][0]['billing'], 'unknown')
        self.assertNotIn('private detail', json.dumps(s))

    def test_accounting_failure_keeps_receipt_and_stops(self):
        def bad(*_): raise OSError()
        calls, directory, s = self.execute([response()], accountant=bad)
        self.assertTrue((directory/'0-response.json').exists())
        self.assertEqual(calls, ['pilot'])
        self.assertEqual(s['jobs'][0]['status'], 'receipt_processing_error')

    def test_preflight_rejects_all_before_dispatch(self):
        for key, value in [('max_calls', 1), ('max_total_output_tokens', 8192),
                           ('max_input_chars_per_call', 1)]:
            p = plan(); p[key] = value
            with self.subTest(key=key), patch.object(d, 'send') as send:
                with tempfile.TemporaryDirectory() as root, self.assertRaises(ValueError):
                    d.run(p, root, send)
                send.assert_not_called()

    def test_bad_thinking_and_timeout(self):
        for change in [dict(thinking_budget=8192), dict(effort='high'), dict(timeout_seconds=601)]:
            p = plan(); p['jobs'][0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError): d.validate(p)

    def test_openai_and_refusal(self):
        j = dict(plan()['jobs'][0], provider='openai')
        r = dict(model='model-a', status='completed', output=[dict(type='message', content=[
            dict(type='output_text', text='Another view.')])])
        self.assertEqual(d.classify(j, r)['status'], 'delivered')
        r['output'][0]['content'] = [dict(type='refusal', refusal='No')]
        self.assertEqual(d.classify(j, r)['status'], 'refused')
        r['status'] = 'incomplete'; r['output'] = []
        self.assertEqual(d.classify(j, r)['status'], 'incomplete')

    def test_adapters_send_once_with_exact_limits_and_no_tools(self):
        for provider in ('anthropic', 'openai'):
            job = dict(plan()['jobs'][0], provider=provider)
            if provider == 'openai':
                job.pop('thinking_budget'); job['effort'] = 'medium'
            client = MagicMock()
            sdk = MagicMock(return_value=MagicMock())
            sdk.return_value.__enter__.return_value = client
            module = SimpleNamespace(**{('Anthropic' if provider == 'anthropic' else 'OpenAI'): sdk})
            route = client.messages.create if provider == 'anthropic' else client.responses.create
            route.return_value.model_dump.return_value = response()
            with patch.dict('sys.modules', {provider: module}):
                d.send(job)
            self.assertEqual(sdk.call_args.kwargs['max_retries'], 0)
            self.assertEqual(sdk.call_args.kwargs['timeout'], 60)
            route.assert_called_once()
            kw = route.call_args.kwargs
            self.assertEqual(kw['model'], job['model'])
            self.assertNotIn('tools', kw)
            if provider == 'anthropic':
                self.assertEqual(kw['thinking']['budget_tokens'], 4096)
                self.assertEqual(kw['max_tokens'], 8192)
            else:
                self.assertFalse(kw['store'])
                self.assertEqual(kw['max_output_tokens'], 8192)

    def test_adaptive_does_not_claim_answer_reservation(self):
        p = plan()
        for j in p['jobs']:
            j.pop('thinking_budget'); j['effort'] = 'high'
        self.assertIn('reserves no answer tokens', d.validate(p)['warning'])

    def test_cli_preflight_does_not_dispatch(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'plan.json'; path.write_text(json.dumps(plan()))
            with patch('sys.argv', ['delegate', str(path)]), patch.object(d, 'run') as run:
                with patch('builtins.print'): d.main()
                run.assert_not_called()

    def test_shared_usage_records_actual_model_and_caching(self):
        with tempfile.TemporaryDirectory() as root, patch.object(Path, 'home', return_value=Path(root)):
            raw = response('actual-model')
            raw['usage']['cache_read_input_tokens'] = 80
            d.account(plan()['jobs'][0], raw)
            row = json.loads((Path(root)/'.cache/vybn-phase/api_usage.jsonl').read_text())
            self.assertEqual(row['model'], 'actual-model')
            self.assertEqual(row['out'], 18000)
            self.assertEqual(row['cache_r'], 80)

    def test_new_run_never_adopts_old_files(self):
        with tempfile.TemporaryDirectory() as root:
            p = plan(); p['jobs'] = p['jobs'][:1]
            a, _ = d.run(p, root, lambda _: response(), lambda *_: None)
            b, s = d.run(p, root, lambda _: response(text=''), lambda *_: None)
            self.assertNotEqual(a, b)
            self.assertEqual(s['status'], 'stopped')

    def test_insecure_root_refused_before_call(self):
        with tempfile.TemporaryDirectory() as root:
            Path(root).chmod(0o755)
            with self.assertRaises(ValueError): d.run(plan(), root, lambda _: self.fail('called'))


if __name__ == '__main__': unittest.main()
