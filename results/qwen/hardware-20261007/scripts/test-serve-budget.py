#!/usr/bin/env python3
"""CPU-only local HTTP mocks and dry-run tests; no SSH, models, GPUs or leases."""
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('budget_client', HERE / 'serve-budget-client.py')
CLIENT = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CLIENT)


class MockHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.0'

    def log_message(self, *_):
        pass

    def send_payload(self, status, payload, content_type='application/json'):
        raw = payload.encode()
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        self.server.calls.append(('GET', self.path))
        value = {'status': 'ok'} if self.path == '/health' else {'data': [{'id': self.server.model}]}
        self.send_payload(200, json.dumps(value))

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.calls.append(('POST', body))
        if any('status' in item and item['status'] != 'completed' for item in body['input']):
            self.send_payload(400, json.dumps({'error': 'replayed input items must have completed status'}))
            return
        if self.server.fail_http:
            self.send_payload(500, json.dumps({'error': 'simulated uncertain GPU error'}))
            return
        stopped = body['input'][0]['content'] == CLIENT.STOP_PROMPT or self.server.stop_early
        tokens = 2 if stopped else body['max_output_tokens']
        text = 'OK' if stopped else ' '.join(str(n) for n in range(1, tokens + 1))
        input_tokens = 100 if len(body['input']) > 1 else 60
        cached = 0 if self.server.no_prefix_cache or len(body['input']) == 1 else self.server.followup_cached
        item = {'id': 'msg-' + str(len(self.server.calls)), 'type': 'message', 'role': 'assistant',
                'status': 'completed' if stopped else 'incomplete',
                'content': [{'type': 'output_text', 'text': text, 'annotations': [], 'logprobs': []}]}
        response = {'id': 'response-unique', 'status': 'completed' if stopped else 'incomplete',
                    'incomplete_details': None if stopped else {'reason': 'max_output_tokens'},
                    'output': [item], 'usage': {'input_tokens': input_tokens,
                    'input_tokens_details': {'cached_tokens': cached, 'cache_write_tokens': 0},
                    'output_tokens': tokens, 'output_tokens_details': {'reasoning_tokens': 0}, 'total_tokens': input_tokens + tokens}}
        if not body['stream']:
            self.send_payload(200, json.dumps(response))
            return
        events = [{'type': 'response.created', 'response': {'id': 'response-unique'}},
                  {'type': 'response.output_text.delta', 'delta': text},
                  {'type': 'response.output_item.done', 'item': item},
                  {'type': 'response.completed' if stopped else 'response.incomplete', 'response': response}]
        payload = ''
        for index, event in enumerate(events):
            event['sequence_number'] = index
            payload += 'event: ' + event['type'] + '\ndata: ' + json.dumps(event) + '\n\n'
        self.send_payload(200, payload, 'text/event-stream')


class ClientRegressionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='serve-budget-test-')
        self.root = Path(self.temp.name)
        self.server = HTTPServer(('127.0.0.1', 0), MockHandler)
        self.server.model = 'test-model'
        self.server.fail_http = self.server.stop_early = False
        self.server.no_prefix_cache = False
        self.server.followup_cached = 61
        self.server.calls = []
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def run_client(self, label, **patch):
        settings = dict(url=f'http://127.0.0.1:{self.server.server_port}', model='test-model',
                        out=self.root / label, budgets=list(range(1, 9)), prompt=CLIENT.PROMPT,
                        pause_seconds=0, request_timeout=2, wall_seconds=30, stop_probe=True,
                        no_prefix_cache=False)
        settings.update(patch)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = CLIENT.run(SimpleNamespace(**settings))
        return code, json.loads((settings['out'] / 'client-summary.json').read_text())

    def test_full_sequence_and_comparison(self):
        for label in ['plain', 'mtp', 'adaptive']:
            code, summary = self.run_client(label)
            self.assertEqual(code, 0, summary)
            self.assertTrue(summary['safe_to_stop'])
            self.assertTrue(summary['server_identity_verified'])
            self.assertEqual(summary['completed_requests'], 37)
            self.assertEqual(summary['followup_cache_hits'], 8)
            self.assertEqual(summary['generated_state_cache_hits'], 7)
        for label in ['mtp', 'adaptive']:
            comparison = CLIENT.compare(self.root / 'plain', self.root / label)
            self.assertEqual(comparison['requests'], 37)
            self.assertTrue(comparison['exact_cached_token_counts'])
        for index, (method, _) in enumerate(self.server.calls):
            if method == 'POST':
                self.assertEqual(self.server.calls[index + 1], ('GET', '/health'))

    def test_prompt_only_cache_hits_do_not_claim_generated_state_coverage(self):
        self.server.followup_cached = 60
        code, summary = self.run_client('prompt-cache-only')
        self.assertEqual(code, 1)
        self.assertTrue(summary['safe_to_stop'])
        self.assertEqual(summary['followup_cache_hits'], 8)
        self.assertEqual(summary['generated_state_cache_hits'], 0)
        self.assertIn('generated-state cache regression untested', summary['error'])

    def test_incomplete_output_is_replayed_as_valid_assistant_input(self):
        code, summary = self.run_client('incomplete-replay', budgets=[1, 2], stop_probe=False)
        self.assertEqual(code, 0, summary)
        rows = [json.loads(line) for line in (self.root / 'incomplete-replay/responses.jsonl').read_text().splitlines()]
        for budget in [1, 2]:
            first = next(row for row in rows if row['budget'] == budget and row['stage'] == 'initial')
            followup = next(row for row in rows if row['budget'] == budget and row['stage'] == 'followup')
            original = first['response']['output'][0]
            replayed = followup['request']['input'][1]
            self.assertEqual(original['status'], 'incomplete')
            self.assertEqual(set(replayed), {'type', 'role', 'content'})
            self.assertEqual((replayed['type'], replayed['role']), ('message', 'assistant'))
            self.assertEqual([part['text'] for part in replayed['content']],
                             [part['text'] for part in original['content']])
        fixture = copy.deepcopy(first['response'])
        fixture['output'][0]['content'][0]['text'] = ' 1,\n二,🙂 '
        unchanged = copy.deepcopy(fixture)
        converted = CLIENT.assistant_input(fixture)
        self.assertEqual(converted[0]['content'][0]['text'], ' 1,\n二,🙂 ')
        self.assertEqual(fixture, unchanged)

    def test_only_one_budget_needs_generated_state_reuse(self):
        self.run_client('cached')
        rows = [json.loads(line) for line in (self.root / 'cached/responses.jsonl').read_text().splitlines()]
        for row in rows:
            if row['stage'] == 'followup' and row['budget'] != 2:
                row['response']['usage']['input_tokens_details']['cached_tokens'] = 60
        coverage = CLIENT.cache_coverage(rows)
        self.assertEqual(coverage['generated_state_cache_hits'], 1)
        CLIENT.validate_cache_coverage(coverage, False)
        rows[6]['response']['usage']['input_tokens_details']['cached_tokens'] = 60
        with self.assertRaisesRegex(AssertionError, 'generated-state cache regression untested'):
            CLIENT.validate_cache_coverage(CLIENT.cache_coverage(rows), False)

    def test_uncached_oracle_ignores_only_cached_tokens(self):
        self.run_client('plain')
        self.server.no_prefix_cache = True
        code, summary = self.run_client('uncached', no_prefix_cache=True)
        self.assertEqual(code, 0, summary)
        self.assertEqual(summary['nonzero_cache_requests'], 0)
        self.assertEqual(summary['completed_requests'], 37)
        comparison = CLIENT.compare(self.root / 'plain', self.root / 'uncached')
        self.assertTrue(comparison['no_prefix_cache_oracle'])
        self.assertFalse(comparison['exact_cached_token_counts'])
        self.assertEqual(comparison['ignored_usage_fields'], ['input_tokens_details.cached_tokens'])
        path = self.root / 'uncached/responses.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[1]['response']['usage']['input_tokens_details']['cache_write_tokens'] = 1
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        with self.assertRaisesRegex(AssertionError, 'usage/cache count differs'):
            CLIENT.compare(self.root / 'plain', self.root / 'uncached')

    def test_uncached_oracle_rejects_any_cached_tokens(self):
        code, summary = self.run_client('incorrectly-cached', no_prefix_cache=True)
        self.assertEqual(code, 1)
        self.assertTrue(summary['safe_to_stop'])
        self.assertIn('no-prefix-cache request reported cached tokens', summary['error'])

    def test_cache_count_difference_is_detected(self):
        self.run_client('plain')
        self.run_client('mtp')
        path = self.root / 'mtp/responses.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[1]['response']['usage']['input_tokens_details']['cached_tokens'] += 1
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        with self.assertRaisesRegex(AssertionError, 'usage/cache count differs'):
            CLIENT.compare(self.root / 'plain', self.root / 'mtp')

    def test_early_stop_does_not_silently_pass_budget_coverage(self):
        self.server.stop_early = True
        code, summary = self.run_client('early-stop')
        self.assertEqual(code, 1)
        self.assertFalse(summary['passed'])
        self.assertTrue(summary['safe_to_stop'])
        self.assertIn('boundary not exercised', summary['error'])

    def test_http_error_never_marks_unknown_gpu_state_safe(self):
        self.server.fail_http = True
        code, summary = self.run_client('http-error')
        self.assertEqual(code, 1)
        self.assertFalse(summary['safe_to_stop'])
        self.assertFalse(summary['last_generation_terminal_verified'])
        self.assertEqual(self.server.calls[-1][0], 'POST')

    def test_wrong_server_identity_is_not_a_safe_barrier(self):
        self.server.model = 'unrelated-server'
        code, summary = self.run_client('wrong-server')
        self.assertEqual(code, 1)
        self.assertFalse(summary['safe_to_stop'])
        self.assertFalse(summary['server_identity_verified'])
        self.assertFalse(any(method == 'POST' for method, _ in self.server.calls))

    def test_runner_dry_runs_all_modes_without_ssh(self):
        mockbin = self.root / 'bin'
        mockbin.mkdir()
        marker = self.root / 'ssh-was-called'
        fake = mockbin / 'ssh'
        fake.write_text('#!/bin/sh\ntouch "' + str(marker) + '"\nexit 97\n')
        fake.chmod(0o755)
        env = dict(os.environ, PATH=str(mockbin) + os.pathsep + os.environ['PATH'])
        for label, mode, extra in [('plain', 'plain', []), ('mtp', 'mtp', []),
                                   ('adaptive', 'adaptive', []), ('uncached', 'plain', ['--no-prefix-cache'])]:
            proc = subprocess.run(['python3', str(HERE / 'run-serve-budget.py'), '--incite', '101', '--invite', '102',
                                   '--binary', 'candidate', '--name', label, '--mode', mode, '--stop-probe',
                                   '--fan-regime', 'dynamic-smc-maximum',
                                   '--out', str(self.root / 'dry'), '--dry-run', *extra], capture_output=True, text=True, env=env)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            manifest = json.loads((self.root / 'dry' / label / 'manifest.json').read_text())
            self.assertEqual(manifest['state'], 'dry-run; no remote operations')
            self.assertFalse(manifest['gpu_commands_started'])
            self.assertTrue(manifest['client_ssh_argv'][-1].startswith('python3 -c '))
            self.assertFalse(manifest['client_ssh_argv'][-1].startswith('gpuq '))
            self.assertEqual(manifest['batch'], 64)
            server = manifest['hosts']['incite']['command']
            self.assertEqual('--mtp' in server, mode != 'plain')
            self.assertEqual('--no-prefix-cache' in server, label == 'uncached')
            self.assertNotIn('--no-prefix-cache', manifest['hosts']['invite']['command'])
            self.assertEqual(manifest['fan_policy']['regime'], 'dynamic-smc-maximum')
            self.assertIn('TRASHCAN_QWEN_ADAPTIVE_MTP=' + ('1' if mode == 'adaptive' else '0'), manifest['hosts']['incite']['environment'])
        self.assertFalse(marker.exists())


    def test_cooled_plan_dry_run_uses_65c_and_sequential_references(self):
        mockbin = self.root / 'tripwire'
        mockbin.mkdir()
        marker = self.root / 'external-command-was-called'
        for name in ['ssh', 'gpuq']:
            fake = mockbin / name
            fake.write_text('#!/bin/sh\ntouch "' + str(marker) + '"\nexit 97\n')
            fake.chmod(0o755)
        out = self.root / 'cooled-dry'
        env = dict(os.environ, PATH=str(mockbin) + os.pathsep + os.environ['PATH'])
        proc = subprocess.run(['python3', str(HERE / 'cooled-serve-plan.py'), '--out', str(out), '--dry-run'],
                              capture_output=True, text=True, env=env)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        summary = json.loads((out / 'summary.json').read_text())
        self.assertEqual(summary['max_c'], 65)
        commands = summary['dry_run_commands']
        self.assertEqual([c[c.index('--mode') + 1] for c in commands], ['plain', 'mtp', 'adaptive', 'plain'])
        self.assertNotIn('--reference', commands[0])
        self.assertIn('--reference', commands[1])
        self.assertIn('--reference', commands[2])
        self.assertIn('--reference', commands[3])
        self.assertIn('--no-prefix-cache', commands[3])
        self.assertTrue(all('--no-prefix-cache' not in command for command in commands[:3]))
        for command in commands:
            self.assertEqual(command[command.index('--max-c') + 1], '65')
            self.assertEqual(command[command.index('--binary') + 1], 'trashcan-ready')
            self.assertIn('--stop-probe', command)
        self.assertFalse(marker.exists())
        rejected = subprocess.run(['python3', str(HERE / 'cooled-serve-plan.py'), '--max-c', '66', '--dry-run'],
                                  capture_output=True, text=True, env=env)
        self.assertEqual(rejected.returncode, 2)
        self.assertFalse(marker.exists())

    def test_reused_release_evidence_requires_finished_restored_scopes(self):
        helper_spec = importlib.util.spec_from_file_location('cooled_release_checks', HERE / 'cooled-plan.py')
        helper = importlib.util.module_from_spec(helper_spec)
        helper_spec.loader.exec_module(helper)
        run = self.root / 'release-evidence'
        run.mkdir()
        (run / 'manifest.json').write_text(json.dumps({'state': 'completed'}))
        scope = {'finished_unix': 1, 'returncode': -15, 'governors_before': {'cpu0': 'schedutil'},
                 'governors_after': {'cpu0': 'schedutil'}, 'restore_errors': {}}
        for host in ['incite', 'invite']:
            (run / host).mkdir()
            (run / host / 'scope.json').write_text(json.dumps(scope))
        self.assertTrue(helper.release_evidence(run)[0])
        damaged = copy.deepcopy(scope)
        damaged['governors_after']['cpu0'] = 'performance'
        (run / 'invite/scope.json').write_text(json.dumps(damaged))
        self.assertFalse(helper.release_evidence(run)[0])
        (run / 'invite/scope.json').write_text(json.dumps(scope))
        (run / 'manifest.json').write_text(json.dumps({'state': 'incomplete; inspect jobs before releasing leases'}))
        self.assertFalse(helper.release_evidence(run)[0])


    def test_scope_accepts_popen_sigterm_but_rejects_unexpected_signal(self):
        helper_spec = importlib.util.spec_from_file_location('server_scope_checks', HERE / 'run-serve-budget.py')
        helper = importlib.util.module_from_spec(helper_spec)
        helper_spec.loader.exec_module(helper)
        scope = {'finished_unix': 1, 'returncode': -15, 'governors_before': {'cpu0': 'schedutil'},
                 'governors_after': {'cpu0': 'schedutil'}, 'restore_errors': {}}
        helper.verify_finished_scope(scope)
        self.assertEqual(scope['returncode'], -15)
        for code in (-9, 1):
            scope['returncode'] = code
            with self.assertRaises(AssertionError):
                helper.verify_finished_scope(scope)


if __name__ == '__main__':
    unittest.main()
