#!/usr/bin/env python3
"""Short greedy Responses requests, full-history continuations, and replay checks.

Run on the server host while the server already holds its granted gpuq lease.
The CPU-only HTTP client never requests a lease. A serial /health request
AFTER each response is the idle barrier: Qwen checkpoints after writing the terminal
response, so receiving that terminal response alone does not prove GPU work ended.
--compare reads completed local captures and performs no HTTP/network operations.
"""
import argparse
import copy
import http.client
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit

PROMPT = 'Write every integer from 1 through 30, separated by commas. Write the entire list and no explanation.'
STOP_PROMPT = 'Reply with exactly one word: OK'
FOLLOWUP = 'Continue the list from the next integer. Write the next 20 integers, separated by commas.'


def normalized(value):
    if isinstance(value, dict):
        return {k: normalized(v) for k, v in value.items() if k not in ('id', 'created_at', 'completed_at')}
    if isinstance(value, list):
        return [normalized(v) for v in value]
    return value


def canonical(response, cache=True):
    result = {key: normalized(response.get(key)) for key in ('status', 'incomplete_details', 'output', 'usage')}
    if not cache:
        result['usage']['input_tokens_details'].pop('cached_tokens', None)
    return result


def parse_response(status, content_type, payload):
    assert status == 200, (status, payload)
    if not content_type.startswith('text/event-stream'):
        return json.loads(payload), None
    events = []
    for block in payload.replace('\r\n', '\n').strip().split('\n\n'):
        fields = dict(line.split(': ', 1) for line in block.splitlines() if ': ' in line)
        value = json.loads(fields['data'])
        assert fields.get('event') == value['type'], fields
        assert value['sequence_number'] == len(events), value
        events.append(value)
    assert events[0]['type'] == 'response.created', events
    assert events[-1]['type'] in ('response.completed', 'response.incomplete'), events[-1]
    response = events[-1]['response']
    assert [e['item'] for e in events if e['type'] == 'response.output_item.done'] == response['output']
    delta = ''.join(e['delta'] for e in events if e['type'] == 'response.output_text.delta')
    assert delta == visible_text(response), (delta, response)
    return response, events


def visible_text(response):
    return ''.join(part['text'] for item in response['output'] if item['type'] == 'message'
                   for part in item['content'] if part['type'] == 'output_text')


def validate_boundary(response, budget, stopped=False):
    usage = response['usage']
    if stopped:
        assert response['status'] == 'completed', 'EOS probe did not terminate within its bounded budget'
        assert response.get('incomplete_details') is None, response
        assert 0 < usage['output_tokens'] <= budget, (budget, usage)
    else:
        assert response['status'] == 'incomplete', 'prompt stopped early; output boundary not exercised'
        assert response['incomplete_details'] == {'reason': 'max_output_tokens'}, response
        assert usage['output_tokens'] == budget, (budget, usage)
    assert usage['total_tokens'] == usage['input_tokens'] + usage['output_tokens'], usage
    assert 0 <= usage['input_tokens_details']['cached_tokens'] < usage['input_tokens'], usage
    assert visible_text(response), 'empty visible output cannot test assistant-history continuation'


def cache_coverage(rows):
    initial = {row['budget']: row['response']['usage']['input_tokens']
               for row in rows if row['stage'] == 'initial'}
    evidence = []
    followup_hits = nonzero = 0
    for row in rows:
        cached = row['response']['usage']['input_tokens_details']['cached_tokens']
        nonzero += cached != 0
        if row['stage'] == 'followup':
            followup_hits += cached > 0
            if row['budget'] >= 2:
                evidence.append(dict(budget=row['budget'], initial_input_tokens=initial[row['budget']],
                                     followup_cached_tokens=cached,
                                     reuses_generated_state=cached > initial[row['budget']]))
    return dict(followup_cache_hits=followup_hits, nonzero_cache_requests=nonzero,
                generated_state_cache_hits=sum(row['reuses_generated_state'] for row in evidence),
                generated_state_cache_evidence=evidence)


def validate_cache_coverage(coverage, disabled):
    if disabled:
        assert coverage['nonzero_cache_requests'] == 0, 'no-prefix-cache request reported cached tokens'
    else:
        assert coverage['generated_state_cache_hits'] > 0, 'no budget>=2 continuation reused generated state; generated-state cache regression untested'


def compare(baseline, candidate):
    left = [json.loads(line) for line in (baseline / 'responses.jsonl').read_text().splitlines()]
    right = [json.loads(line) for line in (candidate / 'responses.jsonl').read_text().splitlines()]
    disabled = []
    for directory, rows in ((baseline, left), (candidate, right)):
        summary = json.loads((directory / 'client-summary.json').read_text())
        assert summary.get('passed') and summary.get('safe_to_stop'), directory
        disabled.append(summary.get('prefix_cache_disabled', False))
        validate_cache_coverage(cache_coverage(rows), disabled[-1])
    compare_cache = not any(disabled)
    assert len(left) == len(right) and left, (len(left), len(right))
    for a, b in zip(left, right):
        label = (a['budget'], a['stage'])
        assert label == (b['budget'], b['stage']), (label, b)
        assert normalized(a['request']['input']) == normalized(b['request']['input']), (label, 'input differs')
        assert canonical(a['response'], cache=compare_cache) == canonical(b['response'], cache=compare_cache), (label, 'output, status, or usage/cache count differs', canonical(a['response']), canonical(b['response']))
    return dict(passed=True, requests=len(left), budgets=sorted({r['budget'] for r in left}),
                exact_visible_output_status_and_usage=True, exact_cached_token_counts=compare_cache,
                no_prefix_cache_oracle=disabled[0] != disabled[1],
                ignored_usage_fields=[] if compare_cache else ['input_tokens_details.cached_tokens'],
                note='Responses does not expose token IDs or internal KV history; this is API/cache-continuation parity.')


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    url = urlsplit(args.url)
    assert url.scheme == 'http' and url.hostname in ('127.0.0.1', 'localhost'), 'use the local dedicated test server'
    summary = dict(started_unix=time.time(), model=args.model, budgets=args.budgets,
                   prompt=args.prompt, followup=FOLLOWUP, passed=False, safe_to_stop=False,
                   completed_requests=0, followup_cache_hits=0, pause_seconds=args.pause_seconds, server_identity_verified=False,
                   last_generation_terminal_verified=True, stop_probe=args.stop_probe,
                   prefix_cache_disabled=args.no_prefix_cache)
    captured = []
    deadline = time.monotonic() + args.wall_seconds

    def save():
        temporary = args.out / 'client-summary.tmp'
        temporary.write_text(json.dumps(summary, indent=2) + '\n')
        temporary.replace(args.out / 'client-summary.json')

    def send(method, path, body=None, timeout=None):
        connection = http.client.HTTPConnection(url.hostname, url.port, timeout=timeout or args.request_timeout)
        started = time.time()
        try:
            connection.request(method, path, None if body is None else json.dumps(body).encode(), {'Content-Type': 'application/json'})
            response = connection.getresponse()
            payload = response.read().decode()
            raw = dict(started_unix=started, finished_unix=time.time(), method=method, path=path,
                       request=body, status=response.status, headers=response.getheaders(), payload=payload)
            with (args.out / 'http.jsonl').open('a') as log:
                log.write(json.dumps(raw) + '\n')
            return response.status, response.getheader('Content-Type', ''), payload
        finally:
            connection.close()

    def idle_barrier():
        assert summary['server_identity_verified'], 'server identity not verified'
        status, _, payload = send('GET', '/health', timeout=5)
        assert status == 200 and json.loads(payload).get('status') == 'ok', payload
        summary['safe_to_stop'] = True
        summary['idle_barrier_unix'] = time.time()
        save()

    def request(budget, stage, history, limit, stream, stopped=False):
        if time.monotonic() >= deadline:
            raise TimeoutError('client wall budget reached between requests')
        body = dict(model=args.model, input=copy.deepcopy(history), temperature=0, top_k=1,
                    top_p=1, seed=1, store=False, reasoning={'effort': 'none'},
                    max_output_tokens=limit, stream=stream)
        summary['safe_to_stop'] = False
        summary['last_generation_terminal_verified'] = False
        save()
        started = time.time()
        status, kind, payload = send('POST', '/v1/responses', body)
        response, events = parse_response(status, kind, payload)
        assert response.get('status') in ('completed', 'incomplete'), 'generation terminal response missing'
        summary['last_generation_terminal_verified'] = True
        # This serial request completes only after checkpoint() and handle() return.
        idle_barrier()
        row = dict(budget=budget, stage=stage, request=body, response=response, events=events,
                   started_unix=started, finished_unix=time.time())
        with (args.out / 'responses.jsonl').open('a') as log:
            log.write(json.dumps(row) + '\n')
        captured.append(row)
        validate_boundary(response, limit, stopped)
        if args.no_prefix_cache:
            assert response['usage']['input_tokens_details']['cached_tokens'] == 0, 'no-prefix-cache request reported cached tokens'
        summary['completed_requests'] += 1
        if stage == 'followup' and response['usage']['input_tokens_details']['cached_tokens'] > 0:
            summary['followup_cache_hits'] += 1
        save()
        print(json.dumps(dict(budget=budget, stage=stage, usage=response['usage'], text=visible_text(response))), flush=True)
        return response

    save()
    code = 1
    try:
        status, _, payload = send('GET', '/v1/models')
        assert status == 200 and json.loads(payload)['data'][0]['id'] == args.model, 'wrong test server'
        summary['server_identity_verified'] = True
        idle_barrier()
        for index, budget in enumerate(args.budgets):
            history = [{'role': 'user', 'content': args.prompt}]
            first = request(budget, 'initial', history, budget, budget % 2 == 0)
            continuation = history + first['output'] + [{'role': 'user', 'content': FOLLOWUP}]
            followup = request(budget, 'followup', continuation, 8, budget % 2 != 0)
            followup_replay = request(budget, 'followup-replay', continuation, 8, budget % 2 == 0)
            assert canonical(followup, cache=False) == canonical(followup_replay, cache=False), ('followup replay mismatch', budget)
            if budget <= 3:
                edited = [{'role': 'user', 'content': args.prompt.replace('30', '31')}]
                request(budget, 'edited-prefix', edited, 8, False)
            replay = request(budget, 'initial-replay', history, budget, budget % 2 != 0)
            assert canonical(first, cache=False) == canonical(replay, cache=False), ('initial replay mismatch', budget)
            if index + 1 < len(args.budgets):
                time.sleep(args.pause_seconds)
        if args.stop_probe:
            stop_history = [{'role': 'user', 'content': STOP_PROMPT}]
            stop = request(8, 'stop-initial', stop_history, 8, False, stopped=True)
            stop_replay = request(8, 'stop-replay', stop_history, 8, True, stopped=True)
            assert canonical(stop, cache=False) == canonical(stop_replay, cache=False), 'EOS replay mismatch'
        # Template re-encoding can invalidate individual short continuations, but
        # at least one must reuse more than its initial prompt to cover generated KV.
        summary.update(cache_coverage(captured))
        validate_cache_coverage(summary, args.no_prefix_cache)
        summary['passed'] = True
        code = 0
    except Exception as error:
        summary['error'] = repr(error)
        print(str(error), file=sys.stderr, flush=True)
    finally:
        # A failed parse/assertion can still be safely shut down after this barrier.
        # A stalled HTTP/GPU operation has no barrier and must be left untouched.
        if not summary['safe_to_stop'] and summary['last_generation_terminal_verified']:
            try:
                idle_barrier()
            except Exception as error:
                summary['idle_barrier_error'] = repr(error)
        summary['finished_unix'] = time.time()
        save()
    return code


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compare', nargs=2, type=Path, metavar=('BASELINE', 'CANDIDATE'))
    p.add_argument('--url', default='http://127.0.0.1:8017')
    p.add_argument('--model', default='mtp-budget-regression')
    p.add_argument('--out', type=Path)
    p.add_argument('--budgets', default='1,2,3,4,5,6,7,8')
    p.add_argument('--prompt', default=PROMPT)
    p.add_argument('--stop-probe', action='store_true', help='also require a greedy EOS within eight output tokens')
    p.add_argument('--no-prefix-cache', action='store_true', help='require zero cached tokens for every response')
    p.add_argument('--pause-seconds', type=float, default=2)
    p.add_argument('--request-timeout', type=float, default=30)
    p.add_argument('--wall-seconds', type=float, default=120)
    args = p.parse_args()
    if args.compare:
        print(json.dumps(compare(*args.compare), indent=2))
        return 0
    if args.out is None:
        p.error('--out is required for HTTP capture')
    try:
        args.budgets = [int(value) for value in args.budgets.split(',')]
        assert args.budgets and len(set(args.budgets)) == len(args.budgets) and all(1 <= n <= 8 for n in args.budgets)
        assert 0 <= args.pause_seconds <= 10 and 0 < args.request_timeout <= 30 and 0 < args.wall_seconds <= 180
    except (ValueError, AssertionError):
        p.error('use unique budgets 1..8, pause 0..10s, request timeout 0..30s, wall time 0..180s')
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
