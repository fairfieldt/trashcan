#!/usr/bin/env python3
"""Tests for server.py's conversations, event log and tool loop, against a fake `trashcan serve` that streams
Responses events the way engine/src/responses.rs does. Standard library only:

    python3 tools/webchat/test_server.py -v

Files go to a directory under tools/webchat/.test-data (removed afterwards); nothing is sent to the network
(web_search and fetch_page are replaced, bash runs `echo`-style commands locally).
"""
import http.client, http.server, json, os, shutil, socketserver, stat, sys, threading, time, unittest
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import server as W  # noqa: E402

DATA = os.path.join(HERE, '.test-data')


# ------------------------------------------------------------------------------------------- a fake model server
def ev_created(model):
    return ('response.created', {'response': {'id': 'resp_1', 'model': model, 'status': 'in_progress', 'output': []}})


def reasoning(text, rid='rs_1'):
    item = {'id': rid, 'type': 'reasoning', 'status': 'completed', 'summary': [{'type': 'summary_text', 'text': text}]}
    evs = [('response.output_item.added', {'item': {'id': rid, 'type': 'reasoning', 'summary': []}})]
    evs += [('response.reasoning_summary_text.delta', {'item_id': rid, 'delta': w}) for w in split(text)]
    evs.append(('response.output_item.done', {'item': item}))
    return evs, item


def message(text, mid='msg_1'):
    item = {'id': mid, 'type': 'message', 'role': 'assistant', 'status': 'completed',
            'content': [{'type': 'output_text', 'text': text}]}
    evs = [('response.output_item.added', {'item': {'id': mid, 'type': 'message', 'content': []}})]
    evs += [('response.output_text.delta', {'item_id': mid, 'delta': w}) for w in split(text)]
    evs.append(('response.output_item.done', {'item': item}))
    return evs, item


def call(name, args, cid, fid=None):
    fid = fid or 'fc_' + cid
    item = {'id': fid, 'type': 'function_call', 'status': 'completed', 'call_id': cid, 'name': name,
            'arguments': json.dumps(args)}
    evs = [('response.output_item.added', {'item': dict(item, arguments='', status='in_progress')}),
           ('response.function_call_arguments.delta', {'item_id': fid, 'delta': item['arguments']}),
           ('response.output_item.done', {'item': item})]
    return evs, item


def split(text):
    out, cur = [], ''
    for ch in text:
        cur += ch
        if ch == ' ':
            out.append(cur)
            cur = ''
    return out + ([cur] if cur else [])


def response(*parts, model='qwen3.8-27b', status='completed', usage=(100, 20)):
    """The event list of one streamed response made of (events, item) parts."""
    evs = [ev_created(model)]
    items = []
    for e, item in parts:
        evs += e
        items.append(item)
    final = {'id': 'resp_1', 'model': model, 'status': status, 'output': items,
             'usage': {'input_tokens': usage[0], 'output_tokens': usage[1], 'total_tokens': sum(usage),
                       'input_tokens_details': {'cached_tokens': 0}}}
    evs.append(('response.completed' if status == 'completed' else 'response.incomplete', {'response': final}))
    return evs


class Fake:
    """Serves /v1/models and plays one scripted reply per POST /v1/responses: a list of events, or
    {'status': N, 'error': msg}, or {'events': [...], 'delay': s, 'hang': True} (stream slowly, or stop sending)."""

    def __init__(self):
        self.model = 'qwen3.8-27b'
        self.script, self.requests = [], []
        self.disconnected = threading.Event()
        fake = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                body = json.dumps({'object': 'list', 'data': [{'id': fake.model, 'object': 'model'}]}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                req = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                fake.requests.append(req)
                step = fake.script.pop(0) if fake.script else {'status': 500, 'error': 'no script'}
                if isinstance(step, list):
                    step = {'events': step}
                if 'status' in step:
                    body = json.dumps({'error': {'message': step['error']}}).encode()
                    self.send_response(step['status'])
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                try:
                    for kind, data in step['events']:
                        self.wfile.write(f'event: {kind}\ndata: {json.dumps(dict(data, type=kind))}\n\n'.encode())
                        self.wfile.flush()
                        time.sleep(step.get('delay', 0))
                    while step.get('hang'):   # the engine stops at its next write once the client has gone
                        self.wfile.write(b'event: x\ndata: {"type":"x"}\n\n')
                        self.wfile.flush()
                        time.sleep(0.05)
                except (BrokenPipeError, ConnectionResetError):
                    fake.disconnected.set()

        class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
            daemon_threads = True

        self.httpd = S(('127.0.0.1', 0), H)
        self.url = f'http://127.0.0.1:{self.httpd.server_address[1]}'
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def fake_tool(name, args, workdir, cancel):
    if name == 'web_search':
        rs = [{'title': f"About {args['query']}", 'url': 'https://example.com/a', 'snippet': 'snippet'}]
        return '1. ' + rs[0]['title'], {'results': rs}, False
    if name == 'slow':
        cancel.wait(10)
        return 'slow done', None, False
    return REAL_RUN_TOOL(name, args, workdir, cancel)


REAL_RUN_TOOL = W.run_tool


class Base(unittest.TestCase):
    def setUp(self):
        shutil.rmtree(DATA, ignore_errors=True)
        self.data = os.path.join(DATA, self.id().split('.')[-1])
        self.fake = Fake()
        W.run_tool = fake_tool
        self.hub = W.Hub(self.data, self.fake.url, HERE)
        self.hub.check_model()

    def tearDown(self):
        W.run_tool = REAL_RUN_TOOL
        self.fake.close()
        shutil.rmtree(DATA, ignore_errors=True)

    def wait_idle(self, s, timeout=10):
        end = time.time() + timeout
        while s.task is not None:
            self.assertLess(time.time(), end, 'turn did not finish')
            time.sleep(0.02)

    def turn(self, text='hi', thinking='medium', s=None):
        s = s or self.hub.create()
        code, out = self.hub.send(s, text, thinking)
        self.assertEqual(code, 202, out)
        self.wait_idle(s)
        return s, s.turns[-1]


# --------------------------------------------------------------------------------------------------------- tests
class TurnTests(Base):
    def test_simple_answer(self):
        self.fake.script.append(response(reasoning('think first'), message('Hello there.')))
        s, t = self.turn('hello')
        self.assertEqual(t['status'], 'done')
        self.assertIsNone(t['error'])
        self.assertEqual([(st['kind'], st['text']) for st in t['steps']],
                         [('thinking', 'think first'), ('text', 'Hello there.')])
        self.assertEqual(s.title, 'hello')
        self.assertEqual(s.context_tokens, 120)
        self.assertEqual(t['model'], 'qwen3.8-27b')
        req = self.fake.requests[0]
        self.assertNotIn('model', req)
        self.assertEqual(req['reasoning'], {'effort': 'medium', 'summary': 'auto'})
        self.assertEqual(req['max_output_tokens'], 16384)
        self.assertEqual(req['input'], [{'type': 'message', 'role': 'user',
                                         'content': [{'type': 'input_text', 'text': 'hello'}]}])
        # stored on disk, and loaded again by a new server
        hub2 = W.Hub(self.data, self.fake.url, HERE)
        s2 = hub2.sessions[s.id]
        self.assertEqual(s2.turns[0]['steps'][1]['text'], 'Hello there.')
        self.assertEqual(s2.seq, s.seq)

    def test_thinking_off(self):
        self.fake.script.append(response(message('ok')))
        self.turn('x', thinking='off')
        self.assertEqual(self.fake.requests[0]['reasoning'], {'effort': 'none'})

    def test_history_replays_reasoning_and_tools(self):
        # Round 1 thinks and calls bash; round 2 answers. A second turn replays all of it, reasoning included, so the
        # engine's prefix cache matches what it generated.
        self.fake.script.append(response(reasoning('need a shell'), call('bash', {'command': 'echo hi-there'}, 'c1')))
        self.fake.script.append(response(message('It printed hi-there.')))
        s, t = self.turn('run echo')
        self.assertEqual(t['status'], 'done')
        tool = t['steps'][1]
        self.assertEqual((tool['kind'], tool['name'], tool['state']), ('tool', 'bash', 'done'))
        self.assertEqual(tool['arguments'], {'command': 'echo hi-there'})
        self.assertEqual(tool['result']['output'].strip(), 'hi-there')
        second = self.fake.requests[1]['input']
        self.assertEqual([i.get('type') for i in second], ['message', 'reasoning', 'function_call', 'function_call_output'])
        self.assertEqual(second[1]['summary'][0]['text'], 'need a shell')
        self.assertEqual(second[3], {'type': 'function_call_output', 'call_id': 'c1', 'output': 'hi-there\n'})
        self.assertNotIn('status', second[2])
        self.fake.script.append(response(message('Sure.')))
        self.turn('again', s=s)
        third = self.fake.requests[2]['input']
        self.assertEqual([i.get('type') for i in third],
                         ['message', 'reasoning', 'function_call', 'function_call_output', 'message', 'message'])
        self.assertEqual(third[4]['role'], 'assistant')
        self.assertEqual(third[5]['content'][0]['text'], 'again')

    def test_parallel_calls_and_narration(self):
        self.fake.script.append(response(message('Let me look both up.'),
                                         call('web_search', {'query': 'a'}, 'c1'),
                                         call('web_search', {'query': 'b'}, 'c2')))
        self.fake.script.append(response(message('Both found.')))
        s, t = self.turn('two facts')
        kinds = [st['kind'] for st in t['steps']]
        self.assertEqual(kinds, ['note', 'tool', 'tool', 'text'])
        self.assertEqual([st['result']['results'][0]['title'] for st in t['steps'][1:3]], ['About a', 'About b'])
        outs = [i for i in self.fake.requests[1]['input'] if i.get('type') == 'function_call_output']
        self.assertEqual([o['call_id'] for o in outs], ['c1', 'c2'])

    def test_bash_error_exit_code(self):
        self.fake.script.append(response(call('bash', {'command': 'echo oops; exit 3'}, 'c1')))
        self.fake.script.append(response(message('failed')))
        s, t = self.turn('x')
        self.assertEqual(t['steps'][0]['state'], 'error')
        self.assertIn('[exit code 3]', self.fake.requests[1]['input'][-1]['output'])

    def test_last_round_cannot_call_tools(self):
        for i in range(W.MAX_ROUNDS):
            self.fake.script.append(response(call('web_search', {'query': str(i)}, f'c{i}')))
        s, t = self.turn('loop')
        self.assertEqual(len(self.fake.requests), W.MAX_ROUNDS)
        self.assertEqual(self.fake.requests[-1]['tool_choice'], 'none')
        self.assertEqual(self.fake.requests[0]['tool_choice'], 'auto')

    def test_stop_mid_answer_closes_upstream(self):
        e, item = message('one two three four five six seven eight nine ten')
        self.fake.script.append({'events': [ev_created('qwen3.8-27b')] + e[:4], 'delay': 0.05, 'hang': True})
        s = self.hub.create()
        self.hub.send(s, 'long', 'low')
        end = time.time() + 5
        while not (s.turns[-1]['steps'] and s.turns[-1]['steps'][0]['text']):
            self.assertLess(time.time(), end)
            time.sleep(0.02)
        self.assertTrue(self.hub.stop(s))
        self.wait_idle(s)
        t = s.turns[-1]
        self.assertEqual(t['status'], 'stopped')
        self.assertTrue(self.fake.disconnected.wait(3), 'the model server kept writing to nobody')
        # the partial answer stays, and is what the model sees next time
        self.assertEqual(t['items'][-1]['role'], 'assistant')
        self.fake.script.append(response(message('ok')))
        self.turn('next', s=s)
        roles = [i.get('role') for i in self.fake.requests[-1]['input']]
        self.assertEqual(roles, ['user', 'assistant', 'user'])

    def test_stop_during_tool(self):
        self.fake.script.append(response(call('slow', {}, 'c1')))
        s = self.hub.create()
        self.hub.send(s, 'slow one', 'low')
        end = time.time() + 5
        while not any(st.get('state') == 'running' for st in s.turns[-1]['steps']):
            self.assertLess(time.time(), end)
            time.sleep(0.02)
        self.hub.stop(s)
        self.wait_idle(s)
        t = s.turns[-1]
        self.assertEqual(t['status'], 'stopped')
        self.assertEqual(t['steps'][0]['state'], 'stopped')
        self.assertEqual([i.get('type') for i in t['items']], ['message', 'function_call', 'function_call_output'])

    def test_context_overflow_400(self):
        self.fake.script.append({'status': 400, 'error': 'prompt has 210000 tokens; context is 204800 and needs room for output'})
        s, t = self.turn('huge')
        self.assertEqual(t['status'], 'error')
        self.assertEqual(t['error']['kind'], 'context')
        self.assertIn('too long for Qwen3.8-27B (204,800 tokens)', t['error']['message'])
        # a turn the model never answered is not sent again
        self.fake.script.append(response(message('ok')))
        self.turn('small', s=s)
        self.assertEqual(len(self.fake.requests[-1]['input']), 1)

    def test_context_full_incomplete(self):
        self.fake.script.append(response(message('partial'), status='incomplete', usage=(204000, 790)))
        s, t = self.turn('x')
        self.assertEqual(t['error']['kind'], 'context')

    def test_length_limit(self):
        self.fake.script.append(response(message('partial'), status='incomplete', usage=(1000, 16384)))
        s, t = self.turn('x')
        self.assertEqual(t['status'], 'done')
        self.assertEqual(t['error']['kind'], 'length')

    def test_model_server_down(self):
        self.fake.close()
        s, t = self.turn('anyone?')
        self.assertEqual(t['status'], 'error')
        self.assertEqual(t['error']['kind'], 'down')
        self.assertIn('not running', t['error']['message'])

    def test_failed_and_dropped_streams(self):
        self.fake.script.append([ev_created('qwen3.8-27b'), ('response.failed', {'response': {'error': {'message': 'GPU hung'}}})])
        s, t = self.turn('x')
        self.assertIn('GPU hung', t['error']['message'])
        e, _ = message('cut short')
        self.fake.script.append([ev_created('qwen3.8-27b')] + e[:2])   # ends without response.completed
        s, t = self.turn('y', s=s)
        self.assertEqual(t['error']['kind'], 'upstream')
        self.assertIn('stopped in the middle', t['error']['message'])

    def test_one_turn_at_a_time(self):
        self.fake.script.append({'events': response(message('slow answer here')), 'delay': 0.05})
        s = self.hub.create()
        self.assertEqual(self.hub.send(s, 'a', 'low')[0], 202)
        self.assertEqual(self.hub.send(s, 'b', 'low')[0], 409)
        self.assertEqual(self.hub.send(s, '   ', 'low')[0], 400)
        self.wait_idle(s)

    def test_restart_marks_running_turn(self):
        s = self.hub.create()
        s.turns.append({'id': 't1', 'user': 'q', 'status': 'running', 'steps': [{'kind': 'tool', 'state': 'running', 'done': False}],
                        'items': [{'type': 'message', 'role': 'user', 'content': []},
                                  {'type': 'function_call', 'call_id': 'c9', 'name': 'bash', 'arguments': '{}'}]})
        s.save()
        hub2 = W.Hub(self.data, self.fake.url, HERE)
        t = hub2.sessions[s.id].turns[0]
        self.assertEqual(t['status'], 'error')
        self.assertEqual(t['error']['kind'], 'interrupted')
        self.assertEqual(t['steps'][0]['state'], 'stopped')
        self.assertEqual(t['items'][-1]['type'], 'function_call_output')


class EventTests(Base):
    def test_log_and_snapshot(self):
        self.fake.script.append(response(reasoning('r'), message('a b c')))
        s, t = self.turn('q')
        seqs = [seq for seq, _ in s.log]
        self.assertEqual(seqs, list(range(1, s.seq + 1)))
        kinds = [ev['type'] for _, ev in s.log]
        self.assertEqual(kinds[0], 'session')     # the title
        self.assertIn('delta', kinds)
        self.assertEqual(kinds[-1], 'turn')
        # replaying the log rebuilds what the snapshot says
        turns = {}
        for _, ev in s.log:
            if ev['type'] == 'turn':
                turns[ev['turn']['id']] = json.loads(json.dumps(ev['turn']))
            elif ev['type'] == 'step':
                steps = turns[ev['turn']]['steps']
                while len(steps) <= ev['index']:
                    steps.append(None)
                steps[ev['index']] = json.loads(json.dumps(ev['step']))
            elif ev['type'] == 'delta':
                turns[ev['turn']]['steps'][ev['index']]['text'] += ev['text']
        self.assertEqual(list(turns.values()), s.public()['turns'])


class HttpTests(Base):
    def setUp(self):
        super().setUp()
        self.httpd = W.serve(self.hub, '127.0.0.1', 0)
        self.base = f'http://127.0.0.1:{self.httpd.server_address[1]}'
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        super().tearDown()

    def api(self, method, path, body=None):
        req = urllib.request.Request(self.base + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                     headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.load(r)
        except urllib.error.HTTPError as e:
            return e.code, json.load(e)

    def events(self, query, until, timeout=10):
        """Read the event stream until until(events) holds; returns [(id, event, data)]."""
        u = urllib.parse.urlsplit(self.base)
        c = http.client.HTTPConnection(u.hostname, u.port, timeout=timeout)
        c.request('GET', '/api/events?' + query)
        r = c.getresponse()
        self.assertEqual(r.status, 200)
        out, cur, end = [], {}, time.time() + timeout
        while time.time() < end:
            line = r.readline().decode().rstrip('\n')
            if line.startswith(('id:', 'event:', 'data:')):
                k, v = line.split(':', 1)
                cur[k] = v.strip()
            elif not line and cur.get('event'):
                out.append((int(cur['id']) if 'id' in cur else None, cur['event'], json.loads(cur['data'])))
                cur = {}
                if until(out):
                    c.close()
                    return out
        c.close()
        self.fail(f'event stream: condition not met; got {[e for _, e, _ in out]}')

    def test_api_flow(self):
        code, st = self.api('GET', '/api/state')
        self.assertEqual(code, 200)
        self.assertEqual(st['status']['model']['id'], 'qwen3.8-27b')
        self.fake.script.append(response(message('Paris.')))
        code, out = self.api('POST', '/api/sessions', {'text': 'Capital of France?', 'thinking': 'off'})
        self.assertEqual(code, 201)
        sid = out['session']['id']
        finished = lambda o: any((e == 'turn' and d['turn']['status'] == 'done') or
                                 (e == 'snapshot' and d['turns'] and d['turns'][-1]['status'] == 'done') for _, e, d in o)
        evs = self.events(f'session={sid}&after=0', finished)
        self.assertEqual(evs[0][1], 'snapshot')
        last = evs[-1][0]
        # reconnecting after the last event: no snapshot, nothing old again
        self.fake.script.append(response(message('Lyon is second.')))
        self.assertEqual(self.api('POST', f'/api/sessions/{sid}/messages', {'text': 'And second?'})[0], 202)
        evs2 = self.events(f'session={sid}&after={last}', lambda o: sum(e == 'turn' and d['turn']['status'] == 'done' for _, e, d in o) >= 1)
        self.assertNotIn('snapshot', [e for _, e, _ in evs2])
        self.assertTrue(all(i is None or i > last for i, _, _ in evs2))
        code, lst = self.api('GET', '/api/sessions')
        self.assertEqual([r['title'] for r in lst['sessions']], ['Capital of France?'])
        self.assertEqual(self.api('PATCH', f'/api/sessions/{sid}', {'title': 'France'})[1]['title'], 'France')
        code, one = self.api('GET', f'/api/sessions/{sid}')
        self.assertEqual(len(one['turns']), 2)
        self.assertNotIn('items', one['turns'][0])
        self.assertEqual(self.api('DELETE', f'/api/sessions/{sid}')[0], 200)
        self.assertEqual(self.api('GET', f'/api/sessions/{sid}')[0], 404)
        self.assertFalse(os.path.exists(os.path.join(self.data, sid + '.json')))

    def test_two_readers_and_a_late_one(self):
        self.fake.script.append({'events': response(message('w1 w2 w3 w4 w5 w6 w7 w8')), 'delay': 0.03})
        code, out = self.api('POST', '/api/sessions', {})
        sid = out['session']['id']
        done = lambda o: any(e == 'turn' and d['turn']['status'] == 'done' for _, e, d in o)
        res = {}
        readers = [threading.Thread(target=lambda k=k: res.__setitem__(k, self.events(f'session={sid}&after=0', done)))
                   for k in ('phone', 'desk')]
        for th in readers:
            th.start()
        time.sleep(0.2)
        self.api('POST', f'/api/sessions/{sid}/messages', {'text': 'go'})
        for th in readers:
            th.join(15)
        text = lambda evs: ''.join(d['text'] for _, e, d in evs if e == 'delta')
        self.assertEqual(text(res['phone']), 'w1 w2 w3 w4 w5 w6 w7 w8')
        self.assertEqual(text(res['phone']), text(res['desk']))
        # what the page builds from step and delta events alone (the final turn event would hide a double count)
        steps = {}
        for _, e, d in res['phone']:
            if e == 'step':
                steps[d['index']] = dict(d['step'])
            elif e == 'delta':
                steps[d['index']]['text'] += d['text']
        self.assertEqual(steps[0]['text'], 'w1 w2 w3 w4 w5 w6 w7 w8')
        # a reader that comes back with a sequence number from before the server kept its log gets a snapshot
        late = self.events(f'session={sid}&after=0', lambda o: len(o) >= 1)
        self.assertEqual(late[0][1], 'snapshot')
        self.assertEqual(late[0][2]['turns'][0]['steps'][0]['text'], 'w1 w2 w3 w4 w5 w6 w7 w8')

    def test_static_and_cache_headers(self):
        with urllib.request.urlopen(self.base + '/') as r:
            self.assertEqual(r.headers['Cache-Control'], 'no-cache')
            self.assertIn(b'<html', r.read())
        with urllib.request.urlopen(self.base + '/?transport=/trashcan-transport.js&ui=win31') as r:
            self.assertEqual(r.status, 200)
        self.assertEqual(self.api('GET', '/api/sessions/000000000000')[0], 404)

    def test_switch(self):
        script = os.path.join(self.data, 'cluster.sh')
        with open(script, 'w') as f:
            f.write('#!/bin/sh\necho "stopped driver on incite"\necho "waiting for the leases"\nsleep 0.3\n'
                    'echo "waiting for invite (queued: blocked by #12)"\nsleep 0.3\necho "start driver on incite"\n'
                    'echo "ready: $2"\n')
        os.chmod(script, os.stat(script).st_mode | stat.S_IEXEC)
        self.hub.cluster = script
        self.fake.model = 'qwen3.8-flash-next'
        code, out = self.api('POST', '/api/switch', {'recipe': '6-flashnext'})
        self.assertEqual(code, 202)
        self.assertEqual(self.api('POST', '/api/switch', {'recipe': '6-flashnext'})[0], 409)
        code, out = self.api('POST', '/api/sessions', {'text': 'hi'})
        self.assertEqual(out.get('error'), 'A model is starting. Send again when it is ready.')
        phases = set()
        end = time.time() + 10
        while self.hub.switch['state'] == 'switching':
            phases.add(self.hub.switch['phase'])
            self.assertLess(time.time(), end)
            time.sleep(0.02)
        self.assertEqual(self.hub.switch['state'], 'idle')
        self.assertIn('Waiting for invite (queued: blocked by #12)', phases)
        st = self.api('GET', '/api/state')[1]['status']
        self.assertEqual((st['model']['id'], st['model']['state'], st['model']['context']), ('qwen3.8-flash-next', 'ready', 32768))
        self.assertEqual(self.api('POST', '/api/switch', {'recipe': 'nope'})[0], 400)

    def test_switch_failure(self):
        script = os.path.join(self.data, 'cluster.sh')
        with open(script, 'w') as f:
            f.write('#!/bin/sh\necho "waiting for the leases"\necho "leases not granted after 10 min: invite (queued)" >&2\nexit 1\n')
        os.chmod(script, os.stat(script).st_mode | stat.S_IEXEC)
        self.hub.cluster = script
        self.api('POST', '/api/switch', {'recipe': '6-flashnext'})
        end = time.time() + 10
        while self.hub.switch['state'] == 'switching':
            self.assertLess(time.time(), end)
            time.sleep(0.02)
        self.assertEqual(self.hub.switch['state'], 'error')
        self.assertIn('leases not granted', self.hub.switch['error'])


if __name__ == '__main__':
    unittest.main()
