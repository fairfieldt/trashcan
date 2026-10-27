#!/usr/bin/env python3
"""The trashcan demo chat: a chat page (static/, the pipe 3.1 look) and the agent behind it, in one process.

Conversations live on this machine (--data, one JSON file each). A turn runs here, not in the browser: rounds of
POST /v1/responses to `trashcan serve` (--upstream), the tools the model calls (web_search, fetch_page, bash on this
machine), more rounds, until the model answers. Every change to a conversation is an event with a sequence number;
a page follows them with GET /api/events?session=ID&after=N (server-sent events) and can reconnect from any point,
from any device, while the turn goes on without it. The model the cluster serves is switched with cluster.sh next to
this file. Python standard library only. See README.md.

usage: server.py [--port 8090] [--host 127.0.0.1] [--upstream http://127.0.0.1:8000] [--data DIR] [--workdir DIR]
"""
import argparse, html.parser, http.client, http.server, json, os, re, signal, socket, socketserver, subprocess, sys
import threading, time, urllib.parse, urllib.request, uuid

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, 'static')
UA = 'Mozilla/5.0 (X11; Linux x86_64) trashcan-webchat'

INSTRUCTIONS = """You are a helpful assistant running on a cluster of 2013 Mac Pros (trashcan).
Tools: web_search, fetch_page, and bash (a shell on the cluster's first Mac).

## Answering questions
When asked a factual question, run 1-3 web_search calls first (fetch_page when the snippets are not enough), then
answer. Structure the answer: the direct answer first, then the key details in short sections or bullets, and the
sources you used at the end."""

TOOLS = [
    {'type': 'function', 'name': 'web_search', 'description': 'Search the web. Returns titles, URLs and snippets.',
     'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}, 'num_results': {'type': 'integer'}},
                    'required': ['query']}},
    {'type': 'function', 'name': 'fetch_page', 'description': 'Read a web page as plain text (the first 8000 characters).',
     'parameters': {'type': 'object', 'properties': {'url': {'type': 'string'}}, 'required': ['url']}},
    {'type': 'function', 'name': 'bash',
     'description': "Run a shell command on the cluster's first Mac (Debian, 60 s limit). Returns stdout and stderr.",
     'parameters': {'type': 'object', 'properties': {'command': {'type': 'string'}}, 'required': ['command']}},
]

# What the cluster can serve: the model id `trashcan serve` reports, the cluster.sh recipe that starts it, its context.
MODELS = [
    {'id': 'qwen3.8-flash-next', 'name': 'Qwen3.8-Flash-Next', 'macs': 6, 'recipe': '6-flashnext', 'context': 32768},
    # 204,800 tokens: what the D700s' VRAM holds for the 27B (cluster.sh passes --context).
    {'id': 'qwen3.8-27b', 'name': 'Qwen3.8-27B', 'macs': 4, 'recipe': '4-qwen27b', 'context': 204800},
]
LEVELS = ('off', 'low', 'medium', 'high')
MAX_ROUNDS = 10
LOG_KEEP = 4000          # events kept in memory per conversation; older readers get a snapshot
PING = 15                # seconds between keepalive events on an idle event stream


def model_info(mid):
    return next((m for m in MODELS if m['id'] == mid), None)


def now_ms():
    return int(time.time() * 1000)


# ------------------------------------------------------------------------------------------------------------ tools
class DuckLite(html.parser.HTMLParser):
    """Results of lite.duckduckgo.com: <a class='result-link' href=...>title</a>, then <td class='result-snippet'>."""

    def __init__(self):
        super().__init__()
        self.results, self.cur, self.mode = [], None, None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == 'a' and a.get('class') == 'result-link':
            self.cur = {'title': '', 'url': a.get('href', ''), 'snippet': ''}
            self.results.append(self.cur)
            self.mode = 'title'
        elif tag == 'td' and a.get('class') == 'result-snippet' and self.cur is not None:
            self.mode = 'snippet'

    def handle_endtag(self, tag):
        if (tag == 'a' and self.mode == 'title') or (tag == 'td' and self.mode == 'snippet'):
            self.mode = None

    def handle_data(self, data):
        if self.mode and self.cur is not None:
            self.cur[self.mode] += data


def web_search(query, n=6):
    if not query.strip():
        raise ValueError('empty query')
    body = urllib.parse.urlencode({'q': query}).encode()
    req = urllib.request.Request('https://lite.duckduckgo.com/lite/', data=body, headers={'User-Agent': UA})
    with urllib.request.urlopen(req, timeout=15) as r:
        p = DuckLite()
        p.feed(r.read().decode('utf-8', 'replace'))
    out = []
    for x in p.results:
        url = x['url']
        if 'duckduckgo.com/y.js' in url:  # ads
            continue
        out.append({'title': ' '.join(x['title'].split()), 'url': url, 'snippet': ' '.join(x['snippet'].split())})
    return out[:max(1, min(n, 10))]


class Text(html.parser.HTMLParser):
    SKIP = {'script', 'style', 'noscript', 'svg', 'nav', 'footer', 'header', 'form'}

    def __init__(self):
        super().__init__()
        self.parts, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag in ('p', 'br', 'li', 'tr', 'h1', 'h2', 'h3', 'h4', 'div'):
            self.parts.append('\n')

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def fetch_page(url, limit=8000):
    if not re.match(r'https?://', url):
        raise ValueError('only http(s) URLs')
    req = urllib.request.Request(url, headers={'User-Agent': UA})
    with urllib.request.urlopen(req, timeout=20) as r:
        kind = r.headers.get('Content-Type', '')
        raw = r.read(2_000_000).decode('utf-8', 'replace')
    if 'html' in kind:
        p = Text()
        p.feed(raw)
        raw = ''.join(p.parts)
    text = re.sub(r'\n\s*\n+', '\n\n', re.sub(r'[ \t]+', ' ', raw)).strip()
    return {'url': url, 'text': text[:limit], 'truncated': len(text) > limit}


def bash(command, workdir, cancel, timeout=60, limit=16000):
    """Runs in its own process group, so a timeout or Stop ends everything it started."""
    p = subprocess.Popen(['bash', '-lc', command], cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    chunks, size = [], [0]

    def read():
        for chunk in iter(lambda: p.stdout.read1(65536), b''):
            if size[0] < limit * 4:
                chunks.append(chunk)
                size[0] += len(chunk)
    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    deadline, note = time.time() + timeout, ''
    while p.poll() is None:
        if cancel.is_set() or time.time() > deadline:
            note = '(stopped)' if cancel.is_set() else f'(stopped after {timeout} s)'
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            break
        time.sleep(0.05)
    code = p.wait()
    reader.join(2)
    out = b''.join(chunks).decode('utf-8', 'replace')
    if len(out) > limit:
        out = out[:limit] + f'\n(output cut at {limit} characters)'
    if note:
        out = (out + '\n' if out else '') + note
        code = 124 if not cancel.is_set() else 130
    return {'output': out, 'exitCode': code}


def run_tool(name, args, workdir, cancel):
    """(text the model reads, structured result the page shows, is_error)."""
    try:
        if name == 'web_search':
            try:
                n = int(args.get('num_results') or 6)
            except (TypeError, ValueError):
                n = 6
            rs = web_search(str(args.get('query', '')), n)
            text = '\n\n'.join(f"{i + 1}. {x['title']}\n   {x['url']}\n   {x['snippet']}" for i, x in enumerate(rs))
            return text or 'No results', {'results': rs}, False
        if name == 'fetch_page':
            out = fetch_page(str(args.get('url', '')))
            return (f"{out['url']}{' (first 8000 characters)' if out['truncated'] else ''}\n\n{out['text']}",
                    out, False)
        if name == 'bash':
            out = bash(str(args.get('command', '')), workdir, cancel)
            text = out['output'] or '(no output)'
            if out['exitCode']:
                text += f"\n[exit code {out['exitCode']}]"
            return text, out, out['exitCode'] != 0
        return f'unknown tool {name!r}', None, True
    except Exception as e:  # the model sees the error and can try something else
        msg = str(e) or e.__class__.__name__
        return f'error: {msg}', {'error': msg}, True


# ---------------------------------------------------------------------------------------------------- conversations
class TurnError(Exception):
    def __init__(self, kind, message):
        super().__init__(message)
        self.kind = kind


class Stopped(Exception):
    pass


class Session:
    """One conversation: its turns, and an in-memory log of the events since this server loaded it."""

    def __init__(self, hub, d):
        self.hub = hub
        self.id = d['id']
        self.title = d.get('title', '')
        self.created = d.get('created', now_ms())
        self.updated = d.get('updated', self.created)
        self.seq = d.get('seq', 0)
        self.context_tokens = d.get('context_tokens', 0)
        self.context_model = d.get('context_model')
        self.turns = d.get('turns', [])
        self.log = []          # [(seq, event)]
        self.task = None       # the running turn's Task
        self.deleted = False

    def stored(self):
        return {'id': self.id, 'title': self.title, 'created': self.created, 'updated': self.updated, 'seq': self.seq,
                'context_tokens': self.context_tokens, 'context_model': self.context_model, 'turns': self.turns}

    def public(self):
        """What a page sees: everything but the replay items."""
        return {'id': self.id, 'title': self.title, 'created': self.created, 'updated': self.updated, 'seq': self.seq,
                'context_tokens': self.context_tokens, 'context_model': self.context_model,
                'running': self.task is not None, 'turns': [public_turn(t) for t in self.turns]}

    def summary(self):
        last = self.turns[-1] if self.turns else None
        return {'id': self.id, 'title': self.title or 'New conversation', 'created': self.created,
                'updated': self.updated, 'turns': len(self.turns), 'running': self.task is not None,
                'status': last['status'] if last else None}

    # Called with hub.cond held.
    def emit(self, kind, **data):
        self.seq += 1
        # Frozen now: the steps it carries keep changing, and a reader may serialize it much later.
        ev = json.loads(json.dumps(dict(data, type=kind, seq=self.seq)))
        self.log.append((self.seq, ev))
        if len(self.log) > LOG_KEEP:
            del self.log[:len(self.log) - LOG_KEEP]
        self.hub.cond.notify_all()
        return ev

    def save(self):
        if self.deleted:
            return
        path = os.path.join(self.hub.data, self.id + '.json')
        tmp = path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(self.stored(), f)
        os.replace(tmp, path)


def public_turn(t):
    return {k: v for k, v in t.items() if k != 'items'}


def replay_item(item):
    """A finished output item as the next request's input: only what the server reads, no ids or statuses."""
    kind = item.get('type')
    if kind == 'reasoning':
        text = ''.join(p.get('text', '') for p in item.get('summary') or [])
        return {'type': 'reasoning', 'summary': [{'type': 'summary_text', 'text': text}]} if text else None
    if kind == 'message':
        text = ''.join(p.get('text', '') for p in item.get('content') or [] if isinstance(p, dict))
        return {'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': text}]} if text else None
    if kind == 'function_call':
        return {'type': 'function_call', 'call_id': item['call_id'], 'name': item['name'],
                'arguments': item.get('arguments') or '{}'}
    return None


def close_items(items):
    """Make a turn's items replayable after it ended early. A round's tool outputs are added together once all its
    tools ran, so calls without outputs can only be the last round's: they get one each, at the end."""
    answered = {i['call_id'] for i in items if i.get('type') == 'function_call_output'}
    out = list(items)
    out += [{'type': 'function_call_output', 'call_id': i['call_id'], 'output': 'Stopped by the user.'}
            for i in items if i.get('type') == 'function_call' and i['call_id'] not in answered]
    while out and out[-1].get('type') == 'reasoning':  # a thought with nothing after it would join the next turn
        out.pop()
    return out


class Task:
    """A running turn: the thread, and what Stop needs to reach."""

    def __init__(self):
        self.cancel = threading.Event()
        self.sock = None
        self.lock = threading.Lock()

    def stop(self):
        self.cancel.set()
        with self.lock:
            sock = self.sock
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)  # the engine stops at its next write
            except OSError:
                pass


# -------------------------------------------------------------------------------------------------------------- hub
class Hub:
    def __init__(self, data, upstream, workdir, cluster=None):
        self.data = data
        os.makedirs(data, exist_ok=True)
        self.upstream = upstream.rstrip('/')
        u = urllib.parse.urlsplit(self.upstream)
        self.up_host, self.up_port = u.hostname, u.port or 80
        self.workdir = workdir
        self.cluster = cluster or os.path.join(HERE, 'cluster.sh')
        self.cond = threading.Condition(threading.RLock())
        self.sessions = {}
        self.model = {'id': None, 'state': 'unknown', 'checked': 0}
        self.switch = {'state': 'idle', 'recipe': None, 'phase': '', 'error': '', 'since': 0, 'log': ''}
        self.status_version = 0
        self.list_version = 0
        self.model_lock = threading.Lock()   # one model request at a time from this server (the engine is serial)
        self.load()

    def load(self):
        for name in sorted(os.listdir(self.data)):
            if not name.endswith('.json'):
                continue
            try:
                with open(os.path.join(self.data, name)) as f:
                    s = Session(self, json.load(f))
            except (OSError, ValueError, KeyError) as e:
                print(f'skipping {name}: {e}', file=sys.stderr)
                continue
            for t in s.turns:  # this server stopped during the turn
                if t.get('status') == 'running':
                    t['status'] = 'error'
                    t['error'] = {'kind': 'interrupted', 'message': 'The chat server restarted during this reply.'}
                    t['items'] = close_items(t.get('items', []))
                    for st in t.get('steps', []):
                        st['done'] = True
                        if st.get('kind') == 'tool' and st.get('state') == 'running':
                            st['state'] = 'stopped'
            self.sessions[s.id] = s

    # Called with cond held.
    def bump_status(self, lists=False):
        self.status_version += 1
        if lists:
            self.list_version += 1
        self.cond.notify_all()

    def status(self):
        with self.cond:
            m = model_info(self.model['id'])
            return {
                'model': {'id': self.model['id'], 'state': self.model['state'],
                          'name': m['name'] if m else self.model['id'], 'context': m['context'] if m else None},
                'switch': {k: self.switch[k] for k in ('state', 'recipe', 'phase', 'error', 'since')},
                'running': [s.id for s in self.sessions.values() if s.task is not None],
                'list_version': self.list_version,
            }

    # ---------------------------------------------------------------- model server
    def check_model(self, timeout=5):
        """Ask the engine what it serves. It answers between requests only, so a timeout means busy, not down."""
        try:
            with urllib.request.urlopen(self.upstream + '/v1/models', timeout=timeout) as r:
                j = json.load(r)
            mid = j['data'][0]['id']
            state = 'ready'
        except (urllib.error.URLError, ConnectionError) as e:
            reason = getattr(e, 'reason', e)
            if isinstance(reason, (socket.timeout, TimeoutError)):
                return
            mid, state = None, 'down'
        except (socket.timeout, TimeoutError):
            return
        except (ValueError, KeyError, IndexError):
            mid, state = None, 'down'
        with self.cond:
            if self.switch['state'] == 'switching':
                return
            if (mid, state) != (self.model['id'], self.model['state']):
                self.model.update(id=mid, state=state)
                self.bump_status()
            self.model['checked'] = time.time()

    def watch_model(self):
        while True:
            with self.cond:
                busy = self.switch['state'] == 'switching' or any(s.task for s in self.sessions.values())
            if not busy:
                self.check_model()
            time.sleep(30 if self.model['state'] == 'ready' else 5)   # notice soon when a stopped model is back

    # ---------------------------------------------------------------- switching
    PHASES = [
        (re.compile(r'^stopped '), lambda m: 'Stopping the current model'),
        (re.compile(r'^waiting for the leases'), lambda m: 'Waiting for the Macs'),
        (re.compile(r'^waiting for (.+)$'), lambda m: f'Waiting for {m.group(1)}'),
        (re.compile(r'^preflight \S+ on '), lambda m: 'Checking the Macs'),
        (re.compile(r'^start '), lambda m: 'Loading the model'),
        (re.compile(r'^driver on \S+ ready'), lambda m: 'Checking the model'),
    ]

    def start_switch(self, recipe):
        target = next((m for m in MODELS if m['recipe'] == recipe), None)
        if target is None:
            return 400, {'error': f'Unknown model {recipe!r}.'}
        with self.cond:
            if self.switch['state'] == 'switching':
                return 409, {'error': 'A model is already starting.'}
            if any(s.task for s in self.sessions.values()):
                return 409, {'error': 'Wait for the reply to finish, or stop it, then switch.'}
            self.switch.update(state='switching', recipe=recipe, phase='Starting', error='', since=time.time(), log='')
            self.model.update(id=target['id'], state='switching')
            self.bump_status()
        threading.Thread(target=self.run_switch, args=(target,), daemon=True).start()
        return 202, self.status()

    def run_switch(self, target):
        try:
            p = subprocess.Popen([self.cluster, 'up', target['recipe']], stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, stdin=subprocess.DEVNULL)
            for line in p.stdout:
                line = line.rstrip()
                with self.cond:
                    self.switch['log'] = (self.switch['log'] + line + '\n')[-4000:]
                    for rx, say in self.PHASES:
                        m = rx.match(line)
                        if m:
                            if say(m) != self.switch['phase']:
                                self.switch['phase'] = say(m)
                                self.bump_status()
                            break
            code = p.wait()
            if code != 0:
                lines = [l for l in self.switch['log'].strip().split('\n') if l.strip()]
                raise RuntimeError(' '.join(lines[-2:]) or f'cluster.sh exited with {code}')
            # The launcher returns once the server answers; make sure it serves the model asked for.
            mid = None
            for _ in range(30):
                try:
                    with urllib.request.urlopen(self.upstream + '/v1/models', timeout=5) as r:
                        mid = json.load(r)['data'][0]['id']
                    break
                except Exception:
                    time.sleep(2)
            if mid != target['id']:
                raise RuntimeError(f'The server came up with {mid}, not {target["id"]}.')
            with self.cond:
                self.switch.update(state='idle', phase='', error='')
                self.model.update(id=mid, state='ready', checked=time.time())
                self.bump_status()
        except Exception as e:
            with self.cond:
                self.switch.update(state='error', phase='', error=f'Could not start {target["name"]}: {e}')
                self.model.update(id=None, state='unknown')
                self.bump_status()
            threading.Thread(target=self.check_model, daemon=True).start()

    # ---------------------------------------------------------------- sessions
    def create(self):
        with self.cond:
            s = Session(self, {'id': uuid.uuid4().hex[:12]})
            self.sessions[s.id] = s
            s.save()
            self.bump_status(lists=True)
            return s

    def get(self, sid):
        with self.cond:
            s = self.sessions.get(sid)
            return s if s and not s.deleted else None

    def rename(self, s, title):
        with self.cond:
            s.title = title.strip()[:120]
            s.emit('session', title=s.title, updated=s.updated)
            s.save()
            self.bump_status(lists=True)

    def delete(self, s):
        with self.cond:
            task = s.task
        if task:
            task.stop()
            for _ in range(100):
                if s.task is None:
                    break
                time.sleep(0.05)
        with self.cond:
            s.deleted = True
            self.sessions.pop(s.id, None)
            try:
                os.remove(os.path.join(self.data, s.id + '.json'))
            except FileNotFoundError:
                pass
            s.emit('deleted')
            self.bump_status(lists=True)

    def send(self, s, text, thinking):
        text = text.strip()
        if not text:
            return 400, {'error': 'Type a message first.'}
        if thinking not in LEVELS:
            thinking = 'medium'
        with self.cond:
            if s.task is not None:
                return 409, {'error': 'A reply is still running. Stop it first.'}
            if self.switch['state'] == 'switching':
                return 409, {'error': 'A model is starting. Send again when it is ready.'}
            turn = {'id': uuid.uuid4().hex[:12], 'user': text, 'thinking': thinking, 'model': self.model['id'],
                    'started': now_ms(), 'ended': None, 'status': 'running', 'error': None, 'waiting': False,
                    'steps': [], 'usage': None,
                    'items': [{'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': text}]}]}
            s.turns.append(turn)
            if not s.title:
                s.title = ' '.join(text.split())[:80]
                s.emit('session', title=s.title, updated=s.updated)
            s.updated = now_ms()
            s.task = Task()
            s.emit('turn', turn=public_turn(turn))
            s.save()
            self.bump_status(lists=True)
        threading.Thread(target=self.run_turn, args=(s, turn, s.task), daemon=True).start()
        return 202, {'turn': turn['id']}

    def stop(self, s):
        with self.cond:
            task = s.task
        if task:
            task.stop()
        return task is not None

    # ---------------------------------------------------------------- the agent loop
    def history(self, s, turn):
        items = []
        for t in s.turns:
            if t is turn:
                items.extend(t['items'])
                break
            # A turn that failed before the model said anything is left out: the page offers to send it again.
            if any(i.get('role') != 'user' for i in t.get('items', [])):
                items.extend(t['items'])
        return items

    def run_turn(self, s, turn, task):
        try:
            for n in range(MAX_ROUNDS):
                if task.cancel.is_set():
                    raise Stopped()
                # The last round may not call tools, so a turn always ends with an answer.
                calls = self.round(s, turn, task, last=n == MAX_ROUNDS - 1)
                if not calls:
                    break
                self.run_calls(s, turn, task, calls)
            status, error = 'done', turn.get('error')
        except Stopped:
            status, error = 'stopped', None
        except TurnError as e:
            status, error = 'error', {'kind': e.kind, 'message': str(e)}
        except Exception as e:  # a bug here must still end the turn visibly
            import traceback
            traceback.print_exc()
            status, error = 'error', {'kind': 'internal', 'message': f'The chat server failed: {e}'}
        with self.cond:
            if task.cancel.is_set() and status == 'done':
                status = 'stopped'
            turn['status'], turn['error'], turn['ended'], turn['waiting'] = status, error, now_ms(), False
            turn['items'] = close_items(turn['items'])
            for i, st in enumerate(turn['steps']):
                if not st.get('done') or (st.get('kind') == 'tool' and st.get('state') in ('running', 'pending')):
                    st['done'] = True
                    if st.get('kind') == 'tool' and st.get('state') in ('running', 'pending'):
                        st['state'] = 'stopped'
            s.task = None
            s.updated = now_ms()
            s.emit('turn', turn=public_turn(turn))
            s.save()
            self.bump_status(lists=True)

    def request_body(self, s, turn, last):
        # No "model": the engine serves one, and names it in its first event (a switch made elsewhere is no error).
        m = model_info(self.model['id'])
        effort = turn['thinking']
        return {'instructions': INSTRUCTIONS, 'input': self.history(s, turn),
                'tools': TOOLS, 'tool_choice': 'none' if last else 'auto', 'stream': True, 'store': False,
                'reasoning': {'effort': 'none'} if effort == 'off' else {'effort': effort, 'summary': 'auto'},
                'max_output_tokens': min(16384, m['context'] // 4) if m else 8192}

    def round(self, s, turn, task, last):
        """One streamed POST /v1/responses. Returns the function calls the model made."""
        body = json.dumps(self.request_body(s, turn, last)).encode()
        # This server sends one model request at a time; another conversation's round waits here.
        if not self.model_lock.acquire(blocking=False):
            with self.cond:
                turn['waiting'] = True
                s.emit('turn', turn=public_turn(turn))
            while not self.model_lock.acquire(timeout=0.25):
                if task.cancel.is_set():
                    raise Stopped()
            with self.cond:
                turn['waiting'] = False
                s.emit('turn', turn=public_turn(turn))
        try:
            return self.stream_round(s, turn, task, body)
        finally:
            self.model_lock.release()

    def stream_round(self, s, turn, task, body):
        conn = http.client.HTTPConnection(self.up_host, self.up_port, timeout=10)
        try:
            try:
                conn.connect()
            except OSError:
                raise TurnError('down', 'The model server is not running. Pick a model to start it.')
            conn.sock.settimeout(3600)   # a long prompt prefills for minutes before the first token
            with task.lock:
                task.sock = conn.sock   # http.client lets go of conn.sock once the response is read to its close
            if task.cancel.is_set():
                raise Stopped()
            try:
                t0 = time.monotonic()
                conn.request('POST', '/v1/responses', body, {'Content-Type': 'application/json'})
                r = conn.getresponse()
            except OSError:
                if task.cancel.is_set():
                    raise Stopped()
                raise TurnError('down', 'The model server closed the connection.')
            if r.status != 200:
                raw = r.read().decode('utf-8', 'replace')
                try:
                    msg = json.loads(raw)['error']['message']
                except (ValueError, KeyError, TypeError):
                    msg = raw.strip() or f'HTTP {r.status}'
                m = re.search(r'prompt has (\d+) tokens; context is (\d+)', msg)
                if m:
                    raise TurnError('context', self.full_message(turn, int(m.group(2))))
                raise TurnError('upstream', f'The model server refused the request: {msg}')
            return self.read_events(s, turn, task, r, t0)
        finally:
            with task.lock:
                task.sock = None
            conn.close()

    def full_message(self, turn, ctx=None):
        mid = turn['model'] or self.model['id']
        m = model_info(mid)
        ctx = ctx or (m['context'] if m else 0)
        name = m['name'] if m else (mid or 'this model')
        return f'This conversation is too long for {name} ({ctx:,} tokens). Start a new conversation.'

    def read_events(self, s, turn, task, r, t0):
        keys = {}          # (item id, kind) -> step index
        final, failed = None, None
        data = []
        # Speed: the first streamed token ends the prefill; tokens after it over the time they took is the decode
        # rate. Streamed text deltas stand in for tokens while the round runs; the engine's usage replaces them at the end.
        clock = {'first': None, 'n': 0, 'shown': 0.0}

        def live_stats():
            now = time.monotonic()
            # A rate from the first few tokens is noise (two can arrive together): wait for 5 and half a second.
            if clock['first'] is None or clock['n'] < 5 or now - clock['first'] < 0.5 or now - clock['shown'] < 1.0:
                return
            clock['shown'] = now
            s.emit('stats', turn=turn['id'], stats=turn.get('stats'),
                   live={'tps': round((clock['n'] - 1) / max(now - clock['first'], 1e-3), 1), 'output': clock['n']})

        def step(key, kind):
            i = keys.get(key)
            if i is None:
                with self.cond:
                    turn['steps'].append({'kind': kind, 'text': '', 'done': False, 'round': turn.get('rounds', 0)})
                    i = keys[key] = len(turn['steps']) - 1
                    s.emit('step', turn=turn['id'], index=i, step=turn['steps'][i])
            return i

        def delta(i, text):
            if text:
                with self.cond:
                    if clock['first'] is None:
                        clock['first'] = time.monotonic()
                    clock['n'] += 1
                    turn['steps'][i]['text'] += text
                    s.emit('delta', turn=turn['id'], index=i, text=text)
                    live_stats()

        def finish(i):
            with self.cond:
                st = turn['steps'][i]
                if not st['done']:
                    st['done'] = True
                    s.emit('step', turn=turn['id'], index=i, step=st)

        try:
            while True:
                line = r.readline()
                if not line:
                    break
                line = line.decode('utf-8', 'replace').rstrip('\r\n')
                if line.startswith('data:'):
                    data.append(line[5:].strip())
                    continue
                if line or not data:
                    continue
                raw, data = ''.join(data), []
                try:
                    ev = json.loads(raw)
                except ValueError:
                    continue
                t = ev.get('type', '')
                item = ev.get('item') or {}
                if t == 'response.created':
                    self.served((ev.get('response') or {}).get('model'), s, turn)
                elif t == 'response.reasoning_summary_text.delta':
                    delta(step((ev.get('item_id'), 'r'), 'thinking'), ev.get('delta', ''))
                elif t == 'response.output_text.delta':
                    delta(step((ev.get('item_id'), 't'), 'text'), ev.get('delta', ''))
                elif t == 'response.output_item.done' and item.get('type') == 'reasoning':
                    if (item.get('id'), 'r') in keys:
                        finish(keys[(item.get('id'), 'r')])
                elif t == 'response.output_item.done' and item.get('type') == 'message':
                    if (item.get('id'), 't') in keys:
                        finish(keys[(item.get('id'), 't')])
                elif t == 'response.output_item.done' and item.get('type') == 'function_call':
                    try:
                        args = json.loads(item.get('arguments') or '{}')
                    except ValueError:
                        args = {}
                    with self.cond:
                        turn['steps'].append({'kind': 'tool', 'call_id': item.get('call_id'), 'name': item.get('name'),
                                              'arguments': args, 'state': 'pending', 'done': False, 'output': '',
                                              'result': None, 'round': turn.get('rounds', 0)})
                        i = len(turn['steps']) - 1
                        s.emit('step', turn=turn['id'], index=i, step=turn['steps'][i])
                elif t == 'response.function_call_arguments.delta':
                    with self.cond:
                        if clock['first'] is None:
                            clock['first'] = time.monotonic()
                        clock['n'] += 1
                        live_stats()
                elif t in ('response.completed', 'response.incomplete'):
                    final = ev.get('response') or {}
                    t_end = time.monotonic()
                elif t == 'response.failed':
                    failed = ((ev.get('response') or {}).get('error') or {}).get('message') or 'generation failed'
        except OSError:
            pass   # Stop shut the socket, or the engine went away: told apart below
        if task.cancel.is_set():
            self.keep_partial(turn)
            raise Stopped()
        for i in keys.values():
            finish(i)
        if failed:
            raise TurnError('upstream', f'The model failed: {failed}')
        if final is None:
            self.keep_partial(turn)
            raise TurnError('upstream', 'The model server stopped in the middle of the answer.')

        out = [x for x in (replay_item(o) for o in final.get('output') or []) if x]
        calls = [o for o in out if o['type'] == 'function_call']
        u = final.get('usage') or {}
        used = (u.get('input_tokens') or 0) + (u.get('output_tokens') or 0)
        with self.cond:
            turn['items'].extend(out)
            turn['rounds'] = turn.get('rounds', 0) + 1
            turn['usage'] = {'input': u.get('input_tokens', 0), 'output': u.get('output_tokens', 0),
                             'cached': (u.get('input_tokens_details') or {}).get('cached_tokens', 0)}
            # The reply's totals over all its rounds; each round sends the whole conversation, so 'input' counts
            # every prompt token processed and 'cached' the ones the engine reused.
            st = turn.get('stats') or {'input': 0, 'output': 0, 'cached': 0, 'gen_tokens': 0, 'gen_s': 0.0,
                                       'prefill_s': 0.0, 'rounds': 0}
            st['input'] += turn['usage']['input']
            st['output'] += turn['usage']['output']
            st['cached'] += turn['usage']['cached']
            st['rounds'] += 1
            if clock['first'] is not None and turn['usage']['output'] > 1:
                st['gen_tokens'] += turn['usage']['output'] - 1
                st['gen_s'] += t_end - clock['first']
                st['prefill_s'] += clock['first'] - t0
            st['tps'] = round(st['gen_tokens'] / st['gen_s'], 1) if st['gen_s'] > 0 else None
            turn['stats'] = st
            # Text written in a round that goes on to call tools is narration of the work, not the answer.
            if calls:
                for st in turn['steps']:
                    if st['kind'] == 'text':
                        st['kind'] = 'note'
            s.context_tokens, s.context_model = used, turn['model']
            s.emit('turn', turn=public_turn(turn))
            s.emit('session', context_tokens=used, context_model=turn['model'])
            s.save()
        if final.get('status') == 'incomplete':
            m = model_info(turn['model'])
            if m and used >= m['context'] - 64:
                raise TurnError('context', self.full_message(turn))
            answered = any(st['kind'] == 'text' and st['text'].strip() for st in turn['steps'])
            with self.cond:
                turn['error'] = {'kind': 'length', 'message': 'The answer reached the length limit and was cut off.'
                                 if answered else 'The model reached its output limit before answering.'}
            return []
        return calls

    def served(self, mid, s, turn):
        if not mid:
            return
        with self.cond:
            if turn['model'] != mid:
                turn['model'] = mid
                s.emit('turn', turn=public_turn(turn))
            if (self.model['id'], self.model['state']) != (mid, 'ready') and self.switch['state'] != 'switching':
                self.model.update(id=mid, state='ready')
                self.bump_status()

    def keep_partial(self, turn):
        """After a stopped round: what the model had written so far goes into the history."""
        r = turn.get('rounds', 0)
        text = ''.join(st['text'] for st in turn['steps'] if st['kind'] == 'text' and st.get('round') == r)
        if text.strip():
            with self.cond:
                turn['items'].append({'type': 'message', 'role': 'assistant',
                                      'content': [{'type': 'output_text', 'text': text}]})

    def run_calls(self, s, turn, task, calls):
        """Run one round's tool calls side by side; their outputs go back in the order they were called."""
        results = {}
        index = {st['call_id']: i for i, st in enumerate(turn['steps']) if st.get('kind') == 'tool'}

        def one(call):
            try:
                args = json.loads(call['arguments'] or '{}')
                if not isinstance(args, dict):
                    raise ValueError
            except ValueError:
                args = None
            if args is None:
                text, result, err = 'error: the arguments are not a JSON object', None, True
            else:
                text, result, err = run_tool(call['name'], args, self.workdir, task.cancel)
            results[call['call_id']] = text
            with self.cond:
                i = index.get(call['call_id'])
                if i is not None and not task.cancel.is_set():
                    st = turn['steps'][i]
                    st.update(state='error' if err else 'done', output=text, result=result, done=True)
                    s.emit('step', turn=turn['id'], index=i, step=st)

        with self.cond:
            for c in calls:
                i = index.get(c['call_id'])
                if i is not None:
                    turn['steps'][i]['state'] = 'running'
                    s.emit('step', turn=turn['id'], index=i, step=turn['steps'][i])
        threads = [threading.Thread(target=one, args=(c,), daemon=True) for c in calls]
        for th in threads:
            th.start()
        for th in threads:
            while th.is_alive():
                th.join(0.2)
                if task.cancel.is_set():
                    break
            if task.cancel.is_set():
                break
        if task.cancel.is_set():
            raise Stopped()
        with self.cond:
            for c in calls:
                turn['items'].append({'type': 'function_call_output', 'call_id': c['call_id'],
                                      'output': results.get(c['call_id'], 'error: no result')})
            s.save()


# -------------------------------------------------------------------------------------------------------------- http
class Handler(http.server.SimpleHTTPRequestHandler):
    hub = None
    server_version = 'trashcan-webchat'

    def __init__(self, *a, **k):
        super().__init__(*a, directory=STATIC, **k)

    def end_headers(self):
        # Deploys change these files: browsers revalidate each load (a cheap 304) instead of keeping a stale copy.
        if not self.path.startswith('/api/'):
            self.send_header('Cache-Control', 'no-cache')
        super().end_headers()

    def log_message(self, fmt, *args):
        sys.stderr.write('%s %s\n' % (self.log_date_time_string(), fmt % args))

    def send_json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def body(self):
        n = int(self.headers.get('Content-Length') or 0)
        if n > 1_000_000:
            raise ValueError('too large')
        raw = self.rfile.read(n) if n else b''
        return json.loads(raw) if raw.strip() else {}

    def route(self):
        path = urllib.parse.urlsplit(self.path).path
        m = re.fullmatch(r'/api/sessions/([0-9a-f]{12})(/[a-z]+)?', path)
        return path, (m.group(1), m.group(2) or '') if m else (None, None)

    def do_GET(self):
        path, (sid, sub) = self.route()
        hub = self.hub
        if path == '/api/events':
            return self.events()
        if path == '/api/state':
            return self.send_json(200, {'models': MODELS, 'levels': LEVELS, 'status': hub.status()})
        if path == '/api/sessions':
            with hub.cond:
                rows = sorted((s.summary() for s in hub.sessions.values()), key=lambda r: -r['updated'])
            return self.send_json(200, {'sessions': rows, 'list_version': hub.list_version})
        if sid and not sub:
            s = hub.get(sid)
            if not s:
                return self.send_json(404, {'error': 'No such conversation.'})
            with hub.cond:
                return self.send_json(200, s.public())
        if path.startswith('/v1/'):
            return self.proxy('GET')
        if path.startswith('/api/'):
            return self.send_json(404, {'error': 'unknown endpoint'})
        if path in ('/', '/index.html'):
            self.path = '/index.html'
        return super().do_GET()

    def do_HEAD(self):
        if self.path.startswith('/api/'):
            return self.send_json(405, {'error': 'method not allowed'})
        if urllib.parse.urlsplit(self.path).path == '/':
            self.path = '/index.html'
        return super().do_HEAD()

    def do_POST(self):
        path, (sid, sub) = self.route()
        hub = self.hub
        if path.startswith('/v1/'):
            return self.proxy('POST')
        try:
            req = self.body()
            if not isinstance(req, dict):
                raise ValueError
        except ValueError:
            return self.send_json(400, {'error': 'invalid JSON'})
        if path == '/api/sessions':
            s = hub.create()
            out = {'session': s.public()}
            if str(req.get('text') or '').strip():
                code, res = hub.send(s, str(req['text']), str(req.get('thinking') or 'medium'))
                out.update(res)
                with hub.cond:
                    out['session'] = s.public()
            return self.send_json(201, out)
        if path == '/api/switch':
            code, out = hub.start_switch(str(req.get('recipe', '')))
            return self.send_json(code, out)
        if sid:
            s = hub.get(sid)
            if not s:
                return self.send_json(404, {'error': 'No such conversation.'})
            if sub == '/messages':
                code, out = hub.send(s, str(req.get('text') or ''), str(req.get('thinking') or 'medium'))
                return self.send_json(code, out)
            if sub == '/stop':
                return self.send_json(200, {'stopped': hub.stop(s)})
            if sub == '/rename':
                hub.rename(s, str(req.get('title') or ''))
                return self.send_json(200, {'title': s.title})
        return self.send_json(404, {'error': 'unknown endpoint'})

    def do_PATCH(self):
        path, (sid, sub) = self.route()
        s = self.hub.get(sid) if sid and not sub else None
        if not s:
            return self.send_json(404, {'error': 'No such conversation.'})
        try:
            req = self.body()
        except ValueError:
            return self.send_json(400, {'error': 'invalid JSON'})
        self.hub.rename(s, str(req.get('title') or ''))
        return self.send_json(200, {'title': s.title})

    def do_DELETE(self):
        path, (sid, sub) = self.route()
        s = self.hub.get(sid) if sid and not sub else None
        if not s:
            return self.send_json(404, {'error': 'No such conversation.'})
        self.hub.delete(s)
        return self.send_json(200, {'deleted': sid})

    # One stream per page: the server's status, and the open conversation's events after the given sequence number
    # (or a snapshot of the whole conversation when the page has none, or asks for more than is kept).
    def events(self):
        hub = self.hub
        q = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
        sid = (q.get('session') or [''])[0]
        try:
            after = int(self.headers.get('Last-Event-ID') or (q.get('after') or ['-1'])[0])
        except ValueError:
            after = -1
        s = hub.get(sid) if sid else None
        if sid and not s:
            return self.send_json(404, {'error': 'No such conversation.'})
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Accel-Buffering', 'no')
        self.end_headers()

        def out(kind, data, seq=None):
            msg = (f'id: {seq}\n' if seq is not None else '') + f'event: {kind}\ndata: {json.dumps(data)}\n\n'
            return msg.encode()

        status_seen, wrote = -1, time.time()
        try:
            self.wfile.write(b'retry: 1500\n\n')
            while True:
                chunks = []
                with hub.cond:
                    if s is not None:
                        if s.deleted:
                            chunks.append(out('deleted', {'session': s.id}))
                        else:
                            base = s.log[0][0] - 1 if s.log else s.seq
                            if after <= 0 or after < base or after > s.seq:
                                chunks.append(out('snapshot', s.public(), s.seq))
                                after = s.seq
                            elif after < s.seq:
                                for seq, ev in s.log:
                                    if seq > after:
                                        chunks.append(out(ev['type'], ev, seq))
                                after = s.seq
                    if hub.status_version != status_seen:
                        status_seen = hub.status_version
                        chunks.append(out('status', hub.status()))
                    if not chunks:
                        hub.cond.wait(PING)
                        if time.time() - wrote >= PING:
                            chunks.append(out('ping', {'t': int(time.time())}))
                if chunks:
                    self.wfile.write(b''.join(chunks))
                    self.wfile.flush()
                    wrote = time.time()
                    if s is not None and s.deleted:
                        return
        except (BrokenPipeError, ConnectionResetError, OSError):
            return

    def proxy(self, method):
        """/v1/* passed through to the engine, for API clients that know this address."""
        n = int(self.headers.get('Content-Length') or 0)
        body = self.rfile.read(n) if n else None
        req = urllib.request.Request(self.hub.upstream + self.path, data=body, method=method,
                                     headers={'Content-Type': self.headers.get('Content-Type', 'application/json')})
        try:
            r = urllib.request.urlopen(req, timeout=3600)
        except urllib.error.HTTPError as e:
            r = e
        except OSError as e:
            return self.send_json(502, {'error': {'message': f'no model server: {e}'}})
        self.send_response(r.status if hasattr(r, 'status') else r.code)
        self.send_header('Content-Type', r.headers.get('Content-Type', 'application/json'))
        self.end_headers()
        try:
            while True:
                chunk = r.read1(65536) if hasattr(r, 'read1') else r.read(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            r.close()


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def serve(hub, host, port):
    handler = type('BoundHandler', (Handler,), {'hub': hub})
    return Server((host, port), handler)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8090)
    ap.add_argument('--upstream', default='http://127.0.0.1:8000')
    ap.add_argument('--data', default=os.path.join(HERE, 'sessions'))
    ap.add_argument('--workdir', default='~', help='where bash commands run')
    ap.add_argument('--cluster', default=os.path.join(HERE, 'cluster.sh'), help='the model switcher (tests use a fake)')
    a = ap.parse_args()
    hub = Hub(os.path.abspath(os.path.expanduser(a.data)), a.upstream,
              os.path.abspath(os.path.expanduser(a.workdir)), a.cluster)
    threading.Thread(target=hub.watch_model, daemon=True).start()
    httpd = serve(hub, a.host, a.port)
    print(f'webchat on http://{a.host}:{a.port}/ -> {hub.upstream}, conversations in {hub.data}', flush=True)
    httpd.serve_forever()


if __name__ == '__main__':
    main()
