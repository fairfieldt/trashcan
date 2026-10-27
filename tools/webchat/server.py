#!/usr/bin/env python3
"""The pipe web UI on a trashcan server, without a pi process: serves the UI (--root, a copy of piPE's web/ plus
trashcan-transport.js; deploy.sh lays it out), passes /v1/* through to `trashcan serve` (same origin, so no CORS), and
runs the tools the page's agent calls: web_search, fetch_page, bash, and switches the served model (POST /switch
{recipe}, GET /switch) with cluster.sh next to this file. Python standard library only.

usage: server.py [--root DIR] [--port 8080] [--host 127.0.0.1] [--upstream http://127.0.0.1:8000] [--workdir DIR]
Then open http://localhost:8080 on the machine running it (it redirects to the UI with the transport; loopback only:
bash runs commands on this machine).
"""
import argparse, html.parser, http.server, json, os, re, socketserver, subprocess, sys, threading, time, urllib.parse, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, 'web')
WORKDIR = os.path.expanduser('~')
UA = 'Mozilla/5.0 (X11; Linux x86_64) trashcan-webchat'


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
    return out[:n]


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


def bash(command, timeout=60, limit=16000):
    try:
        p = subprocess.run(['bash', '-lc', command], cwd=WORKDIR, capture_output=True, text=True, timeout=timeout,
                           stdin=subprocess.DEVNULL)
        out, code = (p.stdout + (('\n' if p.stdout and p.stderr else '') + p.stderr)), p.returncode
    except subprocess.TimeoutExpired as e:
        out, code = (e.stdout or '') if isinstance(e.stdout, str) else '', 124
        out += f'\n(stopped after {timeout} s)'
    if len(out) > limit:
        out = out[:limit] + f'\n(output cut at {limit} characters)'
    return {'output': out or '(no output)', 'exitCode': code}


RECIPES = ('6-flashnext', '4-qwen27b')
switch = {'state': 'idle', 'recipe': None, 'log': '', 'since': 0.0}
switch_lock = threading.Lock()


def run_switch(recipe):
    # The log is shared as it grows, so the page can show what the switch is waiting for.
    p = subprocess.Popen([os.path.join(HERE, 'cluster.sh'), 'up', recipe], stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, stdin=subprocess.DEVNULL)
    for line in p.stdout:
        with switch_lock:
            switch['log'] = (switch['log'] + line)[-3000:]
    code = p.wait()
    with switch_lock:
        switch['state'] = 'idle' if code == 0 else 'error'


def start_switch(recipe):
    if recipe not in RECIPES:
        return 400, {'error': f'unknown recipe {recipe!r}'}
    with switch_lock:
        if switch['state'] == 'switching':
            return 409, {'error': f"already switching to {switch['recipe']}"}
        switch.update(state='switching', recipe=recipe, log='', since=time.time())
    threading.Thread(target=run_switch, args=(recipe,), daemon=True).start()
    return 202, dict(switch)


class Handler(http.server.SimpleHTTPRequestHandler):
    upstream = 'http://127.0.0.1:8000'

    def __init__(self, *a, **k):
        super().__init__(*a, directory=ROOT, **k)

    def end_headers(self):
        # The page's files change with deploys: browsers revalidate each load (a cheap 304 when unchanged) instead of
        # keeping a stale stylesheet for days.
        if self.command == 'GET' and not self.path.startswith('/v1/'):
            self.send_header('Cache-Control', 'no-cache')
        super().end_headers()

    def log_message(self, fmt, *args):
        sys.stderr.write('%s %s\n' % (self.log_date_time_string(), fmt % args))

    def send_json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def proxy(self, method):
        n = int(self.headers.get('Content-Length') or 0)
        body = self.rfile.read(n) if n else None
        req = urllib.request.Request(self.upstream + self.path, data=body, method=method,
                                     headers={'Content-Type': self.headers.get('Content-Type', 'application/json')})
        try:
            r = urllib.request.urlopen(req, timeout=3600)
        except urllib.error.HTTPError as e:
            r = e
        except OSError as e:
            return self.send_json(502, {'error': {'message': f'no trashcan server at {self.upstream}: {e}'}})
        self.send_response(r.status if hasattr(r, 'status') else r.code)
        self.send_header('Content-Type', r.headers.get('Content-Type', 'application/json'))
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        try:  # pass SSE through as it arrives
            while True:
                chunk = r.read1(65536) if hasattr(r, 'read1') else r.read(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # the page stopped the request; closing upstream stops generation
        finally:
            r.close()

    def do_GET(self):
        if self.path == '/':  # the chat: the pipe UI on this server's transport, pipe 3.1 look, working sections open
            self.send_response(302)
            self.send_header('Location', '/?transport=/trashcan-transport.js&ui=win31&working=open')
            self.end_headers()
            return
        if self.path.startswith('/v1/'):
            return self.proxy('GET')
        if self.path == '/switch':
            with switch_lock:
                return self.send_json(200, dict(switch))
        return super().do_GET()

    def do_POST(self):
        if self.path.startswith('/v1/'):
            return self.proxy('POST')
        if self.path == '/switch':
            try:
                req = json.loads(self.rfile.read(int(self.headers.get('Content-Length') or 0)) or b'{}')
            except ValueError:
                req = {}
            code, out = start_switch(str(req.get('recipe', '')))
            return self.send_json(code, out)
        if self.path == '/tool':
            try:
                req = json.loads(self.rfile.read(int(self.headers.get('Content-Length') or 0)))
                args = req.get('arguments') or {}
                if req.get('name') == 'web_search':
                    out = {'results': web_search(str(args.get('query', '')), int(args.get('num_results') or 6))}
                elif req.get('name') == 'fetch_page':
                    out = fetch_page(str(args.get('url', '')))
                elif req.get('name') == 'bash':
                    out = bash(str(args.get('command', '')))
                else:
                    out = {'error': f"unknown tool {req.get('name')!r}"}
            except Exception as e:  # the model sees the error and can try something else
                out = {'error': str(e)}
            return self.send_json(200, out)
        self.send_error(404)


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    global ROOT, WORKDIR
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8080)
    ap.add_argument('--upstream', default='http://127.0.0.1:8000')
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--workdir', default=WORKDIR)
    a = ap.parse_args()
    ROOT, WORKDIR = os.path.abspath(a.root), os.path.abspath(os.path.expanduser(a.workdir))
    Handler.upstream = a.upstream.rstrip('/')
    print(f'webchat on http://{a.host}:{a.port}/ -> {Handler.upstream}', flush=True)
    Server((a.host, a.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
