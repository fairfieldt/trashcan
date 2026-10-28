# Ask the demo chat on incite one question, as the page does, and report the turn's timing and stats.
import json, sys, time, urllib.request
BASE = 'http://127.0.0.1:8080'
def call(method, path, body=None):
    req = urllib.request.Request(BASE + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={'content-type': 'application/json'})
    return json.load(urllib.request.urlopen(req, timeout=30))
t0 = time.time()
if sys.argv[1] == '--follow':
    sid = sys.argv[2]
else:
    s = call('POST', '/api/sessions', {'text': sys.argv[1], 'thinking': 'medium'})
    sid = s.get('id') or (s.get('session') or {}).get('id')
    if not sid:
        sys.exit(f'no session id in {str(s)[:300]}')
while True:
    time.sleep(2)
    d = call('GET', f'/api/sessions/{sid}')
    t = d['turns'][-1]
    if t['status'] != 'running':
        break
st = t.get('stats') or {}
tools = [f"{x['name']}" for x in t['steps'] if x['kind'] == 'tool']
ans = ''.join(x['text'] for x in t['steps'] if x['kind'] == 'text')
print(json.dumps({'session': sid, 'status': t['status'], 'error': t.get('error'), 'wall_s': round(time.time() - t0, 1),
                  'turn_s': round((t['ended'] - t['started']) / 1000, 1), 'rounds': t.get('rounds'), 'tools': tools,
                  'stats': st, 'answer_chars': len(ans), 'answer_head': ans[:160]}, indent=1))
