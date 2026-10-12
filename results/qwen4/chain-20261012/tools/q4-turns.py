#!/usr/bin/env python3
"""Two chat turns against a Responses server: the second must reuse the first's state (cached_tokens). Prints each
turn's time to first token, decode rate and text."""
import argparse, http.client, json, time
from urllib.parse import urlsplit

p = argparse.ArgumentParser()
p.add_argument('--url', default='http://127.0.0.1:8011')
a = p.parse_args()
u = urlsplit(a.url)
conn = http.client.HTTPConnection(u.hostname, u.port, timeout=600)


def stream(body):
    t0 = time.time()
    conn.request('POST', '/v1/responses', json.dumps({**body, 'stream': True}).encode(), {'Content-Type': 'application/json'})
    r = conn.getresponse()
    first, deltas, final, buf = None, 0, None, b''
    while True:
        line = r.readline()
        if not line:
            break
        if line.startswith(b'data: '):
            v = json.loads(line[6:])
            if v['type'] == 'response.output_text.delta':
                deltas += 1
                first = first or time.time()
            if v['type'] in ('response.completed', 'response.incomplete'):
                final = v['response']
    t1 = time.time()
    return final, (first or t1) - t0, t1 - (first or t1)


model = json.loads((conn.request('GET', '/v1/models'), conn.getresponse().read())[1])['data'][0]['id']
history = [{'role': 'user', 'content': 'Name three rivers in Europe, one line each.'}]
first, ttft, dec = stream({'model': model, 'input': history, 'temperature': 0, 'max_output_tokens': 64})
text = ''.join(c['text'] for i in first['output'] if i['type'] == 'message' for c in i['content'])
u1 = first['usage']
print(f"turn 1: {u1['input_tokens']} prompt tokens, cached {u1['input_tokens_details']['cached_tokens']}, "
      f"ttft {ttft:.2f} s, {u1['output_tokens']} tokens in {dec:.2f} s; {text!r}", flush=True)
history += first['output'] + [{'role': 'user', 'content': 'Which of them is the longest?'}]
second, ttft, dec = stream({'model': model, 'input': history, 'temperature': 0, 'max_output_tokens': 64})
text = ''.join(c['text'] for i in second['output'] if i['type'] == 'message' for c in i['content'])
u2 = second['usage']
print(f"turn 2: {u2['input_tokens']} prompt tokens, cached {u2['input_tokens_details']['cached_tokens']}, "
      f"ttft {ttft:.2f} s, {u2['output_tokens']} tokens in {dec:.2f} s; {text!r}", flush=True)
assert u2['input_tokens_details']['cached_tokens'] > u1['input_tokens'], 'the second turn did not reuse the first'
print('PASS: the second turn reused the first turn\'s state', flush=True)
