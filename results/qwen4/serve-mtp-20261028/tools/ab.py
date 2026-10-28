# Decode speed of one streamed Responses request at a given temperature (omit for the server default).
import json, sys, time, urllib.request
temp = None if sys.argv[1] == 'default' else float(sys.argv[1])
prompt = sys.argv[2] if len(sys.argv) > 2 else 'Explain how a Thunderbolt 2 cable link works, in about 300 words.'
effort = sys.argv[3] if len(sys.argv) > 3 else 'medium'
body = {'input': prompt, 'stream': True, 'store': False, 'max_output_tokens': 1500,
        'reasoning': {'effort': 'none'} if effort == 'none' else {'effort': effort, 'summary': 'auto'}}
if temp is not None:
    body['temperature'] = temp
req = urllib.request.Request('http://127.0.0.1:8000/v1/responses', data=json.dumps(body).encode(),
                             headers={'content-type': 'application/json'})
t0 = time.time(); first = None; usage = None
with urllib.request.urlopen(req, timeout=300) as r:
    for line in r:
        line = line.decode().strip()
        if not line.startswith('data:'):
            continue
        ev = json.loads(line[5:])
        if ev.get('type', '').endswith('.delta') and first is None:
            first = time.time()
        if ev.get('type') == 'response.completed':
            usage = ev['response'].get('usage')
end = time.time()
out = usage['output_tokens']
print(json.dumps(usage)); print(f"temperature={sys.argv[1]}: {out} tokens, prefill {first - t0:.2f} s, {(out - 1) / (end - first):.1f} tok/s")
