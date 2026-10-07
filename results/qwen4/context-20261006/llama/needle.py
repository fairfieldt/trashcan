"""Needle in a haystack for llama.cpp K/V cache types: three English facts hidden at 10%, 50% and 90% depth of
I promessi sposi, cut to about 100K and 200K tokens; one llama-server per (length, cache type), one question per fact
(the shared prefix is cached, so only the first question pays the prefill). Greedy, thinking disabled."""
import json
import os
import subprocess
import sys
import time
import urllib.request

BIN = '/Users/fairfieldt/dev/trashcan/llama.cpp/build-mac/bin'
MODEL = '/Users/fairfieldt/models/qwen3.8-flash-next/Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf'
HERE = '/Users/fairfieldt/dev/claude-work/trashcan-20261006/qwen-flash/longctx'
PORT = 18091
NEEDLES = [
    (0.10, 'The secret number of the Lecco harbour master is 4718.',
     'What is the secret number of the Lecco harbour master?', '4718'),
    (0.50, "The password to Don Rodrigo's cellar is blue heron 93.",
     "What is the password to Don Rodrigo's cellar?", 'blue heron 93'),
    (0.90, "Renzo's favourite colour, as the archivist recorded it, is vermilion.",
     "What is Renzo's favourite colour, as the archivist recorded it?", 'vermilion'),
]
CHARS_PER_TOKEN = 1329139 / 383792


def haystack(tokens):
    text = open(f'{HERE}/promessi_sposi.txt', encoding='utf-8').read()[:int(tokens * CHARS_PER_TOKEN)]
    out, last = [], 0
    for depth, fact, _, _ in NEEDLES:
        at = text.find('\n\n', int(len(text) * depth))  # a paragraph boundary
        out += [text[last:at], f'\n\n{fact}\n\n']
        last = at
    out.append(text[last:])
    return ''.join(out)


def ask(doc, question):
    content = ('Below is a long Italian novel. A few English sentences are hidden in it.\n\n<document>\n' + doc
               + f'\n</document>\n\nAnswer with the answer only, no explanation. {question}')
    body = {'messages': [{'role': 'user', 'content': content}], 'temperature': 0, 'max_tokens': 32,
            'cache_prompt': True, 'chat_template_kwargs': {'enable_thinking': False}}
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/chat/completions', json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    start = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        res = json.load(r)
    return res['choices'][0]['message']['content'], res.get('usage', {}), time.time() - start


def run(tokens, ctk, ctv, log):
    ctx = tokens + 4096
    srv = subprocess.Popen([f'{BIN}/llama-server', '-m', MODEL, '-c', str(ctx), '-ngl', '99', '-fa', 'on', '-ctk', ctk,
                            '-ctv', ctv, '-np', '1', '-t', '8', '--port', str(PORT), '--host', '127.0.0.1', '--jinja'],
                           stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(600):
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=5) as r:
                    if json.load(r).get('status') == 'ok':
                        break
            except Exception:
                pass
            time.sleep(2)
        doc = haystack(tokens)
        rows = []
        for depth, _, question, answer in NEEDLES:
            text, usage, secs = ask(doc, question)
            ok = answer.lower() in text.lower()
            rows.append({'tokens': tokens, 'k': ctk, 'v': ctv, 'depth': depth, 'answer': text.strip(), 'ok': ok,
                         'prompt_tokens': usage.get('prompt_tokens'), 'seconds': round(secs, 1)})
            print(json.dumps(rows[-1]), flush=True)
        return rows
    finally:
        srv.terminate()
        srv.wait(timeout=120)


if __name__ == '__main__':
    lengths = [int(x) for x in sys.argv[1:]] or [100000, 200000]
    results = []
    with open(f'{HERE}/needle-server.log', 'a') as log:
        for tokens in lengths:
            # f16 K + q8_0 V runs its attention on the CPU in this llama.cpp build (five times slower at 64K), so
            # it is tested at the shorter length only.
            configs = [('f16', 'f16'), ('q8_0', 'q8_0')] + ([('f16', 'q8_0')] if tokens <= 100000 else [])
            for ctk, ctv in configs:
                results += run(tokens, ctk, ctv, log)
    with open(f'{HERE}/needle-results.json', 'w') as f:
        json.dump(results, f, indent=1)
    for r in results:
        print(f"{r['tokens']:>7} K {r['k']:5} V {r['v']:5} depth {r['depth']:.0%}: {'ok  ' if r['ok'] else 'MISS'} "
              f"{r['answer']!r} ({r['prompt_tokens']} prompt tokens, {r['seconds']} s)")
