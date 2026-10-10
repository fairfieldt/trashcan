#!/usr/bin/env python3
"""One row per qwen-bench run in run order (incite's commands.log): the payload the driver logged, prefill and decode
tok/s of each repetition, exchange and GPU wait per generated token, MTP verification per round and tokens per round,
whether the generated tokens equal the first run's of the same command, and the driver's and node's clocks.
usage: runs.py INCITE_DIR INVITE_DIR PREFIX      (the pulled results/ of each Mac)"""
import json, os, re, sys

incite, invite, prefix = sys.argv[1:4]


def rows(path):
    out = []
    for line in open(path):
        line = line.strip()
        if line.startswith('{'):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def clocks(path):
    # with-clocks.sh: "[clocks] N samples card0 (..) sclk/mclk: 850/1370 xA 450/1370 xB ... card1 ..."
    text = open(path).read() if os.path.exists(path) else ''
    cards = re.findall(r'sclk/mclk: (.*?) max', text)
    return '?' if not cards else 'ok' if all('850/1370' in c for c in cards) else 'LOW'


order = [l.split()[1] for l in open(os.path.join(incite, 'commands.log')) if l.split()[1].startswith(prefix) and 'qwen-bench' in l]
first = {}
print('| run | payload | pp512 tok/s | tok/s | exchange ms/token | GPU wait ms/token | verification ms/round | tokens/round | tokens | clocks (incite, invite) |')
print('|---|---|---|---|---|---|---|---|---|---|')
for name in order:
    err = os.path.join(incite, name + '.err')
    payload = re.search(r'(\w+) all-reduce payload', open(err).read())
    bench = [r for r in rows(os.path.join(incite, name + '.json')) if 'tg_tok_s' in r]
    if not bench:
        print(f'| {name} | {payload and payload[1]} | failed: {open(err).read().strip().splitlines()[-1]} |')
        continue
    key = bool(bench[0].get('mtp'))
    same = all(first.setdefault((key, r['tg']), r['token_ids']) == r['token_ids'] for r in bench)
    mtp = [r['mtp'] for r in bench if r.get('mtp')]
    fmt = lambda v, d: ', '.join(f'{x:.{d}f}' for x in v)
    print(f"| {name} | {payload and payload[1]} | {fmt([r['pp_tok_s'] for r in bench], 1)} | {fmt([r['tg_tok_s'] for r in bench], 2)} | "
          f"{fmt([r['tg_tp_ms']['exchange'] for r in bench], 2)} | {fmt([r['tg_tp_ms']['wait'] for r in bench], 2)} | "
          f"{fmt([m['verify_ms_per_round'] for m in mtp], 1) or '-'} | {fmt([m['tokens_per_round'] for m in mtp], 2) or '-'} | "
          f"{'same' if same else 'DIFF'} | {clocks(err)}, {clocks(os.path.join(invite, f'node-{name}-invite.log'))} |")
