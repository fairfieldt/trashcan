#!/usr/bin/env python3
"""Prefill/decode table of six-Mac `qwen4 bench` results named PREFIX-ppN-BINARY[-R].json.
usage: ab.py DIR PREFIX BINARY_A BINARY_B [...]"""
import json, sys, os, re, collections
d, prefix, bins = sys.argv[1], sys.argv[2], sys.argv[3:]
res = collections.defaultdict(lambda: collections.defaultdict(list))
for f in sorted(os.listdir(d)):
    m = re.match(rf'{re.escape(prefix)}-pp(\d+)-(.+?)(?:-(\d+))?\.json$', f)
    if not m:
        continue
    pp, b = int(m.group(1)), m.group(2)
    try:
        j = json.load(open(os.path.join(d, f)))
    except Exception:
        continue
    res[pp][b].append((j['pp_tok_s'], j['tg_tok_s'], j['pp_seconds']))
print('| prompt | ' + ' | '.join(f'{b} prefill tok/s (decode)' for b in bins) + ' | change |')
print('|---|' + '---|' * (len(bins) + 1))
for pp in sorted(res):
    cells, means = [], []
    for b in bins:
        v = res[pp].get(b, [])
        cells.append(', '.join(f'{x[0]:.1f}' for x in v) + (' (' + ', '.join(f'{x[1]:.2f}' for x in v) + ')' if v else ''))
        means.append(sum(x[0] for x in v) / len(v) if v else float('nan'))
    ch = f'{(means[-1] / means[0] - 1) * 100:+.1f}%' if len(means) > 1 else ''
    print(f'| {pp} | ' + ' | '.join(cells) + f' | {ch} |')
