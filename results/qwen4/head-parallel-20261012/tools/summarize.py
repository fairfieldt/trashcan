#!/usr/bin/env python3
"""Summarize the head A/B: one line per bench run (decode tok/s, ms/token, prefill) and the clocks every Mac's GPUs
held during that run's decode window (driver clock log end minus tg_seconds), then per-variant means.
usage: summarize.py DIR   (DIR/incite/head-tg128-*.json, DIR/<mac>/clocks/*.txt)"""
import glob, json, os, statistics, sys
from collections import Counter

d = sys.argv[1]
macs = ['incite', 'indite', 'inflight', 'alight', 'invite', 'enlight']


def clocks(path):
    rows = []
    for line in open(path).read().splitlines()[1:]:
        f = line.split()
        if len(f) == 7:
            rows.append((float(f[0]), f[1] + '/' + f[2], f[4] + '/' + f[5]))
    return rows


by = {}
for f in sorted(glob.glob(f'{d}/incite/head-tg128-*.json')):
    name = os.path.basename(f)[:-5]
    j = json.load(open(f))
    variant = name.split('-')[2]
    by.setdefault(variant, []).append((j['tg_tok_s'], j['pp_tok_s']))
    drv = clocks(f'{d}/incite/clocks/{name}.txt')
    end = drv[-1][0]
    lo = end - j['tg_seconds']
    states = Counter()
    for m in macs:
        p = f'{d}/{m}/clocks/{name}.txt' if m == 'incite' else f'{d}/{m}/clocks/node-{name}-{m}.txt'
        if not os.path.exists(p):
            continue
        for t, a, b in clocks(p):
            if lo <= t <= end:
                states[a] += 1
                states[b] += 1
    print(f"{name:22s} tg128 {j['tg_tok_s']:6.2f} tok/s {1000 / j['tg_tok_s']:6.2f} ms/token  pp512 {j['pp_tok_s']:6.1f} tok/s  "
          f"decode-window GPU samples: " + ', '.join(f'{k} x{v}' for k, v in states.most_common()))
for v, runs in by.items():
    tg = [r[0] for r in runs]
    pp = [r[1] for r in runs]
    print(f"{v:5s} n={len(tg)} tg mean {statistics.mean(tg):.3f} tok/s ({1000 / statistics.mean(tg):.2f} ms/token) "
          f"range {min(tg):.2f}-{max(tg):.2f}; pp mean {statistics.mean(pp):.1f}")
