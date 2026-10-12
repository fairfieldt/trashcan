#!/usr/bin/env python3
"""Summarize the prefill-overlap A/B: one line per bench run (prefill tok/s and seconds, decode tok/s) with the clock
states every Mac's GPUs were sampled in during that run's prefill window (driver clock log end minus tg_seconds and
pp_seconds), then per-variant means.
usage: summarize.py DIR   (DIR/incite/ovl-pp*.json, DIR/<mac>/clocks/*.txt)"""
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
for f in sorted(glob.glob(f'{d}/incite/ovl-pp*.json')):
    name = os.path.basename(f)[:-5]
    j = json.load(open(f))
    variant = name.split('-')[2]
    by.setdefault((j['pp'], variant), []).append(j)
    drv = clocks(f'{d}/incite/clocks/{name}.txt')
    hi = drv[-1][0] - j['tg_seconds']
    lo = hi - j['pp_seconds']
    states = Counter()
    for m in macs:
        p = f'{d}/{m}/clocks/{name}.txt' if m == 'incite' else f'{d}/{m}/clocks/node-{name}-{m}.txt'
        if not os.path.exists(p):
            continue
        for t, a, b in clocks(p):
            if lo <= t <= hi:
                states[a] += 1
                states[b] += 1
    print(f"{name:20s} pp{j['pp']:<5d} {j['pp_tok_s']:6.1f} tok/s {j['pp_seconds']:6.2f} s  tg{j['tg']:<3d} {j['tg_tok_s']:6.2f} tok/s  "
          f"prefill-window GPU samples: " + ', '.join(f'{k} x{v}' for k, v in states.most_common()))
for (pp, v), runs in sorted(by.items()):
    p = [r['pp_tok_s'] for r in runs]
    t = [r['tg_tok_s'] for r in runs]
    print(f"pp{pp:<5d} {v:5s} n={len(runs)} prefill mean {statistics.mean(p):6.1f} tok/s (range {min(p):.1f}-{max(p):.1f}, "
          f"{statistics.mean(r['pp_seconds'] for r in runs):.2f} s); decode mean {statistics.mean(t):.2f} tok/s "
          f"(range {min(t):.2f}-{max(t):.2f})")
