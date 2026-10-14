#!/usr/bin/env python3
"""Clock summary of a six-Mac session's logs (scripts/with-clocks.sh format: t sclk0 mclk0 temp0 sclk1 mclk1 temp1):
per node, samples at each GPU's sclk/mclk pair over all runs, and the highest temperature.
usage: nodeclocks.py SESSION_DIR (driver logs TAG-*.txt in it, node logs in NODE/node-*.txt)"""
import collections, glob, os, sys

d = sys.argv[1]
groups = {'driver (incite)': sorted(glob.glob(os.path.join(d, '*.txt')))}
for sub in sorted(os.listdir(d)):
    p = os.path.join(d, sub)
    if os.path.isdir(p) and not sub.endswith(('val14', 'val100')):
        groups[sub] = sorted(f for f in glob.glob(os.path.join(p, 'node-*.txt')))
for name, files in groups.items():
    cnt = [collections.Counter(), collections.Counter()]
    hot = 0
    n = 0
    for f in files:
        if not open(f).readline().startswith('t sclk0'):  # not a clock log
            continue
        for line in open(f):
            x = line.split()
            if len(x) != 7 or x[0] == 't':
                continue
            n += 1
            cnt[0][f'{x[1]}/{x[2]}'] += 1
            cnt[1][f'{x[4]}/{x[5]}'] += 1
            hot = max(hot, int(x[3]), int(x[6]))
    if not n:
        continue
    def fmt(c):
        return ' '.join(f'{k} x{v}' for k, v in c.most_common())
    print(f'{name:16s} {len(files)} logs, {n} samples; GPU0 {fmt(cnt[0])}; GPU1 {fmt(cnt[1])}; max {hot} C')
