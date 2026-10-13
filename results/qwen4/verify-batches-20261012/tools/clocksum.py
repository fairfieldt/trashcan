#!/usr/bin/env python3
"""Clock log summary: sclk/mclk pairs seen per GPU (excluding idle 300/150), max temperature, per file.
usage: clocksum.py LOG..."""
import collections, sys
for f in sys.argv[1:]:
    pairs = [collections.Counter(), collections.Counter()]
    tmax = [0, 0]
    for line in open(f):
        p = line.split()
        if len(p) < 7 or p[0] == 't':
            continue
        for g in range(2):
            s, m, t = p[1 + 3 * g], p[2 + 3 * g], p[3 + 3 * g]
            if (s, m) != ('300', '150'):
                pairs[g][f"{s}/{m}"] += 1
            tmax[g] = max(tmax[g], int(t))
    cells = [' '.join(f"{k} x{v}" for k, v in c.most_common()) + f" max {t}C" for c, t in zip(pairs, tmax)]
    print(f"{f.split('/')[-1]}: " + ' | '.join(cells))
