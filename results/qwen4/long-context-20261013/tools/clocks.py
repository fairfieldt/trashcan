#!/usr/bin/env python3
"""Clock and temperature summary of fetched chain runs (runs/NAME/clocks-MAC.txt, clockmon format: t sclk0 mclk0 temp0
sclk1 mclk1 temp1): per Mac and GPU, the samples at each sclk/mclk pair and the maximum temperature.
usage: clocks.py RUN_DIR..."""
import collections, os, sys

for d in sys.argv[1:]:
    print(f"== {os.path.basename(d.rstrip('/'))}")
    for mac in ('incite', 'indite', 'inflight', 'alight', 'invite', 'enlight'):
        f = os.path.join(d, f'clocks-{mac}.txt')
        if not os.path.exists(f):
            continue
        rows = [l.split() for l in open(f) if l[:1].isdigit()]
        out = []
        for g in (0, 1):
            c = collections.Counter(f"{r[1 + 3 * g]}/{r[2 + 3 * g]}" for r in rows if len(r) >= 7)
            tmax = max((int(r[3 + 3 * g]) for r in rows if len(r) >= 7), default=0)
            out.append(f"GPU{g} " + ' '.join(f"{k} x{v}" for k, v in c.most_common(4)) + f" max {tmax}C")
        print(f"  {mac:8s} {len(rows):4d} samples  " + ' | '.join(out))
