#!/usr/bin/env python3
"""Mean SMC power components per Mac between two local times: PSTR (the whole Mac), PCPC (CPU package), PG0R and
PG1R (the two GPUs), from the netconsole telemetry (CRASHCAPTURE lines).
usage: components.py NETCONSOLE_LOG DATE START END    e.g. components.py net.log 2026-10-13 15:26:00 15:26:13"""
import json, statistics, sys
from collections import defaultdict

log, day, start, end = sys.argv[1:5]
t0, t1 = f'{day}T{start}', f'{day}T{end}'
keys = ['PSTR', 'PCPC', 'PG0R', 'PG1R']
w = defaultdict(lambda: defaultdict(list))
for line in open(log, errors='replace'):
    if 'CRASHCAPTURE' not in line or 'telemetry' not in line or not (t0 <= line[:19] <= t1):
        continue
    d = json.loads(line.split('CRASHCAPTURE ', 1)[1])
    smc = d.get('smc') or {}
    for k in keys:
        v = (smc.get(k) or {}).get('watts')
        if v is not None:
            w[d['host']][k].append(v)
total = defaultdict(float)
for h in sorted(w):
    means = {k: statistics.mean(w[h][k]) for k in keys if w[h][k]}
    for k, v in means.items():
        total[k] += v
    print(f'{h:9s} ' + ' '.join(f'{k} {means[k]:6.1f}' for k in keys if k in means))
print(f'{"six Macs":9s} ' + ' '.join(f'{k} {total[k]:6.1f}' for k in keys))
