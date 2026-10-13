#!/usr/bin/env python3
"""One line per bench run (decode tok/s, ms per token, prefill) with the clocks every Mac's GPUs held in that run's
decode window (the driver's clock log end minus tg_seconds), then per-variant means. With --power LOG, also each
Mac's mean SMC PSTR watts over the decode window from the netconsole telemetry (CRASHCAPTURE lines, local time).
usage: summary.py DIR PATTERN [--power NETCONSOLE_LOG]    e.g. summary.py . 'host-s3-*' (run names in DIR/incite/)"""
import argparse, datetime, glob, json, os, re, statistics
from collections import Counter, defaultdict

p = argparse.ArgumentParser()
p.add_argument('dir')
p.add_argument('pattern')
p.add_argument('--power')
a = p.parse_args()
macs = ['incite', 'indite', 'inflight', 'alight', 'invite', 'enlight']


def clocks(path):
    rows = []
    for line in open(path).read().splitlines()[1:]:
        f = line.split()
        if len(f) == 7:
            rows.append((float(f[0]), f[1] + '/' + f[2], f[4] + '/' + f[5]))
    return rows


def power(log, lo, hi):
    """Mean PSTR watts per host between two epoch times."""
    w = defaultdict(list)
    t0 = datetime.datetime.fromtimestamp(lo).strftime('%Y-%m-%dT%H:%M:%S')
    t1 = datetime.datetime.fromtimestamp(hi).strftime('%Y-%m-%dT%H:%M:%S')
    for line in open(log, errors='replace'):
        if 'CRASHCAPTURE' not in line or 'telemetry' not in line or not (t0 <= line[:19] <= t1):
            continue
        d = json.loads(line.split('CRASHCAPTURE ', 1)[1])
        v = ((d.get('smc') or {}).get('PSTR') or {}).get('watts')
        if v is not None:
            w[d['host']].append(v)
    return {h: statistics.mean(v) for h, v in w.items()}


by = defaultdict(list)
for f in sorted(glob.glob(f'{a.dir}/incite/{a.pattern}.json')):
    name = os.path.basename(f)[:-5]
    try:
        j = json.load(open(f))
    except json.JSONDecodeError:
        continue
    if 'tg_tok_s' not in j:
        continue
    variant = re.sub(r'-\d+$', '', name)
    drv = clocks(f'{a.dir}/incite/clocks/{name}.txt')
    end = drv[-1][0]
    lo = end - j['tg_seconds']
    states = Counter()
    for m in macs:
        path = f'{a.dir}/{m}/clocks/{name}.txt' if m == 'incite' else f'{a.dir}/{m}/clocks/node-{name}-{m}.txt'
        if os.path.exists(path):
            for t, g0, g1 in clocks(path):
                if lo <= t <= end:
                    states[g0] += 1
                    states[g1] += 1
    watts = power(a.power, lo + 1, end - 1) if a.power else {}
    by[variant].append((j['tg_tok_s'], j['pp_tok_s'], sum(watts.values()) if len(watts) == 6 else None))
    print(f"{name:24s} tg{j['tg']} {j['tg_tok_s']:6.2f} tok/s {1000 / j['tg_tok_s']:6.2f} ms/token  pp{j['pp']} "
          f"{j['pp_tok_s']:6.1f} tok/s  GPU clocks in the decode window: " + ', '.join(f'{k} x{v}' for k, v in states.most_common())
          + (f"  PSTR W: " + ' '.join(f'{h} {watts[h]:.0f}' for h in macs if h in watts) if watts else ''))
for v, runs in by.items():
    tg = [r[0] for r in runs]
    pw = [r[2] for r in runs if r[2] is not None]
    print(f"{v:22s} n={len(tg)} tg mean {statistics.mean(tg):.3f} tok/s ({1000 / statistics.mean(tg):.2f} ms/token) "
          f"range {min(tg):.2f}-{max(tg):.2f}; pp mean {statistics.mean([r[1] for r in runs]):.1f}"
          + (f"; six Macs' PSTR in decode {statistics.mean(pw):.0f} W" if pw else ''))
