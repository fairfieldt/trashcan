#!/usr/bin/env python3
"""The six-Mac session's bench runs: prefill and decode tok/s per run and arm (base, cand), and each size's means.
usage: chain_sum.py DIR TAG"""
import glob, json, os, re, statistics, sys

d, tag = sys.argv[1], sys.argv[2]
rows = {}
for f in sorted(glob.glob(os.path.join(d, f'{tag}-pp*.json'))):
    name = os.path.basename(f)[:-5]
    m = re.match(rf'{tag}-pp(\d+)-(base|cand)(?:-(\d+))?$', name)
    if not m:
        continue
    try:
        j = json.loads(open(f).read().strip().split('\n')[-1])
    except (ValueError, IndexError):
        print(f"{name}: no result")
        continue
    print(f"{name:26s} pp{j['pp']} {j['pp_tok_s']:7.1f} tok/s  tg{j['tg']} {j['tg_tok_s']:6.2f} tok/s")
    rows.setdefault((int(m.group(1)), m.group(2)), []).append((j['pp_tok_s'], j['tg_tok_s']))
print()
for pp in sorted({k[0] for k in rows}):
    b, c = rows.get((pp, 'base'), []), rows.get((pp, 'cand'), [])
    if not b or not c:
        continue
    bt, ct = statistics.mean(x[1] for x in b), statistics.mean(x[1] for x in c)
    bp, cp = statistics.mean(x[0] for x in b), statistics.mean(x[0] for x in c)
    print(f"pp{pp}: decode {bt:.2f} -> {ct:.2f} tok/s ({ct / bt - 1:+.1%}), prefill {bp:.1f} -> {cp:.1f} tok/s "
          f"({cp / bp - 1:+.1%}), n = {len(b)}, {len(c)}")
