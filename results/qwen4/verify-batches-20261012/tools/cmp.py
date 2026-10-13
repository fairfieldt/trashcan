#!/usr/bin/env python3
"""Side-by-side per-dispatch microseconds of one layer (stage 1 by default) across stage-bench runs, matched by the
dispatch's position in the layer, for decode (n=1) or the prefill chunk.
usage: cmp.py [--chunk] [--layer L] [--stage S] JSON..."""
import json, sys, collections
args = sys.argv[1:]
chunk, layer, stage = False, 5, 1
while args and args[0].startswith('--'):
    if args[0] == '--chunk': chunk = True; args = args[1:]
    elif args[0] == '--layer': layer = int(args[1]); args = args[2:]
    elif args[0] == '--stage': stage = int(args[1]); args = args[2:]
cols = []
for path in args:
    j = json.load(open(path))
    disp = (j['prefill_dispatches'] if chunk else j['dispatches'])[stage]
    out, cur = [], None
    for lab, ms, gap in disp:
        if lab.startswith('layer '):
            cur = int(lab.split()[1]); continue
        if cur == layer:
            out.append((lab, ms * 1000))
    cols.append((path.split('/')[-1].replace('.json', ''), out))
n = max(len(c[1]) for c in cols)
print(f"layer {layer} {'chunk' if chunk else 'decode'}: " + " | ".join(name for name, _ in cols))
tot = [0.0] * len(cols)
for k in range(n):
    cells = []
    for i, (name, out) in enumerate(cols):
        if k < len(out):
            lab, us = out[k]
            tot[i] += us
            cells.append(f"{lab.split(' ')[0][6:30]:24s} {us:6.1f}")
        else:
            cells.append(' ' * 31)
    print('  ' + ' | '.join(cells))
print('  sum ' + ' | '.join(f"{t:30.1f}" for t in tot))
