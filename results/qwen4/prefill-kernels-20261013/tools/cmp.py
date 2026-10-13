#!/usr/bin/env python3
"""Compare stage-bench runs: wall and layer sums (mean of the timed chunks) and, from the profiled chunk, the ms of
dispatch labels matching any PATTERN (deferred groups as one span).
usage: cmp.py RUN.json [...] [-p PATTERN ...]"""
import json, sys, collections
sys.argv, _argv = sys.argv[:1], sys.argv
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from prof import timeline, groups
sys.argv = _argv
args = sys.argv[1:]
pats = args[args.index('-p') + 1:] if '-p' in args else []
files = args[:args.index('-p')] if '-p' in args else args
for path in files:
    j = json.load(open(path))
    lm = sum(x for s in j['prefill'] for x in s['layer_ms'])
    tot = collections.OrderedDict()
    for stage in j['prefill_dispatches']:
        for layer, labs, s, e in groups(timeline(stage)):
            if layer is None:
                continue
            lab = ' + '.join(l.split()[0] for l in labs)
            if not pats or any(p in lab for p in pats):
                tot[lab] = tot.get(lab, 0) + (e - s)
    print(f"{path.split('/')[-1]:24s} b{j['batch']:<4d} wall {j['prefill_wall_ms']:7.1f} layers {lm:7.1f} | "
          + ', '.join(f"{k} {v:.2f}" for k, v in tot.items()))
