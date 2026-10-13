#!/usr/bin/env python3
"""Per-GPU (stage) totals of stage-bench runs: layer ms of each stage (mean of the timed chunks) and, from the
profiled chunk, the per-category ms of each stage. usage: stages.py RUN.json [...]"""
import json, sys, collections
sys.argv, _argv = sys.argv[:1], sys.argv
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from prof import timeline, groups, category
sys.argv = _argv
for path in sys.argv[1:]:
    j = json.load(open(path))
    print(f"== {path.split('/')[-1]} batch {j['batch']}: wall {j['prefill_wall_ms']:.1f}")
    for si, (st, disp) in enumerate(zip(j['prefill'], j['prefill_dispatches'])):
        cats = collections.Counter()
        for layer, labs, s, e in groups(timeline(disp)):
            cats[category(layer if layer is not None else -1, labs) if layer is not None else 'before layers'] += e - s
        print(f"   GPU{si}: layers {sum(st['layer_ms']):6.2f} ms ({', '.join(f'{x:.1f}' for x in st['layer_ms'])}) | "
              + ', '.join(f"{k} {v:.2f}" for k, v in cats.most_common(8)))
