#!/usr/bin/env python3
"""Mean GPU microseconds per dispatch label over the DeltaNet layers 5, 6 and the QSA layer 7 (stage 1 of a
stage-bench run with TRASHCAN_Q4_PROFILE=1). Deferred groups overlap: a group's members are listed with '+'."""
import json, sys, collections

def layer_dispatches(disp):
    layers, cur = {}, None
    for lab, ms, gap in disp:
        if lab.startswith('layer '):
            cur = int(lab.split()[1]); layers[cur] = []
            continue
        if cur is not None:
            layers[cur].append((lab, ms * 1000, gap * 1000))
    return layers

for path in sys.argv[1:]:
    j = json.load(open(path))
    disp = j['dispatches'][1]
    L = layer_dispatches(disp)
    print(f"== {path}")
    for group, ids in (('DeltaNet (layers 5, 6)', [5, 6]), ('QSA (layer 7)', [7])):
        rows = collections.OrderedDict()
        for li in ids:
            for k, (lab, us, gap) in enumerate(L[li]):
                key = (k, lab.split(' ')[0] + ' ' + (lab.split(' ')[1] if ' ' in lab else ''))
                rows.setdefault(key, []).append((us, gap if gap < 1e6 else float('nan')))
        tot = 0
        print(f"  {group}:")
        for (k, lab), v in rows.items():
            us = sum(x[0] for x in v) / len(v)
            gap = sum(x[1] for x in v) / len(v)
            print(f"    {lab:42s} {us:7.1f} us   gap {gap:5.1f}")
        print(f"    layer ms: {[round(x, 3) for x in j['decode'][1]['layer_ms']]}")
