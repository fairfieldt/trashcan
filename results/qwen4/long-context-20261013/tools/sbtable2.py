#!/usr/bin/env python3
"""README table from stage-bench JSON pairs (main, new): GPU 0's late 128-token chunk and decode step, all layers and
the QSA layer split into projections with block scoring and selection (qkv + index: the indexer projections moved
from the index phase to the projections' group) and attention.
usage: sbtable2.py 'LABEL|MAC=MAIN.json,NEW.json' ..."""
import json, sys

def load(f):
    r = json.load(open(f))
    out = {}
    for which in ('late_chunk', 'decode'):
        s = r[which][0]
        ph = {}
        for li, p, ms in s['phases']:
            if li == 3:
                ph[p] = ph.get(p, 0.0) + ms
        out[which] = (sum(s['layer_ms']), s['layer_ms'][3], ph.get('qkv', 0) + ph.get('index', 0), ph.get('attention', 0))
    return out

print('| positions | Mac | chunk, GPU 0 | QSA layer | its projections, scoring, selection | its attention '
      '| decode step, GPU 0 | QSA layer | its projections, scoring, selection | its attention |')
print('|---|---|---|---|---|---|---|---|---|---|')
for arg in sys.argv[1:]:
    label, files = arg.split('=', 1)
    pos, mac = label.split('|')
    b, n = (load(f) for f in files.split(','))
    c = lambda w, i, d: f"{b[w][i]:.{d}f} -> {n[w][i]:.{d}f}"
    print(f"| {pos} | {mac} | {c('late_chunk', 0, 1)} | {c('late_chunk', 1, 1)} | {c('late_chunk', 2, 2)} | "
          f"{c('late_chunk', 3, 2)} | {c('decode', 0, 2)} | {c('decode', 1, 2)} | {c('decode', 2, 3)} | {c('decode', 3, 3)} |")
