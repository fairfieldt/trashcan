#!/usr/bin/env python3
"""Decode wall and per-layer GPU ms of stage-bench JSON files.
usage: walls.py RUN.json..."""
import json, sys
for f in sys.argv[1:]:
    try:
        r = json.loads(open(f).read().strip().split('\n')[-1])
    except Exception as e:
        print(f"{f}: {e}")
        continue
    for i, s in enumerate(r['decode']):
        lm = s['layer_ms']
        print(f"{f:40s} stage {i} wall {r['decode_wall_ms']:.3f} ms  layers {[round(x, 3) for x in lm]}  sum {sum(lm):.3f}")
