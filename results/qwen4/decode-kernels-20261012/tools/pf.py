#!/usr/bin/env python3
"""Prefill-chunk (batch N) GPU times of stage-bench runs: wall per chunk and per-layer ms, both GPUs."""
import json, sys
for path in sys.argv[1:]:
    j = json.load(open(path))
    lm = [x for s in j['prefill'] for x in s['layer_ms']]
    ph = {}
    for s in j['prefill']:
        for li, name, ms in s['phases']:
            ph[name] = ph.get(name, 0) + ms
    print(f"{path.split('/')[-1]:28s} batch {j['batch']:3d} wall {j['prefill_wall_ms']:7.2f} ms  sum {sum(lm):7.2f}  layers "
          + " ".join(f"{x:.2f}" for x in lm) + "  | " + ", ".join(f"{k} {v:.2f}" for k, v in ph.items()))
