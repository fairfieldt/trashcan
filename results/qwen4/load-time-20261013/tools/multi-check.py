#!/usr/bin/env python3
"""Compare stage-bench timings of several --batch sizes on one load (one JSON line per size) with separate runs.
usage: multi-check.py MULTI.json SINGLE-16.json SINGLE-32.json ..."""
import json, sys

multi = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
single = {}
for p in sys.argv[2:]:
    lines = [l for l in open(p) if l.strip()]
    assert len(lines) == 1, p
    d = json.loads(lines[0])
    single[d['batch']] = d


def stage_ms(d, key):
    return [round(sum(s['layer_ms']), 3) for s in d[key]]


print('batch  prefill wall ms (one load / separate)  prefill GPU ms per stage  late chunk GPU ms per stage  decode wall ms')
for m in multi:
    b = m['batch']
    s = single.get(b)
    if s is None:
        continue
    print(f"{b:5d}  {m['prefill_wall_ms']:8.2f} / {s['prefill_wall_ms']:8.2f}   "
          f"{stage_ms(m, 'prefill')} / {stage_ms(s, 'prefill')}   "
          f"{stage_ms(m, 'late_chunk')} / {stage_ms(s, 'late_chunk')}   "
          f"{m['decode_wall_ms']:6.2f} / {s['decode_wall_ms']:6.2f}")
