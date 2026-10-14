#!/usr/bin/env python3
"""Summarize long-position stage-bench JSON: per stage, the prefill chunk at 0, the late chunk and decode (GPU ms of
the layers and the QSA layer's phases), and the late chunk's / decode's slowest dispatches.
usage: sbsum.py FILE.json [--disp N]"""
import json, sys, collections

f = sys.argv[1]
ndisp = int(sys.argv[sys.argv.index('--disp') + 1]) if '--disp' in sys.argv else 12
r = json.load(open(f))
print(f"{f}: batch {r['batch']} late chunk at {r['late_chunk_pos']}, decode from {r['decode_start']}; "
      f"prefill wall {r['prefill_wall_ms']:.1f} ms, decode wall {r['decode_wall_ms']:.2f} ms")
for name in ('prefill', 'late_chunk', 'decode'):
    for si, s in enumerate(r[name]):
        lm = s['layer_ms']
        ph = collections.defaultdict(float)
        for li, p, ms in s['phases']:
            ph[(li, p)] += ms
        qsa = {p: ms for (li, p), ms in ph.items() if li == 3}
        print(f"  {name:10s} stage {si}: layers {' '.join(f'{x:.2f}' for x in lm)} sum {sum(lm):.2f}; QSA layer: "
              + ' '.join(f'{p} {ms:.2f}' for p, ms in qsa.items()))
for name in ('late_dispatches', 'dispatches'):
    for si, d in enumerate(r.get(name, [])):
        # QSA layer dispatches: between 'layer 3' / 'layer 7' and the next layer label.
        tot = collections.defaultdict(float)
        cnt = collections.Counter()
        cur = None
        for label, ms, gap in d:
            if label.startswith('layer '):
                cur = int(label.split()[1])
                continue
            if cur is not None and cur % 4 == 3:
                key = label.split()[0]
                tot[key] += ms
                cnt[key] += 1
        print(f"  {name} stage {si} QSA layer, by kernel (ms): " +
              ', '.join(f'{k} {v:.3f}' for k, v in sorted(tot.items(), key=lambda x: -x[1])[:ndisp]))
