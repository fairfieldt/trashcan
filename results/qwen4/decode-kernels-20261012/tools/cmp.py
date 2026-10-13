#!/usr/bin/env python3
"""Compare dispatch profiles of several stage-bench runs: stage 1 (layers 4-7), one column per run.
Dispatch times in a deferred group overlap: the group's time is its last member's."""
import json, sys

runs = [(p.split('/')[-1].replace('.json', ''), json.load(open(p))) for p in sys.argv[1:]]
print(f"{'':44s}" + "".join(f"{n[-12:]:>13s}" for n, _ in runs))
for si in (0, 1):
    rows = []
    for n, j in runs:
        d = j.get('dispatches', [])
        rows.append(d[si] if len(d) > si else [])
    if not any(rows):
        continue
    L = max(len(r) for r in rows)
    for i in range(L):
        labs = [r[i][0] if i < len(r) else '' for r in rows]
        print(f"{labs[0][:44]:44s}" + "".join(f"{(r[i][1]*1000 if i < len(r) else 0):13.1f}" for r in rows)
              + ("" if len(set(labs)) == 1 else "   | " + " / ".join(l[:30] for l in labs[1:])))
for n, j in runs:
    tot = [sum(s['layer_ms']) for s in j['decode']]
    ph = {}
    for s in j['decode']:
        for li, name, ms in s['phases']:
            ph.setdefault(name, []).append(ms)
    print(f"{n:24s} stage ms {' '.join(f'{t:.3f}' for t in tot)}  layers " +
          " ".join(f"{x:.3f}" for s in j['decode'] for x in s['layer_ms']) + "  phases " +
          ", ".join(f"{k} {sum(v)/len(v):.3f}" for k, v in ph.items()))
