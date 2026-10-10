#!/usr/bin/env python3
"""Tabulate qwen-matvec-bench JSON lines: shape rows, kernel columns, µs and % vs the first kernel."""
import json, sys
for f in sys.argv[1:]:
    rows = {}
    for l in open(f):
        d = json.loads(l)
        rows.setdefault((d['rows'], d['k'], d['group']), {})[d['kernel']] = (d['us'], d['mismatches'])
    ks = []
    for r in rows.values():
        for k in r:
            if k not in ks:
                ks.append(k)
    print(f'{f}\nshape | ' + ' | '.join(ks))
    for sh, r in rows.items():
        ref = r[ks[0]][0]
        print(f'{sh[0]}x{sh[1]} g{sh[2]} | ' + ' | '.join(f"{r[k][0]:.1f} ({(r[k][0]/ref-1)*100:+.0f}%){' MISMATCH' if r[k][1] else ''}" if k in r else '-' for k in ks))
