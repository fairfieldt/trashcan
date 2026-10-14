#!/usr/bin/env python3
"""Compare two stage dumps (every token's rows, TRASHCAN_Q4_DUMP_ALL=1): per tensor, bit-identical or the relative L2
error and the worst row; the selected blocks (indexer_sel-*) as u32 sets per token.
usage: dumpcmp.py DIR_A DIR_B [--width-sel 512]"""
import sys, os, math, struct
from array import array

a, b = sys.argv[1], sys.argv[2]
def load(d, name, typ='f'):
    x = array(typ)
    with open(os.path.join(d, name + '.f32'), 'rb') as f:
        x.frombytes(f.read())
    return x
names = [l.split('\t')[0] for l in open(os.path.join(b, 'index.tsv'))]
for name in names:
    if not os.path.exists(os.path.join(a, name + '.f32')):
        print(f"{name:28s} missing in {a}")
        continue
    if name.startswith('indexer_sel'):
        x, y = load(a, name, 'I'), load(b, name, 'I')
        m = 512
        rows = len(y) // m
        bad = [r for r in range(rows) if list(x[r * m:(r + 1) * m]) != list(y[r * m:(r + 1) * m])]
        sets = [r for r in bad if set(x[r * m:(r + 1) * m]) != set(y[r * m:(r + 1) * m])]
        print(f"{name:28s} {rows} tokens: {len(bad)} rows differ ({len(sets)} as sets)"
              + (f"; first {bad[0]}: {sorted(set(x[bad[0]*m:(bad[0]+1)*m]) ^ set(y[bad[0]*m:(bad[0]+1)*m]))[:10]}" if bad else ""))
        continue
    x, y = load(a, name), load(b, name)
    if len(x) != len(y):
        print(f"{name:28s} length {len(x)} vs {len(y)}")
        continue
    if x.tobytes() == y.tobytes():
        print(f"{name:28s} identical ({len(y)} values)")
        continue
    num = sum((p - q) ** 2 for p, q in zip(x, y))
    den = sum(q * q for q in y) or 1.0
    diff = sum(1 for p, q in zip(x, y) if p != q)
    mx = max(abs(p - q) for p, q in zip(x, y))
    print(f"{name:28s} rel {math.sqrt(num / den):.3e}  max abs {mx:.3e}  {diff}/{len(y)} values differ")
