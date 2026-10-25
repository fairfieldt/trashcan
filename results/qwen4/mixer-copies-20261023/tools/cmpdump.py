#!/usr/bin/env python3
"""Compare two stage dumps tensor by tensor: identical tensors, the worst relative errors, and the MoE tensors.
usage: cmpdump.py A B [N]   (B is the reference)
Exits nonzero unless every reference tensor has identical bytes."""
import math, os, sys
from array import array


def load(path):
    a = array('f')
    with open(path, 'rb') as f:
        a.frombytes(f.read())
    return a


da, db = sys.argv[1], sys.argv[2]
top = int(sys.argv[3]) if len(sys.argv) > 3 else 12
names = [line.split('\t')[0] for line in open(os.path.join(db, 'index.tsv'))]
if not names:
    sys.exit('reference dump contains no tensors')
rows, same = [], 0
for n in names:
    pa, pb = os.path.join(da, n + '.f32'), os.path.join(db, n + '.f32')
    if not os.path.exists(pa):
        rows.append((float('inf'), n, 'missing'))
        continue
    a, b = load(pa), load(pb)
    if a.tobytes() == b.tobytes():
        same += 1
        continue
    if len(a) != len(b):
        rows.append((float('inf'), n, f'length {len(a)} != {len(b)}'))
        continue
    num = sum((x - y) ** 2 for x, y in zip(a, b))
    den = sum(y * y for y in b)
    rel = math.sqrt(num / den) if den > 0 else float('inf')
    rows.append((rel, n, ''))
print(f'== {os.path.basename(da.rstrip("/"))} vs {os.path.basename(db.rstrip("/"))}: {same} of {len(names)} tensors identical')
moe = [r for r in rows if r[1].startswith(('ffn_moe_logits_all', 'moe_weights'))]
print(f'   differing router/top-k tensors: {len(moe)} ' + ' '.join(f'{n} {rel:.2e}' for rel, n, _ in moe[:8]))
for rel, n, note in sorted(rows, reverse=True)[:top]:
    print(f'   {n:32s} rel {rel:9.2e} {note}')

sys.exit(1 if rows else 0)
