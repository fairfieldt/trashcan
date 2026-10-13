#!/usr/bin/env python3
"""Whether two `qwen4 mtp spec` outputs (one JSON line per run and prompt) generated the same tokens per prompt, and
where they first differ.
usage: mtp_same.py A.json B.json"""
import json, sys

def load(f):
    out = {}
    for line in open(f):
        if line.strip():
            r = json.loads(line)
            out.setdefault(r['prompt'], []).append(r)
    return out

a, b = load(sys.argv[1]), load(sys.argv[2])
for p in a:
    ia = [r['mtp']['ids'] if 'mtp' in r else r['ids'] for r in a[p]]
    ib = [r['mtp']['ids'] if 'mtp' in r else r['ids'] for r in b.get(p, [])]
    runs_a = all(x == ia[0] for x in ia)
    runs_b = all(x == ib[0] for x in ib)
    if ib and ia[0] == ib[0]:
        print(f"{p}: the same {len(ia[0])} tokens (runs agree: {runs_a}, {runs_b})")
    elif ib:
        k = next((i for i, (x, y) in enumerate(zip(ia[0], ib[0])) if x != y), min(len(ia[0]), len(ib[0])))
        print(f"{p}: first difference at token {k} of {len(ia[0])} (runs agree: {runs_a}, {runs_b})")
