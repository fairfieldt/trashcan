#!/usr/bin/env python3
"""Instructions per basic block of ACO programs printed by isa.py (RADV_DEBUG=shaders on the null device).
usage: isablocks.py LABEL=FILE.isa ..."""
import re, sys

for arg in sys.argv[1:]:
    label, fn = arg.split('=', 1)
    blocks, cur = [], None
    for line in open(fn).read().split('\n'):
        m = re.match(r'^BB(\d+)', line)
        if m:
            cur = [int(m.group(1)), 0, [], '']
            blocks.append(cur)
            continue
        if cur is None:
            continue
        if line.startswith('/*'):
            cur[3] = line
            continue
        s = line.strip()
        if not s:
            continue
        cur[1] += 1
        op = re.search(r'= (\w+)', s)
        cur[2].append(op.group(1) if op else s.split()[0])
    print(f"== {label}: {len(blocks)} blocks, {sum(b[1] for b in blocks)} instructions")
    for b in blocks:
        kind = re.search(r'kind: ([^*]*)', b[3])
        kind = kind.group(1).strip() if kind else ''
        print(f"BB{b[0]:3d} {b[1]:4d} {kind[:34]:34s} {' '.join(b[2][:6])[:90]}")
    print()
