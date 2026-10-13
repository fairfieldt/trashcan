#!/usr/bin/env python3
"""Expert sharing inside windows of consecutive tokens (MTP verification batches of 2-4 tokens) from stage dumps
holding u32 expert ids [t][slots] (moe_ids_all-L.f32, slot 10 = the shared expert).
usage: share_stats.py DUMP_DIR [DUMP_DIR...]"""
import collections, glob, os, re, struct, sys

slots = 11
for d in sys.argv[1:]:
    print(f'## {d}')
    tot = collections.defaultdict(lambda: [0, 0, 0])
    for path in sorted(glob.glob(os.path.join(d, 'moe_ids_all-*.f32')), key=lambda p: int(re.findall(r'-(\d+)\.f32', p)[0])):
        raw = open(path, 'rb').read()
        ids = struct.unpack(f'<{len(raw) // 4}I', raw)
        n = len(ids) // slots
        rows = [ids[t * slots:t * slots + slots - 1] for t in range(n)]
        line = []
        for w in (2, 3, 4):
            slots_all, distinct, wins = 0, 0, 0
            for t0 in range(0, n - w + 1):
                win = rows[t0:t0 + w]
                slots_all += sum(len(r) for r in win)
                distinct += len(set(e for r in win for e in r))
                wins += 1
            tot[w][0] += slots_all; tot[w][1] += distinct; tot[w][2] += wins
            line.append(f'w{w}: {distinct / wins:.1f} distinct of {slots_all // wins} routed ({1 - distinct / slots_all:.0%} shared)')
        print(f'  {os.path.basename(path)} n={n}: ' + '; '.join(line))
    for w, (s, dd, nw) in sorted(tot.items()):
        print(f'  all layers, windows of {w}: {dd / nw:.2f} distinct routed experts of {s / nw:.0f} '
              f'({1 - dd / s:.1%} of routed slots reuse an expert of an earlier token in the window)')
