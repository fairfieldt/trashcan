#!/usr/bin/env python3
"""Routing statistics from a stage dump of the experiment build (a temporary capture, since removed, wrote
moe_ids_all-L.f32, u32 expert ids [t][slots]): per layer the distinct experts, the slots per used routed
expert, and the tiles of 8 / 4 slots.
usage: route_stats.py DUMP_DIR [SLOTS]"""
import collections, glob, os, re, struct, sys

d = sys.argv[1]
slots = int(sys.argv[2]) if len(sys.argv) > 2 else 11
tot = collections.Counter()
for path in sorted(glob.glob(os.path.join(d, 'moe_ids_all-*.f32')), key=lambda p: int(re.findall(r'-(\d+)\.f32', p)[0])):
    raw = open(path, 'rb').read()
    ids = struct.unpack(f'<{len(raw) // 4}I', raw)
    n = len(ids) // slots
    c = collections.Counter(e for j, e in enumerate(ids) if j % slots != slots - 1)
    hist = collections.Counter(c.values())
    tiles8 = sum((v + 7) // 8 for v in c.values()) + (n + 7) // 8
    tiles4 = sum((v + 3) // 4 for v in c.values()) + (n + 3) // 4
    tot.update(c.values())
    print(f'{os.path.basename(path)}: n={n} distinct={len(c)} mean={sum(c.values()) / len(c):.2f} max={max(c.values())} '
          f'tiles8={tiles8} tiles4={tiles4} hist={dict(sorted(hist.items()))}')
w = sum(k * v for k, v in tot.items())
print('slots by expert count:', {k: round(k * v / w, 3) for k, v in sorted(tot.items())})
