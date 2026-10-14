#!/usr/bin/env python3
"""Per-dispatch first and second copy (TRASHCAN_Q4_DIAG=twice) of a profiled stage-bench run, per layer: the first copy
runs with its code cold, the second right after it (code warm in the instruction cache if the barrier keeps it).
usage: twice.py RUN.json [--layers 5,7] [--stats STATS.txt --vulkan VULKAN_RS]"""
import json, sys, argparse, os
sys.path.insert(0, os.path.dirname(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('run')
ap.add_argument('--layers', default='')
ap.add_argument('--stats')
ap.add_argument('--vulkan')
a = ap.parse_args()
import importlib.util
spec = importlib.util.spec_from_file_location('prof', os.path.join(os.path.dirname(__file__), 'prof.py'))
src = open(os.path.join(os.path.dirname(__file__), 'prof.py')).read().replace('\nmain()\n', '\n')
ns = {}
exec(compile(src, 'prof.py', 'exec'), ns)
code = ns['code_sizes'](a.stats, a.vulkan) if a.stats else {}
run = json.loads(open(a.run).read().strip().split('\n')[-1])
want = [int(x) for x in a.layers.split(',') if x]
tot1 = tot2 = 0
for stage in ns['layers'](run):
    for layer, ds in stage:
        if want and layer not in want:
            continue
        print(f"\n== layer {layer}")
        print(f"{'dispatch':44s} {'first':>7s} {'second':>7s} {'diff':>6s} {'code':>6s} {'us/KB':>6s}")
        i = 0
        while i < len(ds):
            label, us, gap = ds[i]
            if i + 1 < len(ds) and ds[i + 1][0] == label + ' #2':
                us2 = ds[i + 1][1]
                c = code.get(label.split()[0], 0)
                d = us - us2
                tot1 += us
                tot2 += us2
                print(f"{label[:44]:44s} {us:7.1f} {us2:7.1f} {d:6.1f} {c:6d} {d / (c / 1024) if c else 0:6.1f}")
                i += 2
            else:
                i += 1
print(f"\nsum of first copies {tot1:.1f} us, of second copies {tot2:.1f} us")
