#!/usr/bin/env python3
"""Print the dispatch timeline (us from segment begin) of the first segment of each kind in one decode step.

usage: tp_timeline.py PROFILE.jsonl RANK STEP
"""
import json
import sys

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from tp_profile import segments, kind  # noqa: E402

path, rank, step = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
labels = {}
for line in open(path):
    d = json.loads(line)
    if "labels" in d:
        labels[(d["rank"], d["graph"])] = d["labels"]
    if d["rank"] != rank or d["n"] != 1 or len(d["release"]) <= 3:
        continue
    if d["step"] < step:
        continue
    segs = segments(labels[(d["rank"], d["graph"])], d["ns"])
    seen = set()
    for i, s in enumerate(segs):
        k = kind(i, len(segs), 4)
        if k in seen:
            continue
        seen.add(k)
        print(f"segment {i} ({k}): {(s['end'] - s['begin']) / 1e3:.1f} us")
        for lab, t0, t1 in s["spans"]:
            print(f"   {lab:32s} {(t0 - s['begin']) / 1e3:8.1f} {(t1 - s['begin']) / 1e3:8.1f}  ({(t1 - t0) / 1e3:6.1f})")
    break
