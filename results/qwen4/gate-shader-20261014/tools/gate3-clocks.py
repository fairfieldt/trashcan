#!/usr/bin/env python3
"""The clocks (sclk/mclk MHz, samples) of the GPU each gate-check step ran on, from its clock log.

usage: gate3-clocks.py DIR [DIR...]
"""
import collections
import json
import pathlib
import sys

for arg in sys.argv[1:]:
    d = pathlib.Path(arg)
    for p in sorted(d.glob("*.json")):
        text = p.read_text().strip()
        log = d / "clocks" / f"{p.stem}.txt"
        if not log.exists():
            continue
        gpu = json.loads(text.splitlines()[-1])["gpu"] if text else (1 if "gpu1" in p.stem else 0)
        seen = collections.Counter()
        for line in log.read_text().splitlines()[1:]:
            f = line.split()
            if len(f) == 7:
                seen[f"{f[1 + 3 * gpu]}/{f[2 + 3 * gpu]}"] += 1
        print(f"{d.name}/{p.stem} gpu {gpu}: " + ", ".join(f"{k} x{v}" for k, v in sorted(seen.items())))
