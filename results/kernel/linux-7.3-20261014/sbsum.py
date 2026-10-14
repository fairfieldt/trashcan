#!/usr/bin/env python3
"""Summarize run.sh's stage-bench runs per kernel directory: wall time of a 128-token chunk and of a decode step, and
the GPU milliseconds per stage (sum over its layers), each as the mean over the measured runs (sb-1.. ; sb-0 is the
warm-up).
usage: sbsum.py DIR [DIR ...]"""
import glob
import json
import os
import statistics
import sys

for d in sys.argv[1:]:
    runs = []
    for f in sorted(glob.glob(os.path.join(d, "sb-[1-9].json"))):
        try:
            runs.append(json.load(open(f)))
        except (OSError, ValueError) as e:
            print(f"{f}: {e}")
    if not runs:
        print(f"{d}: no runs")
        continue

    def col(get):
        v = [get(r) for r in runs]
        return f"{statistics.mean(v):7.2f} ({min(v):.2f}-{max(v):.2f})"

    print(f"{os.path.basename(os.path.normpath(d))}: {len(runs)} runs")
    print("  chunk of 128, wall ms   ", col(lambda r: r["prefill_wall_ms"]))
    print("  decode step, wall ms    ", col(lambda r: r["decode_wall_ms"]))
    for s in range(len(runs[0]["prefill"])):
        print(f"  stage {s} chunk GPU ms    ", col(lambda r: sum(r["prefill"][s]["layer_ms"])))
        print(f"  stage {s} decode GPU ms   ", col(lambda r: sum(r["decode"][s]["layer_ms"])))
