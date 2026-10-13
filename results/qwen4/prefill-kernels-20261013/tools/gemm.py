#!/usr/bin/env python3
"""Summarize qwen-gemm-bench JSON lines: time, share of the D700's FMA peak at 850 MHz, bit mismatches against the
first kernel. usage: gemm.py FILE.jsonl [...]"""
import json, sys
peak = 850e6 * 2048 * 2 / 1e12
for path in sys.argv[1:]:
    print(f"== {path.split('/')[-1]}")
    for l in open(path):
        j = json.loads(l)
        print(f"  {j['rows']:6d}x{j['k']:5d} n={j['n']:4d} {j['kernel']:30s} {j['ms'] * 1000:8.1f} us "
              f"{j['tflops'] / peak * 100:5.1f}%  mism {j['mismatches']} maxd {j['max_diff']:.1e}")
